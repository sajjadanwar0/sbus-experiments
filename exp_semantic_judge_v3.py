import csv
import json
import os
import sys
import uuid
import time
import argparse
import socket
from dataclasses import dataclass, asdict
from urllib.request import Request, ProxyHandler, build_opener
from urllib.parse import urlencode
from urllib.error import HTTPError

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai"); sys.exit(1)

try:
    import anthropic as _anthropic_test
except ImportError:
    _anthropic_test = None

SBUS_URL    = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE    = "gpt-4o-mini"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

COND_FRESH    = "structural_fresh"
COND_STALE    = "structural_stale"
COND_NO_ORI   = "no_ori"

_opener = build_opener(ProxyHandler({}))

def http_get(url: str, params: dict = None) -> tuple[int, dict]:
    if params:
        url += "?" + urlencode(params)
    try:
        with _opener.open(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def http_post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def reset_bus() -> None:
    status, resp = http_post(f"{SBUS_URL}/admin/reset", {})
    if status != 200:
        print(f"  WARNING: reset returned {status}: {resp}")
    time.sleep(0.3)


def inject_stale(agent_id: str, key: str, stale_version: int) -> bool:
    status, resp = http_post(f"{SBUS_URL}/admin/inject-stale", {
        "agent_id":      agent_id,
        "key":           key,
        "stale_version": stale_version,
    })
    return status == 200


def health_check() -> bool:
    try:
        s = socket.create_connection(("localhost", 7000), timeout=3)
        s.close()
    except Exception:
        return False
    status, body = http_get(f"{SBUS_URL}/admin/health")
    return status == 200

JUDGE_PROMPT = """\
You are a strict technical evaluator assessing whether a software engineering
agent has successfully addressed a code task.

TASK DESCRIPTION:
{task_desc}

FINAL SHARD CONTENT (agent output to evaluate):
{content}

Evaluate whether the content meaningfully and correctly addresses the task.
Consider:
  1. Does it identify the correct technical problem described in the task?
  2. Does it propose a concrete, technically plausible solution?
  3. Is the proposed solution consistent with the codebase context (Django/astropy/sympy)?
  4. Is the content internally consistent (no contradictory approaches)?

Reply with EXACTLY one of:
  CORRECT     — content adequately addresses the task
  INCOMPLETE  — content is on the right track but lacks key detail
  CORRUPTED   — content is wrong, contradictory, or addresses the wrong problem

Then on a new line: one sentence explaining your verdict.

IMPORTANT: Judge only the semantic correctness, not stylistic quality."""


def judge_content(
    task_desc: str,
    content: str,
    oai: OpenAI,
    use_claude: bool = False,
) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(
        task_desc=task_desc[:400],
        content=content[:600] if content else "[empty — no content produced]",
    )
    if use_claude and _anthropic_test is not None:
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=120,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
        except Exception as e:
            text = f"INCOMPLETE\nJudge error: {e}"
    else:
        try:
            r = oai.chat.completions.create(
                model=BACKBONE,
                max_tokens=120,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a strict code reviewer. Reply CORRECT, INCOMPLETE, or CORRUPTED then a newline then one sentence."},
                    {"role": "user",   "content": prompt},
                ],
            )
            text = r.choices[0].message.content.strip()
        except Exception as e:
            text = f"INCOMPLETE\nJudge error: {e}"

    lines = text.strip().split("\n", 1)
    verdict_raw = lines[0].strip().upper()
    if "CORRECT" in verdict_raw and "IN" not in verdict_raw:
        verdict = "CORRECT"
    elif "CORRUPT" in verdict_raw:
        verdict = "CORRUPTED"
    else:
        verdict = "INCOMPLETE"
    reason = lines[1].strip() if len(lines) > 1 else ""
    return verdict, reason

def agent_step(
    oai: OpenAI,
    agent_id: str,
    shard_key: str,
    task_desc: str,
    step: int,
    context: str,
    expected_version: int,
    condition: str,
    use_occ: bool,
) -> tuple[bool, int]:
    try:
        resp = oai.chat.completions.create(
            model=BACKBONE,
            max_tokens=120,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a software engineering agent working on this task:\n"
                    f"TASK: {task_desc[:300]}\n\n"
                    f"Current state of the shared design document (step {step}):\n"
                    f"{context[:400]}\n\n"
                    f"Write exactly ONE concrete technical improvement or correction "
                    f"to the design document. Be specific. Output ONLY the change, "
                    f"no preamble."
                ),
            }],
        )
        delta = f"[{agent_id} step{step}] {resp.choices[0].message.content.strip()}"
    except Exception as e:
        delta = f"[{agent_id} step{step}] ERROR: {e}"

    if use_occ:
        commit_url = f"{SBUS_URL}/commit/v2"
        payload = {
            "key":              shard_key,
            "expected_version": expected_version,
            "delta":            delta,
            "agent_id":         agent_id,
            "read_set":         [{"key": shard_key, "version_at_read": expected_version}],
        }
    else:
        commit_url = f"{SBUS_URL}/commit"
        payload = {
            "key":              shard_key,
            "expected_version": 0,
            "delta":            delta,
            "agent_id":         agent_id,
        }

    status, data = http_post(commit_url, payload)
    succeeded = (status == 200)
    new_version = data.get("new_version", expected_version) if succeeded else expected_version
    return succeeded, new_version

def run_condition_a_fresh(
    oai: OpenAI,
    task_id: str,
    task_desc: str,
    n_agents: int,
    n_steps: int,
) -> tuple[str, int, int]:
    reset_bus()
    run_id = uuid.uuid4().hex[:8]
    shard  = f"design_{run_id}"
    agents = [f"agent_{i}_{run_id}" for i in range(n_agents)]

    http_post(f"{SBUS_URL}/shard", {
        "key":      shard,
        "content":  f"Initial design document for task: {task_id}",
        "goal_tag": task_id,
    })
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})

    commits_ok = 0
    commits_total = 0

    for step in range(n_steps):
        for agent in agents:
            status, data = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": agent})
            if status != 200:
                continue
            current_version = data.get("version", 0)
            current_content = data.get("content", "")

            ok, _ = agent_step(
                oai, agent, shard, task_desc, step,
                context=current_content,
                expected_version=current_version,
                condition=COND_FRESH,
                use_occ=True,
            )
            commits_total += 1
            if ok:
                commits_ok += 1

    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", ""), commits_ok, commits_total


def run_condition_b_stale(
    oai: OpenAI,
    task_id: str,
    task_desc: str,
    n_agents: int,
    n_steps: int,
    injection_step: int,
) -> tuple[str, int, int, int]:
    reset_bus()
    run_id = uuid.uuid4().hex[:8]
    shard  = f"design_{run_id}"
    agents = [f"agent_{i}_{run_id}" for i in range(n_agents)]

    http_post(f"{SBUS_URL}/shard", {
        "key":      shard,
        "content":  f"Initial design document for task: {task_id}",
        "goal_tag": task_id,
    })
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})

    _, snap = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "snapshot"})
    stale_content = snap.get("content", "Initial design document")
    stale_version = snap.get("version", 0)  # = 0

    commits_ok = 0
    commits_total = 0
    n_injections = 0
    stale_agent = agents[0]

    for step in range(n_steps):
        for agent in agents:
            status, data = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": agent})
            if status != 200:
                continue
            current_version = data.get("version", 0)

            if agent == stale_agent and step >= injection_step:
                context_used = (
                    f"{stale_content[:300]}\n"
                    f"[NOTE: This agent is working from its memory/context, "
                    f"not from the freshly fetched state. It may be unaware of "
                    f"changes made by other agents since step 0.]"
                )
                inject_stale(agent, shard, stale_version)
                n_injections += 1
                ok, _ = agent_step(
                    oai, agent, shard, task_desc, step,
                    context=context_used,
                    expected_version=current_version,
                    condition=COND_STALE,
                    use_occ=True,
                )
            else:
                ok, _ = agent_step(
                    oai, agent, shard, task_desc, step,
                    context=data.get("content", ""),
                    expected_version=current_version,
                    condition=COND_STALE,
                    use_occ=True,
                )
            commits_total += 1
            if ok:
                commits_ok += 1

    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", ""), commits_ok, commits_total, n_injections


def run_condition_c_no_ori(
    oai: OpenAI,
    task_id: str,
    task_desc: str,
    n_agents: int,
    n_steps: int,
) -> tuple[str, int, int]:
    reset_bus()
    run_id = uuid.uuid4().hex[:8]
    shard  = f"design_{run_id}"
    agents = [f"agent_{i}_{run_id}" for i in range(n_agents)]

    http_post(f"{SBUS_URL}/shard", {
        "key":      shard,
        "content":  f"Initial design document for task: {task_id}",
        "goal_tag": task_id,
    })

    commits_ok = 0
    commits_total = 0

    for step in range(n_steps):
        for agent in agents:
            status, data = http_get(f"{SBUS_URL}/shard/{shard}", {})
            if status != 200:
                continue
            current_content = data.get("content", "")

            try:
                resp = oai.chat.completions.create(
                    model=BACKBONE, max_tokens=120, temperature=0.3,
                    messages=[{"role": "user", "content": (
                        f"TASK: {task_desc[:300]}\n"
                        f"Current state:\n{current_content[:400]}\n"
                        f"Agent {agent}: write ONE concrete technical improvement. "
                        f"No preamble, just the change."
                    )}],
                )
                delta = f"[{agent} step{step}] {resp.choices[0].message.content.strip()}"
            except Exception as e:
                delta = f"[{agent} step{step}] ERROR: {e}"

            status, data = http_post(f"{SBUS_URL}/commit", {
                "key":              shard,
                "expected_version": 0,
                "delta":            delta,
                "agent_id":         agent,
            })
            commits_total += 1
            if status == 200:
                commits_ok += 1

    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", ""), commits_ok, commits_total

@dataclass
class SJv3Result:
    run_id:          str
    task_id:         str
    run_idx:         int
    condition:       str
    commits_ok:      int
    commits_total:   int
    commit_rate:     float
    n_stale_injected: int
    verdict:         str
    is_corrupted:    bool
    is_complete:     bool
    judge_reason:    str
    surviving_content: str
    wall_secs:       float

def main():
    parser = argparse.ArgumentParser(description="Exp. SJ-v3: Semantic Judge (fully powered)")
    parser.add_argument("--tasks",          default="datasets/tasks_30_multidomain.json")
    parser.add_argument("--n-tasks",        type=int, default=5,
                        help="Number of tasks to use (prefer Django tasks)")
    parser.add_argument("--n-runs",         type=int, default=25,
                        help="Runs per condition per task (n=25 → ~375 total; n=50 → ~750)")
    parser.add_argument("--n-agents",       type=int, default=4)
    parser.add_argument("--n-steps",        type=int, default=20,
                        help="Steps per agent (≥20 recommended for task completion)")
    parser.add_argument("--injection-step", type=int, default=8,
                        help="Step at which to inject stale context (Condition B)")
    parser.add_argument("--conditions",     nargs="+",
                        default=[COND_FRESH, COND_STALE, COND_NO_ORI],
                        help="Conditions to run")
    parser.add_argument("--use-claude-judge", action="store_true",
                        help="Use Claude Haiku as judge (matches paper Exp. B; requires ANTHROPIC_API_KEY)")
    parser.add_argument("--output",         default="results/sj_v3_results.csv")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)

    if args.use_claude_judge and not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: --use-claude-judge set but ANTHROPIC_API_KEY not set; falling back to GPT-4o-mini")
        args.use_claude_judge = False

    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}")
        print("  Start with: SBUS_ADMIN_ENABLED=1 cargo run --release")
        sys.exit(1)

    oai = OpenAI(api_key=api_key)

    with open(args.tasks) as f:
        all_tasks = json.load(f)

    django = [t for t in all_tasks if "django" in t.get("task_id", "").lower()]
    other  = [t for t in all_tasks if "django" not in t.get("task_id", "").lower()]
    tasks  = (django + other)[:args.n_tasks]

    print("=" * 70)
    print("Exp. SJ-v3: Properly Powered Semantic Judge")
    print("=" * 70)
    print(f"Tasks ({len(tasks)}): {[t['task_id'] for t in tasks]}")
    print(f"Conditions: {args.conditions}")
    print(f"Runs/condition/task: {args.n_runs}")
    print(f"Total runs: {len(tasks) * args.n_runs * len(args.conditions)}")
    print(f"Agents/run: {args.n_agents}, Steps/agent: {args.n_steps}")
    print(f"Injection step (Cond B): {args.injection_step}")
    print(f"Judge: {'Claude Haiku' if args.use_claude_judge else 'GPT-4o-mini'}")
    print(f"Backbone: {BACKBONE}")
    print()
    print("DESIGN RATIONALE:")
    print("  Condition A (FRESH):    ORI active, R_obs reads. Control.")
    print("  Condition B (STALE):    ORI active, R_hidden injected. Experimental.")
    print("  Condition C (NO_ORI):   No ORI, LWW. Structural ceiling.")
    print()
    print("HYPOTHESIS:")
    print("  H1: Corruption rate B > A (R_hidden causes semantic failures)")
    print("  H2: Corruption rate C > B (structural ORI adds value beyond semantic gap)")
    print()

    results: list[SJv3Result] = []
    counts = {c: {"corrupted": 0, "incomplete": 0, "correct": 0, "total": 0}
              for c in args.conditions}

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    out_f = open(args.output, "w", newline="")
    writer = None

    for task in tasks:
        tid  = task["task_id"]
        desc = task.get("problem_statement", task.get("description", tid))

        for run_idx in range(args.n_runs):
            for condition in args.conditions:
                label = f"[{tid[:30]:<30}] run={run_idx:02d} cond={condition:<20}"
                print(f"  {label} ...", end=" ", flush=True)
                t0 = time.time()
                run_id = uuid.uuid4().hex[:6]

                try:
                    if condition == COND_FRESH:
                        content, ok, total = run_condition_a_fresh(
                            oai, tid, desc, args.n_agents, args.n_steps)
                        n_inj = 0

                    elif condition == COND_STALE:
                        content, ok, total, n_inj = run_condition_b_stale(
                            oai, tid, desc, args.n_agents, args.n_steps,
                            args.injection_step)

                    else:
                        content, ok, total = run_condition_c_no_ori(
                            oai, tid, desc, args.n_agents, args.n_steps)
                        n_inj = 0

                    wall = time.time() - t0

                    verdict, reason = judge_content(
                        desc, content, oai, use_claude=args.use_claude_judge)

                    r = SJv3Result(
                        run_id=run_id,
                        task_id=tid,
                        run_idx=run_idx,
                        condition=condition,
                        commits_ok=ok,
                        commits_total=total,
                        commit_rate=round(ok / max(1, total), 4),
                        n_stale_injected=n_inj,
                        verdict=verdict,
                        is_corrupted=(verdict == "CORRUPTED"),
                        is_complete=(verdict == "CORRECT"),
                        judge_reason=reason[:200],
                        surviving_content=content[:300],
                        wall_secs=round(wall, 1),
                    )
                    results.append(r)

                    c = counts[condition]
                    c["total"] += 1
                    if verdict == "CORRUPTED":   c["corrupted"]  += 1
                    elif verdict == "CORRECT":   c["correct"]    += 1
                    else:                        c["incomplete"] += 1

                    row = asdict(r)
                    if writer is None:
                        writer = csv.DictWriter(out_f, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    out_f.flush()

                    print(f"commits={ok}/{total} verdict={verdict:<10} {wall:.0f}s")

                except KeyboardInterrupt:
                    print("\nInterrupted — partial results saved.")
                    out_f.close()
                    _print_summary(counts, args)
                    sys.exit(0)
                except Exception as e:
                    print(f"ERROR: {e}")

    out_f.close()

    _print_summary(counts, args)
    _run_statistical_tests(results, args)


def _print_summary(counts: dict, args) -> None:
    print("\n" + "=" * 70)
    print("EXP. SJ-v3 RESULTS SUMMARY")
    print("=" * 70)
    for cond, c in counts.items():
        t = c["total"]
        if t == 0:
            continue
        corr_rate = c["corrupted"] / t
        comp_rate = c["correct"]   / t
        print(f"  {cond:<25}: n={t:3d} | "
              f"CORRECT={c['correct']:3d}({comp_rate*100:5.1f}%) | "
              f"INCOMPLETE={c['incomplete']:3d} | "
              f"CORRUPTED={c['corrupted']:3d}({corr_rate*100:5.1f}%)")

    a = counts.get(COND_FRESH,  {"corrupted": 0, "total": 1})
    b = counts.get(COND_STALE,  {"corrupted": 0, "total": 1})
    c_ = counts.get(COND_NO_ORI, {"corrupted": 0, "total": 1})
    p_a  = a["corrupted"] / max(1, a["total"])
    p_b  = b["corrupted"] / max(1, b["total"])
    p_c  = c_["corrupted"] / max(1, c_["total"])

    print()
    print(f"  H1 (B vs A, R_hidden corruption lift):  "
          f"{(p_b-p_a)*100:+.1f}pp  "
          f"(A={p_a*100:.1f}%, B={p_b*100:.1f}%)")
    print(f"  H2 (C vs B, structural ORI adds value): "
          f"{(p_c-p_b)*100:+.1f}pp  "
          f"(B={p_b*100:.1f}%, C={p_c*100:.1f}%)")
    if p_b > p_a:
        print("  -> POS model CONSISTENT: R_hidden stale context causes semantic corruption.")
    else:
        print("  -> INCONCLUSIVE: stale context does not significantly increase corruption rate.")
        print("     Consider: longer injection (earlier injection_step), more tasks, longer steps.")


def _run_statistical_tests(results: list[SJv3Result], args) -> None:
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("\n  NOTE: scipy not installed. Install for statistical tests:")
        print("  pip install scipy numpy")
        return

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    def get_counts(cond: str):
        cond_results = [r for r in results if r.condition == cond]
        corrupted = sum(1 for r in cond_results if r.is_corrupted)
        n = len(cond_results)
        return corrupted, n - corrupted, n

    for (cond1, cond2, label) in [
        (COND_FRESH, COND_STALE,  "H1: FRESH vs STALE (R_hidden effect)"),
        (COND_STALE, COND_NO_ORI, "H2: STALE vs NO_ORI (structural ORI value)"),
        (COND_FRESH, COND_NO_ORI, "H3: FRESH vs NO_ORI (total ORI value)"),
    ]:
        if cond1 not in [r.condition for r in results]:
            continue
        if cond2 not in [r.condition for r in results]:
            continue

        corr1, ok1, n1 = get_counts(cond1)
        corr2, ok2, n2 = get_counts(cond2)

        if n1 == 0 or n2 == 0:
            continue

        table = [[corr1, ok1], [corr2, ok2]]
        _, p_fisher = scipy_stats.fisher_exact(table, alternative="less")

        p1 = corr1 / n1
        p2 = corr2 / n2
        diff = p2 - p1

        print(f"\n  {label}")
        print(f"    {cond1}: {corr1}/{n1} corrupted = {p1*100:.1f}%")
        print(f"    {cond2}: {corr2}/{n2} corrupted = {p2*100:.1f}%")
        print(f"    Difference: {diff*100:+.1f}pp")
        print(f"    Fisher's exact (one-sided, {cond2}>{cond1}): p = {p_fisher:.4f}")
        if p_fisher < 0.05:
            print("    -> SIGNIFICANT at α=0.05")
        elif p_fisher < 0.10:
            print("    -> MARGINAL (0.05 < p < 0.10)")
        else:
            print(f"    -> NOT SIGNIFICANT (p={p_fisher:.3f})")

        se = ((p1 * (1 - p1) / max(1, n1)) + (p2 * (1 - p2) / max(1, n2))) ** 0.5
        ci_lo = diff - 1.96 * se
        ci_hi = diff + 1.96 * se
        print(f"    95% CI for difference: [{ci_lo*100:.1f}pp, {ci_hi*100:.1f}pp]")

if __name__ == "__main__":
    main()
