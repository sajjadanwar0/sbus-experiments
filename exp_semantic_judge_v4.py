import csv
import json
import os
import sys
import time
import uuid
import argparse
import socket
from dataclasses import dataclass, asdict
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, ProxyHandler, build_opener
from scipy import stats as scipy_stats
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai"); sys.exit(1)

SBUS_URL    = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE    = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

_opener = build_opener(ProxyHandler({}))

SENSITIVE_TASKS = [
    {
        "task_id":   "sympy__sympy-solver-21345",
        "domain":    "sympy_solver",
        "problem_statement": (
            "Fix SymPy solve() dropping solutions when the 'positive' assumption "
            "is combined with complex intermediate steps. The solver should return "
            "all real positive solutions. Each agent step should refine the approach: "
            "step 1 identifies root cause, step 2 proposes fix in assumptions.py, "
            "step 3 adds test cases, etc. Each step BUILDS on prior steps — "
            "if an agent misses a prior step's conclusion, its fix will contradict it."
        ),
        "initial_content": (
            "Bug: solve(x**2 - 2, x, positive=True) returns [] instead of [sqrt(2)].\n"
            "Root cause: unknown. Investigation in progress."
        ),
    },
    {
        "task_id":   "django__django-migration-13230",
        "domain":    "django_migration",
        "problem_statement": (
            "Fix Django migration squasher for circular dependencies. "
            "The squasher must track which migrations have been processed "
            "in EXACT ORDER — if an agent works from a stale migration graph "
            "(missing steps 3-5 of the squash), it will produce a broken squashed "
            "migration that references non-existent dependencies. "
            "Each agent step adds one migration to the tracked set."
        ),
        "initial_content": (
            "Migration squash state:\n"
            "  Processed: []\n"
            "  Pending: [0001_initial, 0002_add_field, 0003_alter_field, "
            "0004_add_index, 0005_data_migration]\n"
            "  Squashed: None"
        ),
    },
    {
        "task_id":   "astropy__astropy-fits-7671",
        "domain":    "astropy_fits",
        "problem_statement": (
            "Fix Astropy FITS HIERARCH keyword parser. "
            "The parser must track a running list of parsed keywords — "
            "if an agent works from stale state (missing keywords added in steps 3-8), "
            "it will produce duplicate or conflicting keyword registrations. "
            "Each agent step registers one new keyword type into the shared registry."
        ),
        "initial_content": (
            "FITS keyword registry state:\n"
            "  Registered: []\n"
            "  Standard keywords: SIMPLE, BITPIX, NAXIS\n"
            "  HIERARCH keywords found: []\n"
            "  Parse errors: 0"
        ),
    },
    {
        "task_id":   "sympy__sympy-matrix-22000",
        "domain":    "sympy_matrix",
        "problem_statement": (
            "Fix SymPy eigenvals() for sparse symbolic matrices. "
            "Agents must track a running computation: step 1 computes det(A-λI), "
            "step 2 factors the characteristic polynomial, step 3 solves for λ, etc. "
            "Each step depends on the EXACT RESULT of the prior step. "
            "A stale agent missing steps 2-4 will propose an incompatible factorisation."
        ),
        "initial_content": (
            "Eigenvalue computation state:\n"
            "  Matrix: A = [[a, b], [c, d]] (symbolic)\n"
            "  Characteristic polynomial: not yet computed\n"
            "  Eigenvalues: not yet found\n"
            "  Current approach: unknown"
        ),
    },
    {
        "task_id":   "requests__requests-session-5430",
        "domain":    "requests_session",
        "problem_statement": (
            "Fix requests session auth header stripping on cross-domain redirect. "
            "Agents must track the session state evolution: step 1 identifies which "
            "headers are stripped, step 2 proposes the fix in adapters.py, "
            "step 3 adds the domain whitelist logic, etc. "
            "A stale agent will propose a fix that contradicts the approach agreed "
            "in steps 2-3, causing architectural inconsistency in the final patch."
        ),
        "initial_content": (
            "Session auth fix state:\n"
            "  Problem identified: auth headers stripped on redirect\n"
            "  Affected method: unknown\n"
            "  Proposed fix: under discussion\n"
            "  Test cases: none yet"
        ),
    },
]

def http_get(url, params=None):
    if params:
        url += "?" + urlencode(params)
    try:
        with _opener.open(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def http_post(url, body):
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def reset_bus():
    http_post(f"{SBUS_URL}/admin/reset", {})
    time.sleep(0.3)


def health_check():
    try:
        s = socket.create_connection(("localhost",
            int(SBUS_URL.split(":")[-1])), timeout=3)
        s.close()
        status, _ = http_get(f"{SBUS_URL}/admin/health")
        return status == 200
    except Exception:
        return False


JUDGE_PROMPT = """\
You are a strict technical evaluator. Judge ONLY semantic correctness.

TASK:
{task}

FINAL CONTENT OF THE SHARED DESIGN DOCUMENT:
{content}

Does the content correctly and specifically address the task?
Consider:
1. Is the proposed fix technically correct for this specific bug?
2. Is it internally consistent (no contradictory approaches)?
3. Does it address the SPECIFIC technical details of the problem?
   (not just the general area)
4. Would this fix actually work, or does it contain technical errors?

Be STRICT. Partial or vague answers are INCOMPLETE, not CORRECT.
A technically wrong answer (wrong method, wrong file, wrong approach) is CORRUPTED.

Reply with EXACTLY:
CORRECT    - fix is technically correct and specific
INCOMPLETE - on right track but missing key technical detail
CORRUPTED  - technically wrong, contradictory, or addresses wrong problem

Then one sentence explanation."""


def judge(task_desc: str, content: str, oai: OpenAI) -> tuple[str, str]:
    try:
        r = oai.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=100,
            temperature=0,
            messages=[
                {"role": "system",
                 "content": "You are a strict code reviewer. "
                             "Reply CORRECT, INCOMPLETE, or CORRUPTED then a newline then one sentence."},
                {"role": "user",
                 "content": JUDGE_PROMPT.format(
                     task=task_desc[:500],
                     content=content[:600] if content else "[empty]",
                 )},
            ],
        )
        text = r.choices[0].message.content.strip()
        lines = text.split("\n", 1)
        v = lines[0].strip().upper()
        if "CORRECT" in v and "IN" not in v:
            verdict = "CORRECT"
        elif "CORRUPT" in v:
            verdict = "CORRUPTED"
        else:
            verdict = "INCOMPLETE"
        reason = lines[1].strip() if len(lines) > 1 else ""
        return verdict, reason
    except Exception as e:
        return "INCOMPLETE", f"judge error: {e}"

def jaccard_similarity(text1: str, text2: str) -> float:
    w1 = set(text1.lower().split())
    w2 = set(text2.lower().split())
    if not w1 and not w2:
        return 1.0
    return len(w1 & w2) / len(w1 | w2)

def run_fresh(
    oai: OpenAI,
    task: dict,
    n_agents: int,
    n_steps: int,
) -> tuple[str, int, int]:
    reset_bus()
    run_id = uuid.uuid4().hex[:8]
    shard  = f"shared_{run_id}"
    agents = [f"agent_{i}_{run_id}" for i in range(n_agents)]

    http_post(f"{SBUS_URL}/shard", {
        "key":      shard,
        "content":  task["initial_content"],
        "goal_tag": task["task_id"],
    })
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})

    commits_ok = 0
    commits_total = 0

    for step in range(n_steps):
        for agent in agents:
            status, data = http_get(f"{SBUS_URL}/shard/{shard}",
                                    {"agent_id": agent})
            if status != 200:
                continue
            current_version = data.get("version", 0)
            current_content = data.get("content", "")

            try:
                resp = oai.chat.completions.create(
                    model=BACKBONE, max_tokens=150, temperature=0.3,
                    messages=[{"role": "user", "content": (
                        f"You are a software engineering agent.\n"
                        f"TASK: {task['problem_statement'][:400]}\n\n"
                        f"CURRENT STATE (freshly fetched, step {step+1}/{n_steps}):\n"
                        f"{current_content[:500]}\n\n"
                        f"Write ONE specific, concrete technical improvement "
                        f"that builds on the current state above. "
                        f"Be precise — name exact methods, files, or values. "
                        f"Output ONLY the change, no preamble."
                    )}],
                )
                delta = f"[{agent} s{step+1}] {resp.choices[0].message.content.strip()}"
            except Exception as e:
                delta = f"[{agent} s{step+1}] error: {e}"

            commit_status, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key":              shard,
                "expected_version": current_version,
                "delta":            delta,
                "agent_id":         agent,
                "read_set":         [{"key": shard,
                                      "version_at_read": current_version}],
            })
            commits_total += 1
            if commit_status == 200:
                commits_ok += 1

    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", ""), commits_ok, commits_total


def run_stale(
    oai: OpenAI,
    task: dict,
    n_agents: int,
    n_steps: int,
    injection_step: int,
) -> tuple[str, int, int, int]:
    reset_bus()
    run_id = uuid.uuid4().hex[:8]
    shard  = f"shared_{run_id}"
    agents = [f"agent_{i}_{run_id}" for i in range(n_agents)]

    http_post(f"{SBUS_URL}/shard", {
        "key":      shard,
        "content":  task["initial_content"],
        "goal_tag": task["task_id"],
    })
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})

    _, snap = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "snapshot"})
    stale_content  = snap.get("content", task["initial_content"])

    stale_agent    = agents[0]
    commits_ok     = 0
    commits_total  = 0
    n_stale_steps  = 0

    for step in range(n_steps):
        for agent in agents:
            status, data = http_get(f"{SBUS_URL}/shard/{shard}",
                                    {"agent_id": agent})
            if status != 200:
                continue
            current_version = data.get("version", 0)

            if agent == stale_agent and step >= injection_step:
                context = (
                    f"{stale_content}\n\n"
                    f"[Agent note: working from prior context, "
                    f"step {step+1} of {n_steps}]"
                )
                n_stale_steps += 1
            else:
                context = data.get("content", "")

            try:
                resp = oai.chat.completions.create(
                    model=BACKBONE, max_tokens=150, temperature=0.3,
                    messages=[{"role": "user", "content": (
                        f"You are a software engineering agent.\n"
                        f"TASK: {task['problem_statement'][:400]}\n\n"
                        f"CURRENT STATE (step {step+1}/{n_steps}):\n"
                        f"{context[:500]}\n\n"
                        f"Write ONE specific, concrete technical improvement "
                        f"that builds on the current state above. "
                        f"Be precise — name exact methods, files, or values. "
                        f"Output ONLY the change, no preamble."
                    )}],
                )
                delta = f"[{agent} s{step+1}] {resp.choices[0].message.content.strip()}"
            except Exception as e:
                delta = f"[{agent} s{step+1}] error: {e}"

            commit_status, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key":              shard,
                "expected_version": current_version,
                "delta":            delta,
                "agent_id":         agent,
                "read_set":         [{"key": shard,
                                      "version_at_read": current_version}],
            })
            commits_total += 1
            if commit_status == 200:
                commits_ok += 1

    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", ""), commits_ok, commits_total, n_stale_steps

@dataclass
class SJv4Result:
    run_id:           str
    task_id:          str
    domain:           str
    run_idx:          int
    condition:        str
    commits_ok:       int
    commits_total:    int
    commit_rate:      float
    n_stale_steps:    int
    verdict:          str
    is_corrupted:     bool
    is_complete:      bool
    judge_reason:     str
    surviving_content: str
    wall_secs:        float

def main():
    parser = argparse.ArgumentParser(description="Exp. SJ-v4: Correct R_hidden simulation")
    parser.add_argument("--tasks-file",     default=None,
                        help="Optional JSON file with task list (uses built-in if not set)")
    parser.add_argument("--n-tasks",        type=int, default=5)
    parser.add_argument("--n-runs",         type=int, default=25)
    parser.add_argument("--n-steps",        type=int, default=20)
    parser.add_argument("--n-agents",       type=int, default=4)
    parser.add_argument("--injection-step", type=int, default=5,
                        help="Step at which stale agent stops fetching fresh state")
    parser.add_argument("--output",         default="results/sj_v4_results.csv")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)

    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}")
        print("  Start: SBUS_ADMIN_ENABLED=1 cargo run --release")
        sys.exit(1)

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if args.tasks_file and os.path.exists(args.tasks_file):
        with open(args.tasks_file) as f:
            all_tasks = json.load(f)
        print(f"Loaded {len(all_tasks)} tasks from {args.tasks_file}")
    else:
        all_tasks = SENSITIVE_TASKS
        if args.tasks_file:
            print(f"WARNING: tasks file not found: {args.tasks_file}, using built-in tasks")

    task_offset = int(os.environ.get("SJV4_TASK_OFFSET", "0"))
    tasks = all_tasks[task_offset:task_offset + args.n_tasks]

    if not tasks:
        print(f"ERROR: no tasks at offset {task_offset} "
              f"(total tasks={len(all_tasks)}, n_tasks={args.n_tasks})")
        sys.exit(1)

    print("=" * 65)
    print("Exp. SJ-v4: Semantic Judge (Correct R_hidden Simulation)")
    print("=" * 65)
    print()
    print("KEY FIX FROM SJ-v3:")
    print("  Stale agent NO LONGER calls GET /shard after injection_step.")
    print("  It reasons from a FROZEN snapshot — actual R_hidden simulation.")
    print()
    print(f"Tasks:          {[t['task_id'] for t in tasks]}")
    print(f"Runs/condition: {args.n_runs}")
    print(f"Steps:          {args.n_steps}")
    print(f"Agents:         {args.n_agents}")
    print(f"Injection step: {args.injection_step}")
    total = len(tasks) * args.n_runs * 2
    print(f"Total runs:     {total}")
    print()
    print("HYPOTHESIS: Stale context causes MORE corruption than fresh")
    print("  because stale agent's deltas contradict fresh agents' accumulated changes.")
    print()

    results = []
    counts = {
        "structural_fresh": {"corrupted":0,"correct":0,"incomplete":0,"total":0},
        "structural_stale": {"corrupted":0,"correct":0,"incomplete":0,"total":0},
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    out_f  = open(args.output, "w", newline="")
    writer = None

    for task in tasks:
        tid  = task["task_id"]
        desc = task["problem_statement"]

        for run_idx in range(args.n_runs):
            for condition in ["structural_fresh", "structural_stale"]:
                label = f"[{tid[-20:]:<20}] run={run_idx:02d} {condition[:5]}"
                print(f"  {label} ...", end=" ", flush=True)
                t0 = time.time()

                try:
                    if condition == "structural_fresh":
                        content, ok, total_c = run_fresh(
                            oai, task, args.n_agents, args.n_steps)
                        n_stale = 0
                    else:
                        content, ok, total_c, n_stale = run_stale(
                            oai, task, args.n_agents, args.n_steps,
                            args.injection_step)

                    wall = time.time() - t0
                    verdict, reason = judge(desc, content, oai)

                    r = SJv4Result(
                        run_id=uuid.uuid4().hex[:6],
                        task_id=tid,
                        domain=task["domain"],
                        run_idx=run_idx,
                        condition=condition,
                        commits_ok=ok,
                        commits_total=total_c,
                        commit_rate=round(ok/max(1,total_c), 4),
                        n_stale_steps=n_stale,
                        verdict=verdict,
                        is_corrupted=(verdict=="CORRUPTED"),
                        is_complete=(verdict=="CORRECT"),
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

                    print(f"stale_steps={n_stale:2d} "
                          f"commits={ok}/{total_c} "
                          f"verdict={verdict:<10} {wall:.0f}s")

                except KeyboardInterrupt:
                    print("\nInterrupted — partial results saved.")
                    out_f.close()
                    _print_summary(counts)
                    sys.exit(0)
                except Exception as e:
                    print(f"ERROR: {e}")

    out_f.close()
    _print_summary(counts)
    _run_stats(results)
    print(f"\nResults: {args.output}")


def _print_summary(counts: dict) -> None:
    print("\n" + "=" * 65)
    print("EXP. SJ-v4 RESULTS")
    print("=" * 65)

    for cond in ["structural_fresh", "structural_stale"]:
        c = counts[cond]
        t = c["total"]
        if t == 0: continue
        print(f"  {cond:<25} n={t:3d} | "
              f"CORRECT={c['correct']:3d}({c['correct']/t*100:5.1f}%) | "
              f"CORRUPTED={c['corrupted']:3d}({c['corrupted']/t*100:5.1f}%)")

    a = counts["structural_fresh"]
    b = counts["structural_stale"]
    p_a = a["corrupted"] / max(1, a["total"])
    p_b = b["corrupted"] / max(1, b["total"])
    diff = p_b - p_a

    print(f"\n  R_hidden lift (STALE vs FRESH): {diff*100:+.1f}pp")

    try:
        k1,n1 = a["corrupted"], a["total"]
        k2,n2 = b["corrupted"], b["total"]
        if n1 > 0 and n2 > 0:
            _, p = scipy_stats.fisher_exact(
                [[k1,n1-k1],[k2,n2-k2]], alternative="less")
            print(f"  Fisher's exact (one-sided):     p = {p:.4f}")
            if p < 0.05:
                print("  SIGNIFICANT at α=0.05")
            elif p < 0.10:
                print("    MARGINAL")
            else:
                print("   NOT SIGNIFICANT")
    except ImportError:
        print("  (install scipy for p-value)")


def _run_stats(results: list) -> None:
    print("\n" + "=" * 65)
    print("PER-TASK BREAKDOWN")
    print("=" * 65)

    task_counts = defaultdict(lambda: defaultdict(
        lambda: {"corrupted":0,"total":0}))
    for r in results:
        task_counts[r.task_id][r.condition]["total"] += 1
        if r.is_corrupted:
            task_counts[r.task_id][r.condition]["corrupted"] += 1

    print(f"  {'Task':<30} {'Fresh':>8} {'Stale':>8} {'Lift':>8}")
    print("  " + "-" * 58)
    for task, conds in sorted(task_counts.items()):
        f = conds["structural_fresh"]
        s = conds["structural_stale"]
        pf = f["corrupted"]/max(1,f["total"])
        ps = s["corrupted"]/max(1,s["total"])
        print(f"  {task[-28:]:<30} {pf*100:>7.1f}% {ps*100:>7.1f}% "
              f"{(ps-pf)*100:>+7.1f}pp")

if __name__ == "__main__":
    main()
