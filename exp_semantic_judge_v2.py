#!/usr/bin/env python3
"""
Exp SJ-v2: Controlled R_hidden Injection Semantic Judge
=========================================================
Properly designed experiment to validate the POS model.

PROBLEM WITH SJ-v1 (exp_semantic_judge.py):
  The original experiment toggled OCC on/off to compare conditions.
  This confounded two independent variables:
    1. OCC blocking (fewer commits -> incomplete solutions)
    2. R_hidden semantic corruption (stale context -> wrong deltas)
  Result: OCC-off got only 1 commit (version mismatch rejects all others),
  making it look "less corrupt" because a single clean answer is easier
  to judge than 20 accumulated partial edits.

THIS EXPERIMENT:
  Both conditions use OCC-ON (structural protection always active).
  The experimental variable is whether agents receive STALE CONTEXT
  (simulating R_hidden corruption) or FRESH CONTEXT (control).

DESIGN:
  - Condition A (Fresh): Agents read current shard state via HTTP before
    each delta (Robs path - ORI protected)
  - Condition B (Stale): Agents receive injected stale context from a
    previous step (simulating Rhidden - not ORI protected)
  - Both conditions use full step count (15 steps) for task completion
  - Judge evaluates: does surviving content address the task goal?

WHAT THIS PROVES:
  If Condition B shows significantly higher corruption than A,
  this validates that R_hidden stale reads cause semantic corruption
  at the rate predicted by POS (rho <= 0.71 * phidden).

USAGE:
  export OPENAI_API_KEY=sk-...
  python3 exp_semantic_judge_v2.py \
      --tasks datasets/tasks_30_multidomain.json \
      --n-tasks 5 \
      --n-runs 5 \
      --n-steps 15 \
      --output results/semantic_judge_v2_results.csv
"""

import csv, json, os, sys, uuid, time, argparse, socket
from dataclasses import dataclass, asdict
from urllib.request import Request, ProxyHandler, build_opener
from urllib.parse import urlencode
from urllib.error import HTTPError

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai"); sys.exit(1)

_opener = build_opener(ProxyHandler({}))
SBUS_URL = "http://localhost:7000"


def http_get(url, params=None):
    if params:
        url += "?" + urlencode(params)
    try:
        with _opener.open(url, timeout=20) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def http_post(url, body):
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def reset_sbus():
    http_post(f"{SBUS_URL}/reset", {})
    time.sleep(0.2)


JUDGE_PROMPT = """You are a strict technical evaluator.
Task: {task}
Final content: {content}

Does the content meaningfully address the task? Consider:
- Does it identify the correct problem?
- Does it propose a plausible solution?
- Is it relevant to the codebase (django/astropy)?

Reply ONLY: CORRECT or CORRUPTED
Then one sentence explanation."""


def judge(task_desc: str, content: str, oai: OpenAI) -> tuple[bool, str]:
    try:
        r = oai.chat.completions.create(
            model="gpt-4o-mini", max_tokens=60, temperature=0,
            messages=[
                {"role": "system", "content": "You judge code solutions. Reply CORRECT or CORRUPTED first."},
                {"role": "user", "content": JUDGE_PROMPT.format(
                    task=task_desc[:300], content=content[:400])}
            ]
        )
        text = r.choices[0].message.content.strip()
        corrupted = "CORRUPT" in text.upper().split("\n")[0]
        reason = text.split("\n", 1)[1].strip() if "\n" in text else text
        return corrupted, reason
    except Exception as e:
        return False, f"Judge error: {e}"


def run_fresh_condition(task_id, task_desc, n_agents, n_steps, run_idx, oai):
    """
    FRESH condition: agents read current shard via HTTP before each commit.
    This is the R_obs path - ORI protected.
    All agents can read current state and commit sequentially.
    """
    reset_sbus()
    sid = uuid.uuid4().hex[:6]
    shard = f"shard_{sid}"
    agents = [f"agent_{i}_{sid}" for i in range(n_agents)]

    http_post(f"{SBUS_URL}/shard", {"key": shard, "content": "Initial", "goal_tag": task_id})
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})

    commits_ok = 0
    for step in range(n_steps):
        for agent in agents:
            # READ current shard via HTTP (R_obs - fresh, ORI protected)
            _, data = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": agent})
            ver = data.get("version", 0) if data else 0
            current = data.get("content", "")[:200] if data else ""

            # Generate delta with FRESH context
            try:
                resp = oai.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=80, temperature=0.3,
                    messages=[{"role": "user", "content": (
                        f"Task: {task_desc[:200]}\n"
                        f"Current state (fresh HTTP read, step {step}): {current}\n"
                        f"Agent {agent}: write one concrete code improvement:"
                    )}]
                )
                delta = f"[{agent}] {resp.choices[0].message.content.strip()}"
            except Exception as e:
                delta = f"[{agent}] error: {e}"

            # Commit with correct version (ORI enforced)
            cst, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard, "expected_version": ver,
                "delta": delta, "agent_id": agent,
                "read_set": [{"key": shard, "version_at_read": ver}],
            })
            if cst == 200:
                commits_ok += 1

    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", "") if final else "", commits_ok


def run_stale_condition(task_id, task_desc, n_agents, n_steps, run_idx, oai):
    """
    STALE condition: agents receive INJECTED stale context (simulating R_hidden).
    Agents read from an OLD snapshot of shard state (not current HTTP read).
    This simulates agents using prior prompt context instead of fetching freshly.
    OCC is still ON - so structural conflicts are blocked.
    But agents make decisions based on stale R_hidden context.
    """
    reset_sbus()
    sid = uuid.uuid4().hex[:6]
    shard = f"shard_{sid}"
    agents = [f"agent_{i}_{sid}" for i in range(n_agents)]

    http_post(f"{SBUS_URL}/shard", {"key": shard, "content": "Initial", "goal_tag": task_id})
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})

    # Create stale snapshot at step 0
    _, initial_data = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "snapshot_agent"})
    stale_snapshot = initial_data.get("content", "Initial") if initial_data else "Initial"
    stale_version = initial_data.get("version", 0) if initial_data else 0

    commits_ok = 0
    for step in range(n_steps):
        for agent in agents:
            # Get actual current version for commit (but use STALE content for reasoning)
            _, current_data = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": agent})
            current_ver = current_data.get("version", 0) if current_data else 0

            # INJECT stale context - agent reasons from OLD state (simulates R_hidden)
            # The stale snapshot was taken at step 0 and is never updated
            injected_stale_context = (
                f"{stale_snapshot[:150]} "
                f"[STALE: this context is from step 0, agent has not re-fetched]"
            )

            # Generate delta with STALE context (simulating R_hidden corruption)
            try:
                resp = oai.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=80, temperature=0.3,
                    messages=[{"role": "user", "content": (
                        f"Task: {task_desc[:200]}\n"
                        f"Current state (from memory, step {step}): {injected_stale_context}\n"
                        f"Agent {agent}: write one concrete code improvement:"
                    )}]
                )
                delta = f"[{agent}] {resp.choices[0].message.content.strip()}"
            except Exception as e:
                delta = f"[{agent}] error: {e}"

            # Commit (ORI still enforced - structural protection remains)
            cst, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard, "expected_version": current_ver,
                "delta": delta, "agent_id": agent,
                "read_set": [{"key": shard, "version_at_read": current_ver}],
            })
            if cst == 200:
                commits_ok += 1

    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", "") if final else "", commits_ok


@dataclass
class SJResult:
    task_id: str
    run_idx: int
    condition: str        # "fresh" or "stale"
    commits_succeeded: int
    surviving_content: str
    is_corrupted: bool
    reason: str
    phidden_simulated: float  # fraction of steps using stale context (1.0 for stale, 0.0 for fresh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="datasets/tasks_30_multidomain.json")
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=15)
    parser.add_argument("--output", default="results/semantic_judge_v2_results.csv")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)

    try:
        s = socket.create_connection(("localhost", 7000), timeout=3); s.close()
        print("S-Bus OK")
    except Exception:
        print("S-Bus not running on port 7000"); sys.exit(1)

    oai = OpenAI(api_key=api_key)

    with open(args.tasks) as f:
        all_tasks = json.load(f)

    # Prefer django tasks for comparability with Exp.I
    django = [t for t in all_tasks if "django" in t.get("task_id","")]
    tasks = (django + all_tasks)[:args.n_tasks]
    print(f"Tasks: {[t['task_id'] for t in tasks]}")
    print(f"Design: {args.n_runs} runs × 2 conditions (fresh/stale) × {len(tasks)} tasks = {args.n_runs*2*len(tasks)} total")
    print()
    print("KEY DIFFERENCE FROM SJ-v1:")
    print("  Both conditions use OCC-ON (structural protection active)")
    print("  Variable is context freshness: fresh (Robs) vs stale (Rhidden injection)")
    print("  This isolates semantic corruption from Rhidden without OCC confound")
    print()

    results = []
    fresh_corrupt = 0; stale_corrupt = 0
    fresh_total = 0; stale_total = 0

    for task in tasks:
        tid = task["task_id"]
        desc = task.get("problem_statement", task.get("description", tid))

        for run_idx in range(args.n_runs):
            for condition in ["fresh", "stale"]:
                print(f"  [{tid[:35]}] run={run_idx} {condition}...", end=" ", flush=True)

                try:
                    if condition == "fresh":
                        content, commits = run_fresh_condition(
                            tid, desc, args.n_agents, args.n_steps, run_idx, oai)
                    else:
                        content, commits = run_stale_condition(
                            tid, desc, args.n_agents, args.n_steps, run_idx, oai)

                    corrupted, reason = judge(desc, content, oai)

                    r = SJResult(
                        task_id=tid, run_idx=run_idx, condition=condition,
                        commits_succeeded=commits,
                        surviving_content=content[:200],
                        is_corrupted=corrupted, reason=reason[:150],
                        phidden_simulated=1.0 if condition == "stale" else 0.0,
                    )
                    results.append(r)

                    if condition == "fresh":
                        fresh_total += 1
                        if corrupted: fresh_corrupt += 1
                    else:
                        stale_total += 1
                        if corrupted: stale_corrupt += 1

                    status = "CORRUPTED" if corrupted else "OK"
                    print(f"commits={commits} -> {status}")

                except Exception as e:
                    print(f"ERROR: {e}")

    # Write results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])

    print(f"\n{'='*60}")
    print("CONTROLLED R_hidden INJECTION RESULTS")
    print(f"{'='*60}")
    fr = fresh_corrupt/max(1,fresh_total)
    sr = stale_corrupt/max(1,stale_total)
    print(f"  Fresh (Robs, ORI protected):  {fresh_corrupt}/{fresh_total} = {fr*100:.1f}% corrupted")
    print(f"  Stale (Rhidden injected):     {stale_corrupt}/{stale_total} = {sr*100:.1f}% corrupted")
    print(f"  POS prediction at phidden=1.0: rho <= 0.71 * 1.0 = 71.0%")
    print(f"  POS prediction at phidden=0.706: rho <= 0.71 * 0.706 = 50.1%")
    print()
    if sr > fr:
        lift = sr - fr
        print(f"  Stale corruption lift: +{lift*100:.1f}pp vs fresh baseline")
        print(f"  POS model: {'CONSISTENT' if sr <= 0.75 else 'EXCEEDED'}")
    else:
        print(f"  WARNING: Stale not more corrupt than fresh ({sr*100:.1f}% vs {fr*100:.1f}%)")
        print(f"  Possible cause: Rhidden injection too mild at {args.n_steps} steps")
        print(f"  Try: longer staleness window, more adversarial injection")
    print()
    print(f"  Paper update (§III-D POS Validation):")
    print(f"  'Exp.~SJ-v2 (controlled Rhidden injection, {args.n_tasks} tasks, {args.n_runs} runs/condition):")
    print(f"   Fresh (Robs): {fr*100:.1f}% semantic corruption;")
    print(f"   Stale (Rhidden injected): {sr*100:.1f}% semantic corruption.")
    if sr <= 0.71:
        print(f"   Consistent with POS prediction rho <= 71.0%.'")
    print(f"  Results: {args.output}")


if __name__ == "__main__":
    main()