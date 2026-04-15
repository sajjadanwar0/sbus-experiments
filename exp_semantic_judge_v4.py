#!/usr/bin/env python3
"""
Exp. SJ-v4: Semantic Judge — Correct R_hidden Simulation
==========================================================

WHY SJ-v3 WAS NULL (post-hoc diagnosis):
  The stale injection in SJ-v3 only wrote a stale version number into
  the DeliveryLog. It did NOT change what the agent actually read.
  The agent still called GET /shard and received fresh content.
  So semantically, both conditions were identical — same prompt content,
  same reasoning, same outputs. The null result was guaranteed by design.

  Commit rate = 1.000 in both conditions confirmed this:
  the stale DeliveryLog entry never triggered a rejection because
  agents committed at the current version, not the stale one.

THE FIX — Three changes from SJ-v3:

  1. ACTUALLY FEED STALE CONTENT to the agent's prompt.
     In the STALE condition: do NOT call GET /shard.
     Instead give the agent a frozen snapshot from step 0.
     This is the actual R_hidden scenario: agent reasons from
     old cached context rather than fetching fresh state.

  2. USE TASKS REQUIRING CUMULATIVE PRECISION.
     Django bug repair is too well-specified — the fix direction
     is clear regardless of intermediate state.
     SJ-v4 uses tasks where intermediate state matters:
       - SymPy algebraic derivation (step N depends on step N-1 result)
       - Schema migration (migration state must track prior steps)
       - Numerical accumulation (running totals, version numbers)

  3. MEASURE CONTENT DIVERGENCE DIRECTLY.
     Instead of only asking "is output corrupted?", also measure:
       - Jaccard distance between fresh and stale agent outputs
       - Semantic similarity (embedding cosine distance)
       - Whether stale agent contradicts fresh agent's changes

DESIGN:
  Condition A (FRESH):  Agent reads current shard via GET before each step.
                        Full fresh context in every prompt. Control.

  Condition B (STALE):  Agent receives frozen snapshot from step 0.
                        Never calls GET /shard after step 0.
                        Simulates agent working from cached prompt context.
                        ORI still active — structural conflicts blocked.
                        But agent's REASONING is from stale state.

  Both conditions: ORI active, commit at current version.
  Judge: blind evaluation of final shard content.
  Also measure: output divergence between conditions at each step.

WHAT THIS PROVES:
  If STALE produces more corruption: R_hidden causes semantic failures.
  The mechanism is now correctly simulated: agent reasons from stale
  content, produces stale-informed deltas, which may contradict
  other agents' fresh-informed changes.

TASKS SELECTED FOR MAXIMUM SENSITIVITY:
  - sympy_solver:    algebraic derivation, step results build on each other
  - sympy_matrix:    eigenvalue computation, precision required
  - django_migration: migration graph, each step depends on prior state
  - astropy_fits:    header accumulation, running state
  - requests_session: session state evolution

USAGE:
  export OPENAI_API_KEY=sk-...
  python3 exp_semantic_judge_v4.py \
      --n-tasks 5 \
      --n-runs 25 \
      --n-steps 20 \
      --injection-step 5 \
      --output results/sj_v4_results.csv

COST: ~$3.00 | TIME: 3-4 hours (use run_sjv4_parallel.py for parallelism)
"""

import csv
import json
import os
import sys
import time
import uuid
import argparse
import socket
import statistics
from dataclasses import dataclass, asdict
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, ProxyHandler, build_opener

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai"); sys.exit(1)

SBUS_URL    = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE    = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

_opener = build_opener(ProxyHandler({}))

# ── Task definitions — selected for cumulative state sensitivity ──────────────

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


# ── HTTP helpers ──────────────────────────────────────────────────────────────

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


# ── Judge ─────────────────────────────────────────────────────────────────────

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


# ── Divergence measurement ────────────────────────────────────────────────────

def jaccard_similarity(text1: str, text2: str) -> float:
    """Word-level Jaccard similarity between two texts."""
    w1 = set(text1.lower().split())
    w2 = set(text2.lower().split())
    if not w1 and not w2:
        return 1.0
    return len(w1 & w2) / len(w1 | w2)


# ── Condition runners ─────────────────────────────────────────────────────────

def run_fresh(
    oai: OpenAI,
    task: dict,
    n_agents: int,
    n_steps: int,
) -> tuple[str, int, int]:
    """
    Condition A: FRESH.
    Every agent reads current shard content via HTTP before each step.
    This is R_obs — ORI protected, fully fresh context.
    """
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
            # FRESH: always read current state via HTTP
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
    """
    Condition B: STALE.
    THE KEY FIX FROM SJ-v3:
    After injection_step, the designated stale agent NO LONGER calls GET /shard.
    Instead it reasons from a frozen snapshot of the shard content at step 0.
    This correctly simulates R_hidden: the agent uses cached prompt context
    rather than fetching fresh state.

    The stale agent still commits at the current version (structural check passes)
    but its REASONING is based on old content — so its delta may contradict
    changes made by other agents since step 0.

    ORI is active and correctly blocks structural conflicts.
    The question is whether the SEMANTIC content becomes corrupted
    despite structural protection.
    """
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

    # Take stale snapshot at step 0 (initial content)
    _, snap = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "snapshot"})
    stale_content  = snap.get("content", task["initial_content"])
    # stale_content is now FROZEN — it will never be updated for the stale agent

    stale_agent    = agents[0]  # agent_0 is the one that goes stale
    commits_ok     = 0
    commits_total  = 0
    n_stale_steps  = 0

    for step in range(n_steps):
        for agent in agents:
            # All agents always get current version for commit
            status, data = http_get(f"{SBUS_URL}/shard/{shard}",
                                    {"agent_id": agent})
            if status != 200:
                continue
            current_version = data.get("version", 0)

            if agent == stale_agent and step >= injection_step:
                # ── THE FIX: use FROZEN stale content in prompt ───────────────
                # Do NOT use data["content"] (which is fresh).
                # Use stale_content (frozen at step 0).
                # This is R_hidden: agent reasons from cached old context.
                context = (
                    f"{stale_content}\n\n"
                    f"[Agent note: working from prior context, "
                    f"step {step+1} of {n_steps}]"
                )
                n_stale_steps += 1
            else:
                # Fresh agents use current content normally
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

            # Commit at current version — structural check passes
            # (stale agent commits correctly structurally,
            #  but its content may be semantically stale)
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


# ── Result dataclass ──────────────────────────────────────────────────────────

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
    n_stale_steps:    int   # how many steps used stale context (0 for fresh)
    verdict:          str
    is_corrupted:     bool
    is_complete:      bool
    judge_reason:     str
    surviving_content: str
    wall_secs:        float


# ── Main ──────────────────────────────────────────────────────────────────────

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

    # ── Checks ────────────────────────────────────────────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)

    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}")
        print("  Start: SBUS_ADMIN_ENABLED=1 cargo run --release")
        sys.exit(1)

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load task list — from JSON file if provided, else built-in SENSITIVE_TASKS
    if args.tasks_file and os.path.exists(args.tasks_file):
        with open(args.tasks_file) as f:
            all_tasks = json.load(f)
        print(f"Loaded {len(all_tasks)} tasks from {args.tasks_file}")
    else:
        all_tasks = SENSITIVE_TASKS
        if args.tasks_file:
            print(f"WARNING: tasks file not found: {args.tasks_file}, using built-in tasks")

    # Apply offset for parallel runner (SJV4_TASK_OFFSET env var)
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
    _generate_paper_text(counts, args)
    print(f"\nResults: {args.output}")


def _print_summary(counts: dict) -> None:
    from scipy import stats as scipy_stats

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
                print("  ✅ SIGNIFICANT at α=0.05")
            elif p < 0.10:
                print("  ⚠️  MARGINAL")
            else:
                print("  ❌ NOT SIGNIFICANT")
    except ImportError:
        print("  (install scipy for p-value)")


def _run_stats(results: list) -> None:
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return

    print("\n" + "=" * 65)
    print("PER-TASK BREAKDOWN")
    print("=" * 65)

    from collections import defaultdict
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


def _generate_paper_text(counts: dict, args) -> None:
    a = counts["structural_fresh"]
    b = counts["structural_stale"]
    p_a = a["corrupted"] / max(1, a["total"])
    p_b = b["corrupted"] / max(1, b["total"])
    diff = p_b - p_a

    print("\n" + "=" * 65)
    print("PAPER TEXT — §9.19 Exp. SJ-v4")
    print("=" * 65)
    print()
    print(f"\\subsection{{Exp.~SJ-v4: Semantic Judge (Correct $R_{{\\text{{hidden}}}}$ Simulation)}}")
    print(f"\\label{{sec:sjv4}}")
    print()
    print(f"\\paragraph{{Design improvement over SJ-v3.}}")
    print(f"Exp.~SJ-v3 (§\\ref{{sec:sjv3}}) produced a null result because")
    print(f"the stale injection only modified the DeliveryLog version record,")
    print(f"not the agent's prompt content. Both conditions received identical")
    print(f"information, making the null result mechanistically guaranteed.")
    print(f"SJ-v4 corrects this: after step~{args.injection_step}, the stale agent")
    print(f"no longer calls \\texttt{{GET /shard}}. Instead it reasons from a frozen")
    print(f"snapshot of the shard content at step~0, correctly simulating")
    print(f"$R_{{\\text{{hidden}}}}$: an agent using cached prompt context rather than")
    print(f"fetching fresh state. We also selected tasks requiring cumulative")
    print(f"precision (SymPy algebraic derivation, Django migration graph,")
    print(f"Astropy FITS header accumulation) where stale context produces")
    print(f"directionally wrong outputs.")
    print()
    print(f"\\paragraph{{Results.}}")
    print(f"Fresh ($R_{{\\text{{obs}}}}$): ${p_a*100:.1f}\\%$~semantic corruption")
    print(f"({a['corrupted']}/{a['total']}~runs).")
    print(f"Stale ($R_{{\\text{{hidden}}}}$ simulated): ${p_b*100:.1f}\\%$~corruption")
    print(f"({b['corrupted']}/{b['total']}~runs).")
    print(f"Lift: ${diff*100:+.1f}$\\,pp.")
    print(f"[INSERT Fisher's exact p-value from script output]")


if __name__ == "__main__":
    main()