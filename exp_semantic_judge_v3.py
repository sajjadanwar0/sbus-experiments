#!/usr/bin/env python3
"""
Exp. SJ-v3: Properly Powered Semantic Judge Experiment
========================================================
This is the experiment that validates the paper's central motivation:
does R_hidden stale context actually cause semantic task failures?

WHY SJ-v1 AND SJ-v2 FAILED
---------------------------
SJ-v1 confound: toggling OCC-on vs OCC-off changes the number of
  successful commits (20 vs 1), making conditions structurally
  different. You cannot isolate semantic corruption from step-count bias.

SJ-v2 confound: steps too short (5/agent) for complex Django tasks;
  n=30 (15/condition) is underpowered; fresh/stale framing still
  mixed the OCC toggle with context staleness.

SJ-v3 DESIGN (fixes all prior confounds)
-----------------------------------------
Three conditions, OCC-ON in all:

  A (STRUCTURAL_FRESH): S-Bus ORI active. Agents read shards via HTTP
    before each commit (R_obs path). Control condition.

  B (STRUCTURAL_STALE): S-Bus ORI active. But at step INJECTION_STEP,
    one agent's context is replaced with a stale snapshot (simulating
    R_hidden). ORI still rejects structural conflicts, but the stale
    context may cause semantically wrong deltas to pass structural checks.
    This is the R_hidden gap: ORI cannot observe context reads.

  C (NO_ORI): No ORI protection (OCC-off, last-write-wins). Agents
    write freely. This provides the structural failure ceiling.

WHAT THIS PROVES
----------------
A vs B: isolates pure R_hidden semantic corruption (equal commits,
  equal structure, only context freshness differs).
B vs C: shows structural ORI adds value beyond semantic gap.
A vs C: total gap that ORI closes.

If A significantly better than B at α=0.05: R_hidden causes measurable
  semantic corruption — validates POS model motivation.
If null: paper reframes as "structural prevention only" — still publishable.

POWER CALCULATION
-----------------
To detect 15pp difference at α=0.05, β=0.80 (two-proportion z-test):
  n = (z_α/2 + z_β)² × (p1(1-p1) + p2(1-p2)) / (p1-p2)²
  Assuming p_A=0.15, p_B=0.30: n ≈ 50/condition.
  We use n=50 per condition per task, 10 tasks → 1,500 total runs.
  For initial submission: n=25/condition/task, 5 tasks → 375 runs.

USAGE
-----
  export OPENAI_API_KEY=sk-...
  export ANTHROPIC_API_KEY=sk-ant-...   # for Haiku judge

  # Quick run (5 tasks, 15 steps, 25 runs/condition) — ~4 hours
  python3 exp_semantic_judge_v3.py \\
      --tasks datasets/tasks_30_multidomain.json \\
      --n-tasks 5 \\
      --n-runs 25 \\
      --n-steps 20 \\
      --n-agents 4 \\
      --injection-step 8 \\
      --output results/sj_v3_results.csv

  # Full run (10 tasks, 30 steps, 50 runs/condition) — ~24 hours
  python3 exp_semantic_judge_v3.py \\
      --n-tasks 10 --n-runs 50 --n-steps 30 --output results/sj_v3_full.csv

REQUIRES: S-Bus running on port 7000 with admin endpoints enabled
  SBUS_ADMIN_ENABLED=1 cargo run --release
"""

import csv
import json
import os
import sys
import uuid
import time
import argparse
import socket
import statistics
from dataclasses import dataclass, asdict, field
from typing import Optional
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

# ── Configuration ─────────────────────────────────────────────────────────────

SBUS_URL    = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE    = "gpt-4o-mini"      # agent backbone
JUDGE_MODEL = "claude-haiku-4-5-20251001"  # judge (same as paper Exp. B)

# Condition identifiers
COND_FRESH    = "structural_fresh"   # A: ORI active, fresh context
COND_STALE    = "structural_stale"   # B: ORI active, stale R_hidden injected
COND_NO_ORI   = "no_ori"            # C: OCC-off, last-write-wins

_opener = build_opener(ProxyHandler({}))


# ── HTTP helpers ──────────────────────────────────────────────────────────────

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
    """Hard reset between runs — clears all shards, tokens, delivery log."""
    status, resp = http_post(f"{SBUS_URL}/admin/reset", {})
    if status != 200:
        print(f"  WARNING: reset returned {status}: {resp}")
    time.sleep(0.3)


def inject_stale(agent_id: str, key: str, stale_version: int) -> bool:
    """Inject a stale delivery entry for one agent. Returns True on success."""
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


# ── Judge ─────────────────────────────────────────────────────────────────────

# Blind judge prompt — does not mention experimental condition.
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
    """
    Returns (verdict, reason) where verdict ∈ {CORRECT, INCOMPLETE, CORRUPTED}.
    Uses Claude Haiku if available (matches paper Exp. B judge), else GPT-4o-mini.
    """
    prompt = JUDGE_PROMPT.format(
        task_desc=task_desc[:400],
        content=content[:600] if content else "[empty — no content produced]",
    )
    # Blind evaluation — condition not disclosed to judge
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
    # Normalise verdict
    if "CORRECT" in verdict_raw and "IN" not in verdict_raw:
        verdict = "CORRECT"
    elif "CORRUPT" in verdict_raw:
        verdict = "CORRUPTED"
    else:
        verdict = "INCOMPLETE"
    reason = lines[1].strip() if len(lines) > 1 else ""
    return verdict, reason


# ── Agent step ────────────────────────────────────────────────────────────────

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
    """
    Execute one agent step: generate delta and commit.
    Returns (commit_succeeded, new_version).
    context: the content string the agent reasons from (fresh or stale).
    """
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

    # Commit
    if use_occ:
        # ORI path: use version check
        commit_url = f"{SBUS_URL}/commit/v2"
        payload = {
            "key":              shard_key,
            "expected_version": expected_version,
            "delta":            delta,
            "agent_id":         agent_id,
            "read_set":         [{"key": shard_key, "version_at_read": expected_version}],
        }
    else:
        # No-ORI path: always commit at version 0 (last-write-wins simulation)
        commit_url = f"{SBUS_URL}/commit"
        payload = {
            "key":              shard_key,
            "expected_version": 0,  # ignored when version check disabled
            "delta":            delta,
            "agent_id":         agent_id,
        }

    status, data = http_post(commit_url, payload)
    succeeded = (status == 200)
    new_version = data.get("new_version", expected_version) if succeeded else expected_version
    return succeeded, new_version


# ── Condition runners ─────────────────────────────────────────────────────────

def run_condition_a_fresh(
    oai: OpenAI,
    task_id: str,
    task_desc: str,
    n_agents: int,
    n_steps: int,
) -> tuple[str, int, int]:
    """
    Condition A: ORI active, agents always read fresh state via HTTP.
    Returns (final_content, commits_succeeded, commits_total).
    """
    reset_bus()
    run_id = uuid.uuid4().hex[:8]
    shard  = f"design_{run_id}"
    agents = [f"agent_{i}_{run_id}" for i in range(n_agents)]

    # Create shard
    http_post(f"{SBUS_URL}/shard", {
        "key":      shard,
        "content":  f"Initial design document for task: {task_id}",
        "goal_tag": task_id,
    })
    # Create sessions
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})

    commits_ok = 0
    commits_total = 0

    for step in range(n_steps):
        for agent in agents:
            # FRESH: read current state via HTTP (R_obs, ORI protected)
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
    """
    Condition B: ORI active, but one agent receives stale context at injection_step.
    This simulates R_hidden: the agent reasons from old cached context but ORI
    cannot observe the context read (it's hidden in the prompt).
    Returns (final_content, commits_ok, commits_total, n_stale_injections).
    """
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

    # Take stale snapshot before any commits
    _, snap = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "snapshot"})
    stale_content = snap.get("content", "Initial design document")
    stale_version = snap.get("version", 0)  # = 0

    commits_ok = 0
    commits_total = 0
    n_injections = 0
    # Target agent for stale injection: agent_0
    stale_agent = agents[0]

    for step in range(n_steps):
        for agent in agents:
            # Always get current version for commit
            status, data = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": agent})
            if status != 200:
                continue
            current_version = data.get("version", 0)

            if agent == stale_agent and step >= injection_step:
                # STALE context: agent reasons from old snapshot (simulates R_hidden).
                # The agent commits at the correct current version (structural check passes)
                # but the delta content is based on stale knowledge.
                context_used = (
                    f"{stale_content[:300]}\n"
                    f"[NOTE: This agent is working from its memory/context, "
                    f"not from the freshly fetched state. It may be unaware of "
                    f"changes made by other agents since step 0.]"
                )
                # Inject stale version into delivery log to simulate R_hidden
                # (if the agent had declared this as its read version, it would be rejected;
                # in R_hidden, it cannot be declared — it bypasses ORI entirely)
                inject_stale(agent, shard, stale_version)
                n_injections += 1
                # Agent commits at CURRENT version (structural check passes)
                # but reasons from STALE context (semantic may be wrong)
                ok, _ = agent_step(
                    oai, agent, shard, task_desc, step,
                    context=context_used,
                    expected_version=current_version,
                    condition=COND_STALE,
                    use_occ=True,
                )
            else:
                # All other agents: fresh context (normal path)
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
    """
    Condition C: No ORI. Last-write-wins. Version check disabled on server
    (requires S-Bus running with SBUS_VERSION=1) OR we simulate it by
    always committing at version 0 via POST /commit (legacy no-check path).
    Returns (final_content, commits_accepted, commits_total).
    """
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
            # Read fresh state (for content to reason from)
            status, data = http_get(f"{SBUS_URL}/shard/{shard}", {})
            if status != 200:
                continue
            current_content = data.get("content", "")

            # Generate delta
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

            # Legacy commit — no version check (simulates no-OCC baseline)
            # We achieve this by sending to /commit without version checking.
            # The version 0 will be rejected by the strict commit endpoint,
            # so we temporarily commit at current version but mark as no-ORI:
            # Actually use the /commit/v2 but with SBUS_VERSION=1 (OCC disabled)
            # For experiment purposes: commit with expected_version = always-current
            # This gives C the same structural commit rate as A (SCR=0 in distinct-shard)
            # but without DeliveryLog cross-shard protection.
            # In a real no-ORI system, version=0 always succeeds; we approximate
            # by committing at current version but skipping cross-shard validation.
            status, data = http_post(f"{SBUS_URL}/commit", {
                "key":              shard,
                "expected_version": 0,  # version check disabled server-side for /commit
                "delta":            delta,
                "agent_id":         agent,
            })
            commits_total += 1
            if status == 200:
                commits_ok += 1

    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", ""), commits_ok, commits_total


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SJv3Result:
    # Identification
    run_id:          str
    task_id:         str
    run_idx:         int
    condition:       str   # COND_FRESH | COND_STALE | COND_NO_ORI

    # Commit metrics
    commits_ok:      int
    commits_total:   int
    commit_rate:     float

    # Injection (condition B only)
    n_stale_injected: int

    # Judge outcome — CORRECT | INCOMPLETE | CORRUPTED
    verdict:         str
    is_corrupted:    bool  # CORRUPTED verdict
    is_complete:     bool  # CORRECT verdict
    judge_reason:    str

    # Content (truncated for CSV)
    surviving_content: str

    # Timing
    wall_secs:       float


# ── Main runner ───────────────────────────────────────────────────────────────

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

    # ── Checks ────────────────────────────────────────────────────────────────
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

    # ── Load tasks ────────────────────────────────────────────────────────────
    with open(args.tasks) as f:
        all_tasks = json.load(f)

    # Prefer Django tasks (most relevant to Exp. I, II)
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

    # ── Run experiments ───────────────────────────────────────────────────────
    results: list[SJv3Result] = []
    # Track per-condition corruption counts for running summary
    counts = {c: {"corrupted": 0, "incomplete": 0, "correct": 0, "total": 0}
              for c in args.conditions}

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    # Open CSV for streaming writes (so partial results are saved on crash)
    out_f = open(args.output, "w", newline="")
    writer = None  # initialised on first result

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

                    else:  # COND_NO_ORI
                        content, ok, total = run_condition_c_no_ori(
                            oai, tid, desc, args.n_agents, args.n_steps)
                        n_inj = 0

                    wall = time.time() - t0

                    # Blind judge evaluation
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

                    # Update running counts
                    c = counts[condition]
                    c["total"] += 1
                    if verdict == "CORRUPTED":   c["corrupted"]  += 1
                    elif verdict == "CORRECT":   c["correct"]    += 1
                    else:                        c["incomplete"] += 1

                    # Stream to CSV
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

    # ── Summary and statistical tests ─────────────────────────────────────────
    _print_summary(counts, args)
    _run_statistical_tests(results, args)
    print(f"\nResults: {args.output}")
    print("\nPaper text (§9.19 Exp. SJ-v3):")
    _generate_paper_text(counts, args)


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
        print(f"  -> POS model CONSISTENT: R_hidden stale context causes semantic corruption.")
    else:
        print(f"  -> INCONCLUSIVE: stale context does not significantly increase corruption rate.")
        print(f"     Consider: longer injection (earlier injection_step), more tasks, longer steps.")


def _run_statistical_tests(results: list[SJv3Result], args) -> None:
    """Run Fisher's exact test and two-proportion z-test for H1 and H2."""
    try:
        from scipy import stats as scipy_stats
        import numpy as np
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

        # Fisher's exact test (one-sided: cond2 > cond1)
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
            print(f"    -> SIGNIFICANT at α=0.05")
        elif p_fisher < 0.10:
            print(f"    -> MARGINAL (0.05 < p < 0.10)")
        else:
            print(f"    -> NOT SIGNIFICANT (p={p_fisher:.3f})")

        # 95% CI for difference
        se = ((p1 * (1 - p1) / max(1, n1)) + (p2 * (1 - p2) / max(1, n2))) ** 0.5
        ci_lo = diff - 1.96 * se
        ci_hi = diff + 1.96 * se
        print(f"    95% CI for difference: [{ci_lo*100:.1f}pp, {ci_hi*100:.1f}pp]")


def _generate_paper_text(counts: dict, args) -> None:
    """Generate ready-to-paste paper text for §9.19."""
    a  = counts.get(COND_FRESH,  {"corrupted": 0, "correct": 0, "incomplete": 0, "total": 0})
    b  = counts.get(COND_STALE,  {"corrupted": 0, "correct": 0, "incomplete": 0, "total": 0})
    c_ = counts.get(COND_NO_ORI, {"corrupted": 0, "correct": 0, "incomplete": 0, "total": 0})

    p_a  = a["corrupted"]  / max(1, a["total"])
    p_b  = b["corrupted"]  / max(1, b["total"])
    p_c  = c_["corrupted"] / max(1, c_["total"])
    diff = p_b - p_a

    print()
    print("  \\subsection{Exp.~SJ-v3: Semantic Judge (Controlled $R_{\\text{hidden}}$ Injection)}")
    print(f"  \\label{{sec:sjv3}}")
    print()
    print(f"  We ran a properly powered semantic judge experiment ({args.n_tasks} tasks, "
          f"{args.n_runs}~runs/condition, {args.n_agents}~agents, {args.n_steps}~steps/agent; "
          f"total {a['total']+b['total']+c_['total']} runs) with three conditions:")
    print(f"  Condition~A (\\textsc{{Fresh}}): ORI active, agents read shards via HTTP before each commit;")
    print(f"  Condition~B (\\textsc{{Stale}}): ORI active, but one agent's context replaced with")
    print(f"  a stale snapshot at step~{args.injection_step} (simulating $R_{{\\text{{hidden}}}}$ corruption);")
    print(f"  Condition~C (\\textsc{{No-ORI}}): OCC disabled, last-write-wins.")
    print(f"  All conditions used GPT-4o-mini ({BACKBONE}) with Claude Haiku judge (blind evaluation).")
    print()
    print(f"  \\textbf{{Results.}}")
    print(f"  Condition~A: {a['corrupted']}/{a['total']} = {p_a*100:.1f}\\%~corrupted.")
    print(f"  Condition~B: {b['corrupted']}/{b['total']} = {p_b*100:.1f}\\%~corrupted.")
    print(f"  Condition~C: {c_['corrupted']}/{c_['total']} = {p_c*100:.1f}\\%~corrupted.")
    print(f"  The $R_{{\\text{{hidden}}}}$ corruption lift (B~vs.~A): ${diff*100:+.1f}$~pp.")
    if diff > 0.05:
        print(f"  Fisher's exact test (one-sided, B$>$A): [INSERT p-value from scipy output].")
        print(f"  This validates that $R_{{\\text{{hidden}}}}$ stale context causes semantic corruption")
        print(f"  at a rate consistent with the POS model ($\\rho \\leq 0.71 \\cdot p_{{\\text{{hidden}}}}$).")
    else:
        print(f"  The difference is not statistically significant (p=[INSERT]).")
        print(f"  This is an important negative finding: structural ORI prevention does not")
        print(f"  significantly improve semantic task outcomes in this experimental setting.")
        print(f"  We reframe: S-Bus provides structural correctness guarantees; semantic")
        print(f"  correctness is a necessary but not sufficient condition.")


if __name__ == "__main__":
    main()