#!/usr/bin/env python3
"""
Exp. SEQUENTIAL: Direct Wall-Time Measurement (S-Bus parallel vs. CrewAI sequential)
======================================================================================
Measures ACTUAL wall-clock time for S-Bus (parallel) vs. CrewAI (sequential-style)
on matched tasks, matched backbone, matched total LLM calls.

MOTIVATION
----------
Table 5 in the paper cites structural calculations for wall-time speedup (16× at N=16).
This experiment measures it directly to confirm the architectural claim empirically.

DESIGN
------
Both conditions use the SAME:
  - LLM backbone (GPT-4o-mini, temperature=0.3)
  - Task set (SWE-bench multi-domain, n_tasks tasks)
  - Number of agents N
  - Number of steps S per agent
  - Hardware (same AWS instance)

S-Bus condition: all N agents run in parallel threads (as in Exp. B)
Sequential condition: N agents run sequentially, each reading the prior agent's output
  (same info-passing as CrewAI, WITHOUT the CrewAI SDK overhead)

This isolates PARALLEL vs. SEQUENTIAL execution, not framework overhead.

METRICS
-------
- wall_time_sbus: total wall time (seconds) from first agent start to last commit
- wall_time_seq:  total wall time for sequential execution
- speedup:        wall_time_seq / wall_time_sbus
- s50_sbus:       task success rate (S-Bus)
- s50_seq:        task success rate (sequential)
- llm_calls_sbus: total LLM API calls (S-Bus)
- llm_calls_seq:  total LLM API calls (sequential) [should be equal]

STATISTICAL ANALYSIS
--------------------
Mann-Whitney U test on wall times across n_tasks runs.
Wilcoxon signed-rank test (paired, since same tasks).
Report 95% CI on speedup via bootstrap (n=2000 resamples).

USAGE
-----
  # Quick smoke test (5 tasks, 4 agents, 10 steps, ~15 min)
  OPENAI_API_KEY=sk-... python3 exp_sequential_wall_time.py \\
      --n-tasks 5 --n-agents 4 --n-steps 10 --workers 4

  # Paper-quality run (30 tasks, N in {4,8,16}, 30 steps, ~2h)
  OPENAI_API_KEY=sk-... python3 exp_sequential_wall_time.py \\
      --n-tasks 30 --n-agents-list 4 8 16 --n-steps 30 --workers 8 \\
      --output results/exp_sequential_wall_time.csv

REQUIRES
--------
  pip install openai anthropic scipy numpy
  S-Bus server running: cargo run --release (port 7000)
"""

import argparse
import csv
import json
import os
import sys
import time
import uuid
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.request import urlopen, Request
import json as jsonlib

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SBUS_BASE = os.environ.get("SBUS_BASE", "http://localhost:7000")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# MINIMAL tasks (so we can run without full SWE-bench datasets)
# ---------------------------------------------------------------------------
MINIMAL_TASKS = [
    {"id": f"task_{i:03d}", "domain": domain, "problem": problem}
    for i, (domain, problem) in enumerate([
        ("django", "Implement a queryset filter that handles timezone-aware datetime comparisons"),
        ("astropy", "Fix coordinate transformation for galactic to ICRS conversion"),
        ("sympy", "Resolve polynomial factoring over finite fields"),
        ("django", "Add proper handling for reverse FK relations in admin inline forms"),
        ("astropy", "Fix WCS header parsing for non-standard axis ordering"),
        ("sympy", "Implement matrix eigenvalue computation for sparse matrices"),
        ("django", "Fix migration autodetector for custom field types"),
        ("astropy", "Handle FITS header continuation cards correctly"),
        ("sympy", "Improve simplification of trigonometric expressions"),
        ("django", "Fix select_related with multi-table inheritance"),
    ])
]

# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def call_llm_openai(messages: list, max_tokens: int = 150) -> tuple:
    """Returns (text, latency_ms). Raises on error."""
    import urllib.request
    import urllib.error
    t0 = time.time()
    body = jsonlib.dumps({
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = jsonlib.loads(resp.read())
    text = data["choices"][0]["message"]["content"]
    latency_ms = (time.time() - t0) * 1000
    return text, latency_ms


def call_claude_judge(content: str) -> str:
    """Simple Claude Haiku judge for semantic consistency."""
    import urllib.request
    body = jsonlib.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": (
                f"Rate the technical coherence of this agent output. "
                f"Respond with only: COMPLETE, INCOMPLETE, or CORRUPTED.\n\n"
                f"Output: {content[:500]}"
            )}
        ]
    }).encode()
    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = jsonlib.loads(resp.read())
        return data["content"][0]["text"].strip()
    except Exception:
        return "INCOMPLETE"


# ---------------------------------------------------------------------------
# S-Bus helpers
# ---------------------------------------------------------------------------

def sbus_post(path: str, payload: dict, timeout: int = 10) -> dict:
    body = jsonlib.dumps(payload).encode()
    req = Request(
        f"{SBUS_BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return jsonlib.loads(resp.read())


def sbus_get(path: str, timeout: int = 10) -> dict:
    req = Request(f"{SBUS_BASE}{path}", method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return jsonlib.loads(resp.read())


def check_sbus_health() -> bool:
    try:
        resp = sbus_get("/stats")
        return True
    except Exception as e:
        print(f"[ERROR] S-Bus not reachable at {SBUS_BASE}: {e}")
        return False


# ---------------------------------------------------------------------------
# Parallel (S-Bus) condition
# ---------------------------------------------------------------------------

@dataclass
class AgentRun:
    agent_id: str
    task_id: str
    shard_key: str
    commits_ok: int = 0
    commits_fail: int = 0
    llm_calls: int = 0
    content: str = ""


def run_sbus_agent(run: AgentRun, task: dict, n_steps: int,
                   llm_call_counter: threading.Lock, llm_total: list) -> AgentRun:
    """Run one S-Bus agent for n_steps. Records LLM calls."""
    # Register agent with DeliveryLog
    try:
        sbus_post("/session", {"agent_id": run.agent_id})
    except Exception:
        pass

    context = [{"role": "system", "content": (
        f"You are a software engineering agent working on: {task['problem']}\n"
        f"Your shard: {run.shard_key}. Write incremental code contributions."
    )}]
    for step in range(n_steps):
        # Read shard — only agent_id as query param (no session_id in API)
        try:
            shard_resp = sbus_get(
                f"/shard/{run.shard_key}?agent_id={run.agent_id}"
            )
            current_version = shard_resp.get("version", 0)
            current_content = shard_resp.get("content", "")
        except Exception:
            current_version = 0
            current_content = ""

        # LLM call
        messages = context + [{"role": "user", "content": (
            f"Step {step+1}/{n_steps}. Current state: {current_content[:200]}\n"
            f"Write a brief code contribution for {task['problem']}:"
        )}]
        try:
            delta, _ = call_llm_openai(messages, max_tokens=100)
            with llm_call_counter:
                llm_total[0] += 1
        except Exception:
            continue

        # Commit — CommitRequest: key, expected_version, delta, agent_id (no session_id)
        try:
            commit_resp = sbus_post("/commit/v2", {
                "key":              run.shard_key,
                "agent_id":         run.agent_id,
                "expected_version": current_version,
                "delta":            delta,
            })
            # Success: response has new_version field (CommitResponse)
            if "new_version" in commit_resp:
                run.commits_ok += 1
                run.content = delta
            else:
                run.commits_fail += 1
        except Exception:
            run.commits_fail += 1

        run.llm_calls += 1

    return run


def run_parallel_sbus(task: dict, n_agents: int, n_steps: int,
                      run_id: str) -> dict:
    """Run N agents in parallel with S-Bus. Returns timing + results."""
    shard_keys = [f"shard_{run_id}_{i}" for i in range(n_agents)]

    # Create shards — CreateShardRequest: key, content, goal_tag
    for key in shard_keys:
        try:
            sbus_post("/shard", {
                "key":      key,
                "content":  "",
                "goal_tag": task["domain"],
            })
        except Exception:
            pass  # shard may already exist from a prior run

    agent_runs = [
        AgentRun(
            agent_id=f"agent_{i}_{run_id}",
            task_id=task["id"],
            shard_key=shard_keys[i],
        )
        for i in range(n_agents)
    ]

    llm_call_counter = threading.Lock()
    llm_total = [0]

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=n_agents) as executor:
        futures = {
            executor.submit(
                run_sbus_agent, r, task, n_steps, llm_call_counter, llm_total
            ): r
            for r in agent_runs
        }
        completed_runs = [f.result() for f in as_completed(futures)]
    t_end = time.time()

    wall_time = t_end - t_start
    total_commits_ok = sum(r.commits_ok for r in completed_runs)
    total_llm = sum(r.llm_calls for r in completed_runs)

    # Judge outcome
    all_content = " ".join(r.content for r in completed_runs if r.content)
    verdict = call_claude_judge(all_content) if all_content else "INCOMPLETE"
    s50 = 1.0 if verdict == "COMPLETE" else 0.0

    return {
        "run_id": run_id,
        "task_id": task["id"],
        "domain": task["domain"],
        "condition": "sbus_parallel",
        "n_agents": n_agents,
        "n_steps": n_steps,
        "wall_time_s": round(wall_time, 3),
        "commits_ok": total_commits_ok,
        "llm_calls": total_llm,
        "verdict": verdict,
        "s50": s50,
    }


# ---------------------------------------------------------------------------
# Sequential condition (same LLM, sequential agent execution)
# ---------------------------------------------------------------------------

def run_sequential(task: dict, n_agents: int, n_steps: int, run_id: str) -> dict:
    """Run N agents sequentially. Each agent reads prior agent's output."""
    t_start = time.time()
    total_llm = 0
    shared_context = ""
    all_outputs = []

    for agent_idx in range(n_agents):
        agent_context = [{"role": "system", "content": (
            f"You are a software engineering agent working on: {task['problem']}\n"
            f"Previous agents have contributed: {shared_context[:300]}\n"
            f"Add your contribution (agent {agent_idx+1}/{n_agents}):"
        )}]
        agent_output = ""
        for step in range(n_steps):
            messages = agent_context + [{"role": "user", "content": (
                f"Step {step+1}/{n_steps}. Write a brief code contribution:"
            )}]
            try:
                delta, _ = call_llm_openai(messages, max_tokens=100)
                total_llm += 1
                agent_output = delta
                agent_context.append({"role": "assistant", "content": delta})
            except Exception:
                pass

        shared_context = (shared_context + "\n" + agent_output)[-600:]
        all_outputs.append(agent_output)

    t_end = time.time()
    wall_time = t_end - t_start

    all_content = " ".join(all_outputs)
    verdict = call_claude_judge(all_content) if all_content else "INCOMPLETE"
    s50 = 1.0 if verdict == "COMPLETE" else 0.0

    return {
        "run_id": run_id,
        "task_id": task["id"],
        "domain": task["domain"],
        "condition": "sequential",
        "n_agents": n_agents,
        "n_steps": n_steps,
        "wall_time_s": round(wall_time, 3),
        "commits_ok": total_llm,
        "llm_calls": total_llm,
        "verdict": verdict,
        "s50": s50,
    }


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def bootstrap_speedup_ci(sbus_times: list, seq_times: list,
                          n_boot: int = 2000, ci: float = 0.95) -> tuple:
    """Bootstrap 95% CI on median speedup (seq/sbus)."""
    import random
    speedups = []
    for _ in range(n_boot):
        s_sample = random.choices(sbus_times, k=len(sbus_times))
        q_sample = random.choices(seq_times, k=len(seq_times))
        speedup = statistics.median(q_sample) / statistics.median(s_sample)
        speedups.append(speedup)
    speedups.sort()
    lo = speedups[int(n_boot * (1 - ci) / 2)]
    hi = speedups[int(n_boot * (1 - (1 - ci) / 2))]
    return statistics.median(speedups), lo, hi


def wilcoxon_signed_rank(x: list, y: list) -> float:
    """Wilcoxon signed-rank test p-value (paired). Returns p-value."""
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(x, y, alternative="greater")
        return float(p)
    except ImportError:
        # Fallback: sign test
        n_pos = sum(1 for a, b in zip(x, y) if a > b)
        n = len([a for a, b in zip(x, y) if a != b])
        if n == 0:
            return 1.0
        # Binomial p-value (one-sided)
        from math import comb
        p = sum(comb(n, k) * (0.5 ** n) for k in range(n_pos, n + 1))
        return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--n-agents-list", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--n-steps", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="results/exp_sequential_wall_time.csv")
    parser.add_argument("--n-repeats", type=int, default=2,
                        help="Repeats per task per condition (for CI)")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set.")
        sys.exit(1)

    if not check_sbus_health():
        print("[ERROR] S-Bus server not running. Start with: cargo run --release")
        sys.exit(1)

    tasks = MINIMAL_TASKS[:args.n_tasks]
    if len(tasks) < args.n_tasks:
        # Try loading from datasets
        ds_path = "datasets/tasks_30_multidomain.json"
        if os.path.exists(ds_path):
            with open(ds_path) as f:
                full_tasks = json.load(f)
            def extract_domain(task_id: str) -> str:
                for d in ("astropy", "django", "sympy", "requests", "scikit"):
                    if d in task_id.lower():
                        return d
                return task_id.split("__")[0] if "__" in task_id else "unknown"
            tasks = [
                {
                    "id": t.get("task_id", t.get("instance_id", f"task_{i}")),
                    "domain": extract_domain(
                        t.get("task_id", t.get("instance_id", ""))
                    ),
                    "problem": (
                        t.get("description", t.get("problem_statement", ""))[:300]
                    ),
                }
                for i, t in enumerate(full_tasks[:args.n_tasks])
            ]
        else:
            tasks = MINIMAL_TASKS[:min(args.n_tasks, len(MINIMAL_TASKS))]

    os.makedirs("results", exist_ok=True)
    fieldnames = [
        "run_id", "task_id", "domain", "condition", "n_agents", "n_steps",
        "wall_time_s", "commits_ok", "llm_calls", "verdict", "s50",
    ]

    all_results = []

    for n_agents in args.n_agents_list:
        print(f"\n{'='*60}")
        print(f"N = {n_agents} agents | {len(tasks)} tasks | {args.n_steps} steps")
        print(f"{'='*60}")

        sbus_times = []
        seq_times = []
        sbus_s50 = []
        seq_s50 = []

        for repeat in range(args.n_repeats):
            for task in tasks:
                run_id = uuid.uuid4().hex[:8]
                print(f"  [{n_agents}] task={task['id']} repeat={repeat} ...", end="", flush=True)

                # S-Bus parallel run
                try:
                    res_sbus = run_parallel_sbus(task, n_agents, args.n_steps, run_id + "s")
                    all_results.append(res_sbus)
                    sbus_times.append(res_sbus["wall_time_s"])
                    sbus_s50.append(res_sbus["s50"])
                    print(f" SBUS={res_sbus['wall_time_s']:.1f}s", end="", flush=True)
                except Exception as e:
                    print(f" SBUS_ERR={e}", end="", flush=True)

                # Sequential run
                try:
                    res_seq = run_sequential(task, n_agents, args.n_steps, run_id + "q")
                    all_results.append(res_seq)
                    seq_times.append(res_seq["wall_time_s"])
                    seq_s50.append(res_seq["s50"])
                    print(f" SEQ={res_seq['wall_time_s']:.1f}s", end="", flush=True)
                except Exception as e:
                    print(f" SEQ_ERR={e}", end="", flush=True)

                print()  # newline

        # Summary statistics
        if sbus_times and seq_times:
            speedup_median = statistics.median(seq_times) / statistics.median(sbus_times)
            speedup_med, speedup_lo, speedup_hi = bootstrap_speedup_ci(sbus_times, seq_times)
            p_wilcoxon = wilcoxon_signed_rank(seq_times, sbus_times)

            print(f"\n  N={n_agents} Summary:")
            print(f"  S-Bus median wall: {statistics.median(sbus_times):.1f}s "
                  f"(n={len(sbus_times)})")
            print(f"  Sequential median wall: {statistics.median(seq_times):.1f}s "
                  f"(n={len(seq_times)})")
            print(f"  Speedup: {speedup_median:.2f}x "
                  f"[{speedup_lo:.2f}, {speedup_hi:.2f}] 95% CI")
            print(f"  Wilcoxon p (seq > sbus): {p_wilcoxon:.4f}")
            print(f"  S-Bus S@50: {sum(sbus_s50)/len(sbus_s50)*100:.1f}%")
            print(f"  Sequential S@50: {sum(seq_s50)/len(seq_s50)*100:.1f}%")
            print()

            # Record summary row
            all_results.append({
                "run_id": f"SUMMARY_N{n_agents}",
                "task_id": "ALL",
                "domain": "summary",
                "condition": "summary",
                "n_agents": n_agents,
                "n_steps": args.n_steps,
                "wall_time_s": round(speedup_median, 3),
                "commits_ok": -1,
                "llm_calls": -1,
                "verdict": f"speedup={speedup_median:.2f}x CI=[{speedup_lo:.2f},{speedup_hi:.2f}] p={p_wilcoxon:.4f}",
                "s50": -1,
            })

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n[DONE] Results written to {args.output}")
    print(f"Total rows: {len(all_results)}")


if __name__ == "__main__":
    main()