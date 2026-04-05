#!/usr/bin/env python3
"""
naive_deadlock_test.py — Experiment R: Re-demonstrate the deadlock result
retracted in S-Bus v19.

Paper context
─────────────
Prior versions claimed "/commit/v2_naive deadlocked at N>=8". That result
was from a DashMap-era multi-lock implementation. The v18 commit_v2_naive
was an alias for commit_delta (single-lock, cannot deadlock). V19 fixes this
with genuine per-shard insertion-order Mutex acquisition (engine_v19.rs FIX-1).

This script confirms that:
  1. /commit/v2        (sorted single-lock) → zero deadlocks at all N.
  2. /commit/v2_naive  (insertion-order per-shard) → deadlock at N>=2
     when two threads hold overlapping shards in opposite order.

Usage
─────
  # Start server first (with v19 engine):
  #   cd sbus && cargo run --release
  python3 naive_deadlock_test.py [--base http://localhost:3000] [--trials 10]

Output
──────
  Table 7 (corrected):
    /commit/v2        N=2  trials=10  deadlocks=0   (0.0%)
    /commit/v2_naive  N=2  trials=10  deadlocks=X   (X0.0%)
"""

import argparse
import concurrent.futures
import requests
import time
import json
import sys
from dataclasses import dataclass, field
from typing import Literal

# ── Config ────────────────────────────────────────────────────────────────────

BASE = "http://localhost:3000"
DEADLOCK_TIMEOUT = 5.0   # seconds — if a commit hangs this long, it's a deadlock
SHARD_INIT_CONTENT = "initial content for shard"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def create_shard(base: str, key: str, content: str = SHARD_INIT_CONTENT) -> dict:
    r = requests.post(f"{base}/shard",
                      json={"key": key, "content": content, "goal_tag": "deadlock_test"},
                      timeout=5.0)
    return r.json()


def read_shard(base: str, key: str) -> dict:
    r = requests.get(f"{base}/shard/{key}", timeout=5.0)
    r.raise_for_status()
    return r.json()


def commit(base: str, endpoint: str, key: str, version: int, delta: str,
           agent_id: str, read_set: list | None = None) -> tuple[bool, dict | str]:
    """
    Returns (success: bool, response: dict | error_str).
    Times out after DEADLOCK_TIMEOUT seconds — a hung request is counted as deadlock.
    """
    body = {
        "key": key,
        "expected_version": version,
        "delta": delta,
        "agent_id": agent_id,
    }
    if read_set:
        body["read_set"] = read_set

    try:
        r = requests.post(f"{base}/{endpoint}", json=body, timeout=DEADLOCK_TIMEOUT)
        return r.status_code == 200, r.json()
    except requests.exceptions.Timeout:
        return False, "TIMEOUT (deadlock)"
    except Exception as e:
        return False, str(e)


# ── Deadlock scenario setup ───────────────────────────────────────────────────
#
# Two shards: "shard_x" and "shard_y".
# Two threads:
#   Thread A: commits to shard_x with read_set=[shard_y]
#             → naive endpoint acquires x_lock first, then y_lock
#   Thread B: commits to shard_y with read_set=[shard_x]
#             → naive endpoint acquires y_lock first, then x_lock
#
# With insertion-order locking: A holds x, waits for y;
#                               B holds y, waits for x → DEADLOCK.
# With sorted locking (/commit/v2): both acquire in order [x, y] → no deadlock.

def run_one_trial(base: str, endpoint: str, trial: int) -> dict:
    """
    Returns {"trial": int, "deadlock": bool, "timeout_thread": str | None}
    """
    # Fresh shards for each trial
    key_x = f"shard_x_{trial}"
    key_y = f"shard_y_{trial}"
    create_shard(base, key_x)
    create_shard(base, key_y)

    sx = read_shard(base, key_x)
    sy = read_shard(base, key_y)
    vx, vy = sx["version"], sy["version"]

    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # Thread A: write x, declare read on y
        fa = pool.submit(commit, base, endpoint,
                         key_x, vx, f"delta_from_A_trial_{trial}", "agent_A",
                         [{"key": key_y, "version_at_read": vy}])
        # Small sleep so both threads are truly concurrent
        time.sleep(0.05)
        # Thread B: write y, declare read on x
        fb = pool.submit(commit, base, endpoint,
                         key_y, vy, f"delta_from_B_trial_{trial}", "agent_B",
                         [{"key": key_x, "version_at_read": vx}])

        ok_a, resp_a = fa.result(timeout=DEADLOCK_TIMEOUT + 1)
        ok_b, resp_b = fb.result(timeout=DEADLOCK_TIMEOUT + 1)

    deadlock = (not ok_a and "TIMEOUT" in str(resp_a)) or \
               (not ok_b and "TIMEOUT" in str(resp_b))
    timeout_thread = None
    if not ok_a and "TIMEOUT" in str(resp_a): timeout_thread = "A"
    if not ok_b and "TIMEOUT" in str(resp_b): timeout_thread = (timeout_thread or "") + "B"

    return {
        "trial": trial,
        "deadlock": deadlock,
        "timeout_thread": timeout_thread,
        "ok_a": ok_a,
        "ok_b": ok_b,
        "resp_a": str(resp_a)[:80],
        "resp_b": str(resp_b)[:80],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment(base: str, endpoint: str, trials: int, label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Endpoint: POST /{endpoint}  |  {label}")
    print(f"Trials: {trials}  |  Timeout per commit: {DEADLOCK_TIMEOUT}s")
    print(f"{'='*60}")

    deadlock_count = 0
    for t in range(1, trials + 1):
        result = run_one_trial(base, endpoint, t)
        status = "DEADLOCK" if result["deadlock"] else "ok"
        if result["deadlock"]:
            deadlock_count += 1
        print(f"  trial {t:2d}: {status}"
              f"  A={'ok' if result['ok_a'] else 'FAIL'}  "
              f"B={'ok' if result['ok_b'] else 'FAIL'}")

    pct = 100 * deadlock_count / trials
    print(f"\n  Result: {deadlock_count}/{trials} deadlocks ({pct:.1f}%)")
    return {"endpoint": endpoint, "trials": trials, "deadlocks": deadlock_count, "pct": pct}


def main():
    parser = argparse.ArgumentParser(description="S-Bus naive deadlock re-demonstration")
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    print("S-Bus Experiment R: Naive Deadlock Re-demonstration (v19)")
    print(f"Server: {args.base}")
    print("\nPre-check: server reachable?")
    try:
        r = requests.get(f"{args.base}/stats", timeout=3)
        r.raise_for_status()
        print(f"  OK — {r.json().get('total_shards', '?')} shards in registry")
    except Exception as e:
        print(f"  FAIL: {e}")
        print("  → Start the S-Bus server first: cd sbus && cargo run --release")
        sys.exit(1)

    results = []

    # Control: sorted single-lock — expect 0 deadlocks
    results.append(run_experiment(
        args.base, "commit/v2", args.trials,
        "CONTROL: sorted single-lock (should be 0 deadlocks)"
    ))

    # Treatment: insertion-order per-shard — expect deadlocks
    results.append(run_experiment(
        args.base, "commit/v2_naive", args.trials,
        "TREATMENT: insertion-order per-shard (should deadlock)"
    ))

    # ── Print Table 7 (corrected) ─────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("TABLE 7 (corrected) — Deadlock Demonstration Results")
    print("="*60)
    print(f"{'Endpoint':<25} {'Trials':>6} {'Deadlocks':>9} {'Rate':>8}")
    print("-"*60)
    for r in results:
        print(f"  {r['endpoint']:<23} {r['trials']:>6} {r['deadlocks']:>9} {r['pct']:>7.1f}%")
    print("="*60)
    print("\nExpected: /commit/v2 = 0%  |  /commit/v2_naive > 0%")
    print("If naive shows 0%: check server is running v19 engine (FIX-1).")

    # Save JSON
    with open("naive_deadlock_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to naive_deadlock_results.json")


if __name__ == "__main__":
    main()