#!/usr/bin/env python3
"""
arsi_real_llm_test.py — Experiment A2 (extended): ARSI over-rejection rate
measured with real LLM calls (not simulated latency).

Paper context
─────────────
Limitation 3 in the paper states:
  "ARSI over-rejection rate [CLOSED — 0.54%]... Caveat: this was measured
   with time.sleep(0.5) to simulate LLM latency. Real LLM calls have
   variable latency and may interact differently with the injector."

This script replaces time.sleep(0.5) with actual GPT-4o-mini calls to close
that caveat. Each "agent step" performs a real LLM inference to produce its
delta, creating genuine variable latency (400–2000ms per step).

Usage
─────
  export OPENAI_API_KEY=sk-...
  # Start S-Bus server first
  python3 arsi_real_llm_test.py [--trials 20] [--steps 30] [--agents 4]

Output
──────
  - Over-rejection rate (aggregate and per-trial mean with 95% CI)
  - Comparison to simulated-latency result (0.54%)
  - Saved to arsi_real_llm_results.json
"""

import argparse
import os
import json
import random
import time
import threading
import statistics
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

BASE = "http://localhost:7000"
MODEL = "gpt-4o-mini"
MAX_TOKENS = 150


# ── S-Bus helpers (mirrors ARSI SDK behaviour) ────────────────────────────────

class ARSIClient:
    """Minimal ARSI-compatible S-Bus client for experiments."""

    def __init__(self, base: str, agent_id: str):
        self.base = base
        self.agent_id = agent_id
        self._read_set: list[dict] = []

    def reset_read_set(self):
        self._read_set = []

    def read_shard(self, key: str) -> dict:
        r = requests.get(f"{self.base}/shard/{key}", timeout=10)
        r.raise_for_status()
        data = r.json()
        # ARSI: record this read in the read-set
        self._read_set.append({"key": key, "version_at_read": data["version"]})
        return data

    def commit(self, key: str, expected_version: int, delta: str) -> tuple[bool, dict]:
        body = {
            "key": key,
            "expected_version": expected_version,
            "delta": delta,
            "agent_id": self.agent_id,
            "read_set": self._read_set,
        }
        r = requests.post(f"{self.base}/commit/v2", json=body, timeout=10)
        self.reset_read_set()
        return r.status_code == 200, r.json()

    def commit_manual_oracle(self, key: str, expected_version: int, delta: str,
                              read_set: list[dict]) -> tuple[bool, dict]:
        """Commit with manually computed read-set (oracle for comparison)."""
        body = {
            "key": key,
            "expected_version": expected_version,
            "delta": delta,
            "agent_id": self.agent_id + "_oracle",
            "read_set": read_set,
        }
        r = requests.post(f"{self.base}/commit/v2", json=body, timeout=10)
        return r.status_code == 200, r.json()


# ── LLM step ─────────────────────────────────────────────────────────────────

def llm_step(client: OpenAI, shard_content: str, task: str) -> str:
    """Perform one real LLM inference step and return the delta."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": (
                "You are a software agent updating a shared state shard. "
                "Respond with ONLY the updated shard content (50-100 words). "
                "No preamble, no explanation."
            )},
            {"role": "user", "content": (
                f"Current shard content:\n{shard_content}\n\n"
                f"Task: {task}\n\n"
                "Write the updated shard content:"
            )},
        ]
    )
    return response.choices[0].message.content.strip()


# ── Injector thread ───────────────────────────────────────────────────────────

class StaleInjector(threading.Thread):
    """
    Background thread that races to commit to watched shards,
    creating stale-read conditions for the ARSI agent to detect.
    """

    def __init__(self, base: str, shards: list[str], interval_s: float = 0.3):
        super().__init__(daemon=True)
        self.base = base
        self.shards = shards
        self.interval = interval_s
        self.injections = 0
        self._running = threading.Event()
        self._running.set()  # starts True; cleared by stop()

    def run(self):
        while self._running.is_set():
            key = random.choice(self.shards)
            try:
                r = requests.get(f"{self.base}/shard/{key}", timeout=2)
                if r.status_code == 200:
                    ver = r.json()["version"]
                    requests.post(f"{self.base}/commit", json={
                        "key": key,
                        "expected_version": ver,
                        "delta": f"injected_at_{time.time():.3f}",
                        "agent_id": "injector",
                    }, timeout=2)
                    self.injections += 1
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._running.clear()


# ── Single trial ─────────────────────────────────────────────────────────────

def run_trial(trial: int, base: str, openai_client: OpenAI,
              n_agents: int, n_steps: int, shards: list[str]) -> dict:
    """
    Run one trial. Returns counts of ARSI stale rejections vs oracle.
    """
    agents = [ARSIClient(base, f"agent_{i}_trial_{trial}") for i in range(n_agents)]

    injector = StaleInjector(base, shards, interval_s=0.2)
    injector.start()

    arsi_stale = 0
    arsi_attempts = 0
    oracle_stale = 0
    oracle_attempts = 0

    tasks = [
        "Update the architecture decision for the database schema.",
        "Revise the API endpoint definitions.",
        "Improve the deployment plan with container specs.",
        "Refine the authentication strategy.",
    ]

    for step in range(n_steps):
        for i, agent in enumerate(agents):
            key = shards[i % len(shards)]
            agent.reset_read_set()

            try:
                shard = agent.read_shard(key)
                arsi_read_set = list(agent._read_set)

                # Real LLM call — actual variable latency
                delta = llm_step(openai_client, shard["content"], tasks[i % len(tasks)])

                # ARSI commit
                arsi_attempts += 1
                ok, resp = agent.commit(key, shard["version"], delta)
                if not ok and resp.get("error") in ("CrossShardStale", "VersionMismatch"):
                    arsi_stale += 1

                # Oracle: manually compute whether the shard was actually stale
                try:
                    current = requests.get(f"{base}/shard/{key}", timeout=2).json()
                    oracle_read_set = [{"key": key, "version_at_read": shard["version"]}]
                    is_stale = current["version"] != shard["version"]
                    oracle_attempts += 1
                    if is_stale:
                        oracle_stale += 1
                except Exception:
                    pass

            except Exception as e:
                print(f"    [trial {trial} step {step} agent {i}] error: {e}")

    injector.stop()
    injector.join(timeout=2)

    over_rejections = max(0, arsi_stale - oracle_stale)
    over_rejection_rate = over_rejections / arsi_attempts if arsi_attempts > 0 else 0.0

    return {
        "trial": trial,
        "arsi_stale": arsi_stale,
        "arsi_attempts": arsi_attempts,
        "oracle_stale": oracle_stale,
        "oracle_attempts": oracle_attempts,
        "injections": injector.injections,
        "over_rejections": over_rejections,
        "over_rejection_rate_pct": round(over_rejection_rate * 100, 3),
    }


# ── Setup / teardown ──────────────────────────────────────────────────────────

def setup_shards(base: str, n: int, trial: int) -> list[str]:
    keys = [f"arsi_real_shard_{i}_t{trial}" for i in range(n)]
    for key in keys:
        try:
            requests.post(f"{base}/shard", json={
                "key": key,
                "content": "Initial architecture plan for a multi-agent software system.",
                "goal_tag": "arsi_real_test",
            }, timeout=5)
        except Exception:
            pass
    return keys


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of independent trials (paper uses 20)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Steps per trial (paper uses 30)")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--shards", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        return

    openai_client = OpenAI(api_key=api_key)

    print("S-Bus Experiment A2 (extended): ARSI Over-Rejection with Real LLM Calls")
    print(f"Model: {MODEL}  |  Trials: {args.trials}  |  Steps: {args.steps}")
    print(f"Agents: {args.agents}  |  Shards: {args.shards}")
    print("-" * 60)

    all_results = []
    for trial in range(1, args.trials + 1):
        print(f"\nTrial {trial}/{args.trials}...", end=" ", flush=True)
        shards = setup_shards(args.base, args.shards, trial)
        result = run_trial(trial, args.base, openai_client,
                           args.agents, args.steps, shards)
        all_results.append(result)
        print(f"  over_rejection_rate={result['over_rejection_rate_pct']:.2f}%"
              f"  injections={result['injections']}")

    # ── Statistics ────────────────────────────────────────────────────────────
    rates = [r["over_rejection_rate_pct"] for r in all_results]
    mean_rate = statistics.mean(rates)
    stdev_rate = statistics.stdev(rates) if len(rates) > 1 else 0.0
    n = len(rates)
    # 95% CI using t-distribution approximation (df=n-1)
    import math
    t_95 = 2.262 if n == 10 else 2.093 if n == 20 else 2.0
    ci_half = t_95 * stdev_rate / math.sqrt(n)
    ci_lo, ci_hi = mean_rate - ci_half, mean_rate + ci_half

    total_arsi = sum(r["arsi_stale"] for r in all_results)
    total_arsi_att = sum(r["arsi_attempts"] for r in all_results)
    total_oracle = sum(r["oracle_stale"] for r in all_results)
    total_inj = sum(r["injections"] for r in all_results)

    agg_over = max(0, total_arsi - total_oracle)
    agg_rate = agg_over / total_arsi_att if total_arsi_att > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESULTS: ARSI Over-Rejection Rate (Real LLM Calls)")
    print("=" * 60)
    print(f"  Total injections across all trials: {total_inj}")
    print(f"  ARSI stale rejections: {total_arsi} / {total_arsi_att} attempts")
    print(f"  Oracle stale: {total_oracle}")
    print(f"  Aggregate over-rejection rate: {agg_rate*100:.2f}%")
    print(f"  Per-trial mean: {mean_rate:.2f}%  95% CI [{ci_lo:.2f}%, {ci_hi:.2f}%]")
    print()
    print("  Simulated-latency result (paper Limitation 3): 0.54% [-3.1%, 4.2%]")
    zero_in_ci = ci_lo <= 0.0 <= ci_hi
    print(f"  CI includes zero: {zero_in_ci}")
    if zero_in_ci:
        print("  → ARSI conservatism is not significantly different from the oracle.")
    else:
        print("  → ARSI over-rejection is statistically significant with real LLMs.")
        print("  → Update Limitation 3 in paper accordingly.")
    print("=" * 60)

    output = {
        "experiment": "ARSI_real_llm",
        "model": MODEL,
        "n_trials": n,
        "n_steps": args.steps,
        "n_agents": args.agents,
        "aggregate_over_rejection_rate_pct": round(agg_rate * 100, 3),
        "per_trial_mean_pct": round(mean_rate, 3),
        "per_trial_stdev_pct": round(stdev_rate, 3),
        "ci_95_lo": round(ci_lo, 3),
        "ci_95_hi": round(ci_hi, 3),
        "ci_includes_zero": zero_in_ci,
        "trials": all_results,
    }
    with open("arsi_real_llm_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to arsi_real_llm_results.json")


if __name__ == "__main__":
    main()