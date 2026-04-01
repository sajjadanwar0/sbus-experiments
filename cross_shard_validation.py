"""
cross_shard_validation.py
─────────────────────────
Experiment to prove that the /commit/v2 endpoint correctly detects and rejects
cross-shard stale reads (phantom reads), closing the §8.9 proof gap.

Matches paper contribution:
  "A controlled validation experiment demonstrating that CrossShardStale
   detection correctly prevents phantom reads across shards with zero state
   corruptions in all N ∈ {4, 8, 16} agent configurations."

Protocol:
  Three shards: db_schema, api_design, deploy_plan
  Agent commits to deploy_plan after reading db_schema + api_design.
  A concurrent injector advances db_schema between the agent's read and commit.
  Expected: CrossShardStale returned, agent retries, zero corruptions.

Run:
  python3 cross_shard_validation.py
  python3 cross_shard_validation.py --agents 8 --trials 50
  python3 cross_shard_validation.py --agents 16 --trials 20 --out results/cross_shard.csv
"""

import argparse
import csv
import time
import threading
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

SBUS_URL = "http://localhost:3000"
MODEL = "gpt-4o-mini-2024-07-18"

# ─── persistent HTTP client ─────────────────────────────────────────────────
# One pooled client shared by all threads — prevents TCP connection exhaustion
# under high concurrency (N=8/16 agents + injector = many simultaneous calls).
_CLIENT = httpx.Client(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
    timeout=httpx.Timeout(30.0, connect=5.0),
)

# ─── helpers ──────────────────────────────────────────────────────────────────

def create_shard(key: str, content: str, goal_tag: str) -> dict:
    r = _CLIENT.post(f"{SBUS_URL}/shard",
                     json={"key": key, "content": content, "goal_tag": goal_tag})
    r.raise_for_status()
    return r.json()


def read_shard(key: str) -> dict:
    r = _CLIENT.get(f"{SBUS_URL}/shard/{key}")
    r.raise_for_status()
    return r.json()


def commit_v2(key: str, expected_version: int, delta: str,
              agent_id: str, read_set: Optional[list] = None) -> tuple[bool, dict]:
    body = {
        "key": key,
        "expected_version": expected_version,
        "delta": delta,
        "agent_id": agent_id,
    }
    if read_set is not None:
        body["read_set"] = read_set
    r = _CLIENT.post(f"{SBUS_URL}/commit/v2", json=body)
    return r.status_code == 200, r.json()


def commit_original(key: str, expected_version: int, delta: str,
                    agent_id: str) -> tuple[bool, dict]:
    """POST /commit (no read_set — original endpoint)."""
    r = _CLIENT.post(f"{SBUS_URL}/commit",
                     json={"key": key, "expected_version": expected_version,
                           "delta": delta, "agent_id": agent_id})
    return r.status_code == 200, r.json()


def server_stats() -> dict:
    return _CLIENT.get(f"{SBUS_URL}/stats").json()


# ─── experiment data structures ────────────────────────────────────────────────

@dataclass
class TrialResult:
    trial_id: int
    n_agents: int
    injections: int           # how many times injector advanced db_schema
    cross_shard_detected: int # how many CrossShardStale errors were returned
    corruptions: int          # how many times deploy_plan had stale content
    retries: int              # total retries by agents
    duration_s: float
    success: bool             # trial completed without corruption


@dataclass
class ExperimentResult:
    mode: str                 # "v2_with_readset" or "v1_no_readset"
    n_agents: int
    n_trials: int
    total_injections: int
    total_cross_shard_detected: int
    total_corruptions: int
    total_retries: int
    detection_rate: float     # cross_shard_detected / injections
    corruption_rate: float    # corruptions / total_injections
    mean_duration_s: float


# ─── injector thread ──────────────────────────────────────────────────────────

class Injector:
    """
    Concurrently advances db_schema between agent reads and commits.
    This simulates the phantom-read scenario described in §8.9.
    """

    def __init__(self, db_schema_key: str, delay_s: float = 0.1):
        self.key = db_schema_key
        self.delay_s = delay_s
        self._stop = threading.Event()
        self._injection_count = 0
        self._lock = threading.Lock()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def injection_count(self) -> int:
        with self._lock:
            return self._injection_count

    def _run(self):
        injector_id = "injector-0"
        while not self._stop.is_set():
            time.sleep(self.delay_s + random.uniform(0, 0.05))
            try:
                shard = read_shard(self.key)
                ok, _ = commit_original(
                    key=self.key,
                    expected_version=shard["version"],
                    delta=f"[injected] PostgreSQL schema v{shard['version']+1} "
                          f"with additional normalization at {datetime.now(timezone.utc).isoformat()}",
                    agent_id=injector_id,
                )
                if ok:
                    with self._lock:
                        self._injection_count += 1
            except Exception:
                pass


# ─── agent logic ──────────────────────────────────────────────────────────────

@dataclass
class AgentStats:
    commits: int = 0
    cross_shard_stale: int = 0
    version_mismatch: int = 0
    retries: int = 0
    corruptions: int = 0


def agent_run_with_readset(
    agent_id: str,
    deploy_key: str,
    db_key: str,
    api_key: str,
    steps: int,
    stats: AgentStats,
    lock: threading.Lock,
):
    """
    Agent that reads db_schema + api_design, then commits to deploy_plan.
    Uses /commit/v2 with read_set.

    Corruption detection (correct method):
      Record db_schema version immediately BEFORE calling POST /commit/v2.
      If commit returns 200 AND pre_commit_db_ver != db_ver, the server
      allowed a stale cross-shard read through — that is a real server-side
      corruption.  This is the only check that cannot produce false positives
      from injector activity that happens AFTER a valid commit.
    """
    for step in range(steps):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Read cross-shard dependencies
                db   = read_shard(db_key)
                api  = read_shard(api_key)
                dep  = read_shard(deploy_key)

                db_ver  = db["version"]
                api_ver = api["version"]
                dep_ver = dep["version"]

                # Simulate LLM call latency (~100ms)
                time.sleep(0.1 + random.uniform(0, 0.05))

                delta = (
                    f"Step {step}: deploy_plan by {agent_id} — "
                    f"references db_schema@v{db_ver} and api_design@v{api_ver}"
                )

                # Snapshot db_schema version immediately before the commit.
                # If the injector advanced db between our read (db_ver) and now
                # (pre_commit_db_ver), the server MUST reject this via
                # CrossShardStale.  A 200 despite pre_commit_db_ver != db_ver
                # would be a genuine server-side protocol failure.
                pre_commit_db_ver = read_shard(db_key)["version"]

                read_set = [
                    {"key": db_key,  "version_at_read": db_ver},
                    {"key": api_key, "version_at_read": api_ver},
                ]
                ok, resp = commit_v2(
                    key=deploy_key,
                    expected_version=dep_ver,
                    delta=delta,
                    agent_id=agent_id,
                    read_set=read_set,
                )

                with lock:
                    if ok:
                        stats.commits += 1
                        # Corruption = server returned 200 but db had already
                        # advanced past db_ver before the commit call was made.
                        # The server should have caught this via CrossShardStale.
                        if pre_commit_db_ver != db_ver:
                            stats.corruptions += 1
                        break
                    else:
                        error_code = resp.get("error", "")
                        if error_code == "CrossShardStale":
                            stats.cross_shard_stale += 1
                            stats.retries += 1
                        elif error_code == "VersionMismatch":
                            stats.version_mismatch += 1
                            stats.retries += 1
                        else:
                            stats.retries += 1
                        time.sleep(0.05 * (attempt + 1))

            except Exception as e:
                with lock:
                    stats.retries += 1
                time.sleep(0.1)


def agent_run_without_readset(
    agent_id: str,
    deploy_key: str,
    db_key: str,
    api_key: str,
    steps: int,
    stats: AgentStats,
    lock: threading.Lock,
):
    """
    Baseline agent using /commit (no read_set).
    Demonstrates the §8.9 failure mode: stale cross-shard commits succeed.

    Corruption detection: same pre_commit_db_ver approach.
    For the baseline, corruptions are EXPECTED (the server has no protection).
    """
    for step in range(steps):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                db   = read_shard(db_key)
                api  = read_shard(api_key)
                dep  = read_shard(deploy_key)

                db_ver  = db["version"]
                dep_ver = dep["version"]

                time.sleep(0.1 + random.uniform(0, 0.05))

                delta = (
                    f"Step {step}: deploy_plan by {agent_id} — "
                    f"references db_schema@v{db_ver}"
                )

                # Record db version just before commit (same method as v2 agent)
                pre_commit_db_ver = read_shard(db_key)["version"]

                ok, resp = commit_original(
                    key=deploy_key,
                    expected_version=dep_ver,
                    delta=delta,
                    agent_id=agent_id,
                )

                with lock:
                    if ok:
                        stats.commits += 1
                        # For baseline: commit succeeds even when db has advanced
                        # (no CrossShardStale protection). This is the corruption.
                        if pre_commit_db_ver != db_ver:
                            stats.corruptions += 1
                        break
                    else:
                        stats.retries += 1
                        time.sleep(0.05 * (attempt + 1))

            except Exception:
                stats.retries += 1
                time.sleep(0.1)


def run_trial(
    trial_id: int,
    n_agents: int,
    steps: int,
    use_readset: bool,
    server_url: str = SBUS_URL,
) -> TrialResult:
    """
    Run one trial:
      - Create 3 shards (db_schema, api_design, deploy_plan)
      - Start injector (advances db_schema concurrently)
      - Run n_agents threads all committing to deploy_plan after reading db+api
      - Measure corruptions, detections, retries
    """
    suffix = f"t{trial_id}_{int(time.time()*1000)}"
    db_key     = f"db_schema_{suffix}"
    api_key    = f"api_design_{suffix}"
    deploy_key = f"deploy_plan_{suffix}"

    create_shard(db_key,     "Initial PostgreSQL schema",   "db_schema")
    create_shard(api_key,    "Initial REST API design",     "api_design")
    create_shard(deploy_key, "Initial deployment plan",     "deploy_plan")

    stats   = AgentStats()
    lock    = threading.Lock()
    injector = Injector(db_key, delay_s=0.08)

    t_start = time.time()
    injector.start()

    threads = []
    agent_fn = agent_run_with_readset if use_readset else agent_run_without_readset
    for i in range(n_agents):
        t = threading.Thread(
            target=agent_fn,
            args=(f"agent-{i}", deploy_key, db_key, api_key, steps, stats, lock),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    injector.stop()
    duration = time.time() - t_start

    injections = injector.injection_count()

    return TrialResult(
        trial_id=trial_id,
        n_agents=n_agents,
        injections=injections,
        cross_shard_detected=stats.cross_shard_stale,
        corruptions=stats.corruptions,
        retries=stats.retries,
        duration_s=round(duration, 2),
        success=(stats.corruptions == 0),
    )


# ─── full experiment ──────────────────────────────────────────────────────────

def run_experiment(
    n_agents: int,
    n_trials: int,
    steps: int,
    use_readset: bool,
) -> ExperimentResult:
    mode = "v2_with_readset" if use_readset else "v1_no_readset"
    print(f"\n{'='*60}")
    print(f"Mode: {mode} | N={n_agents} agents | {n_trials} trials | {steps} steps/agent")
    print(f"{'='*60}")

    trial_results = []
    for i in range(n_trials):
        r = run_trial(i, n_agents, steps, use_readset)
        trial_results.append(r)
        status = "✓" if r.success else "✗ CORRUPTION"
        print(
            f"  trial {i:02d}: injections={r.injections:3d} "
            f"detected={r.cross_shard_detected:3d} "
            f"corruptions={r.corruptions} "
            f"retries={r.retries:3d} "
            f"dur={r.duration_s:.1f}s  {status}"
        )

    total_inj   = sum(r.injections for r in trial_results)
    total_det   = sum(r.cross_shard_detected for r in trial_results)
    total_corr  = sum(r.corruptions for r in trial_results)
    total_retry = sum(r.retries for r in trial_results)
    mean_dur    = sum(r.duration_s for r in trial_results) / len(trial_results)
    det_rate    = total_det  / max(total_inj, 1)
    corr_rate   = total_corr / max(total_inj, 1)

    print(f"\nSummary ({mode}):")
    print(f"  Total injections   : {total_inj}")
    print(f"  CrossShardStale    : {total_det}  ({det_rate:.1%} of injections)")
    print(f"  Corruptions        : {total_corr}  ← MUST BE ZERO for /v2")
    print(f"  Retries            : {total_retry}")
    print(f"  Mean trial duration: {mean_dur:.1f}s")

    return ExperimentResult(
        mode=mode,
        n_agents=n_agents,
        n_trials=n_trials,
        total_injections=total_inj,
        total_cross_shard_detected=total_det,
        total_corruptions=total_corr,
        total_retries=total_retry,
        detection_rate=round(det_rate, 4),
        corruption_rate=round(corr_rate, 4),
        mean_duration_s=round(mean_dur, 2),
    )


# ─── CSV output ───────────────────────────────────────────────────────────────

def write_csv(results: list[ExperimentResult], path: str):
    fields = [
        "mode", "n_agents", "n_trials", "total_injections",
        "total_cross_shard_detected", "total_corruptions", "total_retries",
        "detection_rate", "corruption_rate", "mean_duration_s",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "mode":                       r.mode,
                "n_agents":                   r.n_agents,
                "n_trials":                   r.n_trials,
                "total_injections":           r.total_injections,
                "total_cross_shard_detected": r.total_cross_shard_detected,
                "total_corruptions":          r.total_corruptions,
                "total_retries":              r.total_retries,
                "detection_rate":             r.detection_rate,
                "corruption_rate":            r.corruption_rate,
                "mean_duration_s":            r.mean_duration_s,
            })
    print(f"\nResults written to {path}")


# ─── paper table printer ──────────────────────────────────────────────────────

def print_paper_table(results: list[ExperimentResult]):
    """
    Prints a LaTeX-ready table matching the format of Table 5 in the paper
    but for the cross-shard validation experiment.
    """
    print("\n" + "="*70)
    print("TABLE: Cross-Shard Validation (for revised §8.9 / new Table X)")
    print("="*70)
    print(f"{'Mode':<20} {'N':>4} {'Injections':>12} {'Detected':>10} "
          f"{'Corruptions':>13} {'Det.Rate':>10}")
    print("-"*70)
    for r in results:
        print(
            f"{r.mode:<20} {r.n_agents:>4} {r.total_injections:>12} "
            f"{r.total_cross_shard_detected:>10} "
            f"{r.total_corruptions:>13} "
            f"{r.detection_rate:>10.1%}"
        )
    print("-"*70)
    print("Note: corruptions column MUST be 0 for v2_with_readset rows.")
    print("      Non-zero corruptions for v1_no_readset confirm §8.9 failure mode.")


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Cross-shard phantom read validation")
    ap.add_argument("--agents",  type=int, nargs="+", default=[4, 8],
                    help="Agent counts to test (default: 4 8)")
    ap.add_argument("--trials",  type=int, default=10,
                    help="Trials per (mode × agent_count) (default: 10)")
    ap.add_argument("--steps",   type=int, default=5,
                    help="Steps per agent per trial (default: 5)")
    ap.add_argument("--out",     type=str, default="results/cross_shard_validation.csv",
                    help="Output CSV path")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip v1_no_readset baseline (saves time if you only need the fix proof)")
    args = ap.parse_args()

    # Sanity-check server
    try:
        _CLIENT.get(f"{SBUS_URL}/stats", timeout=3).raise_for_status()
    except Exception:
        print(f"ERROR: Cannot reach S-Bus server at {SBUS_URL}")
        print("  Start it with:  cd sbus && cargo run")
        sys.exit(1)

    all_results: list[ExperimentResult] = []

    for n in args.agents:
        # Experiment A: /commit/v2 WITH read_set — should have zero corruptions
        r_v2 = run_experiment(n, args.trials, args.steps, use_readset=True)
        all_results.append(r_v2)

        if not args.skip_baseline:
            # Experiment B: /commit WITHOUT read_set — expected to have corruptions
            # This is the §8.9 failure mode that /v2 fixes.
            r_v1 = run_experiment(n, args.trials, args.steps, use_readset=False)
            all_results.append(r_v1)

    print_paper_table(all_results)
    import os; os.makedirs("results", exist_ok=True)
    write_csv(all_results, args.out)

    # Final pass/fail assertion for CI
    v2_corruptions = sum(
        r.total_corruptions for r in all_results if r.mode == "v2_with_readset"
    )
    if v2_corruptions == 0:
        print("\n✓ PASS: /commit/v2 produced zero corruptions across all trials.")
        print("  The §8.9 phantom-read gap is closed. Corollary 1 holds for multi-shard ops.")
    else:
        print(f"\n✗ FAIL: {v2_corruptions} corruptions detected with /commit/v2.")
        print("  Check engine.rs CrossShardStale validation logic.")
        sys.exit(1)


if __name__ == "__main__":
    main()