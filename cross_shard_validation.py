from __future__ import annotations

import argparse
import csv
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

SBUS_URL = "http://localhost:7000"

_CLIENT = httpx.Client(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
    timeout=httpx.Timeout(30.0, connect=5.0),
)

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
    r = _CLIENT.post(f"{SBUS_URL}/commit",
                     json={"key": key, "expected_version": expected_version,
                           "delta": delta, "agent_id": agent_id})
    return r.status_code == 200, r.json()

@dataclass
class TrialResult:
    trial_id:             int
    n_agents:             int
    injections:           int
    cross_shard_detected: int
    corruptions:          int
    retries:              int
    duration_s:           float
    success:              bool


@dataclass
class ExperimentResult:
    mode:                       str
    n_agents:                   int
    n_trials:                   int
    total_injections:           int
    total_cross_shard_detected: int
    total_corruptions:          int
    total_retries:              int
    detection_rate:             float
    corruption_rate:            float
    mean_duration_s:            float


class Injector:
    def __init__(self, db_schema_key: str, delay_s: float = 0.08):
        self.key   = db_schema_key
        self.delay = delay_s
        self._stop = threading.Event()
        self._count = 0
        self._lock  = threading.Lock()

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)

    def injection_count(self) -> int:
        with self._lock:
            return self._count

    def _run(self):
        while not self._stop.is_set():
            time.sleep(self.delay + random.uniform(0, 0.05))
            try:
                s  = read_shard(self.key)
                ok, _ = commit_original(
                    key=self.key,
                    expected_version=s["version"],
                    delta=(
                        f"[injected] schema v{s['version']+1} "
                        f"at {datetime.now(timezone.utc).isoformat()}"
                    ),
                    agent_id="injector-0",
                )
                if ok:
                    with self._lock:
                        self._count += 1
            except Exception:
                pass

@dataclass
class AgentStats:
    commits:           int = 0
    cross_shard_stale: int = 0
    version_mismatch:  int = 0
    retries:           int = 0
    corruptions:       int = 0


def _agent_with_readset(
    agent_id: str,
    deploy_key: str,
    db_key: str,
    api_key_shard: str,
    steps: int,
    stats: AgentStats,
    lock: threading.Lock,
):
    for step in range(steps):
        for attempt in range(6):
            try:
                db  = read_shard(db_key)
                api = read_shard(api_key_shard)
                dep = read_shard(deploy_key)

                db_ver, api_ver, dep_ver = (
                    db["version"], api["version"], dep["version"]
                )
                time.sleep(0.10 + random.uniform(0, 0.05))

                delta = (
                    f"step {step}: deploy by {agent_id} — "
                    f"refs db@v{db_ver} api@v{api_ver}"
                )

                pre_commit_db_ver = read_shard(db_key)["version"]

                ok, resp = commit_v2(
                    key=deploy_key,
                    expected_version=dep_ver,
                    delta=delta,
                    agent_id=agent_id,
                    read_set=[
                        {"key": db_key,        "version_at_read": db_ver},
                        {"key": api_key_shard, "version_at_read": api_ver},
                    ],
                )
                with lock:
                    if ok:
                        stats.commits += 1
                        if pre_commit_db_ver != db_ver:
                            stats.corruptions += 1
                        break
                    else:
                        err = resp.get("error", "")
                        if err == "CrossShardStale":
                            stats.cross_shard_stale += 1
                        elif err == "VersionMismatch":
                            stats.version_mismatch += 1
                        stats.retries += 1
                time.sleep(0.05 * (attempt + 1))
            except Exception:
                with lock:
                    stats.retries += 1
                time.sleep(0.1)


def _agent_without_readset(
    agent_id: str,
    deploy_key: str,
    db_key: str,
    api_key_shard: str,
    steps: int,
    stats: AgentStats,
    lock: threading.Lock,
):
    for step in range(steps):
        for attempt in range(6):
            try:
                db  = read_shard(db_key)
                dep = read_shard(deploy_key)

                db_ver, dep_ver = db["version"], dep["version"]
                time.sleep(0.10 + random.uniform(0, 0.05))

                delta = (
                    f"step {step}: deploy by {agent_id} — "
                    f"refs db@v{db_ver}"
                )
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
                        if pre_commit_db_ver != db_ver:
                            stats.corruptions += 1  # expected for baseline
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
) -> TrialResult:
    suffix     = f"t{trial_id}_{int(time.time()*1000)}"
    db_key     = f"db_schema_{suffix}"
    api_key    = f"api_design_{suffix}"
    deploy_key = f"deploy_plan_{suffix}"

    create_shard(db_key,     "Initial PostgreSQL schema", "db_schema")
    create_shard(api_key,    "Initial REST API design",   "api_design")
    create_shard(deploy_key, "Initial deployment plan",   "deploy_plan")

    stats    = AgentStats()
    lock     = threading.Lock()
    injector = Injector(db_key, delay_s=0.08)

    t_start = time.time()
    injector.start()

    agent_fn = _agent_with_readset if use_readset else _agent_without_readset
    threads  = []
    for i in range(n_agents):
        t = threading.Thread(
            target=agent_fn,
            args=(f"agent-{i}", deploy_key, db_key, api_key,
                  steps, stats, lock),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    injector.stop()
    dur = time.time() - t_start

    return TrialResult(
        trial_id=trial_id,
        n_agents=n_agents,
        injections=injector.injection_count(),
        cross_shard_detected=stats.cross_shard_stale,
        corruptions=stats.corruptions,
        retries=stats.retries,
        duration_s=round(dur, 2),
        success=(stats.corruptions == 0),
    )

def run_experiment(
    n_agents: int,
    n_trials: int,
    steps: int,
    use_readset: bool,
) -> ExperimentResult:
    mode = "v2_with_readset" if use_readset else "v1_no_readset"
    print(f"\n{'='*60}")
    print(f"Mode: {mode} | N={n_agents} | {n_trials} trials | {steps} steps/agent")
    print(f"{'='*60}")

    results = []
    for i in range(n_trials):
        r = run_trial(i, n_agents, steps, use_readset)
        results.append(r)
        status = "✓" if r.success else "✗ CORRUPTION"
        print(
            f"  trial {i:02d}: inj={r.injections:3d} "
            f"detected={r.cross_shard_detected:3d} "
            f"corrupt={r.corruptions} "
            f"retries={r.retries:3d} "
            f"dur={r.duration_s:.1f}s  {status}"
        )

    total_inj   = sum(r.injections for r in results)
    total_det   = sum(r.cross_shard_detected for r in results)
    total_corr  = sum(r.corruptions for r in results)
    total_retry = sum(r.retries for r in results)
    mean_dur    = sum(r.duration_s for r in results) / len(results)
    det_rate    = total_det  / max(total_inj, 1)
    corr_rate   = total_corr / max(total_inj, 1)

    print(f"\nSummary ({mode} N={n_agents}):")
    print(f"  Injections       : {total_inj}")
    print(f"  CrossShardStale  : {total_det} ({det_rate:.1%})")
    print(f"  Corruptions      : {total_corr}  ← MUST BE ZERO for /v2")
    print(f"  Retries          : {total_retry}")
    print(f"  Mean duration    : {mean_dur:.1f}s")

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


def print_paper_table(results: list[ExperimentResult]):
    print("\n" + "="*70)
    print("TABLE: Cross-Shard Validation (Table 6 / Table 9 in paper)")
    print("="*70)
    print(f"{'Mode':<22} {'N':>4} {'Injections':>12} {'Detected':>10} "
          f"{'Corruptions':>13} {'Rate':>8}")
    print("-"*70)
    for r in results:
        print(
            f"{r.mode:<22} {r.n_agents:>4} {r.total_injections:>12} "
            f"{r.total_cross_shard_detected:>10} "
            f"{r.total_corruptions:>13} "
            f"{r.corruption_rate:>8.1%}"
        )
    print("-"*70)
    print("v2_with_readset corruptions MUST be 0. Non-zero for v1 is expected.")

def main():
    ap = argparse.ArgumentParser(
        description="Cross-shard phantom-read validation (no LLM calls needed)"
    )
    ap.add_argument("--agents",        type=int, nargs="+", default=[4, 8, 16],
                    help="Agent counts to test")
    ap.add_argument("--trials",        type=int, default=10,
                    help="Trials per condition (use >=3 for N=16)")
    ap.add_argument("--steps",         type=int, default=5,
                    help="Agent steps per trial")
    ap.add_argument("--out",           type=str,
                    default="results/cross_shard_validation.csv")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip v1_no_readset (saves time)")
    ap.add_argument("--server",        type=str, default="http://localhost:7000")
    args = ap.parse_args()

    global SBUS_URL
    SBUS_URL = args.server

    try:
        _CLIENT.get(f"{SBUS_URL}/stats", timeout=3).raise_for_status()
    except Exception:
        print(f"ERROR: Cannot reach S-Bus server at {SBUS_URL}")
        print("  Start it with:  cd sbus && cargo run")
        sys.exit(1)

    all_results: list[ExperimentResult] = []

    for n in args.agents:
        r_v2 = run_experiment(n, args.trials, args.steps, use_readset=True)
        all_results.append(r_v2)

        if not args.skip_baseline:
            r_v1 = run_experiment(n, args.trials, args.steps, use_readset=False)
            all_results.append(r_v1)

    print_paper_table(all_results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(all_results, args.out)

    v2_corruptions = sum(
        r.total_corruptions for r in all_results if r.mode == "v2_with_readset"
    )
    if v2_corruptions == 0:
        print("\n✓ PASS: /commit/v2 produced zero corruptions. ARSI holds.")
    else:
        print(f"\n✗ FAIL: {v2_corruptions} corruptions detected with /commit/v2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
