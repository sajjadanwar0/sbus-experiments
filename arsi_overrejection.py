"""
arsi_overrejection.py — ARSI over-rejection rate measurement.
==============================================================

Closes §VIII-C limitation #6 in the S-Bus paper and provides the
core experiment for Paper B ("ARSI: Automatic Read-Set Inference
for Black-Box LLM Agents").

Three conditions tested simultaneously:

    Mode A: arsi     — SDK auto-declares ALL HTTP reads (conservative).
                       This is what the paper's ARSI result uses.

    Mode B: manual   — Only true causal dependencies declared.
                       Represents the optimal/oracle read-set.

    Mode C: none     — No read-set (baseline, A1 disabled).

KEY METRIC:
    over_rejection_rate = (arsi_stale - manual_stale) / arsi_attempts

This measures how often ARSI's conservative declaration causes a
spurious CrossShardStale rejection that the oracle mode would NOT
generate — i.e., how much overhead ARSI's conservatism costs.

Experiment design:
  - N agents share K shards.
  - A concurrent injector advances non-target shards (simulating
    independent agent activity on other shards).
  - Each agent reads all shards, reasons, commits to its target.
  - In Mode A: all reads go into read_set (ARSI behavior).
  - In Mode B: only the shards the agent actually *depended on*
    for its delta are declared (one shard, the one it just read
    to get context for the target).
  - In Mode C: empty read_set.
  - We record CrossShardStale rejections per commit attempt.

Usage:
    # S-Bus server must be running on :7000
    cargo run --release &
    python arsi_overrejection.py

    # Vary parameters:
    python arsi_overrejection.py --n-agents 8 --n-shards 6 --trials 20

Output:
    results/arsi_overrejection.csv
    Printed summary matching §VIII-C paper format.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev

import httpx

BUS_URL = os.environ.get("SBUS_URL", "http://localhost:7000")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


# ── HTTP client helpers ───────────────────────────────────────────────────────

def _bus() -> httpx.Client:
    return httpx.Client(base_url=BUS_URL, timeout=10.0)


def create_shard(c: httpx.Client, key: str, content: str) -> None:
    r = c.post("/shard", json={"key": key, "content": content, "goal_tag": "arsi_exp"})
    r.raise_for_status()


def read_shard(c: httpx.Client, key: str) -> dict:
    r = c.get(f"/shard/{key}")
    r.raise_for_status()
    return r.json()


def commit_v2(c: httpx.Client, key: str, ver: int, content: str,
              agent: str, read_set: list[dict]) -> dict:
    """POST /commit/v2 — returns {"conflict": True, "code": "..."} on 409."""
    r = c.post("/commit/v2", json={
        "key": key,
        "expected_ver": ver,
        "content": content,
        "agent_id": agent,
        "read_set": read_set,
    })
    if r.status_code == 409:
        body = r.json()
        return {"conflict": True, "code": body.get("error", "unknown")}
    r.raise_for_status()
    return r.json()


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    mode:              str    # "arsi" | "manual" | "none"
    trial:             int
    n_agents:          int
    n_shards:          int
    commit_attempts:   int = 0
    cross_shard_stale: int = 0   # spurious OR real — distinguishable by mode
    version_mismatch:  int = 0
    token_conflict:    int = 0
    successes:         int = 0
    injections:        int = 0   # how many times injector advanced non-target shards
    duration_s:        float = 0.0

    @property
    def stale_rate(self) -> float:
        if self.commit_attempts == 0:
            return 0.0
        return self.cross_shard_stale / self.commit_attempts

    @property
    def success_rate(self) -> float:
        if self.commit_attempts == 0:
            return 0.0
        return self.successes / self.commit_attempts


# ── Injector thread ───────────────────────────────────────────────────────────

class Injector(threading.Thread):
    """
    Continuously advances non-target shards to create stale-read opportunities.

    In Mode A (arsi), the agent declares ALL reads, including shards the injector
    is advancing. When those shards advance between the agent's read and commit,
    ARSI correctly detects the staleness — but if the agent's delta didn't
    actually depend on that shard, the rejection is spurious (over-rejection).

    In Mode B (manual), only the truly-depended-on shard is declared. The
    injector advancing an undeclared shard does NOT cause a rejection, even
    if the agent happened to read it for context.
    """
    def __init__(self, client: httpx.Client, shard_keys: list[str],
                 target_key: str, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.c = client
        # Inject only on NON-target shards (simulating other agents)
        self.inject_keys = [k for k in shard_keys if k != target_key]
        self.target_key = target_key
        self.stop = stop_event
        self.injection_count = 0

    def run(self) -> None:
        if not self.inject_keys:
            return
        i = 0
        while not self.stop.is_set():
            key = self.inject_keys[i % len(self.inject_keys)]
            try:
                s = read_shard(self.c, key)
                resp = commit_v2(
                    self.c, key, s["version"],
                    f"[injected v{s['version']+1}]",
                    "injector",
                    []   # injector declares no read-set
                )
                if not resp.get("conflict"):
                    self.injection_count += 1
            except Exception:
                pass
            i += 1
            time.sleep(0.02)   # 50 injections/s per shard slot


# ── Single trial ─────────────────────────────────────────────────────────────

def run_trial(
    mode: str,
    trial: int,
    n_agents: int,
    n_shards: int,
    steps: int,
    pfx: str,
    step_delay: float = 0.0,
) -> TrialResult:
    """
    Run one trial in the given mode.

    mode="arsi"   → read_set = ALL shards read (ARSI behavior)
    mode="manual" → read_set = only the one shard the agent reads for
                    its target (true causal dependency only)
    mode="none"   → read_set = [] (A1 disabled)
    """
    result = TrialResult(mode=mode, trial=trial, n_agents=n_agents, n_shards=n_shards)
    t0 = time.time()

    with _bus() as c:
        # Create shards
        skeys = [f"{pfx}_{mode}_{i}" for i in range(n_shards)]
        for sk in skeys:
            create_shard(c, sk, f"[{sk}: initial state]")

        agents = [f"agent-{i}" for i in range(n_agents)]

        # Start injector: advances all shards EXCEPT the first agent's target
        stop_evt = threading.Event()
        inj_c = _bus()          # injector uses its own connection
        target_for_agent0 = skeys[0]
        injector = Injector(inj_c, skeys, target_for_agent0, stop_evt)
        injector.start()

        for step in range(steps):
            agent = agents[step % n_agents]
            # Each agent "owns" a rotating subset of shards
            owned_idx = step % n_shards
            target_key = skeys[owned_idx]

            # Read ALL shards (as ARSI would see it — every HTTP GET is recorded)
            all_reads: list[dict] = []
            for sk in skeys:
                try:
                    s = read_shard(c, sk)
                    all_reads.append({"key": sk, "version_at_read": s["version"],
                                      "is_target": sk == target_key})
                except Exception:
                    pass

            # Identify the target shard's read
            target_read = next((r for r in all_reads if r["key"] == target_key), None)
            if target_read is None:
                continue

            # Build read_set according to mode
            if mode == "arsi":
                # Declare ALL reads (conservative — ARSI behavior)
                read_set = [
                    {"key": r["key"], "version_at_read": r["version_at_read"]}
                    for r in all_reads
                    if r["key"] != target_key   # exclude write target (SDK does this)
                ]
            elif mode == "manual":
                # Declare only the ONE shard the agent truly depends on:
                # in this experiment, the agent reads one context shard
                # (the shard immediately before the target in the list) and
                # uses it to inform the delta. All other reads are "ambient."
                context_idx = (owned_idx - 1) % n_shards
                context_key = skeys[context_idx]
                context_read = next(
                    (r for r in all_reads if r["key"] == context_key), None
                )
                read_set = (
                    [{"key": context_read["key"],
                      "version_at_read": context_read["version_at_read"]}]
                    if context_read and context_key != target_key
                    else []
                )
            else:  # none
                read_set = []

            # Attempt commit with retry budget = 3
            for attempt in range(3):
                result.commit_attempts += 1
                delta = f"[{agent} step {step+1} attempt {attempt+1}]"
                resp = commit_v2(c, target_key, target_read["version_at_read"],
                                 delta, agent, read_set)

                if not resp.get("conflict"):
                    result.successes += 1
                    break

                code = resp.get("code", "")
                if "CrossShardStale" in code:
                    result.cross_shard_stale += 1
                elif "VersionMismatch" in code:
                    result.version_mismatch += 1
                elif "TokenConflict" in code:
                    result.token_conflict += 1

                # Re-read target for retry
                try:
                    s = read_shard(c, target_key)
                    target_read = {"key": target_key,
                                   "version_at_read": s["version"],
                                   "is_target": True}
                    # In ARSI mode, re-read all shards too (conservative)
                    if mode == "arsi":
                        all_reads = []
                        for sk in skeys:
                            try:
                                s2 = read_shard(c, sk)
                                all_reads.append({"key": sk,
                                                  "version_at_read": s2["version"],
                                                  "is_target": sk == target_key})
                            except Exception:
                                pass
                        read_set = [
                            {"key": r["key"], "version_at_read": r["version_at_read"]}
                            for r in all_reads if r["key"] != target_key
                        ]
                except Exception:
                    break

            # Simulate LLM inference latency — gives injector time to advance shards
            if step_delay > 0:
                time.sleep(step_delay)

        stop_evt.set()
        injector.join(timeout=2.0)
        result.injections = injector.injection_count
        inj_c.close()

    result.duration_s = time.time() - t0
    return result


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(results: list[TrialResult]) -> None:
    """
    Print the over-rejection analysis.

    Over-rejection rate = (arsi_stale - manual_stale) / arsi_attempts

    This is the key metric for §VIII-C limitation #6 and the ARSI paper.
    """
    print()
    print("=" * 68)
    print("ARSI OVER-REJECTION ANALYSIS")
    print("=" * 68)

    for mode in ["arsi", "manual", "none"]:
        trials = [r for r in results if r.mode == mode]
        if not trials:
            continue
        stale_rates = [r.stale_rate for r in trials]
        success_rates = [r.success_rate for r in trials]
        print(f"\nMode: {mode.upper()} (n={len(trials)} trials)")
        print(f"  CrossShardStale/attempt : mean={mean(stale_rates):.4f}  "
              f"median={median(stale_rates):.4f}  "
              f"stdev={stdev(stale_rates) if len(stale_rates)>1 else 0:.4f}")
        print(f"  Success rate            : mean={mean(success_rates):.3f}")
        print(f"  Total injections        : {sum(r.injections for r in trials)}")

    arsi_t   = [r for r in results if r.mode == "arsi"]
    manual_t = [r for r in results if r.mode == "manual"]

    if arsi_t and manual_t:
        arsi_stale   = sum(r.cross_shard_stale for r in arsi_t)
        manual_stale = sum(r.cross_shard_stale for r in manual_t)
        arsi_att     = sum(r.commit_attempts   for r in arsi_t)

        over_rej = (arsi_stale - manual_stale) / max(arsi_att, 1)
        print()
        print("─" * 68)
        print(f"ARSI over-rejection rate : {over_rej:.4f}  "
              f"({over_rej*100:.2f}% of ARSI commits rejected spuriously)")
        print(f"  ARSI  CrossShardStale  : {arsi_stale}")
        print(f"  Manual CrossShardStale : {manual_stale}")
        print(f"  ARSI  commit attempts  : {arsi_att}")
        print()
        if over_rej < 0.05:
            print("Verdict: ARSI over-rejection is LOW (<5%). "
                  "Conservative read-set has negligible practical overhead.")
        elif over_rej < 0.15:
            print("Verdict: ARSI over-rejection is MODERATE (5–15%). "
                  "Acceptable for most deployments; quantify per task length.")
        else:
            print("Verdict: ARSI over-rejection is HIGH (>15%). "
                  "Recommend read-set pruning or dependency tagging for long tasks.")
        print("─" * 68)


# ── CSV output ────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "mode", "trial", "n_agents", "n_shards",
    "commit_attempts", "cross_shard_stale", "version_mismatch",
    "token_conflict", "successes", "injections",
    "stale_rate", "success_rate", "duration_s",
]


def save_csv(results: list[TrialResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in results:
            w.writerow({
                "mode":             r.mode,
                "trial":            r.trial,
                "n_agents":         r.n_agents,
                "n_shards":         r.n_shards,
                "commit_attempts":  r.commit_attempts,
                "cross_shard_stale": r.cross_shard_stale,
                "version_mismatch": r.version_mismatch,
                "token_conflict":   r.token_conflict,
                "successes":        r.successes,
                "injections":       r.injections,
                "stale_rate":       f"{r.stale_rate:.6f}",
                "success_rate":     f"{r.success_rate:.6f}",
                "duration_s":       f"{r.duration_s:.2f}",
            })
    print(f"\nResults saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-agents", type=int, default=4,
                    help="Number of concurrent agents per trial (default: 4)")
    ap.add_argument("--n-shards", type=int, default=4,
                    help="Number of shared shards (default: 4)")
    ap.add_argument("--steps",    type=int, default=30,
                    help="Commit steps per trial (default: 30)")
    ap.add_argument("--trials",   type=int, default=10,
                    help="Independent trials per mode (default: 10)")
    ap.add_argument("--step-delay", type=float, default=0.5,
                    help="Seconds to sleep per step to simulate LLM latency (default: 0.5).\n"
                         "At 0.5s/step with 30 steps: ~15s window -> ~750 injection attempts.\n"
                         "Set to 0 to disable (reverts to CPU-speed mode, always underpowered).")
    ap.add_argument("--out",      type=Path,
                    default=OUT_DIR / "arsi_overrejection.csv")
    args = ap.parse_args()

    # Verify server is up
    try:
        with _bus() as c:
            c.get("/stats").raise_for_status()
    except Exception as e:
        print(f"ERROR: S-Bus server not reachable at {BUS_URL}: {e}")
        print("Start with:  cargo run --release")
        sys.exit(1)

    print(f"ARSI over-rejection experiment")
    print(f"  agents={args.n_agents}  shards={args.n_shards}  "
          f"steps={args.steps}  trials={args.trials} per mode")
    print(f"  step_delay={args.step_delay}s  "
          f"window_per_trial={args.steps * args.step_delay:.0f}s  "
          f"expected_injections={int(args.steps * args.step_delay * 50)}/trial")
    print(f"  Total trials: {args.trials * 3} "
          f"(arsi + manual + none)")
    print()

    all_results: list[TrialResult] = []
    modes = ["arsi", "manual", "none"]

    for trial in range(args.trials):
        for mode in modes:
            pfx = uuid.uuid4().hex[:8]
            print(f"  Trial {trial+1}/{args.trials}  mode={mode:<6}  pfx={pfx}", end="", flush=True)
            r = run_trial(
                mode=mode,
                trial=trial,
                n_agents=args.n_agents,
                n_shards=args.n_shards,
                steps=args.steps,
                pfx=pfx,
                step_delay=args.step_delay,
            )
            all_results.append(r)
            print(f"  stale={r.cross_shard_stale}/{r.commit_attempts} "
                  f"({r.stale_rate:.3f})  inj={r.injections}  "
                  f"t={r.duration_s:.1f}s")

    analyse(all_results)
    save_csv(all_results, args.out)


if __name__ == "__main__":
    main()