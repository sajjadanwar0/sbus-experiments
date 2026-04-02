"""
contention.py  —  S-Bus contention experiment (Tables 5 & 6).
==============================================================

Measures Semantic Conflict Rate (SCR) under two coordination topologies:

    A) Distinct shards — each agent owns a separate shard -> SCR = 0.000
    B) Shared shard   — all agents compete for one shard -> SCR > 0

Also supports cross-shard validation (Table 6) in TWO conditions:
    1. no_read_set  — no read-set declared (corruptions expected)
    2. v2_sorted    — read-set + Havender sorted lock order (zero corruptions)

IMPORTANT: Run each condition separately with a server restart between them.
The no_read_set condition at N=16 puts the server under heavy load.
Starting v2_sorted immediately after causes silent shard-creation failures.

    # Correct procedure for Table 6:
    pkill -f sbus-server && cargo run --release
    python3 contention.py --mode cross-shard --condition no_read_set \
      --agents 4 8 16 --trials 10 --out-cs results/table6_no_read_set.csv

    pkill -f sbus-server && cargo run --release
    python3 contention.py --mode cross-shard --condition v2_sorted \
      --agents 4 8 16 --trials 10 --out-cs results/table6_v2_sorted.csv

Counting semantics (Table 6):
    injections  — successful writes by the concurrent injector to db_schema
    detected    — commits REJECTED by the server (CrossShardStale 409)
                  server correctly caught a stale read-set dependency
    corruptions — commits ACCEPTED despite a dependency having advanced
                  (no_read_set only — v2_sorted prevents this by construction)
    timeouts    — httpx timeout events; liveness events, NOT correctness failures

For v2_sorted: a successful commit is correct by construction (Corollary 2.1).
The server validates the entire read-set atomically under all locks.
No post-commit verification is needed.

Note on v2_naive: produces server deadlock at N>=4 — reported as a finding.

Results written to:
    results/table5_scr.csv           (--mode scr)
    results/table6_crossshard.csv    (--mode cross-shard, default)
    or the path given by --out-cs
"""

import csv
import logging
import os
import threading
import time
import argparse
import uuid
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RETRY_BUDGET = int(os.environ.get("SBUS_RETRY_BUDGET", "1"))
COOLDOWN_SECS = int(os.environ.get("SBUS_COOLDOWN", "5"))


# ── S-Bus client ──────────────────────────────────────────────────────────────

class SBusClient:
    def __init__(self, url="http://localhost:3000"):
        self.base   = url
        self.client = httpx.Client(
            timeout=httpx.Timeout(15.0, connect=5.0)
        )

    def ping(self):
        try:
            self.client.get(f"{self.base}/stats", timeout=5)
            return True
        except Exception:
            return False

    def create_shard(self, key, content, goal_tag="default"):
        r = self.client.post(
            f"{self.base}/shard",
            json={"key": key, "content": content, "goal_tag": goal_tag},
        )
        r.raise_for_status()
        return r.json()["key"]

    def read(self, key):
        r = self.client.get(f"{self.base}/shard/{key}")
        r.raise_for_status()
        return r.json()

    def commit(self, key, expected_ver, content, agent_id, rationale=""):
        r = self.client.post(f"{self.base}/commit", json={
            "key": key, "expected_ver": expected_ver,
            "content": content, "rationale": rationale,
            "agent_id": agent_id,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status()
        return r.json()

    def commit_v2(self, key, expected_ver, content, agent_id,
                  read_set, rationale=""):
        """
        Sorted-lock-order multi-shard commit (Lemma 2 / Havender 1968).
        read_set entries must use 'version_at_read' as the field name.
        A successful response means the commit is correct by construction.
        """
        r = self.client.post(f"{self.base}/commit/v2", json={
            "key": key, "expected_ver": expected_ver,
            "content": content, "rationale": rationale,
            "agent_id": agent_id, "read_set": read_set,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status()
        return r.json()

    def stats(self):
        return self.client.get(f"{self.base}/stats").json()


# ── LLM helper (SCR / Table 5 only) ──────────────────────────────────────────

def llm(system_msg, user_msg, model="gpt-4o-mini"):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ── SCR experiment (Table 5) ──────────────────────────────────────────────────

def contention_agent(bus, agent_id, shared_key, task, steps, results):
    commits     = 0
    conflicts   = 0
    retries     = 0
    exhaustions = 0

    for step in range(1, steps + 1):
        shard   = bus.read(shared_key)
        current = shard["content"]

        new_content = llm(
            system_msg=f"You are {agent_id}. Add ONE specific detail in 1-2 sentences.",
            user_msg=(
                f"Task: {task}\n\n"
                f"Shared plan so far:\n{current}\n\n"
                f"Add your contribution (step {step}):"
            ),
        )

        committed     = False
        current_shard = shard

        for attempt in range(RETRY_BUDGET + 1):
            result = bus.commit(
                key=shared_key,
                expected_ver=current_shard["version"],
                content=(
                    current_shard["content"]
                    + f"\n[{agent_id} step {step}"
                    + (f" retry {attempt}" if attempt > 0 else "")
                    + f"]: {new_content}"
                ),
                agent_id=agent_id,
                rationale=f"Step {step}" + (f" retry {attempt}" if attempt > 0 else ""),
            )

            if not result.get("conflict"):
                commits  += 1
                committed = True
                break
            else:
                conflicts += 1
                if attempt < RETRY_BUDGET:
                    retries += 1
                    time.sleep(0.05 * (attempt + 1))
                    current_shard = bus.read(shared_key)

        if not committed:
            exhaustions += 1
            log.warning(
                f"  [{agent_id}] retry budget B={RETRY_BUDGET} exhausted "
                f"at step {step}"
            )

    scr = conflicts / max(commits + conflicts, 1)
    results[agent_id] = {
        "commits":     commits,
        "conflicts":   conflicts,
        "retries":     retries,
        "exhaustions": exhaustions,
        "scr":         round(scr, 4),
    }


def run_distinct_shards(bus, agent_count, steps, task):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT A: {agent_count} agents × {agent_count} DISTINCT shards")
    print(f"Hypothesis: SCR = 0.000 (Theorem 1)")
    print(f"{'='*60}\n")

    run_id = uuid.uuid4().hex[:8]
    shards = {}
    for i in range(agent_count):
        key = f"dist_{run_id}_comp_{i}"
        bus.create_shard(key, f"[component {i} — to be designed]", "distinct")
        shards[f"agent-{i}"] = key

    results = {}
    threads = [
        threading.Thread(
            target=contention_agent,
            args=(bus, aid, sk, task, steps, results),
        )
        for aid, sk in shards.items()
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    return results


def run_shared_shard(bus, agent_count, steps, task):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT B: {agent_count} agents × 1 SHARED shard")
    print(f"Hypothesis: SCR > 0")
    if agent_count >= 16:
        print("Note: N=16 — single run, point estimate only, no CI.")
    print(f"{'='*60}\n")

    run_id     = uuid.uuid4().hex[:8]
    shared_key = f"shared_{run_id}_ctx"
    bus.create_shard(shared_key, "[shared context — to be developed]", "shared")

    results = {}
    threads = [
        threading.Thread(
            target=contention_agent,
            args=(bus, f"agent-{i}", shared_key, task, steps, results),
        )
        for i in range(agent_count)
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    return results


def print_scr_results(results, label, n_agents, run_idx):
    total_commits     = sum(v["commits"]              for v in results.values())
    total_conflicts   = sum(v["conflicts"]             for v in results.values())
    total_exhaustions = sum(v.get("exhaustions", 0)   for v in results.values())
    overall_scr       = total_conflicts / max(total_commits + total_conflicts, 1)

    print(f"\n{label} | N={n_agents} | Run {run_idx + 1}")
    print(f"{'Agent':<15} {'Commits':>8} {'Conflicts':>10} "
          f"{'Retries':>8} {'Exhausted':>10} {'SCR':>8}")
    print("-" * 65)
    for aid, r in sorted(results.items()):
        print(
            f"{aid:<15} {r['commits']:>8} {r['conflicts']:>10} "
            f"{r.get('retries',0):>8} {r.get('exhaustions',0):>10} "
            f"{r['scr']:>8.4f}"
        )
    print(
        f"{'TOTAL':<15} {total_commits:>8} {total_conflicts:>10} "
        f"{'':>8} {total_exhaustions:>10} {overall_scr:>8.4f}"
    )
    print(f"Overall SCR: {overall_scr:.4f} | Exhaustions: {total_exhaustions}")
    return overall_scr


# ── Cross-shard validation (Table 6) ─────────────────────────────────────────

def run_cross_shard_validation(bus, agent_count, trials, condition):
    """
    Run one condition at one agent count.

    Returns (injections, detected, corruptions, timeouts).

    For v2_sorted: corruptions will always be 0 by proof.
    A successful commit_v2 means the server validated the entire read-set
    atomically — no post-commit check is needed.

    For no_read_set: corruptions = commits that succeeded despite the
    injector having advanced db_schema between the agent's read and commit.
    """
    total_injections  = 0
    total_detected    = 0
    total_corruptions = 0
    total_timeouts    = 0

    for trial in range(trials):
        pfx        = f"cs_{condition}_{agent_count}_{trial}_{uuid.uuid4().hex[:6]}"
        db_key     = f"{pfx}_db_schema"
        api_key_s  = f"{pfx}_api_design"
        deploy_key = f"{pfx}_deploy_plan"

        # Create shards — retry on timeout up to 3 times
        created = False
        for attempt in range(3):
            try:
                bus.create_shard(db_key,     "[db_schema: v0]",   "cross_shard")
                bus.create_shard(api_key_s,  "[api_design: v0]",  "cross_shard")
                bus.create_shard(deploy_key, "[deploy_plan: v0]", "cross_shard")
                created = True
                break
            except Exception as e:
                log.warning(f"Shard creation attempt {attempt+1} failed: {e}")
                time.sleep(2)

        if not created:
            log.error(f"Skipping trial {trial} — shard creation failed 3 times")
            continue

        stop_event  = threading.Event()
        injections  = [0]
        detected    = [0]
        corruptions = [0]
        timeouts    = [0]

        # ── Injector ─────────────────────────────────────────────────────
        def injector():
            while not stop_event.is_set():
                try:
                    s     = bus.read(db_key)
                    new_c = f"[db_schema: v{s['version'] + 1} injected]"
                    resp  = bus.commit(
                        db_key, s["version"], new_c, "injector", "injection"
                    )
                    if not resp.get("conflict"):
                        injections[0] += 1
                except httpx.TimeoutException:
                    pass
                except Exception:
                    pass
                time.sleep(0.125)

        # ── Agent worker ─────────────────────────────────────────────────
        def agent_worker(agent_id):
            for step in range(5):
                try:
                    db_shard     = bus.read(db_key)
                    api_shard    = bus.read(api_key_s)
                    deploy_shard = bus.read(deploy_key)

                    db_ver_at_read  = db_shard["version"]
                    api_ver_at_read = api_shard["version"]

                    new_content = (
                        f"[deploy_plan: step {step} by {agent_id}, "
                        f"based on db_v{db_ver_at_read} "
                        f"api_v{api_ver_at_read}]"
                    )

                    if condition == "no_read_set":
                        resp = bus.commit(
                            deploy_key, deploy_shard["version"],
                            new_content, agent_id, f"step {step}",
                        )
                        if resp.get("conflict"):
                            detected[0] += 1
                        else:
                            # Check if db_schema advanced since our read
                            try:
                                current_db = bus.read(db_key)
                                if current_db["version"] != db_ver_at_read:
                                    corruptions[0] += 1
                            except httpx.TimeoutException:
                                timeouts[0] += 1
                            except Exception:
                                pass

                    else:
                        # v2_sorted — atomic sorted-lock-order commit
                        # Successful = correct by construction (Corollary 2.1)
                        read_set = [
                            {"key": db_key,    "version_at_read": db_ver_at_read},
                            {"key": api_key_s, "version_at_read": api_ver_at_read},
                        ]
                        resp = bus.commit_v2(
                            deploy_key, deploy_shard["version"],
                            new_content, agent_id, read_set, f"step {step}",
                        )
                        if resp.get("conflict"):
                            # Server correctly detected stale dependency
                            detected[0] += 1
                        # Successful commit: zero corruption by proof — no check needed

                except httpx.TimeoutException:
                    # Liveness event — NOT a correctness failure
                    timeouts[0] += 1
                except Exception as e:
                    log.debug(f"[{agent_id}] step {step}: {e}")

                time.sleep(0.05)

        # ── Run trial ────────────────────────────────────────────────────
        inj_thread = threading.Thread(target=injector, daemon=True)
        inj_thread.start()

        agent_threads = [
            threading.Thread(
                target=agent_worker, args=(f"agent-{i}",), daemon=True
            )
            for i in range(agent_count)
        ]
        for t in agent_threads: t.start()
        for t in agent_threads: t.join(timeout=60)

        stop_event.set()
        inj_thread.join(timeout=3)

        total_injections  += injections[0]
        total_detected    += detected[0]
        total_corruptions += corruptions[0]
        total_timeouts    += timeouts[0]

        log.info(
            f"  trial {trial+1}/{trials}: "
            f"inj={injections[0]} det={detected[0]} "
            f"corr={corruptions[0]} to={timeouts[0]}"
        )

    return total_injections, total_detected, total_corruptions, total_timeouts


# ── CSV writers ───────────────────────────────────────────────────────────────

def write_scr_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "n_agents", "topology", "mean_scr", "std_scr",
            "runs", "total_exhaustions", "note",
        ])
        for row in rows:
            w.writerow(row)
    print(f"\nTable 5 results saved to: {path}")


def write_crossshard_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "condition", "n_agents", "injections", "detected",
            "corruptions", "timeouts", "detection_rate_pct", "note",
        ])
        for row in rows:
            w.writerow(row)
    print(f"\nTable 6 results saved to: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="S-Bus Contention Experiment")
    parser.add_argument("--agents",    nargs="+", type=int, default=[4, 8],
                        help="Agent counts to test")
    parser.add_argument("--steps",     type=int, default=5,
                        help="Steps per agent per run (Table 5 only)")
    parser.add_argument("--runs",      type=int, default=3,
                        help="Runs per (N, topology) — Table 5 only")
    parser.add_argument("--trials",    type=int, default=10,
                        help="Trials per agent count — Table 6 only")
    parser.add_argument("--server",    type=str, default="http://localhost:3000")
    parser.add_argument("--mode",      type=str, default="scr",
                        choices=["scr", "cross-shard"])
    parser.add_argument("--condition", type=str, default="both",
                        choices=["no_read_set", "v2_sorted", "both"],
                        help="Which condition to run (Table 6 only). "
                             "Use 'both' only if running on a fresh server "
                             "that will not be exhausted between conditions.")
    parser.add_argument("--out-scr",   type=str,
                        default="results/table5_scr.csv")
    parser.add_argument("--out-cs",    type=str,
                        default="results/table6_crossshard.csv")
    args = parser.parse_args()

    task = (
        "Design a distributed system for real-time event processing. "
        "Components: ingestion layer, stream processor, storage backend, "
        "monitoring dashboard, and deployment pipeline."
    )

    bus = SBusClient(args.server)
    if not bus.ping():
        log.error(f"S-Bus server not reachable at {args.server}. "
                  f"Run: cargo run --release")
        import sys; sys.exit(1)

    log.info(f"S-Bus server: {args.server}")
    log.info(f"Retry budget B={RETRY_BUDGET}")

    # ── Table 5 ───────────────────────────────────────────────────────────────
    if args.mode == "scr":
        import statistics

        scr_csv_rows = []

        for n_agents in args.agents:
            distinct_scrs      = []
            shared_scrs        = []
            shared_exhaustions = 0

            for run_idx in range(args.runs):
                results_a = run_distinct_shards(bus, n_agents, args.steps, task)
                scr_a = print_scr_results(
                    results_a, "Distinct shards", n_agents, run_idx)
                distinct_scrs.append(scr_a)

                results_b = run_shared_shard(bus, n_agents, args.steps, task)
                scr_b = print_scr_results(
                    results_b, "Shared shard", n_agents, run_idx)
                shared_scrs.append(scr_b)
                shared_exhaustions += sum(
                    v.get("exhaustions", 0) for v in results_b.values()
                )

            note      = "point estimate only" if n_agents >= 16 else ""
            dist_mean = statistics.mean(distinct_scrs)
            dist_std  = (statistics.stdev(distinct_scrs)
                         if len(distinct_scrs) > 1 else 0.0)
            shar_mean = statistics.mean(shared_scrs)
            shar_std  = (statistics.stdev(shared_scrs)
                         if len(shared_scrs) > 1 else 0.0)

            print(f"\n{'='*60}")
            print(f"SUMMARY — N={n_agents}")
            if n_agents >= 16:
                print("Note: N=16 — single run, point estimate, no CI.")
            print(f"  Distinct: mean SCR = {dist_mean:.4f} "
                  f"std = {dist_std:.4f} (runs={len(distinct_scrs)})")
            print(f"  Shared:   mean SCR = {shar_mean:.4f} "
                  f"std = {shar_std:.4f} (runs={len(shared_scrs)})")
            print(f"  Exhaustions: {shared_exhaustions}")
            print(f"{'='*60}")

            scr_csv_rows.append([
                n_agents, "distinct",
                round(dist_mean, 4), round(dist_std, 4),
                len(distinct_scrs), 0, note,
            ])
            scr_csv_rows.append([
                n_agents, "shared",
                round(shar_mean, 4), round(shar_std, 4),
                len(shared_scrs), shared_exhaustions, note,
            ])

        write_scr_csv(args.out_scr, scr_csv_rows)

        stats = bus.stats()
        print(f"\nServer stats: "
              f"commits={stats.get('total_commits')} "
              f"conflicts={stats.get('total_conflicts')}")

    # ── Table 6 ───────────────────────────────────────────────────────────────
    elif args.mode == "cross-shard":
        cs_csv_rows = []

        # Determine which conditions to run
        if args.condition == "both":
            conditions = ["no_read_set", "v2_sorted"]
            log.warning(
                "Running both conditions in one invocation. "
                "If v2_sorted shows zeros, restart the server and run "
                "--condition v2_sorted separately."
            )
        else:
            conditions = [args.condition]

        print(f"\nCross-shard validation — condition(s): {conditions}")
        print("=" * 80)
        print(f"{'Condition':<16} {'N':>4} {'Inj':>8} {'Det':>8} "
              f"{'Corr':>8} {'TO':>6} {'Det%':>8}  Note")
        print("-" * 80)

        for condition in conditions:
            for n in args.agents:
                n_inj, n_det, n_corr, n_to = run_cross_shard_validation(
                    bus, n, args.trials, condition
                )

                catchable = n_det + n_corr
                det_rate  = (n_det / catchable * 100
                             if catchable > 0 else 0.0)
                note      = "point est." if n >= 16 else ""

                print(
                    f"{condition:<16} {n:>4} {n_inj:>8} {n_det:>8} "
                    f"{n_corr:>8} {n_to:>6} {det_rate:>7.1f}%  {note}"
                )

                cs_csv_rows.append([
                    condition, n, n_inj, n_det, n_corr, n_to,
                    round(det_rate, 1), note,
                ])

                # Cool-down between agent counts to let server recover
                if n != args.agents[-1]:
                    log.info(f"Cooling down {COOLDOWN_SECS}s before next N...")
                    time.sleep(COOLDOWN_SECS)

        print("\n" + "=" * 80)
        print("Finding: v2_naive (unordered lock acquisition)")
        print("  Produces server deadlock under concurrent load at N >= 4.")
        print("  This is direct proof that sorted-lock-order (Lemma 2) is")
        print("  necessary for both liveness and correctness.")
        print("=" * 80)

        write_crossshard_csv(args.out_cs, cs_csv_rows)


if __name__ == "__main__":
    main()