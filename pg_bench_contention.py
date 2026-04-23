#!/usr/bin/env python3
"""
pg_bench_contention.py — Contention-arm complement to pg_bench_full.py
========================================================================
This addresses the reviewer critique that pg_bench_full.py's zero-conflict
result proves nothing about the CC mechanism. The v47.1 experiment used
(a) dedicated-shard topology and (b) sequential within-run agent execution,
either of which alone guarantees zero contention.

This harness fixes both:

  1. TOPOLOGY — every agent writes to the SAME "hotspot" shard each step,
     forcing write-write conflicts at the CC layer. Each agent also owns
     one side shard (for DeliveryLog cross-shard read-set entries).

  2. EXECUTION — the N agents in a single step run CONCURRENTLY via
     asyncio.gather, so the CC layer actually sees N in-flight commits
     competing for the hotspot lock / transaction slot / WATCH guard.

Expected result per step under contention:
  - Exactly one commit wins.
  - N-1 commits return 409 (CrossShardStale / VersionMismatch on S-Bus,
    serialization_failure on PG-SER, null EXEC on Redis-WATCH).
  - Zero Type-I corruptions — any accepted commit must satisfy
    new_version == expected_version + 1.

If S-Bus catches conflicts that PG-SER or Redis-WATCH miss, that's the
cross-shard read-set reconstruction contribution showing up. If all
three detect the same conflicts (the expected outcome) that's CC-parity
under contention — which is what the paper actually wants to claim.

The three-way divergence is the paper story either way.

Run pattern:
  # Pre-flight (same as pg_bench_full.py)
  cargo run --release --manifest-path ../rust-server/Cargo.toml   # port 7000
  python3 pg_sbus_server.py                                        # port 7001
  python3 redis_sbus_server.py                                     # port 7002

  # Smoke test first (5 min):
  python3 pg_bench_contention.py --smoke

  # Full sweep (~3-6h on modern laptop, purely localhost HTTP):
  python3 pg_bench_contention.py --backends sbus pg redis \\
      --tasks 30 --agent-counts 4 16 64 --repeats 3 \\
      --out results/pg_comparison_contention.csv

Output CSV schema is IDENTICAL to pg_comparison_full.csv so the existing
analyse() logic and any downstream plotting code works unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import traceback
from pathlib import Path

# Reuse everything we can from the dedicated-shard harness.
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from pg_bench_full import (  # type: ignore
    BACKENDS,
    Backend,
    RunResult,
    load_tasks,
    run_key,
    load_completed,
    append_result,
    write_progress,
    preflight,
    analyse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pg_bench_contention")


# ── Contention workload ──────────────────────────────────────────────────────

# Every run creates these shards. The agents all hammer HOTSPOT; each agent
# also owns one SIDE shard for DeliveryLog cross-shard read-set entries.
HOTSPOT_KEY_PREFIX = "hotspot"
SIDE_KEY_PREFIX    = "side"


async def _create_shard(client, backend: Backend, key: str, initial: str,
                        goal_tag: str) -> None:
    """Create a shard via admin endpoint, falling back to public /shard."""
    payload = {"key": key, "content": initial, "goal_tag": goal_tag}
    r = await client.post("/admin/shard", json=payload)
    if r.status_code in (403, 404, 405):
        # admin disabled or endpoint not present (old PG adapter)
        r = await client.post("/shard", json=payload)
    if r.status_code not in (200, 201, 409):
        raise RuntimeError(
            f"shard create {key} -> {r.status_code}: {r.text[:200]}"
        )


async def _reset(client) -> None:
    """Best-effort reset; ignored if admin disabled."""
    try:
        r = await client.post("/admin/reset")
        if r.status_code not in (200, 403, 404):
            log.debug(f"admin/reset -> {r.status_code}: {r.text[:100]}")
    except Exception as e:
        log.debug(f"admin/reset skipped: {e}")


async def _agent_step(client, backend: Backend, agent_id: str,
                      hotspot_key: str, side_key: str,
                      step: int, task_id: str) -> dict:
    """
    One contending step for one agent:
      1. GET hotspot -> records hotspot version in DeliveryLog
      2. GET own side shard -> adds side_key to read-set
      3. POST /commit/v2 on hotspot with expected=hotspot_version and
         read_set=[(side_key, side_version)]

    Returns a dict describing the outcome:
      { "attempted": True, "committed": bool,
        "conflict_409": bool, "corruption_signal": bool,
        "err": Optional[str], "latency_ms": float }
    """
    t0 = time.time()
    result = {
        "attempted":         True,
        "committed":         False,
        "conflict_409":      False,
        "corruption_signal": False,
        "err":               None,
        "latency_ms":        0.0,
    }

    try:
        # 1. GET hotspot -- records version in DeliveryLog
        r1 = await client.get(f"/shard/{hotspot_key}",
                              params={"agent_id": agent_id})
        if r1.status_code != 200:
            raise RuntimeError(
                f"GET {hotspot_key} -> {r1.status_code}: {r1.text[:120]}"
            )
        hs = r1.json()
        hot_version = hs.get("version", hs.get("v", 0))

        # 2. GET own side shard -- cross-shard read-set entry
        r2 = await client.get(f"/shard/{side_key}",
                              params={"agent_id": agent_id})
        if r2.status_code != 200:
            raise RuntimeError(
                f"GET {side_key} -> {r2.status_code}: {r2.text[:120]}"
            )
        side_state = r2.json()
        side_version = side_state.get("version", side_state.get("v", 0))

        # 3. COMMIT hotspot with expected version + cross-shard read-set
        delta = f"step{step}.agent{agent_id}.delta task={task_id}"
        payload = {
            "key":              hotspot_key,
            "expected_version": hot_version,
            "delta":            delta,
            "agent_id":         agent_id,
            "read_set":         [{"key": side_key,
                                  "version_at_read": side_version}],
        }
        r3 = await client.post("/commit/v2", json=payload)

        if r3.status_code == 200:
            # Commit accepted. Now the corruption check:
            # the new version MUST be exactly expected_version + 1.
            body = r3.json()
            new_version = body.get(
                "new_version",
                body.get("version", body.get("v", -1))
            )
            if new_version != hot_version + 1:
                # Got accepted but version jump is inconsistent with
                # "this agent committed on top of the version it read"
                # => Type-I corruption signal.
                result["corruption_signal"] = True
            result["committed"] = True

        elif r3.status_code == backend.expected_409_status:
            # Conflict detected — the CC mechanism rejected.
            # Under hotspot contention this is the *expected* outcome
            # for N-1 of the N agents.
            result["conflict_409"] = True

        else:
            raise RuntimeError(
                f"COMMIT {hotspot_key} -> {r3.status_code}: {r3.text[:200]}"
            )

    except Exception as e:
        result["err"] = f"{type(e).__name__}: {str(e)[:200]}"

    result["latency_ms"] = (time.time() - t0) * 1000.0
    return result


async def run_one_contention(task: dict, backend: Backend, n_agents: int,
                             repeat: int, steps: int,
                             timeout_s: float) -> RunResult:
    """
    Contention variant: N agents concurrently hammer one hotspot shard
    per step. Each agent has a dedicated side shard for the cross-shard
    read-set entry so that ORI's DeliveryLog actually has a non-trivial
    read-set to validate.

    Returns the SAME RunResult schema as pg_bench_full.run_one so that
    analyse() and any downstream tools keep working.
    """
    import httpx  # lazy import keeps module importable offline

    started     = time.time()
    started_iso = time.strftime("%FT%TZ", time.gmtime(started))

    # Use task id so shard keys are unique per run — avoids stale-state
    # cross-contamination in backends that don't implement admin/reset.
    session_id  = f"{task['id']}:{backend.name}:N{n_agents}:r{repeat}"
    hotspot_key = f"{HOTSPOT_KEY_PREFIX}__{session_id.replace(':', '_')}"
    side_keys   = [f"{SIDE_KEY_PREFIX}__{session_id.replace(':', '_')}__a{i:02d}"
                   for i in range(n_agents)]
    agent_ids   = [f"a{i:02d}" for i in range(n_agents)]

    res = RunResult(
        task_id=task["id"],
        backend=backend.name,
        n_agents=n_agents,
        repeat=repeat,
        wall_time_s=0.0,
        commit_attempts=0,
        commits_succeeded=0,
        conflicts_409=0,
        type_i_corruptions=0,
        final_versions={},
        success=False,
        started_at=started_iso,
    )

    try:
        async with httpx.AsyncClient(base_url=backend.url,
                                     timeout=timeout_s) as client:
            # ── setup ─────────────────────────────────────────────────────
            await _reset(client)

            # Hotspot + one side shard per agent
            await _create_shard(client, backend, hotspot_key,
                                "hotspot initial", "hotspot")
            for sk in side_keys:
                await _create_shard(client, backend, sk,
                                    "side initial", "side")

            # Sessions — best-effort; PG/Redis adapters may ignore
            for aid in agent_ids:
                try:
                    await client.post("/session", json={"agent_id": aid})
                except Exception:
                    pass

            # ── contention loop ───────────────────────────────────────────
            for step in range(steps):
                # All N agents race for the hotspot concurrently
                coros = [
                    _agent_step(client, backend, agent_ids[i],
                                hotspot_key, side_keys[i],
                                step, task["id"])
                    for i in range(n_agents)
                ]
                outcomes = await asyncio.gather(*coros,
                                                return_exceptions=False)

                step_commits = 0
                for oc in outcomes:
                    res.commit_attempts += 1
                    if oc["err"] is not None:
                        # Hard transport-level failure on this attempt;
                        # record as non-commit, non-409.
                        log.debug(f"  err: {oc['err']}")
                        continue
                    if oc["committed"]:
                        res.commits_succeeded += 1
                        step_commits += 1
                        if oc["corruption_signal"]:
                            res.type_i_corruptions += 1
                            log.warning(
                                f"Type-I CORRUPTION: {backend.name} step={step} "
                                f"session={session_id}"
                            )
                    elif oc["conflict_409"]:
                        res.conflicts_409 += 1

                # Sanity: under true contention at most 1 commit per step
                # should succeed. Log if more than 1 does, but don't fail
                # the run — this could legitimately happen with PG-SER's
                # SSI retry amplification inside its own adapter.
                if step_commits > 1:
                    log.info(
                        f"  multi-commit step: {backend.name} step={step} "
                        f"commits={step_commits} (investigate)"
                    )

            # ── final state ───────────────────────────────────────────────
            r = await client.get(f"/shard/{hotspot_key}",
                                 params={"agent_id": "harness"})
            if r.status_code == 200:
                js = r.json()
                res.final_versions[hotspot_key] = js.get(
                    "version", js.get("v", -1)
                )

            res.success = True

    except Exception as e:
        res.error = f"{type(e).__name__}: {str(e)[:300]}"
        log.error(f"[{session_id}] FAIL: {res.error}")
        log.debug(traceback.format_exc())

    res.wall_time_s = time.time() - started
    res.ended_at    = time.strftime("%FT%TZ", time.gmtime(time.time()))
    return res


# ── Driver ───────────────────────────────────────────────────────────────

def planned_runs(tasks, backends, agent_counts, repeats):
    plan = []
    for ti, t in enumerate(tasks):
        for b in backends:
            for n in agent_counts:
                for r in range(repeats):
                    plan.append((ti, b, n, r))
    return plan


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backends", nargs="+", default=["sbus", "pg", "redis"],
                   choices=list(BACKENDS.keys()))
    p.add_argument("--tasks", type=int, default=30)
    p.add_argument("--agent-counts", nargs="+", type=int, default=[4, 16, 64])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--out", default="results/pg_comparison_contention.csv")
    p.add_argument("--progress",
                   default="results/pg_comparison_contention_progress.json")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--analyse-only", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Quick smoke test: N=4, 2 tasks, 1 repeat, "
                        "all backends. Finishes in a few minutes.")
    args = p.parse_args()

    if args.smoke:
        args.tasks = 2
        args.agent_counts = [4]
        args.repeats = 1
        args.out = "results/pg_comparison_contention_smoke.csv"
        args.progress = "results/pg_comparison_contention_smoke_progress.json"
        log.info("SMOKE TEST: N=4, 2 tasks, 1 repeat, all backends")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress)

    if args.analyse_only:
        return analyse(out_path)

    tasks = load_tasks(args.tasks)
    plan = planned_runs(tasks, args.backends, args.agent_counts, args.repeats)
    log.info(
        f"Planned {len(plan)} contention runs: {len(tasks)} tasks × "
        f"{len(args.agent_counts)} N × {args.repeats} repeats × "
        f"{len(args.backends)} backends"
    )
    log.info(
        "  Topology: HOTSPOT (all N agents write to one shard concurrently)"
    )
    log.info(
        "  Expected per step under true contention: 1 commit, "
        "N-1 conflicts_409, 0 corruptions"
    )

    if not args.skip_preflight:
        log.info("Pre-flight: checking backend reachability …")
        if not preflight(args.backends):
            log.error("Pre-flight failed. Use --skip-preflight to override.")
            return 2

    completed = load_completed(out_path) if args.resume else set()
    if args.resume and completed:
        log.info(f"Resume: {len(completed)} runs already in {out_path}")

    todo = []
    for ti, b, n, r in plan:
        key = run_key(tasks[ti]["id"], b, n, r)
        if key in completed:
            continue
        todo.append((ti, b, n, r))
    log.info(f"To run: {len(todo)} (skipping {len(plan)-len(todo)} completed)")

    started_overall = time.time()
    done_count = len(completed)
    total = len(plan)

    for idx, (ti, b, n, r) in enumerate(todo):
        task = tasks[ti]
        backend = BACKENDS[b]
        key = run_key(task["id"], b, n, r)
        log.info(f"[{idx+1}/{len(todo)}] {key}")
        write_progress(progress_path, done_count, total, current=key)
        try:
            res = asyncio.run(run_one_contention(
                task, backend, n, r, args.steps, args.timeout))
        except Exception as e:
            log.error(f"run_one_contention raised: {e}")
            log.debug(traceback.format_exc())
            # Write a failure row so --resume doesn't re-run it indefinitely
            res = RunResult(
                task_id=task["id"], backend=b, n_agents=n, repeat=r,
                wall_time_s=0.0, commit_attempts=0, commits_succeeded=0,
                conflicts_409=0, type_i_corruptions=0,
                final_versions={}, success=False,
                error=f"{type(e).__name__}: {str(e)[:200]}",
                started_at=time.strftime("%FT%TZ", time.gmtime()),
                ended_at=time.strftime("%FT%TZ", time.gmtime()),
            )
        append_result(out_path, res)
        done_count += 1

        # Per-run summary — visible progress signal
        expected_conflicts = (n - 1) * args.steps
        conflict_ratio = (res.conflicts_409 / expected_conflicts
                          if expected_conflicts else 0)
        elapsed = time.time() - started_overall
        avg = elapsed / (idx + 1)
        eta_h = avg * (len(todo) - idx - 1) / 3600.0
        log.info(
            f"  -> success={res.success}  attempts={res.commit_attempts}  "
            f"commits={res.commits_succeeded}  409={res.conflicts_409}/"
            f"{expected_conflicts} ({conflict_ratio:.0%})  "
            f"corrupt={res.type_i_corruptions}  wall={res.wall_time_s:.1f}s  "
            f"ETA={eta_h:.1f}h"
        )

    write_progress(progress_path, done_count, total, current="DONE")
    log.info(f"DONE. {done_count}/{total} runs in CSV.")
    return analyse(out_path)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("Interrupted. Use --resume to continue.")
        sys.exit(130)
