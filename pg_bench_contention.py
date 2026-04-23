from __future__ import annotations
import argparse
import asyncio
import logging
import sys
import time
import traceback
from pathlib import Path
import httpx

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

HOTSPOT_KEY_PREFIX = "hotspot"
SIDE_KEY_PREFIX    = "side"


async def _create_shard(client, backend: Backend, key: str, initial: str,
                        goal_tag: str) -> None:
    payload = {"key": key, "content": initial, "goal_tag": goal_tag}
    r = await client.post("/admin/shard", json=payload)
    if r.status_code in (403, 404, 405):
        r = await client.post("/shard", json=payload)
    if r.status_code not in (200, 201, 409):
        raise RuntimeError(
            f"shard create {key} -> {r.status_code}: {r.text[:200]}"
        )


async def _reset(client) -> None:
    try:
        r = await client.post("/admin/reset")
        if r.status_code not in (200, 403, 404):
            log.debug(f"admin/reset -> {r.status_code}: {r.text[:100]}")
    except Exception as e:
        log.debug(f"admin/reset skipped: {e}")


async def _agent_step(client, backend: Backend, agent_id: str,
                      hotspot_key: str, side_key: str,
                      step: int, task_id: str) -> dict:
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
        r1 = await client.get(f"/shard/{hotspot_key}",
                              params={"agent_id": agent_id})
        if r1.status_code != 200:
            raise RuntimeError(
                f"GET {hotspot_key} -> {r1.status_code}: {r1.text[:120]}"
            )
        hs = r1.json()
        hot_version = hs.get("version", hs.get("v", 0))

        r2 = await client.get(f"/shard/{side_key}",
                              params={"agent_id": agent_id})
        if r2.status_code != 200:
            raise RuntimeError(
                f"GET {side_key} -> {r2.status_code}: {r2.text[:120]}"
            )
        side_state = r2.json()
        side_version = side_state.get("version", side_state.get("v", 0))

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
            body = r3.json()
            new_version = body.get(
                "new_version",
                body.get("version", body.get("v", -1))
            )
            if new_version != hot_version + 1:
                result["corruption_signal"] = True
            result["committed"] = True

        elif r3.status_code == backend.expected_409_status:
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
    started     = time.time()
    started_iso = time.strftime("%FT%TZ", time.gmtime(started))
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
            await _reset(client)

            await _create_shard(client, backend, hotspot_key,
                                "hotspot initial", "hotspot")
            for sk in side_keys:
                await _create_shard(client, backend, sk,
                                    "side initial", "side")
            for aid in agent_ids:
                try:
                    await client.post("/session", json={"agent_id": aid})
                except Exception:
                    pass

            for step in range(steps):
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

                if step_commits > 1:
                    log.info(
                        f"  multi-commit step: {backend.name} step={step} "
                        f"commits={step_commits} (investigate)"
                    )

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
