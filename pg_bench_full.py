import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import httpx

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pg_bench_full")


def load_tasks(n_tasks: int) -> list[dict]:
    try:
        from swe_bench_lite import TASKS as SWE_TASKS  # type: ignore
        log.info(f"Loaded {len(SWE_TASKS)} tasks from swe_bench_lite.py")
        if n_tasks > len(SWE_TASKS):
            log.warning(
                f"Requested {n_tasks} tasks but swe_bench_lite has only "
                f"{len(SWE_TASKS)}. Using all available."
            )
            n_tasks = len(SWE_TASKS)
        return SWE_TASKS[:n_tasks]
    except ImportError:
        log.warning(
            "swe_bench_lite.py not importable — using synthetic stubs."
        )
        return _synthetic_tasks(n_tasks)


def _synthetic_tasks(n: int) -> list[dict]:
    domains = [
        ("django.queryset.ordering", "ORM analyst", "Patch designer", "Test engineer"),
        ("astropy.fits.headers",     "FITS analyst", "Header patcher", "Test engineer"),
        ("sympy.solver.diophantine", "Solver analyst", "Algorithm patcher", "Test engineer"),
        ("scikit.estimator.predict", "Estimator analyst", "API patcher", "Test engineer"),
        ("requests.session.adapter", "Session analyst", "Adapter patcher", "Test engineer"),
    ]
    out = []
    for i in range(n):
        d, *roles = domains[i % len(domains)]
        out.append({
            "id": f"synthetic__{d}__{i:03d}",
            "repo": "synthetic",
            "description": f"Synthetic coordination workload #{i:03d} on {d}.",
            "shards": [
                {"key": "bug_analysis",  "role": roles[0], "initial": "No analysis yet."},
                {"key": "patch_plan",    "role": roles[1], "initial": "No patch yet."},
                {"key": "test_strategy", "role": roles[2], "initial": "No strategy yet."},
            ],
            "min_steps": 6,
        })
    return out


@dataclass
class Backend:
    name: str
    url: str
    expected_409_status: int = 409


BACKENDS = {
    "sbus":  Backend(name="sbus",  url=os.getenv("SBUS_URL",  "http://localhost:7000")),
    "pg":    Backend(name="pg",    url=os.getenv("PG_URL",    "http://localhost:7001")),
    "redis": Backend(name="redis", url=os.getenv("REDIS_URL", "http://localhost:7002")),
}

@dataclass
class RunResult:
    task_id: str
    backend: str
    n_agents: int
    repeat: int
    wall_time_s: float
    commit_attempts: int
    commits_succeeded: int
    conflicts_409: int
    type_i_corruptions: int
    final_versions: dict
    success: bool
    error: Optional[str] = None
    started_at: str = ""
    ended_at: str = ""

    def csv_row(self) -> dict:
        d = asdict(self)
        d["final_versions"] = json.dumps(d["final_versions"])
        return d


CSV_FIELDS = [
    "task_id", "backend", "n_agents", "repeat",
    "wall_time_s", "commit_attempts", "commits_succeeded",
    "conflicts_409", "type_i_corruptions", "final_versions",
    "success", "error", "started_at", "ended_at",
]

def run_one(task: dict, backend: Backend, n_agents: int, repeat: int,
            steps: int, openai_key: str, timeout_s: float) -> RunResult:
    started = time.time()
    started_iso = time.strftime("%FT%TZ", time.gmtime(started))

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

    shard_keys = [s["key"] for s in task["shards"]]
    session_id = f"{task['id']}:{backend.name}:N{n_agents}:r{repeat}"

    try:
        with httpx.Client(base_url=backend.url, timeout=timeout_s) as client:
            try:
                r_reset = client.post("/admin/reset")
                if r_reset.status_code not in (200, 403, 404):
                    log.debug(
                        f"admin/reset -> {r_reset.status_code}: {r_reset.text[:100]}"
                    )
            except Exception as e:
                log.debug(f"admin/reset skipped: {e}")

            for s in task["shards"]:
                payload = {
                    "key":      s["key"],
                    "content":  s["initial"],
                    "goal_tag": s.get("goal_tag", s["key"]),
                }
                r = client.post("/admin/shard", json=payload)
                if r.status_code in (403, 404, 405):
                    r = client.post("/shard", json=payload)
                if r.status_code not in (200, 201, 409):
                    raise RuntimeError(
                        f"shard create {s['key']} -> {r.status_code}: "
                        f"{r.text[:200]}"
                    )

            agent_ids = [f"a{i:02d}" for i in range(n_agents)]
            agent_shard = [shard_keys[i % len(shard_keys)] for i in range(n_agents)]
            for aid in agent_ids:
                try:
                    client.post("/session", json={"agent_id": aid})
                except Exception:
                    pass

            for step in range(steps):
                for ai in range(n_agents):
                    own_key = agent_shard[ai]
                    sib_key = shard_keys[(shard_keys.index(own_key) + 1)
                                          % len(shard_keys)]
                    aid = agent_ids[ai]

                    r1 = client.get(
                        f"/shard/{own_key}",
                        params={"agent_id": aid},
                    )
                    if r1.status_code != 200:
                        raise RuntimeError(
                            f"GET {own_key} -> {r1.status_code}: {r1.text[:200]}"
                        )
                    own_state = r1.json()
                    own_version = own_state.get("version", own_state.get("v", 0))

                    r2 = client.get(
                        f"/shard/{sib_key}",
                        params={"agent_id": aid},
                    )
                    if r2.status_code != 200:
                        raise RuntimeError(
                            f"GET {sib_key} -> {r2.status_code}: {r2.text[:200]}"
                        )
                    sib_state = r2.json()
                    sib_version = sib_state.get("version", sib_state.get("v", 0))

                    delta = (
                        f"step{step}.agent{ai}.delta: synthetic content, "
                        f"task={task['id']}"
                    )
                    commit_payload = {
                        "key": own_key,
                        "expected_version": own_version,
                        "delta": delta,
                        "agent_id": aid,
                        "read_set": [
                            {"key": sib_key, "version_at_read": sib_version},
                        ],
                    }
                    r3 = client.post("/commit/v2", json=commit_payload)
                    res.commit_attempts += 1

                    if r3.status_code == 200:
                        res.commits_succeeded += 1
                        new_state = r3.json()
                        new_version = new_state.get(
                            "new_version",
                            new_state.get("version", new_state.get("v", -1)),
                        )
                        if new_version != own_version + 1:
                            res.type_i_corruptions += 1
                            log.warning(
                                f"Type-I signal: {backend.name} {aid} "
                                f"committed {own_key} expected v{own_version} "
                                f"+ 1 = v{own_version+1}, got v{new_version}"
                            )
                    elif r3.status_code == backend.expected_409_status:
                        res.conflicts_409 += 1
                    else:
                        raise RuntimeError(
                            f"COMMIT {own_key} -> {r3.status_code}: {r3.text[:200]}"
                        )

            for k in shard_keys:
                r = client.get(
                    f"/shard/{k}",
                    params={"agent_id": "harness"},
                )
                if r.status_code == 200:
                    js = r.json()
                    res.final_versions[k] = js.get("version", js.get("v", -1))

            res.success = True

    except Exception as e:
        res.error = f"{type(e).__name__}: {str(e)[:300]}"
        log.error(f"[{session_id}] FAIL: {res.error}")
        log.debug(traceback.format_exc())

    res.wall_time_s = time.time() - started
    res.ended_at = time.strftime("%FT%TZ", time.gmtime(time.time()))
    return res


def planned_runs(tasks: list[dict], backends: list[str],
                 agent_counts: list[int], repeats: int) -> list[tuple]:
    plan = []
    for ti, t in enumerate(tasks):
        for b in backends:
            for n in agent_counts:
                for r in range(repeats):
                    plan.append((ti, b, n, r))
    return plan


def run_key(task_id: str, backend: str, n: int, r: int) -> str:
    return f"{task_id}|{backend}|N{n}|r{r}"


def load_completed(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            done.add(run_key(row["task_id"], row["backend"],
                             int(row["n_agents"]), int(row["repeat"])))
    return done


def append_result(csv_path: Path, res: RunResult) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(res.csv_row())


def write_progress(progress_path: Path, completed: int, total: int,
                   current: Optional[str] = None) -> None:
    progress_path.write_text(json.dumps({
        "completed": completed,
        "total": total,
        "fraction": completed / total if total else 0.0,
        "current": current or "",
        "updated": time.strftime("%FT%TZ", time.gmtime()),
    }, indent=2))


def preflight(backends: list[str]) -> bool:
    import httpx
    ok = True
    for bname in backends:
        b = BACKENDS[bname]
        try:
            r = httpx.get(f"{b.url}/stats", timeout=5.0)
            if r.status_code == 200:
                log.info(f"  {bname:6s} reachable at {b.url} (stats OK)")
            else:
                log.warning(
                    f"  {bname:6s} at {b.url} returned {r.status_code} "
                    f"on /stats — proceeding anyway"
                )
        except Exception as e:
            log.error(f"  {bname:6s} at {b.url} NOT REACHABLE: {e}")
            ok = False
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", nargs="+", default=["sbus", "pg"],
                   choices=list(BACKENDS.keys()))
    p.add_argument("--tasks", type=int, default=30,
                   help="Number of tasks to use from swe_bench_lite (default 30)")
    p.add_argument("--agent-counts", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--steps", type=int, default=6,
                   help="Coordination steps per run (default 6)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="HTTP timeout per request in seconds")
    p.add_argument("--out", default="results/pg_comparison_full.csv")
    p.add_argument("--progress", default="results/pg_comparison_progress.json")
    p.add_argument("--resume", action="store_true",
                   help="Skip runs already in --out")
    p.add_argument("--analyse-only", action="store_true",
                   help="Print summary from existing CSV, do not run")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Parallel workers (default 1; backends are stateful "
                        "single-process — increasing this risks corruption "
                        "of OTHER concurrent runs sharing keyspace)")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY", ""))

    args = p.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress)

    if args.analyse_only:
        return analyse(out_path)

    tasks = load_tasks(args.tasks)
    plan = planned_runs(tasks, args.backends, args.agent_counts, args.repeats)
    log.info(f"Planned {len(plan)} runs: "
             f"{len(tasks)} tasks × {len(args.agent_counts)} N × "
             f"{args.repeats} repeats × {len(args.backends)} backends")

    if not args.skip_preflight:
        log.info("Pre-flight: checking backend reachability …")
        if not preflight(args.backends):
            log.error("Pre-flight failed. Use --skip-preflight to override.")
            return 2

    completed = load_completed(out_path) if args.resume else set()
    if args.resume and completed:
        log.info(f"Resume: {len(completed)} runs already in {out_path}, "
                 f"will skip them")

    todo = []
    for ti, b, n, r in plan:
        key = run_key(tasks[ti]["id"], b, n, r)
        if key in completed:
            continue
        todo.append((ti, b, n, r))
    log.info(f"To run: {len(todo)} (skipping {len(plan) - len(todo)} already done)")

    if args.max_workers > 1:
        log.warning(
            f"max_workers={args.max_workers} — backends are stateful "
            "and shared between runs; using >1 worker risks cross-run "
            "interference. Proceed only if you know your backends "
            "isolate by session_id."
        )

    started_overall = time.time()
    done_count = len(completed)
    total = len(plan)

    for idx, (ti, b, n, r) in enumerate(todo):
        task = tasks[ti]
        backend = BACKENDS[b]
        key = run_key(task["id"], b, n, r)
        log.info(f"[{idx+1}/{len(todo)}] {key}")
        write_progress(progress_path, done_count, total, current=key)
        res = run_one(task, backend, n, r, args.steps,
                      args.openai_key, args.timeout)
        append_result(out_path, res)
        done_count += 1
        elapsed = time.time() - started_overall
        if idx > 0:
            avg = elapsed / (idx + 1)
            eta = avg * (len(todo) - idx - 1)
            log.info(
                f"  -> success={res.success}, attempts={res.commit_attempts}, "
                f"409={res.conflicts_409}, corruptions={res.type_i_corruptions}, "
                f"wall={res.wall_time_s:.1f}s, "
                f"avg={avg:.1f}s, ETA={eta/3600:.1f}h"
            )

    write_progress(progress_path, done_count, total, current="DONE")
    log.info(f"DONE. {done_count}/{total} runs in CSV.")
    return analyse(out_path)


def analyse(csv_path: Path) -> int:
    if not csv_path.exists():
        log.error(f"{csv_path} not found.")
        return 1

    rows = list(csv.DictReader(csv_path.open()))
    log.info(f"Loaded {len(rows)} runs from {csv_path}")

    print("\n" + "=" * 78)
    print(f"{'Backend':<8} {'N':>3} {'runs':>5} {'attempts':>9} "
          f"{'commits':>9} {'409s':>6} {'corrupt':>8} {'mean_wall':>10}")
    print("-" * 78)
    grouped = {}
    for row in rows:
        if row.get("success") not in ("True", "true", True):
            continue
        key = (row["backend"], int(row["n_agents"]))
        g = grouped.setdefault(key, {
            "runs": 0, "attempts": 0, "commits": 0,
            "conflicts": 0, "corruptions": 0, "wall": 0.0,
        })
        g["runs"] += 1
        g["attempts"] += int(row["commit_attempts"])
        g["commits"]  += int(row["commits_succeeded"])
        g["conflicts"] += int(row["conflicts_409"])
        g["corruptions"] += int(row["type_i_corruptions"])
        g["wall"] += float(row["wall_time_s"])

    for (b, n), g in sorted(grouped.items()):
        mean_wall = g["wall"] / g["runs"] if g["runs"] else 0
        print(f"{b:<8} {n:>3} {g['runs']:>5} {g['attempts']:>9} "
              f"{g['commits']:>9} {g['conflicts']:>6} "
              f"{g['corruptions']:>8} {mean_wall:>9.1f}s")
    print("=" * 78)

    total_corruptions = sum(g["corruptions"] for g in grouped.values())
    total_attempts = sum(g["attempts"] for g in grouped.values())
    print(f"\nGRAND TOTALS: {total_attempts} commit attempts, "
          f"{total_corruptions} Type-I corruptions across all backends.")
    if total_corruptions == 0:
        print("  -> Property 1 (ORI Safety) holds empirically across the full "
              "PG-Comparison sweep.")
        print("  -> Refutes the 'any DB CC system would suffice' null "
              "hypothesis can ONLY be made on the PG side; safety-parity "
              "is established on the SBUS side.")

    fail_rows = [r for r in rows if r.get("success") not in ("True", "true", True)]
    if fail_rows:
        print(f"\n  WARNING: {len(fail_rows)} runs failed (excluded from "
              "analysis above):")
        for r in fail_rows[:10]:
            print(f"    {r['task_id']:<50s} {r['backend']:<6s} "
                  f"N{r['n_agents']:>2s} r{r['repeat']:<2s} "
                  f"{r.get('error', '')[:60]}")
        if len(fail_rows) > 10:
            print(f"    ... and {len(fail_rows) - 10} more")

    return 0 if total_corruptions == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("Interrupted. Use --resume to continue from where you left off.")
        sys.exit(130)
