import argparse
import concurrent.futures as cf
import csv
import json
import os
import random
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
import httpx

BACKENDS = {
    "sbus":          os.getenv("SBUS_URL", "http://localhost:7000"),
    "pg_ser_rs":     os.getenv("PG_URL",   "http://localhost:7001"),
    "redis_watch_rs": os.getenv("REDIS_URL", "http://localhost:7002"),
}
N_AGENTS_LIST = [int(x) for x in os.getenv("N_AGENTS", "4,8,16,32").split(",")]
N_REPEATS = int(os.getenv("N_REPEATS", "3"))
N_ATTEMPTS_PER_AGENT = int(os.getenv("N_ATTEMPTS", "100"))
RETRY_BUDGET = int(os.getenv("RETRY_BUDGET", "20"))

OUT_CSV = os.getenv("OUT_CSV", "results/pg_contention.csv")
OUT_SUMMARY = os.getenv("OUT_SUMMARY", "results/pg_contention_summary.json")

def http_get(base: str, path: str, params: dict | None = None) -> tuple[int, dict]:
    try:
        r = httpx.get(f"{base}{path}", params=params, timeout=10.0)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text[:200]}
    except httpx.RequestError as e:
        return -1, {"err": str(e)}

def http_post(base: str, path: str, body: dict) -> tuple[int, dict]:
    try:
        r = httpx.post(f"{base}{path}", json=body, timeout=30.0)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text[:200]}
    except httpx.RequestError as e:
        return -1, {"err": str(e)}

def backend_healthcheck(base: str) -> bool:
    s, _ = http_get(base, "/stats")
    if s == 200:
        return True
    s, _ = http_get(base, "/health")
    return s == 200

def reset_backend(base: str) -> None:
    http_post(base, "/admin/reset", {})


@dataclass
class AgentOutcome:
    agent_id: str
    n_attempts: int
    n_success: int
    n_409: int
    n_other_err: int
    n_retries: int
    wall_time_s: float

def run_agent(backend_base: str, shard_key: str, agent_id: str,
               n_attempts: int, retry_budget: int) -> AgentOutcome:
    t0 = time.time()
    n_success = n_409 = n_other = n_retries = 0

    for i in range(n_attempts):
        attempts_this_commit = 0
        committed = False
        while attempts_this_commit < retry_budget and not committed:
            attempts_this_commit += 1
            s, data = http_get(backend_base, f"/shard/{shard_key}",
                                {"agent_id": agent_id})
            if s != 200:
                n_other += 1
                break
            expected_v = data.get("version", 0)
            delta = f"[agent={agent_id} attempt={i} retry={attempts_this_commit}]"
            s, resp = http_post(backend_base, "/commit/v2", {
                "key": shard_key,
                "expected_version": expected_v,
                "delta": delta,
                "agent_id": agent_id,
                "read_set": [],
            })
            if s == 200:
                n_success += 1
                committed = True
            elif s == 409:
                n_409 += 1
                if attempts_this_commit < retry_budget:
                    n_retries += 1
                    time.sleep(random.uniform(0.001, 0.005))
                    continue
                break
            else:
                n_other += 1
                break

    return AgentOutcome(
        agent_id=agent_id,
        n_attempts=n_attempts,
        n_success=n_success,
        n_409=n_409,
        n_other_err=n_other,
        n_retries=n_retries,
        wall_time_s=round(time.time() - t0, 3),
    )

def safety_check(backend_base: str, shard_key: str,
                  expected_successes: int, starting_version: int) -> dict:
    s, data = http_get(backend_base, f"/shard/{shard_key}")
    final_v = data.get("version", -1) if s == 200 else -1
    expected_final = starting_version + expected_successes
    return {
        "final_version": final_v,
        "expected_final": expected_final,
        "version_gap": final_v - expected_final,
        "type1_corruption": final_v != expected_final,
    }

def run_one_cell(backend_name: str, backend_base: str, n_agents: int,
                  repeat: int, n_attempts_per_agent: int) -> dict:
    shard_key = f"contention_{backend_name}_{n_agents}_{repeat}_{uuid.uuid4().hex[:6]}"
    reset_backend(backend_base)
    http_post(backend_base, "/shard", {
        "key": shard_key,
        "content": "[initial]",
        "goal_tag": f"pg_contention_{n_agents}",
    })
    s, data = http_get(backend_base, f"/shard/{shard_key}")
    start_v = data.get("version", 0) if s == 200 else 0

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n_agents) as pool:
        futures = [
            pool.submit(run_agent, backend_base, shard_key,
                         f"agent_{i:03d}",
                         n_attempts_per_agent, RETRY_BUDGET)
            for i in range(n_agents)
        ]
        outcomes = [f.result() for f in futures]
    wall = time.time() - t0

    total_attempts = sum(o.n_attempts for o in outcomes)
    total_success = sum(o.n_success for o in outcomes)
    total_409 = sum(o.n_409 for o in outcomes)
    total_retries = sum(o.n_retries for o in outcomes)
    total_other = sum(o.n_other_err for o in outcomes)

    safety = safety_check(backend_base, shard_key, total_success, start_v)

    scr = total_409 / max(1, total_attempts + total_retries)
    commit_rate = total_success / max(0.001, wall)
    mean_retry = total_retries / max(1, total_success)

    return {
        "backend": backend_name,
        "n_agents": n_agents,
        "repeat": repeat,
        "shard_key": shard_key,
        "n_attempts": total_attempts,
        "n_success": total_success,
        "n_409": total_409,
        "n_retries": total_retries,
        "n_other_err": total_other,
        "scr": round(scr, 4),
        "commit_rate_per_s": round(commit_rate, 3),
        "mean_retries_per_success": round(mean_retry, 3),
        "wall_time_s": round(wall, 3),
        "final_version": safety["final_version"],
        "expected_final_version": safety["expected_final"],
        "version_gap": safety["version_gap"],
        "type1_corruption": safety["type1_corruption"],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Only do a quick healthcheck, no full sweep")
    args = ap.parse_args()

    ok = True
    for name, base in BACKENDS.items():
        h = backend_healthcheck(base)
        print(f"  {name:16s} at {base}: {'OK' if h else 'DOWN'}")
        ok = ok and h
    if not ok:
        print("ERROR: one or more backends are down.", file=sys.stderr)
        sys.exit(1)
    if args.dry_run:
        return

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)

    all_rows: list[dict] = []
    for backend_name, backend_base in BACKENDS.items():
        for n_agents in N_AGENTS_LIST:
            for rep in range(N_REPEATS):
                print(f"\n=== {backend_name} | N={n_agents} | rep={rep+1}/{N_REPEATS} ===",
                      flush=True)
                row = run_one_cell(backend_name, backend_base, n_agents,
                                    rep, N_ATTEMPTS_PER_AGENT)
                print(f"  attempts={row['n_attempts']:5d}  success={row['n_success']:5d}  "
                      f"409s={row['n_409']:5d}  SCR={row['scr']:.3f}  "
                      f"rate={row['commit_rate_per_s']:.1f}/s  "
                      f"type1={row['type1_corruption']}", flush=True)
                all_rows.append(row)

    if all_rows:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nWrote {len(all_rows)} rows to {OUT_CSV}")

    summary: dict = {
        "experiment": "pg_contention",
        "total_attempts": sum(r["n_attempts"] + r["n_retries"] for r in all_rows),
        "total_success": sum(r["n_success"] for r in all_rows),
        "total_409": sum(r["n_409"] for r in all_rows),
        "total_type1_corruptions": sum(1 for r in all_rows if r["type1_corruption"]),
        "per_backend_per_n": {},
    }
    for backend_name in BACKENDS:
        summary["per_backend_per_n"][backend_name] = {}
        for n in N_AGENTS_LIST:
            rows = [r for r in all_rows if r["backend"] == backend_name
                                         and r["n_agents"] == n]
            if not rows:
                continue
            summary["per_backend_per_n"][backend_name][str(n)] = {
                "n_repeats": len(rows),
                "total_attempts": sum(r["n_attempts"] + r["n_retries"] for r in rows),
                "total_success": sum(r["n_success"] for r in rows),
                "total_409": sum(r["n_409"] for r in rows),
                "mean_scr": round(statistics.mean(r["scr"] for r in rows), 4),
                "mean_commit_rate": round(statistics.mean(r["commit_rate_per_s"] for r in rows), 3),
                "mean_retries_per_success": round(statistics.mean(r["mean_retries_per_success"] for r in rows), 3),
                "type1_corruptions": sum(1 for r in rows if r["type1_corruption"]),
            }

    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {OUT_SUMMARY}")
    print(json.dumps(summary, indent=2))

    print("\n=== HEADLINE ===")
    print(f"Total commit attempts (incl. retries): {summary['total_attempts']}")
    print(f"Total successes:                        {summary['total_success']}")
    print(f"Total 409 conflicts:                    {summary['total_409']}")
    print(f"Total Type-I corruptions:               {summary['total_type1_corruptions']}")
    if summary["total_type1_corruptions"] == 0:
        print("RESULT: all three backends maintain safety under shared-shard contention.")
        print("  => 'safety parity under contention' claim empirically supported.")
    else:
        print("RESULT: Type-I corruption detected — investigate before claiming parity.")

if __name__ == "__main__":
    main()
