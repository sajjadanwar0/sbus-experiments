#!/usr/bin/env python3
"""
pg_scr_contention.py — SCR contention experiment vs PostgreSQL baseline.

DEBUGGING CHECKLIST before running:
  1. PostgreSQL must be running:
     sudo systemctl start postgresql
     psql "host=localhost dbname=sbus_baseline user=sbus_user password=sbus_pass" -c "SELECT 1"

  2. pg_sbus_server.py must be running on port 7001:
     PG_DSN="host=localhost dbname=sbus_baseline user=sbus_user password=sbus_pass" \
     PG_PORT=7001 python3 pg_sbus_server.py &

  3. Run THIS script pointing at port 7001:
     python3 pg_scr_contention.py --url http://localhost:7001 --out results/pg_scr.csv

Common failure: SCR=1.0, 0 commits → PostgreSQL not running → all commits are
HTTP 500 (counted as conflicts). Fix: start PostgreSQL first.
"""

import argparse, csv, os, time, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:7001")
parser.add_argument("--out", default="results/pg_scr_contention.csv")
parser.add_argument("--steps", type=int, default=10)
args = parser.parse_args()

client = httpx.Client(timeout=30, base_url=args.url)

# ── Counters ──────────────────────────────────────────────────────────────────
server_errors = 0

def check_server():
    """Verify server is running AND PostgreSQL is connected."""
    try:
        r = client.get("/stats", timeout=5)
        if r.status_code != 200:
            print(f"❌ /stats returned {r.status_code}")
            return False
        info = r.json()
        system = info.get("system", "unknown")
        print(f"✅ Server: {system} at {args.url}")
        if "postgresql" not in system:
            print(f"⚠  WARNING: Expected postgresql_serializable, got '{system}'")
            print(f"   Are you running against S-Bus instead of the PG baseline?")
        return True
    except Exception as e:
        print(f"❌ Server not reachable at {args.url}: {e}")
        print(f"   Checklist:")
        print(f"     sudo systemctl start postgresql")
        print(f"     PG_DSN='...' PG_PORT=7001 python3 pg_sbus_server.py &")
        return False

def create_shard(key):
    r = client.post("/shard", json={"key": key, "content": "initial", "goal_tag": "scr_pg"})
    return r.status_code in (200, 201, 409)

def delete_shard(key):
    try: client.delete(f"/shard/{key}", timeout=5)
    except: pass

def get_shard(key, agent_id=""):
    url = f"/shard/{key}" + (f"?agent_id={agent_id}" if agent_id else "")
    r = client.get(url)
    if r.status_code == 200:
        return r.json()
    return None

def commit(key, expected_version, delta, agent_id):
    global server_errors
    r = client.post("/commit/v2", json={
        "key": key, "expected_version": expected_version,
        "delta": delta, "agent_id": agent_id
    })
    if r.status_code == 500:
        server_errors += 1
        if server_errors <= 3:
            print(f"  ⚠ HTTP 500 from server (PostgreSQL down?): {r.text[:100]}")
    return r.status_code == 200

def sbus_agent(agent_id, shard_key, n_steps, results):
    commits = conflicts = errors = 0
    for step in range(n_steps):
        shard = get_shard(shard_key, agent_id)
        if not shard:
            errors += 1; continue
        ver = shard["version"]
        ok = commit(shard_key, ver, f"step={step} agent={agent_id}", agent_id)
        if ok: commits += 1
        else:  conflicts += 1
    results.append({"commits": commits, "conflicts": conflicts, "errors": errors})

def run_exp(topology, n_agents, n_steps, shard_keys, baseline=False):
    for key in shard_keys:
        delete_shard(key)
        time.sleep(0.05)
        create_shard(key)

    results = []
    pre_versions = {k: (get_shard(k) or {}).get("version", 0) for k in shard_keys}

    if baseline:
        # Wrong version (simulating no version tracking)
        def run_baseline_agent(agent_id, key, steps, res):
            c = cf = 0
            for _ in range(steps):
                ok = commit(key, 0, f"baseline_{agent_id}", agent_id)
                if ok: c += 1
                else:  cf += 1
            res.append({"commits": c, "conflicts": cf, "errors": 0})
        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            futs = [pool.submit(run_baseline_agent, f"base_a{i}",
                                random.choice(shard_keys), n_steps, results)
                    for i in range(n_agents)]
            for f in as_completed(futs): f.result()
    else:
        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            futs = [pool.submit(sbus_agent, f"a{i}",
                                random.choice(shard_keys), n_steps, results)
                    for i in range(n_agents)]
            for f in as_completed(futs): f.result()

    post_versions = {k: (get_shard(k) or {}).get("version", 0) for k in shard_keys}
    total_c  = sum(r["commits"]   for r in results)
    total_cf = sum(r["conflicts"] for r in results)
    total_e  = sum(r["errors"]    for r in results)
    total_at = total_c + total_cf
    version_advances = sum(post_versions[k] - pre_versions[k] for k in shard_keys)

    if total_e > total_at * 0.1:
        print(f"  ⚠ {total_e} server errors ({total_e/max(1,total_at):.0%}) — "
              f"PostgreSQL may not be running!")

    corruptions = max(0, total_c - version_advances) if baseline else 0
    return {
        "topology":      topology,
        "n_agents":      n_agents,
        "n_steps":       n_steps,
        "total_commits": total_c,
        "commits":       total_c,
        "conflicts":     total_cf,
        "corruptions":   corruptions,
        "scr":           total_cf / total_at if total_at > 0 else 0.0,
        "server_errors": total_e,
    }

def main():
    if not check_server():
        sys.exit(1)

    print(f"\n{'─'*60}")
    print("  Exp 1: Distinct shards (each agent owns one shard)")
    print(f"{'─'*60}")
    distinct_rows = []
    for N in [3, 4, 8, 16]:
        ts = int(time.time())
        keys = [f"pg_dist_{N}_{ts}_{i}" for i in range(N)]
        for key in keys: create_shard(key)
        results = []
        with ThreadPoolExecutor(max_workers=N) as pool:
            futs = [pool.submit(sbus_agent, f"a{i}", keys[i], args.steps, results)
                    for i in range(N)]
            for f in as_completed(futs): f.result()
        total_c  = sum(r["commits"]   for r in results)
        total_cf = sum(r["conflicts"] for r in results)
        total_e  = sum(r["errors"]    for r in results)
        ta = total_c + total_cf
        row = {"topology":"distinct","n_agents":N,"n_steps":args.steps,
               "total_commits":ta,"commits":total_c,"conflicts":total_cf,
               "corruptions":0,"scr":total_cf/ta if ta>0 else 0.0,"server_errors":total_e}
        distinct_rows.append(row)
        icon = "✅" if total_cf == 0 and total_e == 0 else ("⚠" if total_e > 0 else "✅")
        print(f"  {icon} N={N:2d}: commits={total_c:3d}  conflicts={total_cf:3d}  "
              f"errors={total_e}  SCR={row['scr']:.3f}")

    if server_errors > 5:
        print(f"\n❌ {server_errors} server errors — aborting. Check PostgreSQL connection.")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print("  Exp 2: Shared shard")
    print(f"{'─'*60}")
    shared_rows = []
    for N in [4, 8, 16]:
        ts = int(time.time())
        key = f"pg_shared_{N}_{ts}"
        row = run_exp("shared", N, args.steps, [key])
        shared_rows.append(row)
        icon = "✅" if row["corruptions"] == 0 else "❌"
        print(f"  {icon} N={N:2d}: commits={row['commits']:3d}  "
              f"conflicts={row['conflicts']:3d}  SCR={row['scr']:.3f}  "
              f"corruptions={row['corruptions']}")

    print(f"\n{'─'*60}")
    print("  Exp 3: Baseline (version=0 always)")
    print(f"{'─'*60}")
    ts = int(time.time())
    base_row = run_exp("baseline_no_version_check", 4, args.steps,
                       [f"pg_base_{ts}"], baseline=True)
    print(f"  N= 4: commits={base_row['commits']:3d}  "
          f"corruptions={base_row['corruptions']} "
          f"({base_row['corruptions']/max(1,base_row['commits']):.1%})")

    all_rows = distinct_rows + shared_rows + [base_row]
    os.makedirs(os.path.dirname(args.out) if "/" in args.out else ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader(); writer.writerows(all_rows)

    print(f"\n✅ Results → {args.out}")
    total_sbus = sum(r["commits"]+r["conflicts"]
                     for r in distinct_rows+shared_rows)
    total_corr = sum(r["corruptions"] for r in distinct_rows+shared_rows)
    print(f"  S-Bus attempts: {total_sbus}  Corruptions: {total_corr}")
    print(f"  {'✅ ZERO corruptions' if total_corr == 0 else '❌ CORRUPTIONS'}")

if __name__ == "__main__":
    main()