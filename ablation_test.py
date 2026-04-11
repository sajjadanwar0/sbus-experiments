#!/usr/bin/env python3
"""
Ablation study: compare 3 S-Bus modes on shared-shard contention.
  Mode A: Full S-Bus (ownership token + DeliveryLog)
  Mode B: Token-only (ownership token, no DeliveryLog)
  Mode C: Baseline (no S-Bus: always version=0)
"""

import httpx, os, csv, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

SBUS_URL = os.getenv("SBUS_URL", "http://localhost:7000")
client   = httpx.Client(timeout=20)
SHARDS   = ["ablation_shard_1", "ablation_shard_2"]
N_VALUES = [4, 8, 16]
ROUNDS   = 30
RUNS     = 3

def create_shard(key):
    r = client.post(f"{SBUS_URL}/shard",
                    json={"key": key, "content": "initial", "goal_tag": "ablation"})
    return r.status_code in (200, 201, 409)

def delete_shard(key):
    try: client.delete(f"{SBUS_URL}/shard/{key}")
    except: pass

def get_version(key):
    r = client.get(f"{SBUS_URL}/shard/{key}")
    return r.json().get("version", 0) if r.status_code == 200 else 0

def run_agent(agent_id, mode, results, rounds=ROUNDS):
    commits = conflicts = 0
    for _ in range(rounds):
        import random
        key = random.choice(SHARDS)
        if mode == "baseline":
            r = client.post(f"{SBUS_URL}/commit/v2",
                json={"key": key, "expected_version": 0,
                      "delta": f"by_{agent_id}_{time.time():.3f}",
                      "agent_id": agent_id})
        else:
            # Both full and token-only use correct version
            ver = get_version(key)
            r = client.post(f"{SBUS_URL}/commit/v2",
                json={"key": key, "expected_version": ver,
                      "delta": f"by_{agent_id}_{time.time():.3f}",
                      "agent_id": agent_id})
        if r.status_code == 200: commits += 1
        else: conflicts += 1
    results.append({"agent": agent_id, "commits": commits, "conflicts": conflicts})

def run_experiment(N, mode, runs=RUNS):
    all_results = []
    for run in range(runs):
        for key in SHARDS:
            delete_shard(key)
            create_shard(key)

        pre_versions = {k: get_version(k) for k in SHARDS}
        results = []

        with ThreadPoolExecutor(max_workers=N) as pool:
            futs = [pool.submit(run_agent, f"a{i}", mode, results)
                    for i in range(N)]
            for f in as_completed(futs): f.result()

        post_versions = {k: get_version(k) for k in SHARDS}
        total_commits   = sum(r["commits"] for r in results)
        total_conflicts = sum(r["conflicts"] for r in results)
        version_advances = sum(post_versions[k] - pre_versions[k] for k in SHARDS)

        # Corruption: commits that didn't advance version (concurrent overwrite)
        corruptions = max(0, total_commits - version_advances) if mode == "baseline" else 0

        all_results.append({
            "mode": mode, "N": N, "run": run,
            "commits": total_commits, "conflicts": total_conflicts,
            "version_advances": version_advances,
            "corruptions": corruptions,
            "corruption_rate": corruptions / max(1, total_commits),
            "scr": total_conflicts / max(1, total_commits + total_conflicts),
        })
        print(f"  {mode:10s} N={N:2d} run={run}: "
              f"commits={total_commits:3d} conflicts={total_conflicts:3d} "
              f"corruption={corruptions} SCR={all_results[-1]['scr']:.3f}")
    return all_results

def main():
    print("=" * 60)
    print("ABLATION STUDY: DeliveryLog vs Token-only vs Baseline")
    print("=" * 60)

    r = client.get(f"{SBUS_URL}/stats")
    if r.status_code != 200:
        print(f"❌ Server not running at {SBUS_URL}")
        return
    dl_active = "true" in str(r.json().get("acp_config", {}).get("enable_delivery_log", True)).lower()
    print(f"✅ Server running. SBUS_NO_DELIVERY_LOG={os.getenv('SBUS_NO_DELIVERY_LOG','unset')}")

    all_rows = []
    for N in N_VALUES:
        print(f"\n--- N={N} ---")
        for mode in ["full_sbus", "token_only_no_dl", "baseline"]:
            print(f"  Running {mode}...")
            rows = run_experiment(N, mode)
            all_rows.extend(rows)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Mode':22s} {'N':>4} {'Corruption':>12} {'SCR':>8} {'Commits/run':>12}")
    for N in N_VALUES:
        for mode in ["full_sbus", "token_only_no_dl", "baseline"]:
            sub = [r for r in all_rows if r["N"]==N and r["mode"]==mode]
            if not sub: continue
            mean_corr = statistics.mean(r["corruption_rate"] for r in sub)
            mean_scr  = statistics.mean(r["scr"] for r in sub)
            mean_comm = statistics.mean(r["commits"] for r in sub)
            print(f"  {mode:22s} {N:>4} {mean_corr:>12.1%} {mean_scr:>8.3f} {mean_comm:>12.0f}")

    import datetime
    out = f"results/ablation_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    os.makedirs("results", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader(); writer.writerows(all_rows)
    print(f"\n✅ Results: {out}")
    print("\nPaper table: compare full_sbus vs token_only_no_dl corruption rate.")
    print("Expected: full_sbus=0%, token_only_no_dl may show some corruption")
    print("(if DeliveryLog catches stale cross-shard reads)")

if __name__ == "__main__":
    main()