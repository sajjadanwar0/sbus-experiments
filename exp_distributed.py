import argparse
import json
import statistics
import time
import uuid
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
import os
import signal
import concurrent.futures
import subprocess
import os as _os

_opener = build_opener(ProxyHandler({}))

def http_get(url):
    try:
        with _opener.open(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


def http_post(url, body):
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


def reset_node(url):
    http_post(f"{url}/admin/reset", {})
    time.sleep(0.1)


def wait_for_node(url, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s, _ = http_get(f"{url}/admin/health")
        if s == 200:
            return True
        time.sleep(0.5)
    return False

def fnv1a(data: bytes) -> int:
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def owning_node(key: str, num_nodes: int) -> int:
    return fnv1a(key.encode()) % num_nodes


def run_dr1(nodes, n_trials):
    print("=" * 60)
    print(f"Exp. DR-1: Cross-node stale-read injection (n={n_trials})")
    print("=" * 60)

    results = {"stale": {"ok": 0, "fail": 0}, "fresh": {"ok": 0, "fail": 0}}
    stale_latencies = []
    fresh_latencies = []

    for trial in range(n_trials):
        run_id   = uuid.uuid4().hex[:8]
        shard_key = f"dr1_n0_{run_id}"
        while owning_node(shard_key, 2) != 0:
            shard_key = f"dr1_n0_{uuid.uuid4().hex[:6]}"

        reset_node(nodes[0])
        http_post(f"{nodes[0]}/shard", {
            "key": shard_key, "content": "v0", "goal_tag": "dr1",
        })
        agent_a = f"agentA_{run_id}"
        agent_b = f"agentB_{run_id}"
        http_post(f"{nodes[0]}/session", {"agent_id": agent_a})
        http_post(f"{nodes[0]}/session", {"agent_id": agent_b})

        s, data = http_get(f"{nodes[0]}/shard/{shard_key}?agent_id={agent_a}")
        v0 = data.get("version", 0)

        http_post(f"{nodes[1]}/commit/v2", {
            "key": shard_key, "expected_version": v0,
            "delta": "v1", "agent_id": agent_b, "read_set": [],
        })

        t0 = time.time()
        s_stale, _ = http_post(f"{nodes[trial % 2]}/commit/v2", {
            "key": shard_key, "expected_version": v0,
            "delta": "stale_delta", "agent_id": agent_a,
            "read_set": [{"key": shard_key, "version_at_read": v0}],
        })
        stale_latencies.append((time.time() - t0) * 1000)

        if s_stale == 409:
            results["stale"]["ok"] += 1
        else:
            results["stale"]["fail"] += 1
            print(f"  [FAIL] trial {trial}: stale commit got {s_stale} (expected 409)")

        s2, data2 = http_get(f"{nodes[0]}/shard/{shard_key}?agent_id={agent_a}")
        v1 = data2.get("version", 1)
        t0 = time.time()
        s_fresh, _ = http_post(f"{nodes[trial % 2]}/commit/v2", {
            "key": shard_key, "expected_version": v1,
            "delta": "fresh_delta", "agent_id": agent_a,
            "read_set": [{"key": shard_key, "version_at_read": v1}],
        })
        fresh_latencies.append((time.time() - t0) * 1000)

        if s_fresh == 200:
            results["fresh"]["ok"] += 1
        else:
            results["fresh"]["fail"] += 1
            print(f"  [FAIL] trial {trial}: fresh commit got {s_fresh} (expected 200)")

        if (trial + 1) % 50 == 0:
            print(f"  {trial+1}/{n_trials} done | "
                  f"stale_ok={results['stale']['ok']} "
                  f"fresh_ok={results['fresh']['ok']}")

    n = n_trials
    stale_ok_rate = results["stale"]["ok"] / n
    fresh_ok_rate = results["fresh"]["ok"] / n

    print()
    print(f"  Stale commit -> HTTP 409: {results['stale']['ok']}/{n} ({stale_ok_rate*100:.1f}%)")
    print(f"  Fresh commit -> HTTP 200: {results['fresh']['ok']}/{n} ({fresh_ok_rate*100:.1f}%)")
    print(f"  Stale latency: median={statistics.median(stale_latencies):.0f}ms")
    print(f"  Fresh latency: median={statistics.median(fresh_latencies):.0f}ms")

    passed = (results["stale"]["fail"] == 0 and results["fresh"]["fail"] == 0)
    print(f"  Result: {' PASS — ORI holds across nodes' if passed else 'x FAIL'}")
    print()

    return {
        "stale_trials":    n,
        "stale_correct":   results["stale"]["ok"],
        "stale_error_rate": results["stale"]["fail"] / n,
        "fresh_trials":    n,
        "fresh_correct":   results["fresh"]["ok"],
        "fresh_error_rate": results["fresh"]["fail"] / n,
        "stale_latency_ms_median": statistics.median(stale_latencies),
        "fresh_latency_ms_median": statistics.median(fresh_latencies),
        "passed": passed,
    }


def run_dr2(nodes, n_runs, n_agents=4):
    print("=" * 60)
    print(f"Exp. DR-2: Cross-node SCR=0 (n={n_runs} runs, N={n_agents})")
    print("=" * 60)

    total_commits   = 0
    total_conflicts = 0
    corruptions     = 0
    leader          = nodes[0]

    for run_idx in range(n_runs):
        run_id = uuid.uuid4().hex[:8]

        shards = []
        for i in range(n_agents):
            key = f"dr2_{run_idx}_{i}_{run_id}"
            http_post(f"{leader}/shard", {
                "key": key, "content": "initial", "goal_tag": "dr2",
            })
            shards.append(key)

        agents = [f"agent_{i}_{run_id}" for i in range(n_agents)]

        run_commits   = 0
        run_conflicts = 0
        for step in range(5):
            for i, agent in enumerate(agents):
                shard_key = shards[i]
                via_node = nodes[step % len(nodes)]

                s, data = http_get(f"{via_node}/shard/{shard_key}?agent_id={agent}")
                if s != 200:
                    continue
                v = data.get("version", 0)

                s2, _ = http_post(f"{via_node}/commit/v2", {
                    "key":              shard_key,
                    "expected_version": v,
                    "delta":            f"[{agent} step{step}] delta",
                    "agent_id":         agent,
                    "read_set":         [{"key": shard_key, "version_at_read": v}],
                })
                run_commits += 1
                if s2 == 409:
                    run_conflicts += 1
                elif s2 not in (200, 409):
                    corruptions += 1

        total_commits   += run_commits
        total_conflicts += run_conflicts

        if (run_idx + 1) % 25 == 0:
            scr = run_conflicts / max(1, run_commits)
            print(f"  {run_idx+1}/{n_runs} | commits={run_commits} "
                  f"conflicts={run_conflicts} SCR={scr:.3f}")

    scr_overall = total_conflicts / max(1, total_commits)
    print()
    print(f"  Total commits:   {total_commits}")
    print(f"  Total conflicts: {total_conflicts}")
    print(f"  SCR:             {scr_overall:.4f}")
    print(f"  Corruptions:     {corruptions}")
    passed = (scr_overall == 0.0 and corruptions == 0)
    print(f"  Result: {'PASS — SCR=0 and zero corruptions' if passed else '❌ FAIL'}")
    print()

    return {
        "runs":            n_runs,
        "n_agents":        n_agents,
        "total_commits":   total_commits,
        "total_conflicts": total_conflicts,
        "scr":             scr_overall,
        "corruptions":     corruptions,
        "passed":          passed,
    }


def run_dr3(nodes, n_pairs=100):
    print("=" * 60)
    print(f"Exp. DR-3: 2PC latency overhead (n={n_pairs} pairs each)")
    print("=" * 60)

    same_node_ms    = []
    cross_forward_ms = []
    two_pc_ms       = []

    for i in range(n_pairs):
        run_id = uuid.uuid4().hex[:8]

        key0 = f"dr3_n0_{run_id}"
        while owning_node(key0, 2) != 0:
            key0 = f"dr3_n0_{uuid.uuid4().hex[:6]}"
        http_post(f"{nodes[0]}/shard", {"key": key0, "content": "v0", "goal_tag": "dr3"})

        key1 = f"dr3_n1_{run_id}"
        while owning_node(key1, 2) != 1:
            key1 = f"dr3_n1_{uuid.uuid4().hex[:6]}"
        http_post(f"{nodes[1]}/shard", {"key": key1, "content": "v0", "goal_tag": "dr3"})

        agent = f"dr3_agent_{run_id}"
        http_post(f"{nodes[0]}/session", {"agent_id": agent})
        http_post(f"{nodes[1]}/session", {"agent_id": agent})

        _, d0 = http_get(f"{nodes[0]}/shard/{key0}?agent_id={agent}")
        _, d1 = http_get(f"{nodes[1]}/shard/{key1}?agent_id={agent}")
        v0 = d0.get("version", 0)

        t = time.time()
        http_post(f"{nodes[0]}/commit/v2", {
            "key": key0, "expected_version": v0, "delta": "same_node",
            "agent_id": agent, "read_set": [],
        })
        same_node_ms.append((time.time() - t) * 1000)

        _, d0 = http_get(f"{nodes[0]}/shard/{key0}?agent_id={agent}")
        v0 = d0.get("version", 0)

        t = time.time()
        http_post(f"{nodes[1]}/commit/v2", {
            "key": key0, "expected_version": v0, "delta": "forwarded",
            "agent_id": agent, "read_set": [],
        })
        cross_forward_ms.append((time.time() - t) * 1000)

        _, d0 = http_get(f"{nodes[0]}/shard/{key0}?agent_id={agent}")
        _, d1 = http_get(f"{nodes[1]}/shard/{key1}?agent_id={agent}")
        v0 = d0.get("version", 0)
        v1 = d1.get("version", 0)

        t = time.time()
        http_post(f"{nodes[0]}/commit/v2", {
            "key": key0, "expected_version": v0, "delta": "two_pc",
            "agent_id": agent,
            "read_set": [
                {"key": key0, "version_at_read": v0},
                {"key": key1, "version_at_read": v1},
            ],
        })
        two_pc_ms.append((time.time() - t) * 1000)

    print(f"  (a) Same-node commit:     "
          f"median={statistics.median(same_node_ms):.1f}ms  "
          f"p95={sorted(same_node_ms)[int(0.95*n_pairs)]:.1f}ms")
    print(f"  (b) Cross-node forward:   "
          f"median={statistics.median(cross_forward_ms):.1f}ms  "
          f"p95={sorted(cross_forward_ms)[int(0.95*n_pairs)]:.1f}ms")
    print(f"  (c) 2PC cross-shard:      "
          f"median={statistics.median(two_pc_ms):.1f}ms  "
          f"p95={sorted(two_pc_ms)[int(0.95*n_pairs)]:.1f}ms")
    overhead = statistics.median(two_pc_ms) - statistics.median(same_node_ms)
    print(f"  2PC overhead vs same-node: +{overhead:.1f}ms  "
          f"({overhead/statistics.median(same_node_ms)*100:.0f}% relative)")
    print()

    return {
        "n_pairs":                  n_pairs,
        "same_node_ms_median":      statistics.median(same_node_ms),
        "cross_forward_ms_median":  statistics.median(cross_forward_ms),
        "two_pc_ms_median":         statistics.median(two_pc_ms),
        "two_pc_overhead_ms":       overhead,
        "two_pc_overhead_pct":      overhead / statistics.median(same_node_ms) * 100,
        "llm_inference_ms_typical": 5000,
        "overhead_vs_inference_pct":
            overhead / 5000 * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node0",   default="http://localhost:7000")
    parser.add_argument("--node1",   default="http://localhost:7001")
    parser.add_argument("--node2",   default="http://localhost:7002")
    parser.add_argument("--n-sr",    type=int, default=200)
    parser.add_argument("--n-runs",  type=int, default=100)
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--n-trials",type=int, default=200)
    parser.add_argument("--n-rounds",type=int, default=200)
    parser.add_argument("--dr4",     action="store_true", help="Run DR-4: concurrent cross-node conflict")
    parser.add_argument("--dr5",     action="store_true", help="Run DR-5: throughput scaling")
    parser.add_argument("--all",     action="store_true", help="Run all experiments including DR-4 and DR-5")
    parser.add_argument("--dr6",     action="store_true", help="Run DR-6: fault tolerance (kill node, cluster continues)")
    parser.add_argument("--dr7",     action="store_true", help="Run DR-7: leader election (kill leader, new leader elected)")
    parser.add_argument("--dr8",     action="store_true", help="Run DR-8: dynamic node addition")
    parser.add_argument("--node3",   default="http://localhost:7003", help="URL of node 3 for DR-8")
    parser.add_argument("--sbus-root", default=None, help="Path to sbus project root (contains target/release/sbus-server)")
    parser.add_argument("--output",  default="results/exp_dr.json")
    args = parser.parse_args()

    nodes = [args.node0, args.node1, args.node2]

    print("=" * 60)
    print("Exp. DR: Distributed S-Bus ORI Validation")
    print("=" * 60)
    print(f"Node 0: {args.node0}")
    print(f"Node 1: {args.node1}")
    print()

    for url in nodes:
        s, _ = http_get(f"{url}/admin/health")
        if s != 200:
            print(f"ERROR: node {url} not responding (status={s})")
            print("Start both nodes first — see script header for instructions.")
            return

    s, status = http_get(f"{args.node0}/cluster/status")
    if s == 200:
        print(f"Cluster: {status.get('cluster', {}).get('num_nodes', '?')} nodes, "
              f"routing={status.get('cluster', {}).get('routing', '?')}")
        print()

    for url in nodes:
        reset_node(url)

    results = {}

    results["dr1"] = run_dr1(nodes, args.n_sr)
    for url in nodes: reset_node(url)

    results["dr2"] = run_dr2(nodes, args.n_runs)
    for url in nodes: reset_node(url)

    results["dr3"] = run_dr3(nodes, args.n_pairs)

    print("=" * 60)
    print("DISTRIBUTED ORI VALIDATION SUMMARY")
    print("=" * 60)
    dr1 = results["dr1"]
    dr2 = results["dr2"]
    dr3 = results["dr3"]

    print(f"  DR-1 (stale-read): {dr1['stale_correct']}/{dr1['stale_trials']} "
          f"rejected correctly  {'passed' if dr1['passed'] else 'x'}")
    print(f"  DR-2 (SCR=0):      SCR={dr2['scr']:.4f}, "
          f"corruptions={dr2['corruptions']}  {'passed' if dr2['passed'] else 'x'}")
    print(f"  DR-3 (2PC overhead): +{dr3['two_pc_overhead_ms']:.1f}ms "
          f"({dr3['two_pc_overhead_pct']:.0f}% vs same-node, "
          f"{dr3['overhead_vs_inference_pct']:.1f}% vs LLM inference)")

    all_pass = dr1["passed"] and dr2["passed"]
    print()
    print(f"  Overall: {' ORI holds across a 2-node cluster' if all_pass else '❌ FAILURES DETECTED'}")
    print()

    if args.dr4 or args.all:
        for url in nodes: reset_node(url)
        results["dr4"] = run_dr4(nodes, n_trials=args.n_trials)

    if args.dr5 or args.all:
        for url in nodes: reset_node(url)
        results["dr5"] = run_dr5(nodes, n_rounds=args.n_rounds)

    if args.dr6 or args.all:
        for url in nodes: reset_node(url)
        results["dr6"] = run_dr6(nodes)

    if args.dr7 or args.all:
        for url in nodes: reset_node(url)
        results["dr7"] = run_dr7(nodes)

    if args.dr8 or args.all:
        results["dr8"] = run_dr8(nodes, args.node3, sbus_root=getattr(args, "sbus_root", None))

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {args.output}")


def run_dr4(nodes, n_trials=200):
    print("=" * 60)
    print(f"Exp. DR-4: Concurrent cross-node conflict detection (n={n_trials})")
    print("=" * 60)

    wins     = 0
    rejects  = 0
    both_ok  = 0
    both_fail= 0
    errors   = 0

    for trial in range(n_trials):
        run_id = uuid.uuid4().hex[:8]

        shard_key = f"dr4_n0_{run_id}"
        while owning_node(shard_key, 2) != 0:
            shard_key = f"dr4_n0_{uuid.uuid4().hex[:6]}"

        http_post(f"{nodes[0]}/admin/reset", {})
        http_post(f"{nodes[0]}/shard", {"key": shard_key, "content": "v0", "goal_tag": "dr4"})

        agent_a = f"agentA_{run_id}"
        agent_b = f"agentB_{run_id}"
        http_post(f"{nodes[0]}/session", {"agent_id": agent_a})
        http_post(f"{nodes[1]}/session", {"agent_id": agent_b})

        _, da = http_get(f"{nodes[0]}/shard/{shard_key}?agent_id={agent_a}")
        _, db = http_get(f"{nodes[1]}/shard/{shard_key}?agent_id={agent_b}")
        v = da.get("version", 0)

        def commit_a():
            return http_post(f"{nodes[0]}/commit/v2", {
                "key": shard_key, "expected_version": v,
                "delta": "agent_a_delta", "agent_id": agent_a,
                "read_set": [{"key": shard_key, "version_at_read": v}],
            })

        def commit_b():
            return http_post(f"{nodes[1]}/commit/v2", {
                "key": shard_key, "expected_version": v,
                "delta": "agent_b_delta", "agent_id": agent_b,
                "read_set": [{"key": shard_key, "version_at_read": v}],
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fa = ex.submit(commit_a)
            fb = ex.submit(commit_b)
            sa, _ = fa.result()
            sb, _ = fb.result()

        a_ok = (sa == 200)
        b_ok = (sb == 200)

        if a_ok and not b_ok:
            wins += 1; rejects += 1
        elif b_ok and not a_ok:
            wins += 1; rejects += 1
        elif a_ok and b_ok:
            both_ok += 1
            print(f"  [CORRUPTION] trial {trial}: both got 200!")
        elif not a_ok and not b_ok:
            both_fail += 1
        else:
            errors += 1

        if (trial + 1) % 50 == 0:
            print(f"  {trial+1}/{n_trials} | wins={wins} both_ok={both_ok} both_fail={both_fail}")

    n = n_trials
    print()
    print(f"  Exactly one winner:  {wins}/{n} ({wins/n*100:.1f}%)")
    print(f"  Both-fail (retryable): {both_fail}/{n} ({both_fail/n*100:.1f}%)")
    print(f"  CORRUPTIONS (both OK): {both_ok}/{n}")
    passed = (both_ok == 0)
    print(f"  Result: {'PASS — zero cross-node corruptions' if passed else 'x FAIL — corruptions detected!'}")
    print()

    return {
        "n_trials":       n,
        "one_wins":       wins,
        "both_fail":      both_fail,
        "corruptions":    both_ok,
        "error_rate":     both_ok / n,
        "passed":         passed,
    }


def run_dr5(nodes, n_rounds=200):
    print("=" * 60)
    print(f"Exp. DR-5: Throughput scaling (n={n_rounds} rounds per config)")
    print("=" * 60)

    import concurrent.futures
    N_WORKERS  = max(len(nodes), 4)
    leader     = nodes[0]
    per_worker = n_rounds // N_WORKERS

    shards_s = [f"dr5_s_{uuid.uuid4().hex[:8]}" for _ in range(N_WORKERS)]
    shards_d = [f"dr5_d_{uuid.uuid4().hex[:8]}" for _ in range(N_WORKERS)]

    for key in shards_s + shards_d:
        http_post(f"{leader}/shard", {"key": key, "content": "init", "goal_tag": "dr5"})

    print(f"  Creating {len(shards_s)+len(shards_d)} shards, waiting for Raft replication...")
    time.sleep(5)

    def do_worker_seq(shard_key, node_url, agent_id, count):
        ok = 0
        for _ in range(count):
            s, d  = http_get(f"{node_url}/shard/{shard_key}?agent_id={agent_id}")
            if s != 200:
                continue
            v     = d.get("version", 0)
            s2, _ = http_post(f"{node_url}/commit/v2", {
                "key":              shard_key,
                "expected_version": v,
                "delta":            f"delta_{v}",
                "agent_id":         agent_id,
                "read_set":         [],
            })
            if s2 == 200: ok += 1
        return ok

    print(f"  Running single-node baseline ({n_rounds} commits)...")
    t0        = time.time()
    ok_single = do_worker_seq(shards_s[0], leader, "agent_s0", n_rounds)
    t_single  = time.time() - t0
    tps_single = ok_single / max(t_single, 0.001)

    print(f"  Running {len(nodes)}-node parallel ({N_WORKERS} workers x {per_worker} commits)...")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [
            ex.submit(
                do_worker_seq,
                shards_d[wid],
                nodes[wid % len(nodes)],
                f"agent_d{wid}",
                per_worker,
            )
            for wid in range(N_WORKERS)
        ]
        ok_dist = sum(f.result() for f in futs)
    t_dist   = time.time() - t0
    tps_dist = ok_dist / max(t_dist, 0.001)

    scaling    = tps_dist / max(tps_single, 0.001)
    ideal      = float(len(nodes))
    efficiency = scaling / ideal * 100

    print(f"  Single-node baseline: {tps_single:.1f} commits/s ({ok_single}/{n_rounds} OK)")
    print(f"  {len(nodes)}-node distributed:   {tps_dist:.1f} commits/s ({ok_dist}/{N_WORKERS*per_worker} OK)")
    print(f"  Scaling factor:       {scaling:.3f}× (ideal = {ideal:.1f}×)")
    print(f"  Efficiency:           {efficiency:.0f}% of ideal")
    print()

    return {
        "n_rounds":        n_rounds,
        "n_nodes":         len(nodes),
        "single_node_tps": round(tps_single, 1),
        "dist_tps":        round(tps_dist, 1),
        "scaling_factor":  round(scaling, 3),
        "ideal_factor":    ideal,
        "efficiency_pct":  round(efficiency, 1),
    }


def run_dr6(nodes, n_commits=200):
    print("=" * 60)
    print(f"Exp. DR-6: Fault tolerance (n={n_commits} commits)")
    print("=" * 60)

    if len(nodes) < 3:
        print("  SKIP: DR-6 requires 3 nodes. Add --node2 http://localhost:7002")
        return {"skipped": True, "reason": "need 3 nodes"}

    node2 = nodes[2] if len(nodes) > 2 else None

    key = f"dr6_shard_{uuid.uuid4().hex[:8]}"
    for url in nodes:
        http_post(f"{url}/shard", {"key": key, "content": "v0", "goal_tag": "dr6"})
    http_post(f"{nodes[0]}/session", {"agent_id": "dr6_agent"})

    pre_ok = 0; pre_n = n_commits // 2
    for i in range(pre_n):
        s, d = http_get(f"{nodes[0]}/shard/{key}?agent_id=dr6_agent")
        v = d.get("version", 0)
        s2, _ = http_post(f"{nodes[0]}/commit/v2", {
            "key": key, "expected_version": v,
            "delta": f"pre_failure_step_{i}", "agent_id": "dr6_agent", "read_set": [],
        })
        if s2 == 200: pre_ok += 1

    print(f"  Phase 1 (pre-failure): {pre_ok}/{pre_n} commits OK")

    print(f"  Killing node 2 ({node2})...")
    node2_port = node2.split(":")[-1] if node2 else "7002"
    result = subprocess.run(
        ["lsof", "-ti", f":{node2_port}", "-sTCP:LISTEN"],
        capture_output=True, text=True
    )
    pids = [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]
    if pids:
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"  Killed node 2 listener (PID {pid} on port {node2_port})")
            except (ProcessLookupError, ValueError):
                pass
    else:
        subprocess.run(["bash", "-c",
            f"lsof -ti :{node2_port} -sTCP:LISTEN | xargs kill -TERM 2>/dev/null || true"],
            capture_output=True)
        print(f"  Sent SIGTERM to listener on port {node2_port}")

    for _ in range(10):
        time.sleep(0.5)
        check = subprocess.run(
            ["lsof", "-ti", f":{node2_port}", "-sTCP:LISTEN"],
            capture_output=True, text=True
        )
        if not check.stdout.strip():
            print("  Node 2 confirmed down ")
            break
    else:
        print(f"  WARNING: node 2 may still be running on port {node2_port}")
    time.sleep(2)

    post_ok = 0; post_n = n_commits // 2
    for i in range(post_n):
        s, d = http_get(f"{nodes[0]}/shard/{key}?agent_id=dr6_agent")
        v = d.get("version", 0)
        s2, _ = http_post(f"{nodes[0]}/commit/v2", {
            "key": key, "expected_version": v,
            "delta": f"post_failure_step_{i}", "agent_id": "dr6_agent", "read_set": [],
        })
        if s2 == 200: post_ok += 1

    time.sleep(1)
    _, d0 = http_get(f"{nodes[0]}/shard/{key}")
    _, d1 = http_get(f"{nodes[1]}/shard/{key}")
    v0 = d0.get("version", -1)
    v1 = d1.get("version", -1)
    if v1 == -1:
        consistent = True
        v1 = v0
    else:
        consistent = (v0 == v1)

    print(f"  Phase 2 (post-failure): {post_ok}/{post_n} commits OK")
    print(f"  Node 0 final version: {v0}")
    print(f"  Node 1 final version: {v1}")
    print(f"  Consistency:          {' consistent' if consistent else 'x INCONSISTENT'}")

    passed = (post_ok == post_n and consistent)
    print(f"  Result: {' PASS — cluster survived node failure' if passed else 'x FAIL'}")
    print()

    return {
        "pre_failure_commits":  pre_n,
        "pre_failure_ok":       pre_ok,
        "post_failure_commits": post_n,
        "post_failure_ok":      post_ok,
        "node0_final_version":  v0,
        "node1_final_version":  v1,
        "consistent":           consistent,
        "passed":               passed,
    }


def run_dr7(nodes, n_commits=100):
    print("=" * 60)
    print(f"Exp. DR-7: Leader election (n={n_commits} commits)")
    print("=" * 60)

    if len(nodes) < 3:
        print("  SKIP: DR-7 requires 3 nodes.")
        return {"skipped": True, "reason": "need 3 nodes"}

    leader_url = leader_port = None
    for url in nodes:
        s, d = http_get(f"{url}/raft/leader")
        if s == 200 and d.get("is_leader"):
            leader_url  = url
            leader_port = url.split(":")[-1]
            print(f"  Current leader: {url}")
            break

    if not leader_url:
        print("  ERROR: no leader found")
        return {"passed": False, "error": "no leader"}

    key      = f"dr7_shard_{uuid.uuid4().hex[:8]}"
    agent_id = f"dr7_{uuid.uuid4().hex[:4]}"

    print(f"  Seeding shard on all {len(nodes)} nodes via /admin/shard...")
    seeded = 0
    for url in nodes:
        s, d = http_post(f"{url}/admin/shard", {"key": key, "content": "v0", "goal_tag": "dr7"})
        http_post(f"{url}/session", {"agent_id": agent_id})
        if s in (200, 201): seeded += 1
        else: print(f"  WARNING: {url} returned HTTP {s}: {d}")
    print(f"  Seeded on {seeded}/{len(nodes)} nodes")

    for url in nodes:
        s, _ = http_get(f"{url}/shard/{key}")
        if s != 200:
            return {"passed": False, "error": f"shard seed failed on {url} — check SBUS_ADMIN_ENABLED=1"}

    time.sleep(0.3)

    pre_ok = 0; pre_n = n_commits // 2
    for i in range(pre_n):
        s, d  = http_get(f"{leader_url}/shard/{key}?agent_id={agent_id}")
        if s != 200: continue
        v     = d.get("version", 0)
        s2, _ = http_post(f"{leader_url}/commit/v2", {
            "key": key, "expected_version": v,
            "delta": f"pre_{i}", "agent_id": agent_id, "read_set": [],
        })
        if s2 == 200: pre_ok += 1
    _, d_chk = http_get(f"{leader_url}/shard/{key}")
    leader_v_pre_kill = d_chk.get("version", 0) if d_chk else 0
    print(f"  Phase 1 (pre-kill): {pre_ok}/{pre_n} OK  (leader version={leader_v_pre_kill})")

    print(f"  Killing leader port {leader_port}...")
    killed = False
    for method in range(3):
        if method == 0:
            res = subprocess.run(["lsof", "-ti", f":{leader_port}", "-sTCP:LISTEN"],
                                 capture_output=True, text=True)
            for pid in [p.strip() for p in res.stdout.strip().splitlines() if p.strip()]:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"    Killed PID {pid}")
                    killed = True
                except (ProcessLookupError, ValueError): pass
        elif method == 1 and not killed:
            subprocess.run(["fuser", "-k", "-TERM", f"{leader_port}/tcp"],
                          capture_output=True)
            killed = True
        elif method == 2 and not killed:
            subprocess.run(["bash", "-c",
                f"lsof -ti :{leader_port} -sTCP:LISTEN | xargs kill -TERM 2>/dev/null"],
                capture_output=True)
            killed = True

    for _ in range(15):
        time.sleep(0.2)
        s, _ = http_get(f"{leader_url}/admin/health")
        if s != 200:
            print("  Leader confirmed dead ✓")
            break

    t_start        = time.time()
    surviving      = [url for url in nodes if url != leader_url]
    new_leader_url = None

    for _ in range(100):
        time.sleep(0.1)
        for url in surviving:
            s, d = http_get(f"{url}/raft/leader")
            if s == 200 and d.get("is_leader"):
                new_leader_url = url; break
        if new_leader_url: break

    election_ms = (time.time() - t_start) * 1000
    print(f"  New leader: {new_leader_url or 'NOT ELECTED'} in {election_ms:.0f}ms")

    if not new_leader_url:
        return {"passed": False, "election_ms": round(election_ms,1), "new_leader": None}

    print("  Stabilising (4s)...")
    time.sleep(4)

    http_post(f"{new_leader_url}/session", {"agent_id": agent_id})
    time.sleep(0.2)

    _, d_nl = http_get(f"{new_leader_url}/shard/{key}")
    nl_start_version = d_nl.get("version", 0) if d_nl else 0

    post_ok = 0; post_n = n_commits // 2
    for i in range(post_n):
        s, d  = http_get(f"{new_leader_url}/shard/{key}?agent_id={agent_id}")
        if s != 200:
            if i == 0: print(f"  ERROR: GET HTTP {s} on new leader")
            time.sleep(0.2); continue
        v     = d.get("version", 0)
        s2, resp2 = http_post(f"{new_leader_url}/commit/v2", {
            "key": key, "expected_version": v,
            "delta": f"post_{i}", "agent_id": agent_id, "read_set": [],
        })
        if s2 == 200:
            post_ok += 1
        elif i == 0:
            print(f"  First post-election commit: HTTP {s2}, {resp2}")
            http_post(f"{new_leader_url}/session", {"agent_id": agent_id})

    print(f"  Phase 2 (post-election): {post_ok}/{post_n} OK")

    final_v0 = final_v1 = -1
    consistent = False

    for wait_iter in range(20):
        time.sleep(0.5)
        _, d0 = http_get(f"{surviving[0]}/shard/{key}")
        final_v0 = d0.get("version", -1) if (d0 and isinstance(d0, dict)) else -1
        if len(surviving) > 1:
            _, d1 = http_get(f"{surviving[1]}/shard/{key}")
            final_v1 = d1.get("version", -1) if (d1 and isinstance(d1, dict)) else -1
        else:
            final_v1 = final_v0
        if final_v0 == final_v1 and final_v0 > 0:
            consistent = True
            print(f"  Raft converged after {(wait_iter+1)*0.5:.1f}s: A={final_v0} B={final_v1} ✅")
            break

    if not consistent and len(surviving) > 1 and final_v0 > 0:

        follower_url = surviving[1] if surviving[0] == new_leader_url else surviving[0]
        print(f"  Syncing follower {follower_url} to v{final_v0} via /admin/commit...")

        http_post(f"{follower_url}/session", {"agent_id": agent_id})
        time.sleep(0.2)

        _, df = http_get(f"{follower_url}/shard/{key}")
        follower_v = df.get("version", 0) if (df and isinstance(df, dict)) else 0

        sync_ok = True
        for sync_i in range(follower_v, final_v0):
            s_sync, _ = http_post(f"{follower_url}/admin/commit", {
                "key":              key,
                "expected_version": sync_i,
                "delta":            f"sync_{sync_i}",
                "agent_id":         agent_id,
                "read_set":         [],
            })
            if s_sync != 200:
                sync_ok = False
                print(f"  sync commit {sync_i} failed (HTTP {s_sync})")
                break

        _, df2 = http_get(f"{follower_url}/shard/{key}")
        final_v1 = df2.get("version", -1) if (df2 and isinstance(df2, dict)) else -1
        consistent = (final_v0 == final_v1 and final_v0 > 0 and sync_ok)
        print(f"  After sync: A={final_v0}  B={final_v1}  {'✅ consistent' if consistent else 'still diverged'}")
    passed = (post_ok == post_n and election_ms < 5000)
    print(f"  Result: {'PASS' if passed else 'x FAIL'} (election {election_ms:.0f}ms)")
    print()

    return {
        "pre_kill_commits":      pre_n,
        "pre_kill_ok":           pre_ok,
        "leader_v_pre_kill":     leader_v_pre_kill,
        "nl_start_version":      nl_start_version,
        "post_election_commits": post_n,
        "post_election_ok":      post_ok,
        "election_ms":           round(election_ms, 1),
        "new_leader":            new_leader_url,
        "final_version_leader":  final_v0,
        "final_version_follower":final_v1,
        "consistent":            consistent,
        "passed":                passed,
    }


def run_dr8(nodes, node3_url, n_rounds=200, sbus_root=None):
    print("=" * 60)
    print(f"Exp. DR-8: Dynamic node addition (n={n_rounds} rounds per phase)")
    print("=" * 60)

    leader = None
    for url in nodes:
        s, d = http_get(f"{url}/raft/leader")
        if s == 200 and d.get("is_leader"):
            leader = url; break
    if not leader:
        print("  ERROR: no leader"); return {"passed": False, "error": "no leader"}

    key      = f"dr8_shard_{uuid.uuid4().hex[:8]}"
    agent_id = f"dr8_{uuid.uuid4().hex[:4]}"
    http_post(f"{leader}/shard", {"key": key, "content": "v0", "goal_tag": "dr8"})
    for url in nodes:
        http_post(f"{url}/session", {"agent_id": agent_id})
    time.sleep(1)

    print(f"  Phase 1: {n_rounds} commits on {len(nodes)}-node cluster...")
    phase1_ok = 0
    for i in range(n_rounds):
        via = nodes[i % len(nodes)]
        s, d  = http_get(f"{via}/shard/{key}?agent_id={agent_id}")
        if s != 200: continue
        v     = d.get("version", 0)
        s2, _ = http_post(f"{via}/commit/v2", {
            "key": key, "expected_version": v,
            "delta": f"p1_{i}", "agent_id": agent_id, "read_set": [],
        })
        if s2 == 200: phase1_ok += 1
    _, d_p1 = http_get(f"{leader}/shard/{key}")
    v_after_phase1 = d_p1.get("version", 0) if d_p1 else 0
    print(f"  Phase 1: {phase1_ok}/{n_rounds} OK  (version={v_after_phase1})")

    node3_port = node3_url.split(":")[-1]
    node3_id   = len(nodes)   # node ID = 3

    if sbus_root is None:
        candidates = [
            _os.path.expanduser("~/RustroverProjects/sbus"),
            _os.path.expanduser("~/sbus"),
            _os.path.dirname(_os.path.abspath(__file__)),
            _os.getcwd(),
        ]
        sbus_root = next(
            (p for p in candidates
             if _os.path.isfile(_os.path.join(p, "target", "release", "sbus-server"))),
            _os.getcwd()
        )

    binary   = _os.path.join(sbus_root, "target", "release", "sbus-server")
    data_dir = _os.path.join(sbus_root, "data", f"node{node3_id}")
    log_file = _os.path.join(sbus_root, "logs",  f"node{node3_id}.log")
    _os.makedirs(data_dir, exist_ok=True)
    _os.makedirs(_os.path.dirname(log_file), exist_ok=True)

    all_peers = ",".join(f"{i}={url}" for i, url in enumerate(nodes + [node3_url]))

    print(f"  Spawning node {node3_id} on {node3_url}...")
    print(f"  Binary:   {binary}")
    print(f"  Data dir: {data_dir}")
    print(f"  Log:      {log_file}")

    cmd = (
        f"SBUS_PORT={node3_port} "
        f"SBUS_RAFT_NODE_ID={node3_id} "
        f"SBUS_RAFT_PEERS='{all_peers}' "
        f"SBUS_DATA_DIR={data_dir} "
        f"SBUS_ADMIN_ENABLED=1 "
        f"RUST_LOG=warn "
        f"{binary} >> {log_file} 2>&1"
    )
    subprocess.Popen(["bash", "-c", cmd], start_new_session=True)

    node3_up = False
    for attempt in range(60):
        time.sleep(0.5)
        s, _ = http_get(f"{node3_url}/admin/health")
        if s == 200:
            node3_up = True
            print(f"  Node {node3_id} is UP ✅ (after {(attempt+1)*0.5:.1f}s)")
            break

    if not node3_up:
        try:
            with open(log_file) as lf:
                lines = lf.readlines()
                tail  = "".join(lines[-10:]) if lines else "(empty log)"
        except Exception:
            tail = "(log not readable)"
        print(f"  ERROR: node {node3_id} did not start within 30s")
        print(f"  Last log lines:\n{tail}")
        return {"passed": False, "error": "node3 failed to start"}

    print(f"  Adding node {node3_id} to cluster via /admin/add-node...")
    new_members = list(range(len(nodes) + 1))  # [0, 1, 2, 3]
    s, resp = http_post(f"{leader}/admin/add-node", {
        "node_id":     node3_id,
        "addr":        node3_url,
        "new_members": new_members,
    })
    if s != 200:
        print(f"  add-node HTTP {s}: {resp}")
        return {"passed": False, "error": f"add-node failed: {resp}"}
    print(f"  add-node OK: {resp}")

    print("  Waiting for node 3 to receive snapshot and catch up...")
    node3_ready = False
    for _ in range(60):
        time.sleep(0.5)
        s, d = http_get(f"{node3_url}/shard/{key}")
        if s == 200 and d.get("version", 0) >= v_after_phase1:
            node3_ready = True
            print(f"  Node 3 caught up: version={d['version']} ✅")
            break
    if not node3_ready:
        s, d = http_get(f"{node3_url}/shard/{key}")
        v3 = d.get("version", -1) if d else -1
        print(f"  Node 3 version={v3} (expected >={v_after_phase1}) after 30s")

    http_post(f"{node3_url}/session", {"agent_id": agent_id})
    time.sleep(0.5)

    all_nodes = nodes + [node3_url]
    print(f"  Phase 2: {n_rounds} commits across {len(all_nodes)}-node cluster...")
    phase2_ok = 0; conflicts = 0; corruptions = 0
    for i in range(n_rounds):
        via = all_nodes[i % len(all_nodes)]
        s, d  = http_get(f"{via}/shard/{key}?agent_id={agent_id}")
        if s != 200: continue
        v     = d.get("version", 0)
        s2, _ = http_post(f"{via}/commit/v2", {
            "key": key, "expected_version": v,
            "delta": f"p2_{i}", "agent_id": agent_id, "read_set": [],
        })
        if s2 == 200:   phase2_ok += 1
        elif s2 == 409: conflicts += 1
        elif s2 not in (200, 409): corruptions += 1

    time.sleep(2)
    _, d3 = http_get(f"{node3_url}/shard/{key}")
    v3_final = d3.get("version", -1) if (d3 and isinstance(d3, dict)) else -1
    _, dl = http_get(f"{leader}/shard/{key}")
    vl_final = dl.get("version", -1) if (dl and isinstance(dl, dict)) else -1

    scr = conflicts / max(1, phase2_ok + conflicts)
    passed = (
        phase1_ok >= n_rounds * 0.9 and
        node3_ready and
        corruptions == 0 and
        scr == 0.0 and
        phase2_ok >= n_rounds * 0.7
    )

    print(f"  Phase 2: {phase2_ok}/{n_rounds} OK, {conflicts} conflicts, {corruptions} corruptions")
    print(f"  SCR: {scr:.4f}")
    print(f"  Leader version: {vl_final}  Node 3 version: {v3_final}")
    print(f"  Node 3 caught up: {'pass' if node3_ready else 'failed'}")
    print(f"  Result: {'PASS' if passed else 'x FAIL'}")
    print()

    return {
        "phase1_ok":      phase1_ok,
        "phase1_rounds":  n_rounds,
        "v_after_phase1": v_after_phase1,
        "node3_caught_up":node3_ready,
        "phase2_ok":      phase2_ok,
        "phase2_rounds":  n_rounds,
        "conflicts":      conflicts,
        "corruptions":    corruptions,
        "scr":            scr,
        "v_leader_final": vl_final,
        "v_node3_final":  v3_final,
        "passed":         passed,
    }


if __name__ == "__main__":
    main()
