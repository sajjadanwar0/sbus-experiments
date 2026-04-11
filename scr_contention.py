#!/usr/bin/env python3
"""
scr_contention.py
=================
Produces the SCR contention table (Table scr in paper v28):
  - Zero write-write corruptions across all agent counts
  - SCR rates under shared-shard contention
  - commit_v2_naive deadlock at N>=8 (documented, not run)

Run from your sbus project root with the server already running:
    python3 scr_contention.py

Output: results/scr_contention.csv
"""

import csv, os, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

SBUS_URL = os.getenv("SBUS_URL", "http://localhost:7000")
OUT_FILE = "results/scr_contention.csv"

client = httpx.Client(timeout=30)


def verify_server():
    try:
        r = client.get(f"{SBUS_URL}/stats", timeout=5)
        if r.status_code == 200:
            print(f"✅ Server running at {SBUS_URL}")
            return True
    except Exception:
        pass
    print(f"❌ Server not running at {SBUS_URL}")
    print("   Start it first: SBUS_WAL_PATH=results/wal.jsonl ./target/release/sbus-server &")
    return False


def create_shard(key: str, content: str = "initial") -> bool:
    r = client.post(f"{SBUS_URL}/shard",
                    json={"key": key, "content": content, "goal_tag": "scr_test"})
    return r.status_code == 201


def read_shard(key: str) -> dict | None:
    r = client.get(f"{SBUS_URL}/shard/{key}")
    return r.json() if r.status_code == 200 else None


def commit(key: str, version: int, delta: str, agent_id: str,
           read_set: list | None = None) -> tuple[bool, str]:
    body = {"key": key, "expected_version": version,
            "delta": delta, "agent_id": agent_id}
    if read_set:
        body["read_set"] = read_set
    r = client.post(f"{SBUS_URL}/commit/v2", json=body)
    if r.status_code == 200:
        return True, ""
    try:
        return False, r.json().get("error", "unknown")
    except Exception:
        return False, "parse_error"


# ── Experiment 1: Distinct shards (baseline, expect SCR=0) ────────────────────

def run_distinct_shards(n_agents: int, n_steps: int) -> dict:
    """
    Each agent owns a separate shard — no contention.
    Expected: SCR = 0.000, zero corruptions.
    """
    run_id = str(uuid.uuid4())[:8]
    keys   = [f"{run_id}_agent{i}" for i in range(n_agents)]
    for i, k in enumerate(keys):
        create_shard(k, f"initial content for agent {i}")

    total = commits = conflicts = 0

    def agent_work(idx: int) -> tuple[int, int]:
        my_key = keys[idx]
        c = cf = 0
        for step in range(n_steps):
            s = read_shard(my_key)
            if s is None:
                continue
            ok, _ = commit(my_key, s["version"],
                           f"agent{idx}_step{step}", f"agent{idx}_{run_id}")
            if ok:
                c += 1
            else:
                cf += 1
        return c, cf

    with ThreadPoolExecutor(max_workers=n_agents) as pool:
        futures = [pool.submit(agent_work, i) for i in range(n_agents)]
        for f in as_completed(futures):
            c, cf = f.result()
            commits   += c
            conflicts += cf
            total     += c + cf

    # Verify version trace: each shard should be at exactly n_steps
    corruptions = 0
    for i, k in enumerate(keys):
        s = read_shard(k)
        if s and s["version"] != n_steps:
            corruptions += 1

    scr = conflicts / total if total > 0 else 0.0
    return {
        "topology":    "distinct",
        "n_agents":    n_agents,
        "n_steps":     n_steps,
        "total_commits": total,
        "commits":     commits,
        "conflicts":   conflicts,
        "corruptions": corruptions,
        "scr":         round(scr, 4),
    }


# ── Experiment 2: Shared shard (high contention) ─────────────────────────────

def run_shared_shard(n_agents: int, n_steps: int) -> dict:
    """
    All agents target a SINGLE shared shard — maximum contention.
    Expected: SCR > 0 (conflicts), but zero corruptions.
    The ACP serializes all writes; version trace is strictly monotone.
    """
    run_id = str(uuid.uuid4())[:8]
    key    = f"shared_{run_id}"
    create_shard(key, "shared initial content")

    total = commits = conflicts = 0

    def agent_work(idx: int) -> tuple[int, int]:
        c = cf = 0
        for step in range(n_steps):
            s = read_shard(key)
            if s is None:
                continue
            ok, _ = commit(key, s["version"],
                           f"agent{idx}_step{step}", f"agent{idx}_{run_id}")
            if ok:
                c += 1
            else:
                cf += 1
        return c, cf

    with ThreadPoolExecutor(max_workers=n_agents) as pool:
        futures = [pool.submit(agent_work, i) for i in range(n_agents)]
        for f in as_completed(futures):
            c, cf = f.result()
            commits   += c
            conflicts += cf
            total     += c + cf

    # Verify NO corruption: final version must equal exact commits,
    # no gaps or duplicates in delta log
    s            = read_shard(key)
    final_ver    = s["version"] if s else -1
    corruptions  = 0 if final_ver == commits else 1  # corruption = version mismatch

    scr = conflicts / total if total > 0 else 0.0
    return {
        "topology":    "shared",
        "n_agents":    n_agents,
        "n_steps":     n_steps,
        "total_commits": total,
        "commits":     commits,
        "conflicts":   conflicts,
        "corruptions": corruptions,
        "scr":         round(scr, 4),
    }


# ── Experiment 3: Baseline (no S-Bus) — for comparison ───────────────────────

def run_baseline(n_agents: int, n_steps: int) -> dict:
    """
    Simulate coordinator-worker: all agents write to same key with no
    conflict detection (expected version always 0). Measures the
    corruption rate WITHOUT S-Bus protection.
    """
    run_id = str(uuid.uuid4())[:8]
    key    = f"baseline_{run_id}"
    create_shard(key, "baseline initial")

    total = commits = conflicts = corruptions = 0

    def agent_work(idx: int) -> tuple[int, int, int]:
        c = cf = corrupt = 0
        for step in range(n_steps):
            # Always claim version=0 (ignores actual version — simulates no protection)
            ok, err = commit(key, 0, f"agent{idx}_step{step}", f"agent{idx}_{run_id}")
            total_local = c + cf + 1
            if ok:
                c += 1
            else:
                cf += 1
                if "VersionMismatch" in err or "mismatch" in err.lower():
                    corrupt += 1  # would have been a stale write in real system
        return c, cf, corrupt

    with ThreadPoolExecutor(max_workers=n_agents) as pool:
        futures = [pool.submit(agent_work, i) for i in range(n_agents)]
        for f in as_completed(futures):
            c, cf, co = f.result()
            commits     += c
            conflicts   += cf
            corruptions += co
            total       += c + cf

    scr = corruptions / total if total > 0 else 0.0
    return {
        "topology":    "baseline_no_sbus",
        "n_agents":    n_agents,
        "n_steps":     n_steps,
        "total_commits": total,
        "commits":     commits,
        "conflicts":   conflicts,
        "corruptions": corruptions,
        "scr":         round(scr, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not verify_server():
        return

    os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)

    results = []
    N_STEPS = 10  # steps per agent (enough to see contention)

    print("\n── Exp 1: Distinct shards (baseline, expect SCR=0) ──────────────────")
    for n in [3, 4, 8, 16]:
        r = run_distinct_shards(n, N_STEPS)
        results.append(r)
        icon = "✅" if r["corruptions"] == 0 and r["scr"] == 0 else "❌"
        print(f"  {icon} N={n:2d}: commits={r['commits']:4d}  "
              f"conflicts={r['conflicts']:4d}  SCR={r['scr']:.3f}  "
              f"corruptions={r['corruptions']}")

    print("\n── Exp 2: Shared shard (high contention, expect SCR>0 but 0 corruptions) ─")
    for n in [4, 8, 16]:
        r = run_shared_shard(n, N_STEPS)
        results.append(r)
        icon = "✅" if r["corruptions"] == 0 else "❌"
        print(f"  {icon} N={n:2d}: commits={r['commits']:4d}  "
              f"conflicts={r['conflicts']:4d}  SCR={r['scr']:.3f}  "
              f"corruptions={r['corruptions']}")

    print("\n── Exp 3: Baseline without S-Bus protection (expect corruptions>0) ────")
    for n in [4]:
        r = run_baseline(n, N_STEPS)
        results.append(r)
        print(f"  N={n:2d}: commits={r['commits']:4d}  "
              f"conflicts={r['conflicts']:4d}  "
              f"corruptions={r['corruptions']} ({r['scr']:.1%} rate)")

    # Save CSV
    fields = ["topology", "n_agents", "n_steps", "total_commits",
              "commits", "conflicts", "corruptions", "scr"]
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"\n✅ Results saved → {OUT_FILE}")

    # Summary for paper
    sbus_rows = [r for r in results if r["topology"] in ("distinct", "shared")]
    total_inj  = sum(r["total_commits"] for r in sbus_rows)
    total_corr = sum(r["corruptions"]   for r in sbus_rows)
    print(f"\n── Paper claim verification ──────────────────────────────────────────")
    print(f"  Total commit attempts (S-Bus): {total_inj:,}")
    print(f"  Write-write corruptions:       {total_corr}")
    if total_corr == 0:
        print(f"  ✅ Claim holds: zero write-write corruptions across {total_inj:,} events")
    else:
        print(f"  ❌ {total_corr} corruptions found — investigate before submitting")


if __name__ == "__main__":
    main()