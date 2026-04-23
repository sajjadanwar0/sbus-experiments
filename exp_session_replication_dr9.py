#!/usr/bin/env python3
"""
exp_session_replication_dr9.py — DR-9: ORI survives leader failover
Fixes: unique agent IDs per trial, longer Raft wait, uniform CSV fields.
"""

import os
import sys
import time
import requests
import csv
import subprocess
import statistics
import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

NODE_URLS = {
    0: os.getenv("SBUS_NODE0", "http://localhost:7000"),
    1: os.getenv("SBUS_NODE1", "http://localhost:7001"),
    2: os.getenv("SBUS_NODE2", "http://localhost:7002"),
}
SBUS_ROOT  = os.getenv("SBUS_ROOT", str(Path.home() / "RustroverProjects/sbus"))
RESTART_SH = os.path.join(SBUS_ROOT, "restart_node.sh")

# All possible CSV fields — ensures uniform rows even for ERROR cases
CSV_FIELDS = ["trial","status","ori_held","alpha1_http","alpha1_error",
              "v0","election_ms","killed_node","new_leader","wall_secs",
              "timestamp","reason"]

def empty_row(trial_id):
    return {f: "" for f in CSV_FIELDS} | {"trial": trial_id, "timestamp": datetime.now(timezone.utc).isoformat()}

# ── Leader detection ──────────────────────────────────────────────────────────
def get_leader():
    for nid, url in NODE_URLS.items():
        try:
            r = requests.get(f"{url}/admin/health", timeout=2)
            if r.ok and "Leader" in str(r.json().get("raft_state", "")):
                return nid, url
        except Exception:
            pass
    for nid, url in NODE_URLS.items():
        try:
            r = requests.get(f"{url}/raft/leader", timeout=2)
            if r.ok:
                lid = r.json().get("current_leader")
                if lid is not None and lid == nid:
                    return nid, NODE_URLS.get(lid, url)
        except Exception:
            pass
    return None, None

def wait_for_leader(timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        nid, url = get_leader()
        if nid is not None:
            return nid, url
        time.sleep(0.3)
    return None, None

def wait_for_new_leader(old_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        nid, url = get_leader()
        if nid is not None and nid != old_id:
            return nid, url
        time.sleep(0.2)
    return None, None

# ── Node control ──────────────────────────────────────────────────────────────
def node_port(nid):
    return int(NODE_URLS[nid].rstrip("/").split(":")[-1])

def kill_node(nid):
    """Kill only the server LISTENING on this port — NOT Python client connections."""
    port = node_port(nid)
    subprocess.run(
        ["bash", "-c",
         f"lsof -ti TCP:{port} -sTCP:LISTEN 2>/dev/null | xargs -r kill -9"],
        capture_output=True, timeout=5
    )
    time.sleep(0.5)

def restart_node(nid):
    if not os.path.exists(RESTART_SH):
        print(f"\n  ⚠  {RESTART_SH} not found")
        return False
    r = subprocess.run(
        ["bash", RESTART_SH, str(nid), SBUS_ROOT],
        capture_output=True, text=True, timeout=20
    )
    return r.returncode == 0

def node_is_up(nid, timeout=10.0):
    url = NODE_URLS[nid]
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/admin/health", timeout=1).ok:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False

def register_session(url, agent_id):
    """Create a fresh session (empty DeliveryLog) for this agent."""
    try:
        requests.post(f"{url}/session",
                      json={"agent_id": agent_id}, timeout=3)
    except Exception:
        pass

# ── S-Bus operations ──────────────────────────────────────────────────────────
def do_create(url, key):
    r = requests.post(f"{url}/shard",
        json={"key": key, "content": "dr9_initial", "goal_tag": "dr9"}, timeout=5)
    return r.status_code in (200, 201, 409)

def do_get(url, key, agent):
    r = requests.get(f"{url}/shard/{key}?agent_id={agent}", timeout=5)
    if r.ok:
        return r.json()["version"]
    raise Exception(f"GET {r.status_code}: {r.text[:80]}")

def do_commit(url, key, ver, agent, delta):
    r = requests.post(f"{url}/commit/v2",
        json={"key": key, "expected_version": ver,
              "delta": delta, "agent_id": agent}, timeout=5)
    return r.status_code, r.json()

# ── Single trial ──────────────────────────────────────────────────────────────
def run_trial(trial_id):
    t0 = time.time()
    row = empty_row(trial_id)

    # Unique per-trial shard key AND agent IDs
    # Agent IDs are unique to avoid stale DeliveryLog entries from prior trials
    key    = f"dr9_t{trial_id}_{int(t0*1000) % 999999}"
    alpha1 = f"dr9_alpha1_t{trial_id}"
    alpha2 = f"dr9_alpha2_t{trial_id}"

    # Find leader
    leader_id, leader_url = wait_for_leader(timeout=10)
    if leader_id is None:
        return row | {"status": "ERROR", "reason": "no_leader_at_start"}

    # Register fresh sessions for both agents on the leader
    register_session(leader_url, alpha1)
    register_session(leader_url, alpha2)

    # Create shard via Raft (replicated to all nodes)
    do_create(leader_url, key)
    time.sleep(1.5)  # wait for Raft replication to all nodes

    # α1 GETs shard → DeliveryLog entry replicated via P1
    try:
        v0 = do_get(leader_url, key, alpha1)
    except Exception as e:
        return row | {"status": "ERROR", "reason": f"alpha1_get:{e}"}

    time.sleep(0.6)  # allow Raft DeliveryEntry to replicate

    # Kill leader
    kill_t = time.time()
    killed_id = leader_id
    kill_node(killed_id)

    # Wait for new leader
    new_id, new_url = wait_for_new_leader(killed_id, timeout=10)
    election_ms = (time.time() - kill_t) * 1000

    if new_id is None:
        restart_node(killed_id)
        node_is_up(killed_id, timeout=12)
        return row | {"status": "ERROR", "reason": "no_new_leader",
                      "election_ms": round(election_ms)}

    # Register alpha2 session on new leader
    register_session(new_url, alpha2)

    # α2 bumps version on new leader: v0 → v1
    try:
        v_cur = do_get(new_url, key, alpha2)
        sc2, r2 = do_commit(new_url, key, v_cur, alpha2, "alpha2_update")
        if sc2 != 200:
            restart_node(killed_id)
            node_is_up(killed_id, timeout=12)
            return row | {"status": "ERROR",
                          "reason": f"alpha2_{sc2}:{r2.get('error','')}",
                          "election_ms": round(election_ms)}
    except Exception as e:
        restart_node(killed_id)
        node_is_up(killed_id, timeout=12)
        return row | {"status": "ERROR", "reason": f"alpha2:{e}",
                      "election_ms": round(election_ms)}

    # α1 commits with stale v0 (shard now at v1)
    # With P1:    new leader has DLog[alpha1]→(key, v0) → detects stale → 409 ✅
    # Without P1: empty DLog → no check → 200 ❌
    try:
        sc1, r1 = do_commit(new_url, key, v0, alpha1, "alpha1_stale")
    except Exception as e:
        restart_node(killed_id)
        node_is_up(killed_id, timeout=12)
        return row | {"status": "ERROR", "reason": f"alpha1:{e}",
                      "election_ms": round(election_ms)}

    ori_held = (sc1 == 409)

    # Restart killed node → back to 3-node cluster
    restart_node(killed_id)
    node_is_up(killed_id, timeout=12)
    time.sleep(2.0)  # Raft catch-up

    return row | {
        "status":       "OK",
        "ori_held":     ori_held,
        "alpha1_http":  sc1,
        "alpha1_error": r1.get("error", ""),
        "v0":           v0,
        "election_ms":  round(election_ms, 1),
        "killed_node":  killed_id,
        "new_leader":   new_id,
        "wall_secs":    round(time.time() - t0, 1),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--output",   default="results/dr9_session_replication.csv")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("DR-9: Session Replication — ORI Survives Failover")
    print("=" * 60)
    print(f"Nodes:    {NODE_URLS}")
    print(f"Restart:  {RESTART_SH}")
    print(f"Trials:   {args.n_trials}")
    print()

    nid, url = wait_for_leader(8)
    if nid is None:
        print("ERROR: No leader. Run ./start_raft_cluster.sh first.")
        sys.exit(1)
    print(f"Initial leader: Node {nid} ({url})\n")

    results, held, errors = [], 0, 0

    for i in range(1, args.n_trials + 1):
        print(f"Trial {i:2d}/{args.n_trials} ... ", end="", flush=True)
        r = run_trial(i)
        results.append(r)

        if r["status"] == "ERROR":
            print(f"ERROR — {r['reason']}")
            errors += 1
            time.sleep(3)
        else:
            if r["ori_held"]:
                held += 1
            icon = "✅" if r["ori_held"] else "❌"
            print(f"{icon} ORI={'HELD' if r['ori_held'] else 'MISSED'}  "
                  f"HTTP={r['alpha1_http']}  "
                  f"election={r['election_ms']:.0f}ms  "
                  f"killed=Node{r['killed_node']}→Node{r['new_leader']}  "
                  f"wall={r['wall_secs']:.0f}s")

    # Write CSV — all rows have identical fields
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    ok = [r for r in results if r["status"] == "OK"]
    ems = [float(r["election_ms"]) for r in ok if r["election_ms"]]

    print(f"\n{'='*60}")
    print(f"Completed:  {len(ok)}/{args.n_trials}")
    print(f"ORI held:   {held}/{len(ok)} "
          f"({100*held/max(len(ok),1):.1f}%)")
    print(f"Errors:     {errors}")
    if ems:
        print(f"Election:   median={statistics.median(ems):.0f}ms  "
              f"min={min(ems):.0f}ms  max={max(ems):.0f}ms")
    print(f"Output:     {args.output}")
    print()
    print("Expected WITH    P1 session replication: 100%")
    print("Expected WITHOUT P1 session replication:   0%")

if __name__ == "__main__":
    main()
