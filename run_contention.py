#!/usr/bin/env python3
"""
run_contention.py — S-Bus contention scale: N=4,8,16,32,64
============================================================
Extends Exp.E to N=32 and N=64. Measures:
  - SCR (Semantic Conflict Rate) by topology
  - Type-I corruptions (MUST be zero)
  - K95 retry budget for 95% success

Two topologies:
  shared   — all N agents write to ONE shard (worst-case contention)
  distinct — each agent writes to its OWN shard (SCR must = 0)

USAGE
------
  # Start S-Bus first:
  SBUS_ADMIN_ENABLED=1 cargo run --release

  # Then run (no LLM needed):
  python3 run_contention.py

  # Results saved to: results/contention_scale_v2.csv
"""

import csv, json, math, os, threading, time, uuid
from urllib.request import Request, urlopen
from urllib.error import HTTPError

URL = os.getenv("SBUS_URL", "http://localhost:7000")

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def post(path, body):
    data = json.dumps(body).encode()
    req  = Request(f"{URL}{path}", data=data,
                   headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:    b = json.loads(e.read())
        except: b = {}
        return e.code, b
    except Exception as e:
        return 0, {"error": str(e)}


def get(path):
    try:
        with urlopen(f"{URL}{path}", timeout=15) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


# ── Health check ──────────────────────────────────────────────────────────────

def health_check():
    st, d = get("/stats")
    if st == 200:
        print(f"  S-Bus running: {d}")
        return True
    print(f"  S-Bus not responding (status={st})")
    print("  Start: SBUS_ADMIN_ENABLED=1 cargo run --release")
    return False


# ── One agent ─────────────────────────────────────────────────────────────────

def run_agent(agent_id, shard_key, n_attempts,
              results, committed_versions, cv_lock, barrier):
    ok = conflicts = errors = corruptions = 0
    try:
        barrier.wait(timeout=30)
    except threading.BrokenBarrierError:
        return

    for attempt in range(n_attempts):
        st, data = get(f"/shard/{shard_key}")
        if st != 200:
            errors += 1
            continue

        cur_ver     = data.get("version", 0)
        new_content = data.get("content", "") + f" [{agent_id}:{attempt}]"

        # Correct field names: expected_ver + content (NOT expected_version/delta)
        st2, resp = post("/commit", {
            "key":          shard_key,
            "expected_ver": cur_ver,
            "content":      new_content,
            "agent_id":     agent_id,
            "rationale":    f"attempt {attempt}",
        })

        if st2 == 200:
            ok += 1
            commit_key = (shard_key, cur_ver)
            with cv_lock:
                if commit_key in committed_versions and \
                   committed_versions[commit_key] != agent_id:
                    corruptions += 1
                else:
                    committed_versions[commit_key] = agent_id
        elif st2 in (409, 423):
            conflicts += 1
        else:
            errors += 1

    with cv_lock:
        results["ok"]          += ok
        results["conflicts"]   += conflicts
        results["errors"]      += errors
        results["corruptions"] += corruptions


# ── Run one combination ───────────────────────────────────────────────────────

def run_one(n_agents, topology, n_attempts, rep):
    run_id = uuid.uuid4().hex[:6]

    if topology == "shared":
        shard_map = {f"a{i}": f"s_{run_id}" for i in range(n_agents)}
    else:
        shard_map = {f"a{i}": f"s_{run_id}_{i}" for i in range(n_agents)}

    for sk in sorted(set(shard_map.values())):
        st, d = post("/shard", {
            "key":      sk,
            "content":  "initial",
            "goal_tag": "contention_scale",
        })
        if st not in (200, 201):
            print(f"    Shard creation failed: status={st} {d}")
            return None

    results            = {"ok": 0, "conflicts": 0, "errors": 0, "corruptions": 0}
    committed_versions = {}
    cv_lock            = threading.Lock()
    barrier            = threading.Barrier(n_agents)

    t0      = time.perf_counter()
    threads = [
        threading.Thread(
            target=run_agent,
            args=(aid, shard_map[aid], n_attempts,
                  results, committed_versions, cv_lock, barrier),
            daemon=True,
        )
        for aid in shard_map
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout=120)
    wall = time.perf_counter() - t0

    total = results["ok"] + results["conflicts"] + results["errors"]
    scr   = results["conflicts"] / total if total > 0 else 0
    safe  = results["corruptions"] == 0
    if   scr <= 0: k95 = 1
    elif scr >= 1: k95 = 9999
    else:          k95 = max(1, math.ceil(math.log(0.05) / math.log(scr)))

    return {
        "n_agents":       n_agents,
        "topology":       topology,
        "repeat":         rep,
        "commits_ok":     results["ok"],
        "attempts_total": total,
        "conflicts":      results["conflicts"],
        "errors":         results["errors"],
        "corruptions":    results["corruptions"],
        "scr":            round(scr, 4),
        "k95":            k95,
        "type1_safe":     safe,
        "wall_secs":      round(wall, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    AGENTS     = [4, 8, 16, 32, 64]
    TOPOLOGIES = ["distinct", "shared"]
    ATTEMPTS   = 100
    REPEATS    = 3

    print("=" * 60)
    print("Contention Scale: N=4,8,16,32,64")
    print("=" * 60)
    print("\nStep 1: S-Bus health check")
    if not health_check():
        return

    os.makedirs("results", exist_ok=True)
    out_path = "results/contention_scale_v2.csv"
    fields   = ["n_agents","topology","repeat","commits_ok","attempts_total",
                "conflicts","errors","corruptions","scr","k95","type1_safe","wall_secs"]
    out    = open(out_path, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()

    print(f"\nStep 2: running ({len(AGENTS)} N values × {len(TOPOLOGIES)} topologies × {REPEATS} repeats)")
    print(f"  {'N':>4}  {'Topology':<10}  {'rep':>3}  "
          f"{'ok':>7}  {'conflict':>9}  {'SCR':>7}  {'K95':>5}  Safe")
    print("  " + "-" * 58)

    all_corrupt = 0
    all_attempts = 0

    for n in AGENTS:
        for topo in TOPOLOGIES:
            for rep in range(1, REPEATS + 1):
                r = run_one(n, topo, ATTEMPTS, rep)
                if r is None:
                    continue
                all_corrupt  += r["corruptions"]
                all_attempts += r["attempts_total"]
                sym = "✅" if r["type1_safe"] else "❌"
                print(f"  {n:>4}  {topo:<10}  {rep:>3}  "
                      f"{r['commits_ok']:>7}  {r['conflicts']:>9}  "
                      f"{r['scr']:>7.4f}  {r['k95']:>5}  {sym}  "
                      f"{r['wall_secs']:.1f}s")
                writer.writerow(r)
                out.flush()

    out.close()
    print()
    print("=" * 60)
    print(f"  Attempts  : {all_attempts:,}")
    print(f"  Corruptions: {all_corrupt}")
    if all_corrupt == 0:
        print(f"  ZERO TYPE-I CORRUPTIONS — Property 3.1 holds at N=32, N=64")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()