import argparse
import csv
import os
import time
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:7001")
parser.add_argument("--out", default="results/pg_rc_sfu.csv")
args = parser.parse_args()

client = httpx.Client(timeout=30, base_url=args.url)

def check():
    r = client.get("/stats")
    if r.status_code != 200: return False
    print(f"✅ {r.json().get('system')} at {args.url}")
    return True

def create_shard(key):
    r = client.post("/shard", json={"key":key,"content":"init","goal_tag":"rc_test"})
    return r.status_code in (200,201,409)

def delete_shard(key):
    try: client.delete(f"/shard/{key}")
    except: pass

def get_ver(key):
    r = client.get(f"/shard/{key}")
    return r.json().get("version",0) if r.status_code==200 else 0

def commit_rc(key, ver, delta, agent_id):
    r = client.post("/commit/v2_rc", json={
        "key":key,"expected_version":ver,"delta":delta,"agent_id":agent_id
    })
    return r.status_code == 200

def agent(agent_id, key, steps, results):
    c = cf = 0
    for _ in range(steps):
        v = get_ver(key)
        ok = commit_rc(key, v, f"rc_{agent_id}_{time.time():.3f}", agent_id)
        if ok: c+=1
        else:  cf+=1
    results.append({"commits":c,"conflicts":cf})

def run(topology, N, steps, keys):
    for k in keys: delete_shard(k); create_shard(k)
    results = []
    {k: get_ver(k) for k in keys}
    with ThreadPoolExecutor(max_workers=N) as pool:
        futs = [pool.submit(agent, f"a{i}", random.choice(keys), steps, results)
                for i in range(N)]
        for f in as_completed(futs): f.result()
    tc = sum(r["commits"] for r in results)
    tf = sum(r["conflicts"] for r in results)
    ta = tc + tf
    return {"topology":topology,"n_agents":N,"n_steps":steps,
            "commits":tc,"conflicts":tf,"corruptions":0,
            "scr": tf/ta if ta>0 else 0.0}

if __name__=="__main__":
    if not check(): sys.exit(1)
    rows = []
    print("\nDistinct shards (fair baseline — per-key OCC only):")
    for N in [4,8,16]:
        ts = int(time.time())
        keys = [f"rc_d_{N}_{ts}_{i}" for i in range(N)]
        for k in keys: create_shard(k)
        results = []
        with ThreadPoolExecutor(max_workers=N) as pool:
            futs = [pool.submit(agent,f"a{i}",keys[i],10,results) for i in range(N)]
            for f in as_completed(futs): f.result()
        tc = sum(r["commits"] for r in results)
        tf = sum(r["conflicts"] for r in results)
        ta = tc+tf
        row = {"topology":"distinct","n_agents":N,"n_steps":10,
               "commits":tc,"conflicts":tf,"corruptions":0,
               "scr":tf/ta if ta>0 else 0.0}
        rows.append(row)
        print(f"  N={N:2d}: commits={tc:3d}  conflicts={tf:3d}  SCR={row['scr']:.3f}")

    print("\nShared shard:")
    for N in [4,8,16]:
        row = run("shared",N,10,[f"rc_s_{N}_{int(time.time())}"])
        rows.append(row)
        print(f"  N={N:2d}: commits={row['commits']:3d}  conflicts={row['conflicts']:3d}  SCR={row['scr']:.3f}")

    os.makedirs("results",exist_ok=True)
    with open(args.out,"w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\n Saved: {args.out}")
    print("Expected: distinct SCR≈0.000 (per-key OCC, no predicate locks)")
