#!/usr/bin/env python3
"""deliverylog_fp_experiment.py v2 — added --delay-ms flag"""
from __future__ import annotations
import argparse, csv, os, time, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import httpx

class Bus:
    def __init__(self, url):
        self.url = url
        self.c = httpx.Client(timeout=60)
    def ping(self):
        try: return self.c.get(f"{self.url}/stats", timeout=5).status_code == 200
        except: return False
    def create(self, key, content):
        return self.c.post(f"{self.url}/shard", json={"key":key,"content":content,"goal_tag":"fp_test"}).status_code == 201
    def read(self, key, agent_id=""):
        r = self.c.get(f"{self.url}/shard/{key}", params={"agent_id":agent_id} if agent_id else {})
        return r.json() if r.status_code==200 else None
    def commit(self, key, version, delta, agent_id, read_set=None):
        body={"key":key,"expected_version":version,"delta":delta,"agent_id":agent_id}
        if read_set is not None: body["read_set"]=read_set
        r=self.c.post(f"{self.url}/commit/v2",json=body)
        if r.status_code==200: return True,""
        try: return False,r.json().get("error","?")
        except: return False,"parse_error"

@dataclass
class FpResult:
    n_agents:int; k_extra:int; delay_ms:int; rounds:int
    total_commits:int=0; fp_count:int=0; vm_count:int=0; success_count:int=0
    @property
    def fp_rate(self): return self.fp_count/self.total_commits if self.total_commits else 0.0
    @property
    def success_rate(self): return self.success_count/self.total_commits if self.total_commits else 0.0

def run_experiment(bus, n_agents, k_extra, rounds, delay_ms, run_id, arsi=False):
    result = FpResult(n_agents=n_agents, k_extra=-1 if arsi else k_extra,
                      delay_ms=delay_ms, rounds=rounds)
    prefix = "arsi" if arsi else "dl"
    keys = [f"{run_id}_{prefix}_a{i}" for i in range(n_agents)]
    for i,k in enumerate(keys): bus.create(k, f"Init {i}")

    def agent_round(idx, rnd):
        aid = f"{prefix}_{idx}"
        mk  = keys[idx]
        s   = bus.read(mk, agent_id=aid)
        if s is None: return False,"not_found"
        ver = s["version"]
        if arsi:
            if delay_ms>0: time.sleep(delay_ms/1000)
            return bus.commit(mk,ver,f"r{rnd}",aid,
                              read_set=[{"key":mk,"version_at_read":ver}])
        else:
            others=[k for k in keys if k!=mk]
            for j in range(min(k_extra,len(others))): bus.read(others[j],agent_id=aid)
            if delay_ms>0: time.sleep(delay_ms/1000)
            return bus.commit(mk,ver,f"r{rnd}",aid,read_set=[])

    for rnd in range(rounds):
        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            for fut in as_completed({pool.submit(agent_round,i,rnd):i for i in range(n_agents)}):
                result.total_commits += 1
                ok,err = fut.result()
                if ok: result.success_count += 1
                elif "CrossShardStale" in err or "stale" in err.lower(): result.fp_count += 1
                elif "Mismatch" in err or "mismatch" in err.lower(): result.vm_count += 1
    return result

def main():
    ap = argparse.ArgumentParser(description="DeliveryLog FP rate experiment")
    ap.add_argument("--url",         default="http://localhost:7000")
    ap.add_argument("--rounds",      type=int, default=100)
    ap.add_argument("--n-agents",    type=int, nargs="+", default=[4,8,16])
    ap.add_argument("--extra-reads", type=int, nargs="+", default=[0,1,2,3])
    ap.add_argument("--delay-ms",    type=int, default=50,
                    help="ms between reads and commit. 50=worst-case; 500=LLM timescale")
    ap.add_argument("--out",         default="results/deliverylog_fp_results.csv")
    args = ap.parse_args()

    bus = Bus(args.url)
    if not bus.ping():
        print(f"ERROR: Cannot reach S-Bus at {args.url}")
        raise SystemExit(1)
    print(f"S-Bus connected | delay_ms={args.delay_ms}")

    results = []
    rid = str(uuid.uuid4())[:8]

    for n in args.n_agents:
        r = run_experiment(bus,n,0,args.rounds,args.delay_ms,f"{rid}_n{n}",arsi=True)
        results.append(r)
        print(f"N={n} ARSI: FP={r.fp_count}/{r.total_commits} ({r.fp_rate:.0%})")
        for k in args.extra_reads:
            r = run_experiment(bus,n,k,args.rounds,args.delay_ms,f"{rid}_n{n}_k{k}")
            results.append(r)
            print(f"N={n} k={k}: FP={r.fp_count}/{r.total_commits} ({r.fp_rate:.0%})")

    fields=["n_agents","k_extra","delay_ms","rounds","total_commits","fp_count","fp_rate","vm_count","success_count","success_rate"]
    os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
    with open(args.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in results:
            w.writerow({"n_agents":r.n_agents,"k_extra":r.k_extra,"delay_ms":r.delay_ms,
                        "rounds":r.rounds,"total_commits":r.total_commits,"fp_count":r.fp_count,
                        "fp_rate":f"{r.fp_rate:.4f}","vm_count":r.vm_count,
                        "success_count":r.success_count,"success_rate":f"{r.success_rate:.4f}"})
    print(f"Results → {args.out}")
    print(f"\nTo run at LLM timescales: python3 {__file__} --delay-ms 500")

if __name__=="__main__":
    main()