#!/usr/bin/env python3
"""
exp_sequential_wall_time_v2.py — Wall-Time + Latency Breakdown
==============================================================
FIXES over v1/v3:
  FIX-1: Seeds each shard with a real task description so agents have
          non-empty context → LLM produces meaningful deltas → commits succeed.
  FIX-2: Tracks llm_calls and commit_ok separately per run.
  FIX-3: Latency breakdown: measures llm_ms vs sbus_ms vs retry_ms per run.
  FIX-4: Semantic judge reads final shard content (non-empty) → valid verdict.
  FIX-5: delta trimmed to 1800 chars (under SBUS_MAX_DELTA=2000).
  FIX-6: Column names consistent with v3 schema.

USAGE
-----
  # Smoke test (~10 min)
  OPENAI_API_KEY=sk-... python3 exp_sequential_wall_time_v2.py \\
      --n-tasks 3 --n-agents-list 4 8 --n-steps 4 --n-repeats 2 \\
      --workers 2 --output results/exp_sequential_v4.csv

  # Paper quality (~90 min)
  OPENAI_API_KEY=sk-... python3 exp_sequential_wall_time_v2.py \\
      --n-tasks 10 --n-agents-list 4 8 16 --n-steps 8 --n-repeats 3 \\
      --workers 2 --output results/exp_sequential_v4.csv
"""

import argparse, csv, json, math, os, random, sys
import threading, time, uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai"); sys.exit(1)

SBUS_URL = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE = "gpt-4o-mini"
MAX_DELTA = 1800   # chars; SBUS_MAX_DELTA default is 2000

TASKS = [
    {"id": f"t{i:02d}", "domain": d, "problem": p,
     "seed": s}                            # FIX-1: seed content for shard
    for i, (d, p, s) in enumerate([
        ("django",
         "Fix Django queryset ordering with select_related FK traversal",
         "Problem: Django ORM select_related() with FK traversal produces wrong ORDER BY. "
         "Root cause is in django/db/models/sql/compiler.py get_order_by(). "
         "Agents should propose a concrete patch."),
        ("astropy",
         "Fix Astropy FITS HIERARCH keyword parsing",
         "Problem: Astropy fails to parse FITS HIERARCH keywords with non-standard spacing. "
         "Root cause in astropy/io/fits/card.py. Agents should propose a fix."),
        ("sympy",
         "Fix SymPy solve() dropping solutions with positive assumptions",
         "Problem: SymPy solve() silently drops valid solutions when assumptions are set. "
         "Root cause in sympy/solvers/solvers.py _solve(). Agents should propose a patch."),
        ("django",
         "Fix Django migration squasher circular dependencies",
         "Problem: Django squashmigrations crashes on circular RunSQL dependencies. "
         "Root cause in django/core/management/commands/squashmigrations.py. Fix needed."),
        ("astropy",
         "Fix Astropy ZEA projection boundary errors near poles",
         "Problem: Astropy ZEA sky projection raises ValueError near declination poles. "
         "Root cause in astropy/wcs/utils.py. Agents should propose a geometric fix."),
        ("sympy",
         "Fix SymPy eigenvals() for sparse Rational matrices",
         "Problem: SymPy eigenvals() returns wrong results for large sparse Rational matrices. "
         "Root cause in sympy/matrices/matrices.py. Agents should propose a fix."),
        ("django",
         "Fix Django admin bulk action permission checks",
         "Problem: Django admin bulk actions bypass per-object permissions. "
         "Root cause in django/contrib/admin/options.py response_action(). Fix needed."),
        ("astropy",
         "Fix Astropy coordinate transformation galactic to ICRS",
         "Problem: Astropy galactic-to-ICRS transformation has numerical errors near equator. "
         "Root cause in astropy/coordinates/builtin_frames/. Agents should fix the matrix."),
        ("sympy",
         "Fix SymPy polynomial factoring over finite fields",
         "Problem: SymPy factor_list() hangs on dense polynomials over GF(p). "
         "Root cause in sympy/polys/factortools.py. Agents should propose a fix."),
        ("django",
         "Fix Django ModelForm field ordering with Meta.fields override",
         "Problem: Django ModelForm ignores Meta.fields ordering when using __all__. "
         "Root cause in django/forms/models.py fields_for_model(). Fix needed."),
    ])
]

# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, rpm=250):
        self._interval = 60.0 / rpm
        self._last = 0.0
        self._lock = threading.Lock()
    def acquire(self):
        with self._lock:
            now = time.monotonic()
            w = self._last + self._interval - now
            if w > 0: time.sleep(w)
            self._last = time.monotonic()

# ── S-Bus API ─────────────────────────────────────────────────────────────────

def _req(method, path, body=None, params=None):
    url = SBUS_URL + path
    if params: url += "?" + urlencode(params)
    data = json.dumps(body).encode() if body else None
    hdrs = {"Content-Type": "application/json"} if data else {}
    req = Request(url, data=data, headers=hdrs, method=method)
    try:
        with urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try: return e.code, json.loads(e.read())
        except: return e.code, {}
    except: return 0, {}

def create_shard(key, domain, seed_content=""):
    # FIX-1: pass seed_content so shard is non-empty from the start
    st, _ = _req("POST", "/shard",
                 {"key": key, "content": seed_content, "goal_tag": domain})
    return st in (200, 201)

def register_agent(agent_id):
    st, _ = _req("POST", "/session", {"agent_id": agent_id})
    return st == 200

def read_shard(key, agent_id):
    st, d = _req("GET", f"/shard/{key}", params={"agent_id": agent_id})
    if st == 200: return d.get("version", 0), d.get("content", "")
    return 0, ""

def do_commit(key, agent_id, expected_version, delta):
    # Trim delta to stay under SBUS_MAX_DELTA  (FIX-5)
    delta = delta[:MAX_DELTA]
    st, resp = _req("POST", "/commit/v2", {
        "key": key, "expected_version": expected_version,
        "delta": delta, "agent_id": agent_id,
    })
    return st == 200 and "new_version" in resp

def health_check():
    import socket
    try: s = socket.create_connection(("localhost", 7000), timeout=3); s.close()
    except: return False
    st, _ = _req("GET", "/stats")
    return st == 200

# ── LLM call ─────────────────────────────────────────────────────────────────

def llm_call(oai, rl, problem, context, agent_id, step):
    rl.acquire()
    t0 = time.perf_counter()
    try:
        r = oai.chat.completions.create(
            model=BACKBONE, max_tokens=120, temperature=0.3,
            messages=[{"role": "user", "content": (
                f"TASK: {problem[:200]}\n"
                f"CURRENT STATE (step {step}):\n{context[:400]}\n"
                f"Add 2-3 sentences: one concrete technical improvement or fix step."
            )}])
        text = r.choices[0].message.content.strip()
    except Exception as e:
        text = f"[ERR:{e}]"
    llm_ms = int((time.perf_counter() - t0) * 1000)
    delta = f"[{agent_id}_s{step}] {text}"
    return delta, llm_ms

# ── Simple semantic judge (FIX-4) ────────────────────────────────────────────

JUDGE_SYS = (
    "You evaluate a collaborative technical document. "
    "Reply with exactly one word: COMPLETE, INCOMPLETE, or CORRUPTED.\n"
    "COMPLETE = substantive content addressing the task.\n"
    "INCOMPLETE = too little content or mostly empty.\n"
    "CORRUPTED = content is contradictory or incoherent."
)

def judge_content(oai, rl, problem, content):
    if not content or len(content.strip()) < 50:
        return "INCOMPLETE"
    rl.acquire()
    try:
        r = oai.chat.completions.create(
            model=BACKBONE, max_tokens=5, temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content":
                    f"TASK: {problem[:150]}\nOutput:\n{content[:600]}"}
            ])
        v = r.choices[0].message.content.strip().upper().split()[0]
        return v if v in ("COMPLETE", "INCOMPLETE", "CORRUPTED") else "INCOMPLETE"
    except:
        return "INCOMPLETE"

# ── Parallel condition ────────────────────────────────────────────────────────

def _step_par(oai, rl, shard, agent_id, problem, step, snap_ver, snap_content):
    delta, llm_ms = llm_call(oai, rl, problem, snap_content, agent_id, step)
    ver = snap_ver
    retry_ms = 0
    for attempt in range(6):
        t0 = time.perf_counter()
        ok = do_commit(shard, agent_id, ver, delta)
        retry_ms += int((time.perf_counter() - t0) * 1000)
        if ok:
            return True, attempt + 1, llm_ms, retry_ms
        ver, _ = read_shard(shard, agent_id)
        time.sleep(0.04 * (attempt + 1))
    return False, 6, llm_ms, retry_ms

def run_parallel(oai, rl, task, n_agents, n_steps, run_id):
    shard = f"wt_{run_id}_p"
    create_shard(shard, task["domain"], task["seed"])   # FIX-1
    agents = [f"a{i}_p_{run_id}" for i in range(n_agents)]
    for a in agents: register_agent(a)

    commits_ok = llm_calls = total_llm_ms = total_sbus_ms = 0
    t0 = time.perf_counter()
    for step in range(n_steps):
        snaps = {}
        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futs = {ex.submit(read_shard, shard, agents[i]): i
                    for i in range(n_agents)}
            for f in as_completed(futs): snaps[futs[f]] = f.result()
        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futs = {
                ex.submit(_step_par, oai, rl, shard, agents[i],
                          task["problem"], step, snaps[i][0], snaps[i][1]): i
                for i in range(n_agents)
            }
            for f in as_completed(futs):
                ok, attempts, llm_ms, sbus_ms = f.result()
                llm_calls += 1
                total_llm_ms += llm_ms
                total_sbus_ms += sbus_ms
                if ok: commits_ok += 1
    wall = time.perf_counter() - t0

    # Read final content for judge
    _, final_content = read_shard(shard, agents[0])
    verdict = judge_content(oai, rl, task["problem"], final_content)

    return {
        "run_id": run_id, "task_id": task["id"], "domain": task["domain"],
        "condition": "sbus_parallel", "n_agents": n_agents, "n_steps": n_steps,
        "wall_time_s": round(wall, 3),
        "commits_ok": commits_ok, "llm_calls": llm_calls,
        "avg_llm_ms": round(total_llm_ms / max(1, llm_calls)),
        "avg_sbus_ms": round(total_sbus_ms / max(1, llm_calls)),
        "verdict": verdict, "s50": 0.0,
    }

# ── Sequential condition ──────────────────────────────────────────────────────

def run_sequential(oai, rl, task, n_agents, n_steps, run_id):
    shard = f"wt_{run_id}_s"
    create_shard(shard, task["domain"], task["seed"])   # FIX-1
    agents = [f"a{i}_s_{run_id}" for i in range(n_agents)]
    for a in agents: register_agent(a)

    commits_ok = llm_calls = total_llm_ms = total_sbus_ms = 0
    t0 = time.perf_counter()
    for a in agents:
        for step in range(n_steps):
            ver, content = read_shard(shard, a)
            delta, llm_ms = llm_call(oai, rl, task["problem"], content, a, step)
            tc = time.perf_counter()
            ok = do_commit(shard, a, ver, delta)
            sbus_ms = int((time.perf_counter() - tc) * 1000)
            llm_calls += 1
            total_llm_ms += llm_ms
            total_sbus_ms += sbus_ms
            if ok: commits_ok += 1
    wall = time.perf_counter() - t0

    _, final_content = read_shard(shard, agents[0])
    verdict = judge_content(oai, rl, task["problem"], final_content)

    return {
        "run_id": run_id, "task_id": task["id"], "domain": task["domain"],
        "condition": "sequential", "n_agents": n_agents, "n_steps": n_steps,
        "wall_time_s": round(wall, 3),
        "commits_ok": commits_ok, "llm_calls": llm_calls,
        "avg_llm_ms": round(total_llm_ms / max(1, llm_calls)),
        "avg_sbus_ms": round(total_sbus_ms / max(1, llm_calls)),
        "verdict": verdict, "s50": 0.0,
    }

# ── Stats ────────────────────────────────────────────────────────────────────

def bootstrap_speedup(par, seq, n=2000, seed=42):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sp = sorted(rng.choices(par, k=len(par)))[len(par)//2]
        ss = sorted(rng.choices(seq, k=len(seq)))[len(seq)//2]
        if sp > 0: vals.append(ss / sp)
    vals.sort()
    return vals[n//2], vals[int(n*0.025)], vals[int(n*0.975)]

def wilcoxon_p(x, y):
    try:
        from scipy.stats import wilcoxon
        diffs = [b - a for a, b in zip(x, y) if a != b]
        if not diffs: return 1.0
        _, p = wilcoxon(diffs, alternative="greater")
        return float(p)
    except ImportError:
        pos = sum(1 for a, b in zip(x, y) if b > a)
        n = sum(1 for a, b in zip(x, y) if a != b)
        return sum(math.comb(n, k) * (0.5**n)
                   for k in range(pos, n + 1)) if n else 1.0

# ── CSV writer ────────────────────────────────────────────────────────────────

class CSVWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = None
        self._lock = threading.Lock()
    def write(self, row):
        with self._lock:
            if self._w is None:
                self._w = csv.DictWriter(self._f, fieldnames=list(row.keys()))
                self._w.writeheader()
            self._w.writerow(row)
            self._f.flush()
    def close(self): self._f.close()

_pl = threading.Lock()
def tprint(*a, **k):
    with _pl: print(*a, **k)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks",       type=int, default=10)
    ap.add_argument("--n-agents-list", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--n-steps",       type=int, default=8)
    ap.add_argument("--n-repeats",     type=int, default=3)
    ap.add_argument("--workers",       type=int, default=2)
    ap.add_argument("--rpm",           type=int, default=250)
    ap.add_argument("--output", default="results/exp_sequential_v4.csv")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)
    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}"); sys.exit(1)

    tasks = TASKS[:args.n_tasks]
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rl  = RateLimiter(rpm=args.rpm)
    out = CSVWriter(args.output)

    work = [(t, n, r) for n in args.n_agents_list
            for t in tasks for r in range(args.n_repeats)]

    print(f"Exp. SEQUENTIAL v2 — {len(work)*2} trials "
          f"(N={args.n_agents_list}, {args.n_tasks} tasks, {args.n_repeats} repeats)")
    print(f"Steps/run: {args.n_steps}  Workers: {args.workers}\n")

    by_n = defaultdict(lambda: {"par": [], "seq": []})
    lock = threading.Lock()

    def _run(unit):
        task, n, rep = unit
        rid = uuid.uuid4().hex[:8]
        rp = rs = None
        try:
            rp = run_parallel(oai, rl, task, n, args.n_steps, rid + "p")
            out.write(rp)
        except Exception as e:
            tprint(f"  ERR par {task['id']} N={n}: {e}")
        try:
            rs = run_sequential(oai, rl, task, n, args.n_steps, rid + "s")
            out.write(rs)
        except Exception as e:
            tprint(f"  ERR seq {task['id']} N={n}: {e}")
        with lock:
            if rp: by_n[n]["par"].append(rp["wall_time_s"])
            if rs: by_n[n]["seq"].append(rs["wall_time_s"])
        ps = f"{rp['wall_time_s']:.1f}s ok={rp['commits_ok']}" if rp else "ERR"
        ss = f"{rs['wall_time_s']:.1f}s ok={rs['commits_ok']}" if rs else "ERR"
        tprint(f"  N={n} {task['id']}: par={ps}  seq={ss}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run, u): u for u in work}
        try:
            for _ in as_completed(futs): pass
        except KeyboardInterrupt:
            print("\nInterrupted."); pool.shutdown(wait=False, cancel_futures=True)

    out.close()

    print("\n" + "="*60)
    print("RESULTS — wall-time speedup (parallel vs sequential)")
    print("="*60)
    summary_rows = []
    for n in args.n_agents_list:
        pw = sorted(by_n[n]["par"])
        sw = sorted(by_n[n]["seq"])
        if not pw or not sw: continue
        med_p = pw[len(pw)//2]
        med_s = sw[len(sw)//2]
        sp, lo, hi = bootstrap_speedup(pw, sw)
        p = wilcoxon_p(pw, sw)
        sig = ("***" if p < 0.001 else "**" if p < 0.01
               else "*" if p < 0.05 else "n.s.")
        print(f"  N={n:2d}: par={med_p:.1f}s  seq={med_s:.1f}s  "
              f"speedup={sp:.2f}x [{lo:.2f},{hi:.2f}]  p={p:.4f} {sig}")
        summary_rows.append({
            "n_agents": n, "par_median_s": round(med_p, 2),
            "seq_median_s": round(med_s, 2),
            "speedup": round(sp, 2),
            "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
            "p": round(p, 4), "sig": sig,
            "n_par": len(pw), "n_seq": len(sw),
        })

    sumpath = args.output.replace(".csv", "_summary.csv")
    if summary_rows:
        with open(sumpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"\nSummary  → {sumpath}")
    print(f"Full data → {args.output}")

if __name__ == "__main__":
    main()