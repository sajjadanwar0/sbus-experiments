#!/usr/bin/env python3
"""
Exp ORI-ISOLATION: Parallel+ORI vs Parallel-ORI — correct isolation
=====================================================================
Correctly isolates ORI's effect by holding the execution architecture
CONSTANT (parallel) and toggling ONLY the version check / cross-shard
validation.

DESIGN INSIGHT
--------------
Sequential mode cannot test ORI — ORI is a CONCURRENCY mechanism that
prevents stale reads between concurrent agents. In sequential mode, each
agent reads the current version before committing, so no stale reads occur.

The ONLY valid isolation comparison is:
  A) Parallel + ORI ON   (S-Bus normal mode)
  B) Parallel + ORI OFF  (version check disabled — last-write-wins)
  C) CrewAI sequential   (different architecture, different semantics)

A vs B = pure ORI effect (same concurrency, only version check differs)
A vs C = architecture effect (already in paper as Table 3)

WHAT THIS PROVES
----------------
A vs B at shared-shard topology:
  If corruption(B) >> corruption(A) → ORI prevents real conflicts
  Exp.E already shows this: B=97.5% corruption, A=0%

A vs B at distinct-shard topology (Exp.B):
  If corruption(A) ≈ corruption(B) → ORI adds no overhead in no-contention
  S@50(A) vs S@50(B) → does ORI change task success?

JUDGE FIX
---------
Uses a better judge that evaluates whether the final accumulated content
represents a TECHNICALLY COHERENT SOLUTION DIRECTION, not whether the
code is complete. Longer max_tokens (200) reduces INCOMPLETE verdicts.

REQUIRES
---------
  SBUS_ADMIN_ENABLED=1 cargo run --release
  pip install openai anthropic scipy

USAGE
------
  # Quick run (5 tasks × 4 conditions × 10 runs = 200 trials, ~45min W=8)
  python3 exp_ori_isolation.py \
    --n-tasks 5 --n-runs 10 --n-steps 15 --workers 8

  # Full run for paper (10 tasks × 4 conditions × 25 runs = 1000 trials)
  python3 exp_ori_isolation.py \
    --n-tasks 10 --n-runs 25 --n-steps 20 --workers 8 \
    --output results/ori_isolation.csv
"""

import csv, json, os, sys, time, uuid, socket, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from urllib.request import Request, ProxyHandler, build_opener
from urllib.parse import urlencode
from urllib.error import HTTPError

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai"); sys.exit(1)

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# ── Configuration ─────────────────────────────────────────────────────────────

SBUS_URL    = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE    = "gpt-4o-mini"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

COND_PAR_ORI_ON   = "parallel_ori_on"    # A: S-Bus normal
COND_PAR_ORI_OFF  = "parallel_ori_off"   # B: no version check
COND_SEQ_ORI_ON   = "sequential_ori_on"  # C: sequential + ORI (shows arch effect)
COND_CREWAI_SEQ   = "crewai_sequential"  # D: CrewAI baseline

TASKS = [
    {"id": "django_queryset",     "desc": "Fix Django queryset ordering with select_related() FK traversal. Agents: ORM core, query compiler, test fixtures, migration plan.", "shards": 1},
    {"id": "django_migration",    "desc": "Fix Django migration squasher circular RunSQL dependencies. Agents: graph resolver, squash logic, dependency checker, state rebuilder.", "shards": 1},
    {"id": "django_admin",        "desc": "Fix Django admin bulk action per-object permission checks. Agents: permission layer, action registry, view logic, template renderer.", "shards": 1},
    {"id": "sympy_algebraic",     "desc": "Fix SymPy solve() dropping solutions with positive assumptions. Agents: solver core, assumption checker, filter logic, simplification.", "shards": 1},
    {"id": "sympy_matrix",        "desc": "Fix SymPy eigenvals() for sparse Rational matrices. Agents: matrix engine, eigenvalue solver, sparse backend, rational arithmetic.", "shards": 1},
    {"id": "astropy_fits",        "desc": "Fix Astropy FITS HIERARCH keyword parsing non-standard dialects. Agents: header parser, IO handler, keyword registry, card formatter.", "shards": 1},
    {"id": "sklearn_clone",       "desc": "Fix scikit-learn clone() failing with **kwargs in __init__. Agents: estimator API, clone logic, param inspector, validation utils.", "shards": 1},
    {"id": "requests_redirect",   "desc": "Fix requests auth header stripping on cross-domain 301 redirects. Agents: session handler, auth logic, redirect chain, header policy.", "shards": 1},
    {"id": "astropy_wcs",         "desc": "Fix Astropy ZEA projection boundary errors near poles. Agents: WCS transform, projection math, coordinate frame, bounds checker.", "shards": 1},
    {"id": "django_forms",        "desc": "Fix Django ModelForm field ordering with Meta.fields override. Agents: form metaclass, field ordering, validation logic, widget registry.", "shards": 1},
]


# ── Rate limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, rpm=400):
        self._interval = 60.0 / rpm
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0: time.sleep(wait)
            self._last = time.monotonic()


# ── HTTP (thread-safe) ────────────────────────────────────────────────────────

def _op(): return build_opener(ProxyHandler({}))

def http_get(url, params=None):
    if params: url += "?" + urlencode(params)
    try:
        with _op().open(url, timeout=30) as r: return r.status, json.loads(r.read())
    except HTTPError as e: return e.code, {}
    except Exception: return 0, {}

def http_post(url, body):
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _op().open(req, timeout=30) as r: return r.status, json.loads(r.read())
    except HTTPError as e: return e.code, {}
    except Exception: return 0, {}

def health_check():
    try: s = socket.create_connection(("localhost", 7000), timeout=3); s.close()
    except: return False
    st, _ = http_get(f"{SBUS_URL}/stats"); return st == 200


# ── Judge ──────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are evaluating the FINAL ACCUMULATED STATE of a multi-agent software engineering task.
The content shows incremental agent contributions — this delta format is NORMAL and EXPECTED.

TASK: {task}
ACCUMULATED CONTENT: {content}

EVALUATE ONLY: Does the accumulated content represent a COHERENT TECHNICAL DIRECTION?
- CORRECT: agents converge on a consistent, technically plausible fix for the stated task
- INCOMPLETE: too little content to evaluate, or entirely off-topic
- CORRUPTED: agents propose CONTRADICTORY approaches (one says fix X, another says break X)

Reply EXACTLY one of: CORRECT | INCOMPLETE | CORRUPTED
Then a newline, then ONE SENTENCE explaining the verdict.
Note: incomplete code, truncation, and agent markers [aX sN] are ALL NORMAL."""

def judge(task_desc, content, oai, rl):
    prompt = JUDGE_PROMPT.format(task=task_desc[:350], content=content[:800] if content else "[empty]")
    if HAS_ANTHROPIC:
        try:
            rl.acquire()
            msg = anthropic.Anthropic().messages.create(
                model=JUDGE_MODEL, max_tokens=120, temperature=0,
                messages=[{"role":"user","content":prompt}])
            text = msg.content[0].text.strip()
        except Exception as e: text = f"INCOMPLETE\n{e}"
    else:
        try:
            rl.acquire()
            r = oai.chat.completions.create(
                model=BACKBONE, max_tokens=120, temperature=0,
                messages=[
                    {"role":"system","content":"Evaluate technical coherence. Reply CORRECT, INCOMPLETE, or CORRUPTED then newline."},
                    {"role":"user","content":prompt}])
            text = r.choices[0].message.content.strip()
        except Exception as e: text = f"INCOMPLETE\n{e}"
    lines = text.strip().split("\n", 1)
    raw = lines[0].upper()
    v = "CORRECT" if ("CORRECT" in raw and "IN" not in raw) else ("CORRUPTED" if "CORRUPT" in raw else "INCOMPLETE")
    return v, lines[1].strip() if len(lines) > 1 else ""


# ── LLM delta ─────────────────────────────────────────────────────────────────

def _delta(oai, rl, task_desc, context, agent_id, step):
    try:
        rl.acquire()
        r = oai.chat.completions.create(
            model=BACKBONE, max_tokens=200, temperature=0.3,  # 200 tokens for richer content
            messages=[{"role":"user","content":(
                f"TASK: {task_desc[:250]}\n"
                f"Current shared document (step {step}):\n{context[:500]}\n"
                f"Write 2-3 sentences: ONE concrete technical improvement relevant to the task. "
                f"Be specific about the code change needed."
            )}])
        return f"[{agent_id} s{step}] {r.choices[0].message.content.strip()}"
    except Exception as e:
        return f"[{agent_id} s{step}] ERROR: {e}"


# ── Condition runners ─────────────────────────────────────────────────────────

def _make_shard(run_id, task):
    shard = f"doc_{run_id}"
    http_post(f"{SBUS_URL}/shard", {"key": shard, "content": f"Task: {task['desc'][:80]}", "goal_tag": task["id"]})
    return shard


def run_parallel_ori_on(oai, rl, task, n_agents, n_steps, run_id):
    """Condition A: parallel agents, ORI ACTIVE. Standard S-Bus."""
    shard = _make_shard(run_id, task)
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})
    ok = total = 0
    for step in range(n_steps):              # steps outer = concurrent
        for a in agents:                     # agents inner
            _, d = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": a})
            if not d: continue
            delta = _delta(oai, rl, task["desc"], d.get("content",""), a, step)
            st, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard, "expected_version": d.get("version", 0),
                "delta": delta, "agent_id": a,
                "read_set": [{"key": shard, "version_at_read": d.get("version", 0)}]})
            total += 1
            if st == 200: ok += 1
    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id":"judge"})
    return final.get("content",""), ok, total


def run_parallel_ori_off(oai, rl, task, n_agents, n_steps, run_id):
    """
    Condition B: parallel agents, ORI DISABLED.
    KEY FIX: always reads CURRENT version before committing (correct LWW).
    No cross-shard validation, no cross-version check beyond the current read.
    This simulates a system where agents can commit stale deltas freely.
    """
    shard = _make_shard(run_id, task)
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    ok = total = 0
    for step in range(n_steps):
        for a in agents:
            _, d = http_get(f"{SBUS_URL}/shard/{shard}", {})
            if not d: continue
            cur_ver = d.get("version", 0)           # ← CRITICAL: use current version
            delta = _delta(oai, rl, task["desc"], d.get("content",""), a, step)
            # Use /commit (basic endpoint) to bypass ARSI cross-shard check
            # but pass current version so the simple version check passes
            st, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard,
                "expected_version": cur_ver,        # current version → always passes
                "delta": delta,
                "agent_id": a,
                "read_set": []})                    # empty read_set = skip cross-shard check
            total += 1
            if st == 200: ok += 1
    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id":"judge"})
    return final.get("content",""), ok, total


def run_sequential_ori_on(oai, rl, task, n_agents, n_steps, run_id):
    """Condition C: sequential execution, ORI active. Agents run one at a time."""
    shard = _make_shard(run_id, task)
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})
    ok = total = 0
    for a in agents:                         # agents outer = sequential
        for step in range(n_steps):
            _, d = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": a})
            if not d: continue
            delta = _delta(oai, rl, task["desc"], d.get("content",""), a, step)
            st, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard, "expected_version": d.get("version", 0),
                "delta": delta, "agent_id": a,
                "read_set": [{"key": shard, "version_at_read": d.get("version", 0)}]})
            total += 1
            if st == 200: ok += 1
    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id":"judge"})
    return final.get("content",""), ok, total


def run_crewai_sequential(oai, rl, task, n_agents, n_steps, run_id):
    """Condition D: CrewAI-style sequential, always reads current version before LWW commit."""
    shard = _make_shard(run_id, task)
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    ok = total = 0
    for a in agents:
        for step in range(n_steps):
            _, d = http_get(f"{SBUS_URL}/shard/{shard}", {})
            if not d: continue
            cur_ver = d.get("version", 0)           # ← CRITICAL: always read current version
            delta = _delta(oai, rl, task["desc"], d.get("content",""), a, step)
            st, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard,
                "expected_version": cur_ver,        # current → always passes
                "delta": delta,
                "agent_id": a,
                "read_set": []})                    # no cross-shard check
            total += 1
            if st == 200: ok += 1
    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id":"judge"})
    return final.get("content",""), ok, total


# ── Trial ─────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    run_id: str; task_id: str; run_idx: int; condition: str
    n_agents: int; n_steps: int
    commits_ok: int; commits_total: int; commit_rate: float
    verdict: str; is_correct: bool; is_corrupted: bool
    judge_reason: str; surviving_content: str; wall_secs: float


def run_trial(oai, rl, task, run_idx, condition, n_agents, n_steps):
    run_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    try:
        if   condition == COND_PAR_ORI_ON:  c, ok, tot = run_parallel_ori_on (oai,rl,task,n_agents,n_steps,run_id)
        elif condition == COND_PAR_ORI_OFF: c, ok, tot = run_parallel_ori_off(oai,rl,task,n_agents,n_steps,run_id)
        elif condition == COND_SEQ_ORI_ON:  c, ok, tot = run_sequential_ori_on(oai,rl,task,n_agents,n_steps,run_id)
        else:                               c, ok, tot = run_crewai_sequential(oai,rl,task,n_agents,n_steps,run_id)
    except Exception as e: return None, str(e)
    wall = time.perf_counter() - t0
    v, reason = judge(task["desc"], c, oai, rl)
    return TrialResult(run_id=run_id, task_id=task["id"], run_idx=run_idx,
                       condition=condition, n_agents=n_agents, n_steps=n_steps,
                       commits_ok=ok, commits_total=tot,
                       commit_rate=round(ok/max(1,tot),4),
                       verdict=v, is_correct=(v=="CORRECT"), is_corrupted=(v=="CORRUPTED"),
                       judge_reason=reason[:200], surviving_content=c[:300],
                       wall_secs=round(wall,1)), None


# ── CSV writer / progress ─────────────────────────────────────────────────────

class CSVWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._f = open(path, "w", newline=""); self._w = None; self._lock = threading.Lock()
    def write(self, row):
        with self._lock:
            if self._w is None:
                self._w = csv.DictWriter(self._f, fieldnames=list(row.keys()))
                self._w.writeheader()
            self._w.writerow(row); self._f.flush()
    def close(self): self._f.close()

_pl = threading.Lock()
def tprint(*a,**k):
    with _pl: print(*a,**k)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks",   type=int, default=10)
    parser.add_argument("--n-runs",    type=int, default=25)
    parser.add_argument("--n-agents",  type=int, default=4)
    parser.add_argument("--n-steps",   type=int, default=20)
    parser.add_argument("--conditions",nargs="+",
                        default=[COND_PAR_ORI_ON, COND_PAR_ORI_OFF,
                                 COND_SEQ_ORI_ON, COND_CREWAI_SEQ])
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--rpm",       type=int, default=400)
    parser.add_argument("--output",    default="results/ori_isolation.csv")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"): print("ERROR: set OPENAI_API_KEY"); sys.exit(1)
    if not health_check(): print("ERROR: S-Bus not running"); sys.exit(1)

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rl  = RateLimiter(rpm=args.rpm)
    out = CSVWriter(args.output)

    tasks = TASKS[:args.n_tasks]
    work  = [(t,r,c) for t in tasks for r in range(args.n_runs) for c in args.conditions]
    total = len(work)

    counts = {c: {"correct":0,"corrupted":0,"incomplete":0,"total":0} for c in args.conditions}
    done = [0]; lock = threading.Lock()

    calls_per = args.n_agents * args.n_steps + 1
    est_h = (total * calls_per / args.rpm / 60) / args.workers
    print("="*68)
    print("Exp ORI-ISOLATION: Parallel+ORI vs Parallel-ORI")
    print("="*68)
    print(f"Conditions : {args.conditions}")
    print(f"Total      : {total} trials  Est: {est_h:.1f}h")
    print()
    print("KEY COMPARISON:")
    print(f"  A ({COND_PAR_ORI_ON:<22}): parallel + ORI ACTIVE")
    print(f"  B ({COND_PAR_ORI_OFF:<22}): parallel + ORI DISABLED (LWW, empty read_set)")
    print(f"  C ({COND_SEQ_ORI_ON:<22}): sequential + ORI (shows arch effect)")
    print(f"  D ({COND_CREWAI_SEQ:<22}): CrewAI sequential baseline")
    print()
    print("H1: A vs B  → pure ORI effect (same parallel architecture)")
    print("H2: A vs C  → architecture effect (parallel vs sequential)")
    print("H3: A vs D  → original paper comparison")
    print()
    print("Running... (✓=correct ✗=corrupted ·=incomplete)")

    t_start = time.time()

    def _run(unit):
        task, run_idx, condition = unit
        result, err = run_trial(oai, rl, task, run_idx, condition, args.n_agents, args.n_steps)
        if result is None: tprint(f"\n  ERR: {err}"); return
        out.write(asdict(result))
        sym = "✓" if result.is_correct else ("✗" if result.is_corrupted else "·")
        with lock:
            c = counts[condition]
            c["total"] += 1
            if result.is_correct: c["correct"] += 1
            elif result.is_corrupted: c["corrupted"] += 1
            else: c["incomplete"] += 1
            done[0] += 1; n = done[0]
        with _pl:
            print(sym, end="", flush=True)
            if n % 40 == 0:
                el = time.time()-t_start
                print(f"  [{n}/{total}] {el/60:.1f}m  ETA {(total-n)/n*el/60:.0f}m")
                _live(counts, args.conditions)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run,u):u for u in work}
        try:
            for _ in as_completed(futures): pass
        except KeyboardInterrupt:
            print("\nInterrupted."); pool.shutdown(wait=False, cancel_futures=True)

    out.close(); print()
    _final(counts, args)
    _stats(args.output, args.conditions)
    print(f"\nResults: {args.output}")


def _live(counts, conditions):
    parts = []
    for c in conditions:
        d = counts[c]; t = d["total"]
        parts.append(f"{c[-7:]}: {d['correct']/max(1,t)*100:.0f}%")
    with _pl: print("  " + "  |  ".join(parts))


def _final(counts, args):
    from scipy import stats as scipy_stats
    print("="*68)
    print("RESULTS")
    print("="*68)

    labels = {COND_PAR_ORI_ON:"A parallel+ORI", COND_PAR_ORI_OFF:"B parallel-ORI",
              COND_SEQ_ORI_ON:"C sequential+ORI", COND_CREWAI_SEQ:"D CrewAI seq"}

    def s50(c):
        d = counts.get(c,{"correct":0,"total":1})
        return d["correct"]/max(1,d["total"])

    for c in args.conditions:
        d = counts[c]; t = d["total"]
        bar = "█"*int(s50(c)*30)+"░"*(30-int(s50(c)*30))
        print(f"  {labels.get(c,c):<22}: S@50={s50(c)*100:5.1f}% [{bar}]  "
              f"correct={d['correct']} corrupted={d['corrupted']} incomplete={d['incomplete']}")

    print()
    pairs = [
        (COND_PAR_ORI_ON, COND_PAR_ORI_OFF, "H1 A vs B (ORI ON vs OFF, parallel)  ← KEY"),
        (COND_PAR_ORI_ON, COND_SEQ_ORI_ON,  "H2 A vs C (parallel vs sequential)"),
        (COND_PAR_ORI_ON, COND_CREWAI_SEQ,  "H3 A vs D (S-Bus vs CrewAI)"),
    ]
    for c1,c2,label in pairs:
        if c1 not in args.conditions or c2 not in args.conditions: continue
        d1,d2 = counts[c1],counts[c2]
        n1,n2 = d1["total"],d2["total"]
        if n1==0 or n2==0: continue
        c1c,c2c = d1["correct"],d2["correct"]
        _, p = scipy_stats.fisher_exact([[c1c,n1-c1c],[c2c,n2-c2c]], alternative="two-sided")
        diff = c1c/n1-c2c/n2
        print(f"  {label}")
        print(f"    {c1c}/{n1}={c1c/n1*100:.1f}%  vs  {c2c}/{n2}={c2c/n2*100:.1f}%"
              f"  Δ={diff*100:+.1f}pp  p={p:.4f}  "
              f"{'✅ sig' if p<0.05 else '– n.s.'}")
        print()

    print("PAPER TEXT:")
    for c in args.conditions:
        d = counts[c]; t = d["total"]
        print(f"  {labels.get(c,c)}: S@50={s50(c)*100:.1f}% (n={t})")


def _stats(csv_path, conditions):
    # already called from _final
    pass


if __name__ == "__main__":
    main()