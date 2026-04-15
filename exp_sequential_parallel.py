#!/usr/bin/env python3
"""
Exp SEQUENTIAL (PARALLEL): Sequential S-Bus vs CrewAI Isolation — Parallelised
================================================================================
Same experiment as exp_sequential_isolation.py but runs all (task × run × condition)
trials concurrently via ThreadPoolExecutor.

WHY THIS IS SAFE TO PARALLELISE
---------------------------------
Each trial uses a unique run_id → unique shard keys → no S-Bus collisions.
Each trial is completely stateless between runs. OpenAI client is thread-safe.
CSV writes protected by a lock.

WORKERS GUIDE
-------------
  --workers 4   tier-1 safe (~3h for full 400-trial run)
  --workers 8   tier-2 safe (~1.5h)
  --workers 12  aggressive tier-2 (~1h)

USAGE
------
  # Quick smoke test — 3 tasks × 4 conditions × 5 runs = 60 trials (~15 min, W=8)
  python3 exp_sequential_parallel.py \
    --n-tasks 3 --n-runs 5 --n-steps 10 --workers 8

  # Full paper run — 10 tasks × 4 conditions × 25 runs = 1000 trials
  python3 exp_sequential_parallel.py \
    --n-tasks 10 --n-runs 25 --n-steps 20 --workers 8 \
    --output results/sequential_parallel.csv

  # Minimal isolation test — just conditions A and D (S-Bus seq vs CrewAI seq)
  python3 exp_sequential_parallel.py \
    --conditions sbus_sequential_ori_on crewai_sequential \
    --n-tasks 10 --n-runs 25 --workers 8

REQUIRES
---------
  SBUS_ADMIN_ENABLED=1 cargo run --release
  pip install openai anthropic scipy
"""

import csv
import json
import os
import sys
import time
import uuid
import socket
import argparse
import threading
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

COND_A = "sbus_sequential_ori_on"    # Sequential + ORI active
COND_B = "sbus_sequential_ori_off"   # Sequential + ORI disabled (pure arch)
COND_C = "sbus_parallel_ori_on"      # Parallel + ORI (original Exp.B)
COND_D = "crewai_sequential"         # Sequential + no ORI (CrewAI baseline)

ALL_CONDITIONS = [COND_A, COND_B, COND_C, COND_D]

# Tasks — prefer Django (most deterministic for code-engineering judge)
TASKS = [
    {"id": "django_queryset",     "desc": "Fix Django queryset ordering with select_related() and FK traversal. Identify the root cause and propose a concrete patch to the ORM layer.", "domain": "django"},
    {"id": "django_migration",    "desc": "Fix Django migration squasher circular dependency on RunSQL operations. Propose a fix to the squash command.", "domain": "django"},
    {"id": "django_admin",        "desc": "Fix Django admin per-object permission checking for custom bulk actions. Patch the ModelAdmin action dispatch.", "domain": "django"},
    {"id": "sympy_algebraic",     "desc": "Fix SymPy solve() dropping solutions when positive assumptions are set. Patch the solution filter.", "domain": "sympy"},
    {"id": "sympy_matrix",        "desc": "Fix SymPy eigenvals() for sparse matrices with Rational coefficients. Patch the eigenvalue solver.", "domain": "sympy"},
    {"id": "astropy_fits_header", "desc": "Fix Astropy FITS HIERARCH keyword parsing for non-standard dialects. Patch the header parser.", "domain": "astropy"},
    {"id": "sklearn_clone",       "desc": "Fix scikit-learn estimator clone() failing with **kwargs in __init__. Patch clone logic.", "domain": "sklearn"},
    {"id": "requests_redirect",   "desc": "Fix requests auth header stripping on cross-domain 301 redirects. Patch session redirect handling.", "domain": "requests"},
    {"id": "astropy_wcs",         "desc": "Fix Astropy ZEA projection boundary errors near poles. Patch the WCS transform.", "domain": "astropy"},
    {"id": "django_forms",        "desc": "Fix Django ModelForm field ordering when Meta.fields overrides ordering. Patch form metaclass.", "domain": "django"},
]


# ── Rate limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, rpm: int = 400):
        self._interval = 60.0 / rpm
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _opener():
    return build_opener(ProxyHandler({}))

def http_get(url, params=None):
    if params: url += "?" + urlencode(params)
    try:
        with _opener().open(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e: return e.code, {}
    except Exception: return 0, {}

def http_post(url, body):
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener().open(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e: return e.code, {}
    except Exception: return 0, {}

def health_check():
    try:
        s = socket.create_connection(("localhost", 7000), timeout=3); s.close()
    except Exception: return False
    st, _ = http_get(f"{SBUS_URL}/stats")
    return st == 200


# ── Judge ──────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are evaluating a multi-agent software engineering process.
The shared document contains INCREMENTAL AGENT CONTRIBUTIONS — this format is normal.
Each line "[agent_X sN] ..." is one agent step. Evaluate the OVERALL DIRECTION only.

TASK: {task}
SHARED DOCUMENT: {content}

Does the accumulated content show agents working TOWARD a correct solution?
Consider: (1) is the problem correctly identified? (2) are proposals consistent
with each other? (3) is the overall direction technically sound?

Reply EXACTLY one of: CORRECT | INCOMPLETE | CORRUPTED
Then a newline, then one sentence on whether agents converge toward a solution."""

def judge(task_desc, content, oai, rl):
    prompt = JUDGE_PROMPT.format(task=task_desc[:350],
                                 content=content[:700] if content else "[empty]")
    if HAS_ANTHROPIC:
        try:
            rl.acquire()
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=JUDGE_MODEL, max_tokens=100, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            text = msg.content[0].text.strip()
        except Exception as e:
            text = f"INCOMPLETE\n{e}"
    else:
        try:
            rl.acquire()
            r = oai.chat.completions.create(
                model=BACKBONE, max_tokens=100, temperature=0,
                messages=[
                    {"role": "system", "content": "Evaluate agent convergence. Reply CORRECT, INCOMPLETE, or CORRUPTED then newline."},
                    {"role": "user", "content": prompt}])
            text = r.choices[0].message.content.strip()
        except Exception as e:
            text = f"INCOMPLETE\n{e}"
    lines = text.strip().split("\n", 1)
    raw = lines[0].upper()
    v = "CORRECT" if ("CORRECT" in raw and "IN" not in raw) else ("CORRUPTED" if "CORRUPT" in raw else "INCOMPLETE")
    return v, lines[1].strip() if len(lines) > 1 else ""


# ── Core trial runners ─────────────────────────────────────────────────────────

def _llm_delta(oai, rl, task_desc, context, agent_id, step):
    """Generate one incremental delta."""
    try:
        rl.acquire()
        r = oai.chat.completions.create(
            model=BACKBONE, max_tokens=150, temperature=0.3,
            messages=[{"role": "user", "content": (
                f"TASK: {task_desc[:250]}\n"
                f"Current shared doc (step {step}):\n{context[:400]}\n"
                f"Write ONE concrete technical improvement. Output ONLY the change, 1-2 sentences."
            )}])
        return f"[{agent_id} s{step}] {r.choices[0].message.content.strip()}"
    except Exception as e:
        return f"[{agent_id} s{step}] ERROR: {e}"


def run_sbus_sequential_ori_on(oai, rl, task, n_agents, n_steps, run_id):
    """Condition A: sequential execution, ORI active. Agents run one at a time."""
    shard = f"doc_{run_id}"
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    http_post(f"{SBUS_URL}/shard", {"key": shard, "content": f"Task: {task['desc'][:80]}", "goal_tag": task["id"]})
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})
    ok = total = 0
    for a in agents:
        for step in range(n_steps):
            _, d = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": a})
            if not d: continue
            delta = _llm_delta(oai, rl, task["desc"], d.get("content", ""), a, step)
            st, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard, "expected_version": d.get("version", 0),
                "delta": delta, "agent_id": a,
                "read_set": [{"key": shard, "version_at_read": d.get("version", 0)}]})
            total += 1
            if st == 200: ok += 1
    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", ""), ok, total


def run_sbus_sequential_ori_off(oai, rl, task, n_agents, n_steps, run_id):
    """Condition B: sequential + no ORI. Uses /commit without version check."""
    shard = f"doc_{run_id}"
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    http_post(f"{SBUS_URL}/shard", {"key": shard, "content": f"Task: {task['desc'][:80]}", "goal_tag": task["id"]})
    ok = total = 0
    for a in agents:
        for step in range(n_steps):
            _, d = http_get(f"{SBUS_URL}/shard/{shard}", {})
            if not d: continue
            delta = _llm_delta(oai, rl, task["desc"], d.get("content", ""), a, step)
            # /commit = last-write-wins, no version check
            st, _ = http_post(f"{SBUS_URL}/commit", {
                "key": shard, "expected_version": 0, "delta": delta, "agent_id": a})
            total += 1
            if st == 200: ok += 1
    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", ""), ok, total


def run_sbus_parallel_ori_on(oai, rl, task, n_agents, n_steps, run_id):
    """Condition C: parallel + ORI. All agents advance step-by-step (original Exp.B)."""
    shard = f"doc_{run_id}"
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    http_post(f"{SBUS_URL}/shard", {"key": shard, "content": f"Task: {task['desc'][:80]}", "goal_tag": task["id"]})
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})
    ok = total = 0
    for step in range(n_steps):               # steps outer = parallel-ish
        for a in agents:                      # agents inner
            _, d = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": a})
            if not d: continue
            delta = _llm_delta(oai, rl, task["desc"], d.get("content", ""), a, step)
            st, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard, "expected_version": d.get("version", 0),
                "delta": delta, "agent_id": a,
                "read_set": [{"key": shard, "version_at_read": d.get("version", 0)}]})
            total += 1
            if st == 200: ok += 1
    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", ""), ok, total


def run_crewai_sequential(oai, rl, task, n_agents, n_steps, run_id):
    """Condition D: CrewAI-style sequential, no ORI. Agents run in turn, last-write-wins."""
    shard = f"doc_{run_id}"
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    http_post(f"{SBUS_URL}/shard", {"key": shard, "content": f"Task: {task['desc'][:80]}", "goal_tag": task["id"]})
    ok = total = 0
    for a in agents:
        for step in range(n_steps):
            _, d = http_get(f"{SBUS_URL}/shard/{shard}", {})
            if not d: continue
            delta = _llm_delta(oai, rl, task["desc"], d.get("content", ""), a, step)
            st, _ = http_post(f"{SBUS_URL}/commit", {
                "key": shard, "expected_version": 0, "delta": delta, "agent_id": a})
            total += 1
            if st == 200: ok += 1
    _, final = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": "judge"})
    return final.get("content", ""), ok, total


# ── Trial runner ───────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    run_id:            str
    task_id:           str
    task_domain:       str
    run_idx:           int
    condition:         str
    n_agents:          int
    n_steps:           int
    commits_ok:        int
    commits_total:     int
    commit_rate:       float
    verdict:           str
    is_correct:        bool
    is_corrupted:      bool
    judge_reason:      str
    surviving_content: str
    wall_secs:         float


def run_trial(oai, rl, task, run_idx, condition, n_agents, n_steps):
    run_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    try:
        if condition == COND_A:
            content, ok, total = run_sbus_sequential_ori_on(oai, rl, task, n_agents, n_steps, run_id)
        elif condition == COND_B:
            content, ok, total = run_sbus_sequential_ori_off(oai, rl, task, n_agents, n_steps, run_id)
        elif condition == COND_C:
            content, ok, total = run_sbus_parallel_ori_on(oai, rl, task, n_agents, n_steps, run_id)
        else:
            content, ok, total = run_crewai_sequential(oai, rl, task, n_agents, n_steps, run_id)
    except Exception as e:
        return None, str(e)

    wall = time.perf_counter() - t0
    v, reason = judge(task["desc"], content, oai, rl)
    return TrialResult(
        run_id=run_id, task_id=task["id"], task_domain=task.get("domain",""),
        run_idx=run_idx, condition=condition,
        n_agents=n_agents, n_steps=n_steps,
        commits_ok=ok, commits_total=total,
        commit_rate=round(ok/max(1,total), 4),
        verdict=v, is_correct=(v=="CORRECT"), is_corrupted=(v=="CORRUPTED"),
        judge_reason=reason[:200], surviving_content=content[:300],
        wall_secs=round(wall, 1)), None


# ── Thread-safe CSV writer ────────────────────────────────────────────────────

class CSVWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
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


_print_lock = threading.Lock()
def tprint(*a, **kw):
    with _print_lock: print(*a, **kw)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp SEQUENTIAL (PARALLEL): Isolate ORI from parallelism")
    parser.add_argument("--n-tasks",    type=int, default=10)
    parser.add_argument("--n-runs",     type=int, default=25)
    parser.add_argument("--n-agents",   type=int, default=4)
    parser.add_argument("--n-steps",    type=int, default=20)
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS)
    parser.add_argument("--workers",    type=int, default=4,
                        help="4=tier1 safe, 8=tier2 safe, 12=aggressive")
    parser.add_argument("--rpm",        type=int, default=400,
                        help="400=tier1, 4000=tier2")
    parser.add_argument("--output",     default="results/sequential_parallel.csv")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)

    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}")
        print("  Start: SBUS_ADMIN_ENABLED=1 cargo run --release"); sys.exit(1)

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rl  = RateLimiter(rpm=args.rpm)
    out = CSVWriter(args.output)

    tasks = TASKS[:args.n_tasks]
    work = [(t, r, c)
            for t in tasks
            for r in range(args.n_runs)
            for c in args.conditions]
    total = len(work)

    # Per-condition counters
    counts = {c: {"correct":0,"corrupted":0,"incomplete":0,"total":0}
              for c in args.conditions}
    done = [0]
    lock = threading.Lock()

    # Estimate time
    calls_per = args.n_agents * args.n_steps + 1
    est_h = (total * calls_per / args.rpm / 60) / args.workers
    print("="*65)
    print("Exp SEQUENTIAL (PARALLEL): Isolate ORI from architecture")
    print("="*65)
    print(f"Tasks      : {[t['id'] for t in tasks]}")
    print(f"Conditions : {args.conditions}")
    print(f"Runs/cond  : {args.n_runs}  Total: {total}")
    print(f"Workers    : {args.workers}  RPM: {args.rpm}")
    print(f"Est. time  : {est_h:.1f}h  (serial: {est_h*args.workers:.1f}h)")
    print()
    print("HYPOTHESES:")
    print(f"  H1: A≈D  → parallelism explains S@50 gap, not ORI")
    print(f"  H2: A>B  → ORI itself improves quality in sequential mode")
    print(f"  H3: A>C  → sequential execution helps quality vs parallel")
    print()
    print("Running... (✓=correct  ✗=corrupted  ·=incomplete)")
    print()
    t_start = time.time()

    def _run(unit):
        task, run_idx, condition = unit
        result, err = run_trial(oai, rl, task, run_idx, condition,
                                args.n_agents, args.n_steps)
        if result is None:
            tprint(f"\n  ERR [{condition}] {err}")
            return

        out.write(asdict(result))
        sym = "✓" if result.is_correct else ("✗" if result.is_corrupted else "·")

        with lock:
            c = counts[condition]
            c["total"] += 1
            if result.is_correct: c["correct"] += 1
            elif result.is_corrupted: c["corrupted"] += 1
            else: c["incomplete"] += 1
            done[0] += 1
            n = done[0]

        with _print_lock:
            print(sym, end="", flush=True)
            if n % 40 == 0:
                el = time.time() - t_start
                print(f"  [{n}/{total}] {el/60:.1f}m  ETA {(total-n)/n*el/60:.0f}m")
                _live(counts, args.conditions)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, u): u for u in work}
        try:
            for _ in as_completed(futures): pass
        except KeyboardInterrupt:
            print("\nInterrupted.")
            pool.shutdown(wait=False, cancel_futures=True)

    out.close()
    print()
    _final(counts, args)
    _stats(args.output, args.conditions)
    print(f"\nResults: {args.output}")


def _live(counts, conditions):
    parts = []
    for c in conditions:
        d = counts[c]; t = d["total"]
        parts.append(f"{c[-6:]}: {d['correct']/max(1,t)*100:.0f}%")
    with _print_lock: print("  " + "  ".join(parts))


def _final(counts, args):
    print("="*65)
    print("RESULTS")
    print("="*65)

    def s50(c):
        d = counts.get(c, {"correct":0,"total":1})
        return d["correct"] / max(1, d["total"])

    cond_labels = {
        COND_A: "A  S-Bus sequential + ORI ON",
        COND_B: "B  S-Bus sequential + ORI OFF",
        COND_C: "C  S-Bus parallel   + ORI ON",
        COND_D: "D  CrewAI sequential (no ORI)",
    }

    for c in args.conditions:
        d = counts[c]; t = d["total"]
        label = cond_labels.get(c, c)
        bar = "█"*int(s50(c)*30) + "░"*(30-int(s50(c)*30))
        print(f"  {label:<38}: S@50={s50(c)*100:5.1f}% [{bar}]  n={t}")

    if all(c in counts for c in [COND_A, COND_D]):
        a, d = s50(COND_A), s50(COND_D)
        print()
        print(f"  H1 A vs D (sequential ORI vs no-ORI): {(a-d)*100:+.1f}pp "
              f"→ {'ORI helps in sequential' if a-d > 0.05 else 'comparable → parallelism was the gap'}")
    if all(c in counts for c in [COND_A, COND_B]):
        a, b = s50(COND_A), s50(COND_B)
        print(f"  H2 A vs B (ORI ON vs OFF sequential): {(a-b)*100:+.1f}pp "
              f"→ {'ORI adds value' if a-b > 0.05 else 'ORI neutral in sequential mode'}")
    if all(c in counts for c in [COND_A, COND_C]):
        a, c_ = s50(COND_A), s50(COND_C)
        print(f"  H3 A vs C (sequential vs parallel):   {(a-c_)*100:+.1f}pp "
              f"→ {'sequential better' if a-c_ > 0.05 else 'parallel competitive'}")

    print()
    print("PAPER TEXT (§7 Exp.SEQUENTIAL):")
    for c in args.conditions:
        d = counts[c]; t = d["total"]
        label = cond_labels.get(c, c)
        print(f"  {label}: S@50={s50(c)*100:.1f}% (n={t})")


def _stats(csv_path, conditions):
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("\nNOTE: pip install scipy for p-values"); return

    rows = []
    try:
        with open(csv_path) as f: rows = list(csv.DictReader(f))
    except Exception: return

    def gc(cond):
        sub = [r for r in rows if r["condition"]==cond]
        c = sum(1 for r in sub if r["verdict"]=="CORRECT")
        return c, len(sub)-c, len(sub)

    print()
    print("="*65)
    print("FISHER'S EXACT TESTS")
    print("="*65)
    pairs = [
        (COND_A, COND_D, "H1: A vs D (sequential ORI vs CrewAI sequential)"),
        (COND_A, COND_B, "H2: A vs B (ORI ON vs OFF, sequential mode)"),
        (COND_A, COND_C, "H3: A vs C (sequential vs parallel, ORI fixed)"),
        (COND_B, COND_D, "H4: B vs D (sequential no-ORI: S-Bus vs CrewAI)"),
    ]
    for c1, c2, label in pairs:
        if c1 not in conditions or c2 not in conditions: continue
        cc1, nc1, n1 = gc(c1); cc2, nc2, n2 = gc(c2)
        if n1==0 or n2==0: continue
        _, p = scipy_stats.fisher_exact([[cc1,nc1],[cc2,nc2]], alternative="two-sided")
        d = cc1/max(1,n1) - cc2/max(1,n2)
        print(f"\n  {label}")
        print(f"    {c1[-6:]}: {cc1}/{n1}={cc1/n1*100:.1f}%  "
              f"{c2[-6:]}: {cc2}/{n2}={cc2/n2*100:.1f}%")
        print(f"    Δ={d*100:+.1f}pp  p={p:.4f}  "
              f"{'✅ sig' if p<0.05 else '– n.s.'}")


if __name__ == "__main__":
    main()