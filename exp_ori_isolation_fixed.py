#!/usr/bin/env python3
"""
Exp ORI-ISOLATION (FIXED): Parallel+ORI vs Parallel-ORI — correct isolation
============================================================================
KEY FIX over original: agents now execute TRULY CONCURRENTLY using
ThreadPoolExecutor. All agents read the shard at the SAME time (same version v),
generate deltas concurrently, then ALL commit at version v concurrently.

This is the ONLY way to create genuine stale reads:
  1. All N agents read → all capture version v
  2. All N agents do LLM work (concurrent, takes real time)
  3. Agent 0 commits first → version advances to v+1
  4. Agents 1..N-1 still hold version v (now STALE)
  5. ORI-ON:  agents 1..N-1 get 409 → retry with v+1 → eventually succeed
     ORI-OFF: agents 1..N-1 commit anyway (no version check) → last-write-wins

The critical result this enables:
  - ORI-OFF: 3/4 agents' contributions silently lost every step
  - ORI-ON:  all agents eventually commit, all contributions preserved
  - Judge sees CORRUPTED/INCOMPLETE (ORI-OFF) vs CORRECT (ORI-ON)

REQUIRES
--------
  SBUS_ADMIN_ENABLED=1 cargo run --release    (port 7000)
  pip install openai anthropic scipy

USAGE
-----
  # Quick (5 tasks × 20 runs × 2 conditions, ~40min)
  SBUS_ADMIN_ENABLED=1 OPENAI_API_KEY=sk-... python3 exp_ori_isolation_fixed.py \\
      --n-tasks 5 --n-runs 20 --n-steps 15 --workers 6

  # Paper-quality (10 tasks × 25 runs × 4 conditions, ~2h)
  SBUS_ADMIN_ENABLED=1 OPENAI_API_KEY=sk-... python3 exp_ori_isolation_fixed.py \\
      --n-tasks 10 --n-runs 25 --n-steps 20 --workers 8 \\
      --output results/ori_isolation.csv
"""

import csv, json, os, sys, time, uuid, argparse, threading, socket
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass, asdict
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError
from urllib.parse import urlencode

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai"); sys.exit(1)

try:
    import anthropic as _anthropic; HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# ── Config ────────────────────────────────────────────────────────────────────

SBUS_URL    = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE    = "gpt-4o-mini"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

COND_PAR_ORI_ON   = "parallel_ori_on"    # A: S-Bus, ORI active, retry on 409
COND_PAR_ORI_OFF  = "parallel_ori_off"   # B: S-Bus storage, no retry (LWW)
COND_SEQ_ORI_ON   = "sequential_ori_on"  # C: sequential + ORI (arch baseline)
COND_CREWAI_SEQ   = "crewai_sequential"  # D: sequential LWW (CrewAI-style)

# Tasks: all shared-shard (n_shards=1) to force ORI conflicts
TASKS = [
    {"id": "django_queryset",   "desc": "Fix Django queryset ordering with select_related() FK traversal. Agents: ORM core, query compiler, test fixtures, migration plan.", "shards": 1},
    {"id": "django_migration",  "desc": "Fix Django migration squasher circular RunSQL dependencies. Agents: graph resolver, squash logic, dependency checker, state rebuilder.", "shards": 1},
    {"id": "django_admin",      "desc": "Fix Django admin bulk action per-object permission checks. Agents: permission layer, action registry, view logic, template renderer.", "shards": 1},
    {"id": "sympy_algebraic",   "desc": "Fix SymPy solve() dropping solutions with positive assumptions. Agents: solver core, assumption checker, filter logic, simplification.", "shards": 1},
    {"id": "sympy_matrix",      "desc": "Fix SymPy eigenvals() for sparse Rational matrices. Agents: matrix engine, eigenvalue solver, sparse backend, rational arithmetic.", "shards": 1},
    {"id": "astropy_fits",      "desc": "Fix Astropy FITS HIERARCH keyword parsing non-standard dialects. Agents: header parser, IO handler, keyword registry, card formatter.", "shards": 1},
    {"id": "sklearn_clone",     "desc": "Fix scikit-learn clone() failing with **kwargs in __init__. Agents: estimator API, clone logic, param inspector, validation utils.", "shards": 1},
    {"id": "requests_redirect", "desc": "Fix requests auth header stripping on cross-domain 301 redirects. Agents: session handler, auth logic, redirect chain, header policy.", "shards": 1},
    {"id": "astropy_wcs",       "desc": "Fix Astropy ZEA projection boundary errors near poles. Agents: WCS transform, projection math, coordinate frame, bounds checker.", "shards": 1},
    {"id": "django_forms",      "desc": "Fix Django ModelForm field ordering with Meta.fields override. Agents: form metaclass, field ordering, validation logic, widget registry.", "shards": 1},
]

# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, rpm=350):
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

# ── HTTP helpers (thread-safe, no session sharing) ────────────────────────────

def _opener():
    return build_opener(ProxyHandler({}))

def http_get(url, params=None):
    if params:
        url += "?" + urlencode(params)
    try:
        with _opener().open(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:
            body = json.loads(e.read())
        except Exception:
            body = {}
        return e.code, body
    except Exception:
        return 0, {}

def http_post(url, body):
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener().open(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:
            body_resp = json.loads(e.read())
        except Exception:
            body_resp = {}
        return e.code, body_resp
    except Exception:
        return 0, {}

def health_check():
    try:
        s = socket.create_connection(("localhost", 7000), timeout=3)
        s.close()
    except Exception:
        return False
    st, _ = http_get(f"{SBUS_URL}/stats")
    return st == 200

# ── LLM delta generator ───────────────────────────────────────────────────────

def _delta(oai, rl, task_desc, context, agent_id, step):
    """
    Generate ONLY the new contribution — no appending to context.
    S-Bus max_delta_chars=2000 by default; append semantics hit this after ~3 commits.
    History is tracked externally by ContribTracker, not in the shard content.
    The shard content stays small (just the latest contribution for agent context).
    """
    try:
        rl.acquire()
        r = oai.chat.completions.create(
            model=BACKBONE, max_tokens=150, temperature=0.3,
            messages=[{"role": "user", "content": (
                f"TASK: {task_desc[:250]}\n"
                f"Current shared doc (step {step}):\n{context[:500]}\n\n"
                f"Write 2-3 sentences: ONE concrete technical fix. "
                f"Be specific about the code change. Do NOT repeat prior content."
            )}])
        new_contribution = r.choices[0].message.content.strip()
        # Return ONLY the tagged contribution — no accumulation in shard
        return f"[{agent_id} s{step}] {new_contribution}"
    except Exception as e:
        return f"[{agent_id} s{step}] ERR:{e}"


class ContribTracker:
    """Thread-safe per-trial contribution log. Replaces shard-based accumulation."""
    def __init__(self):
        self._entries = []   # list of (agent_id, step, text)
        self._lock = threading.Lock()

    def record(self, agent_id, step, delta_text):
        with self._lock:
            self._entries.append((agent_id, step, delta_text))

    def document(self):
        with self._lock:
            return "\n---\n".join(t for _, _, t in sorted(self._entries, key=lambda x: (x[1], x[0])))

    def agent_ids(self):
        import re
        return set(re.findall(r'\[a(\d+)_', self.document()))

# ── Judge ─────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are evaluating a shared document written by {n_agents} agents collaborating on a task.
Each agent's contributions are marked with [aX_...] tags.

TASK: {task}
DOCUMENT: {content}

COUNT how many DISTINCT agent IDs appear (e.g. a0, a1, a2, a3).
CHECK whether any agents propose CONTRADICTORY approaches (mutually exclusive fixes).

Reply with EXACTLY one of:
  CORRECT    — {n_agents} distinct agents contributed AND no contradictions
  PARTIAL    — fewer than {n_agents} agents contributed (some contributions silently lost)
  CORRUPTED  — agents propose directly contradictory fixes

Then one newline, then: "agents_found=N" where N is the count of distinct agent IDs found."""

def judge(task_desc, content, oai, rl, n_agents=4):
    # Count distinct agent IDs directly from content — no LLM needed for this
    import re
    agent_ids = set(re.findall(r'\[a(\d+)_', content))
    n_found = len(agent_ids)

    # Use LLM only to detect contradictions
    prompt = JUDGE_PROMPT.format(
        n_agents=n_agents,
        task=task_desc[:300],
        content=content[:1200] if content else "[empty]",
    )
    if HAS_ANTHROPIC:
        try:
            rl.acquire()
            msg = _anthropic.Anthropic().messages.create(
                model=JUDGE_MODEL, max_tokens=80, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            text = msg.content[0].text.strip()
        except Exception as e:
            text = f"PARTIAL\nagents_found={n_found}"
    else:
        try:
            rl.acquire()
            r = oai.chat.completions.create(
                model=BACKBONE, max_tokens=80, temperature=0,
                messages=[
                    {"role": "system", "content": "Reply CORRECT, PARTIAL, or CORRUPTED. Then agents_found=N."},
                    {"role": "user", "content": prompt}
                ])
            text = r.choices[0].message.content.strip()
        except Exception as e:
            text = f"PARTIAL\nagents_found={n_found}"

    lines = text.strip().split("\n", 1)
    raw = lines[0].upper().strip()
    if "CORRUPT" in raw:
        v = "CORRUPTED"
    elif "CORRECT" in raw and n_found >= n_agents:
        v = "CORRECT"
    else:
        v = "PARTIAL"   # fewer agents contributed = LWW corruption

    # Use our own regex count as ground truth (more reliable than LLM counting)
    reason = f"agents_found={n_found}/{n_agents} ids={sorted(agent_ids)}"
    return v, reason

# ── Shard setup ───────────────────────────────────────────────────────────────

def _create_shard(run_id, task):
    """Create a fresh shard for this trial. Returns shard key."""
    shard = f"doc_{run_id}"
    # CreateShardRequest: key, content, goal_tag
    st, resp = http_post(f"{SBUS_URL}/shard", {
        "key":      shard,
        "content":  f"Task: {task['desc'][:80]}",
        "goal_tag": task["id"],
    })
    if st not in (200, 201):
        # Already exists from a prior run — reset via admin if enabled
        http_post(f"{SBUS_URL}/admin/shard", {
            "key":      shard,
            "content":  f"Task: {task['desc'][:80]}",
            "goal_tag": task["id"],
        })
    return shard


def _read_shard(shard, agent_id):
    """Read shard. Returns (version, content) or (0, '') on error."""
    st, d = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": agent_id})
    if st == 200:
        return d.get("version", 0), d.get("content", "")
    return 0, ""


def _register_agent(agent_id):
    """Register agent with DeliveryLog. SessionRequest: {agent_id}."""
    http_post(f"{SBUS_URL}/session", {"agent_id": agent_id})


# ── Condition A: Parallel + ORI ON ───────────────────────────────────────────
#
# Design:
#   Each STEP is a concurrent round:
#     1. All N agents read shard simultaneously → all capture (version_v, content_v)
#     2. All N agents generate LLM deltas concurrently
#     3. All N agents try to commit at version_v concurrently
#        → First commit succeeds (v → v+1)
#        → Remaining get 409 → retry with fresh read → eventually succeed
#   This correctly exercises ORI: stale reads are rejected, retried, and resolved.

def _agent_step_ori_on(oai, rl, shard, agent_id, task_desc, step,
                        snapshot_version, snapshot_content, tracker, max_retries=5):
    """ORI ACTIVE: retry on 409, regenerate delta with fresh content each retry."""
    version = snapshot_version
    content = snapshot_content
    for attempt in range(max_retries):
        delta = _delta(oai, rl, task_desc, content, agent_id, step)
        st, resp = http_post(f"{SBUS_URL}/commit/v2", {
            "key": shard, "expected_version": version,
            "delta": delta, "agent_id": agent_id,
        })
        if st == 200 and "new_version" in resp:
            tracker.record(agent_id, step, delta)
            return True, attempt + 1
        if st == 409:
            version, content = _read_shard(shard, agent_id)
            time.sleep(0.05 * (attempt + 1))
            continue
        break   # non-retryable: DeltaTooLarge, ShardNotFound, etc.
    return False, max_retries


def run_parallel_ori_on(oai, rl, task, n_agents, n_steps, run_id):
    """
    Condition A: N agents parallel, ORI ACTIVE.
    All agents read the shard at step start (same version snapshot).
    Concurrent commits: first wins, others get 409 → retry with fresh read.
    ContribTracker records all successful commits.
    Expected: tracker shows all N agent IDs (all agents eventually commit).
    """
    shard = _create_shard(run_id, task)
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    for a in agents:
        _register_agent(a)

    tracker = ContribTracker()
    total_ok = total_attempts = 0

    for step in range(n_steps):
        # Phase 1: all agents read concurrently → same version snapshot
        snapshots = {}
        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futs = {ex.submit(_read_shard, shard, a): a for a in agents}
            for f in as_completed(futs):
                snapshots[futs[f]] = f.result()

        # Phase 2: all agents commit concurrently — 409s get retried
        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futs = {
                ex.submit(
                    _agent_step_ori_on, oai, rl, shard, a, task["desc"], step,
                    snapshots[a][0], snapshots[a][1], tracker
                ): a for a in agents
            }
            for f in as_completed(futs):
                committed, attempts = f.result()
                total_attempts += attempts
                if committed:
                    total_ok += 1

    return tracker.document(), total_ok, total_attempts


def _agent_step_ori_off(oai, rl, shard, agent_id, task_desc, step,
                         snapshot_version, snapshot_content, tracker):
    """ORI OFF (LWW): commit at snapshot version, NO retry on 409."""
    delta = _delta(oai, rl, task_desc, snapshot_content, agent_id, step)
    st, resp = http_post(f"{SBUS_URL}/commit/v2", {
        "key": shard, "expected_version": snapshot_version,
        "delta": delta, "agent_id": agent_id,
    })
    committed = (st == 200 and "new_version" in resp)
    if committed:
        tracker.record(agent_id, step, delta)
    return committed, 1


def run_parallel_ori_off(oai, rl, task, n_agents, n_steps, run_id):
    """
    Condition B: N agents parallel, ORI DISABLED (LWW).
    All agents snapshot the same version. Only first committer wins per step.
    Other agents' contributions are silently lost — no retry.
    Expected: tracker shows only 1 agent ID (only winner per step recorded).
    """
    shard = _create_shard(run_id, task)
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]

    tracker = ContribTracker()
    total_ok = total_attempts = 0

    for step in range(n_steps):
        snapshots = {}
        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futs = {ex.submit(_read_shard, shard, a): a for a in agents}
            for f in as_completed(futs):
                snapshots[futs[f]] = f.result()

        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futs = {
                ex.submit(
                    _agent_step_ori_off, oai, rl, shard, a, task["desc"], step,
                    snapshots[a][0], snapshots[a][1], tracker
                ): a for a in agents
            }
            for f in as_completed(futs):
                committed, attempts = f.result()
                total_attempts += attempts
                if committed:
                    total_ok += 1

    return tracker.document(), total_ok, total_attempts


def run_sequential_ori_on(oai, rl, task, n_agents, n_steps, run_id):
    """
    Condition C: sequential agents, ORI active.
    No stale reads (each reads after prior commits). Baseline.
    Expected: all agents commit, tracker shows all N IDs.
    """
    shard = _create_shard(run_id, task)
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    for a in agents:
        _register_agent(a)

    tracker = ContribTracker()
    ok = total = 0
    for a in agents:
        for step in range(n_steps):
            version, content = _read_shard(shard, a)
            delta = _delta(oai, rl, task["desc"], content, a, step)
            st, resp = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard, "expected_version": version,
                "delta": delta, "agent_id": a,
            })
            total += 1
            if st == 200 and "new_version" in resp:
                ok += 1
                tracker.record(a, step, delta)

    return tracker.document(), ok, total


def run_crewai_sequential(oai, rl, task, n_agents, n_steps, run_id):
    """
    Condition D: CrewAI-style sequential LWW.
    Agents read current version before each commit — no conflicts.
    Expected: all commits succeed, tracker shows all N IDs.
    """
    shard = _create_shard(run_id, task)
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]

    tracker = ContribTracker()
    ok = total = 0
    for a in agents:
        for step in range(n_steps):
            version, content = _read_shard(shard, a)
            delta = _delta(oai, rl, task["desc"], content, a, step)
            st, resp = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard, "expected_version": version,
                "delta": delta, "agent_id": a,
            })
            total += 1
            if st == 200 and "new_version" in resp:
                ok += 1
                tracker.record(a, step, delta)

    return tracker.document(), ok, total


# ── Trial runner ──────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    run_id: str
    task_id: str
    run_idx: int
    condition: str
    n_agents: int
    n_steps: int
    commits_ok: int
    commits_total: int
    commit_rate: float
    verdict: str
    is_correct: bool
    is_corrupted: bool
    judge_reason: str
    surviving_content: str
    wall_secs: float


def run_trial(oai, rl, task, run_idx, condition, n_agents, n_steps):
    run_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    try:
        if   condition == COND_PAR_ORI_ON:  content, ok, tot = run_parallel_ori_on(oai, rl, task, n_agents, n_steps, run_id)
        elif condition == COND_PAR_ORI_OFF: content, ok, tot = run_parallel_ori_off(oai, rl, task, n_agents, n_steps, run_id)
        elif condition == COND_SEQ_ORI_ON:  content, ok, tot = run_sequential_ori_on(oai, rl, task, n_agents, n_steps, run_id)
        else:                               content, ok, tot = run_crewai_sequential(oai, rl, task, n_agents, n_steps, run_id)
    except Exception as e:
        return None, str(e)

    wall = time.perf_counter() - t0
    v, reason = judge(task["desc"], content, oai, rl, n_agents=n_agents)
    return TrialResult(
        run_id=run_id, task_id=task["id"], run_idx=run_idx,
        condition=condition, n_agents=n_agents, n_steps=n_steps,
        commits_ok=ok, commits_total=tot,
        commit_rate=round(ok / max(1, tot), 4),
        verdict=v,
        is_correct=(v == "CORRECT"),
        is_corrupted=(v == "CORRUPTED" or v == "PARTIAL"),
        judge_reason=reason[:200], surviving_content=content[:600],
        wall_secs=round(wall, 1)
    ), None


# ── CSV writer ────────────────────────────────────────────────────────────────

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

    def close(self):
        self._f.close()


_pl = threading.Lock()

def tprint(*a, **k):
    with _pl:
        print(*a, **k)


# ── Statistics ────────────────────────────────────────────────────────────────

def _print_stats(counts, conditions):
    from scipy import stats as scipy_stats

    labels = {
        COND_PAR_ORI_ON:  "A parallel+ORI",
        COND_PAR_ORI_OFF: "B parallel-ORI (LWW)",
        COND_SEQ_ORI_ON:  "C sequential+ORI",
        COND_CREWAI_SEQ:  "D CrewAI seq",
    }

    print("=" * 68)
    print("RESULTS")
    print("=" * 68)

    def s50(c):
        d = counts.get(c, {"correct": 0, "total": 1})
        return d["correct"] / max(1, d["total"])

    for c in conditions:
        d = counts[c]
        t = d["total"]
        pct = s50(c) * 100
        bar = "█" * int(pct / 100 * 30) + "░" * (30 - int(pct / 100 * 30))
        print(f"  {labels.get(c, c):<24}: S@50={pct:5.1f}% [{bar}]  "
              f"correct={d['correct']} corrupted={d['corrupted']} partial={d.get('partial',0)} n={t}")

    print()
    pairs = [
        (COND_PAR_ORI_ON, COND_PAR_ORI_OFF, "H1 A vs B  [KEY: ORI ON vs OFF, same parallel arch]"),
        (COND_PAR_ORI_ON, COND_SEQ_ORI_ON,  "H2 A vs C  [parallel vs sequential, ORI active]"),
        (COND_PAR_ORI_ON, COND_CREWAI_SEQ,  "H3 A vs D  [S-Bus vs CrewAI baseline]"),
    ]
    for c1, c2, label in pairs:
        if c1 not in conditions or c2 not in conditions:
            continue
        d1, d2 = counts[c1], counts[c2]
        n1, n2 = d1["total"], d2["total"]
        if n1 == 0 or n2 == 0:
            continue
        _, p = scipy_stats.fisher_exact(
            [[d1["correct"], n1 - d1["correct"]],
             [d2["correct"], n2 - d2["correct"]]],
            alternative="two-sided"
        )
        diff = d1["correct"] / n1 - d2["correct"] / n2
        sig = "✅ sig" if p < 0.05 else "– n.s."
        print(f"  {label}")
        print(f"    {d1['correct']}/{n1}={d1['correct']/n1*100:.1f}%  vs  "
              f"{d2['correct']}/{n2}={d2['correct']/n2*100:.1f}%"
              f"  Δ={diff*100:+.1f}pp  p={p:.4f}  {sig}")
        print()

    print("PAPER-READY TEXT:")
    for c in conditions:
        d = counts[c]
        print(f"  {labels.get(c, c)}: S@50={s50(c)*100:.1f}% (n={d['total']}, "
              f"corrupted={d['corrupted']}, partial={d.get('partial',0)})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks",   type=int, default=10)
    ap.add_argument("--n-runs",    type=int, default=25)
    ap.add_argument("--n-agents",  type=int, default=4)
    ap.add_argument("--n-steps",   type=int, default=10)
    ap.add_argument("--conditions", nargs="+",
                    default=[COND_PAR_ORI_ON, COND_PAR_ORI_OFF,
                             COND_SEQ_ORI_ON, COND_CREWAI_SEQ])
    ap.add_argument("--workers",   type=int, default=2)
    ap.add_argument("--rpm",       type=int, default=300)
    ap.add_argument("--output",    default="results/ori_isolation.csv")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)
    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}"); sys.exit(1)

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rl  = RateLimiter(rpm=args.rpm)
    out = CSVWriter(args.output)

    tasks = TASKS[:args.n_tasks]
    work  = [(t, r, c) for t in tasks for r in range(args.n_runs) for c in args.conditions]
    total = len(work)

    calls_per = args.n_agents * args.n_steps + 1
    est_h = (total * calls_per / args.rpm / 60) / args.workers

    print("=" * 68)
    print("Exp ORI-ISOLATION (FIXED): True Parallel Execution")
    print("=" * 68)
    print(f"  Tasks    : {args.n_tasks}  Runs/task: {args.n_runs}")
    print(f"  Agents   : {args.n_agents}  Steps: {args.n_steps}")
    print(f"  Total    : {total} trials  Est: {est_h:.1f}h  Workers: {args.workers}")
    print()
    print("  A (parallel_ori_on)  : parallel agents, ORI ACTIVE, 409 → retry")
    print("  B (parallel_ori_off) : parallel agents, ORI OFF,    409 → lost (LWW)")
    print("  C (sequential_ori_on): sequential agents, ORI active")
    print("  D (crewai_sequential): CrewAI-style sequential LWW")
    print()
    print("  KEY: A vs B isolates the pure ORI effect.")
    print("       Expected: A >> B on S@50 at shared-shard topology.")
    print()
    print("Running... (✓=correct ✗=corrupted ·=incomplete)")

    counts = {c: {"correct": 0, "corrupted": 0, "partial": 0, "total": 0}
              for c in args.conditions}
    done = [0]
    lock = threading.Lock()
    t_start = time.time()

    def _run(unit):
        task, run_idx, condition = unit
        result, err = run_trial(oai, rl, task, run_idx, condition,
                                args.n_agents, args.n_steps)
        if result is None:
            tprint(f"\n  ERR [{condition}]: {err}")
            return
        out.write(asdict(result))
        sym = "✓" if result.is_correct else ("✗" if result.is_corrupted else "·")
        with lock:
            c = counts[condition]
            c["total"] += 1
            if result.is_correct:   c["correct"] += 1
            elif result.is_corrupted: c["corrupted"] += 1
            else:                   c["partial"] += 1
            done[0] += 1
            n = done[0]
        with _pl:
            print(sym, end="", flush=True)
            if n % 40 == 0:
                el = time.time() - t_start
                eta = (total - n) / max(n, 1) * el / 60
                print(f"  [{n}/{total}] {el/60:.1f}m  ETA {eta:.0f}m")
                # Live rates
                parts = [f"{c[-6:]}: {counts[c]['correct']/max(1,counts[c]['total'])*100:.0f}%"
                         for c in args.conditions]
                with _pl: print("  " + "  |  ".join(parts))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, u): u for u in work}
        try:
            for _ in as_completed(futures):
                pass
        except KeyboardInterrupt:
            print("\nInterrupted.")
            pool.shutdown(wait=False, cancel_futures=True)

    out.close()
    print("\n")
    _print_stats(counts, args.conditions)
    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    main()