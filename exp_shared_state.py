#!/usr/bin/env python3
"""
exp_shared_state.py — Shared-State Multi-Agent Evaluation
==========================================================
Replaces SWE-bench with tasks that have GENUINE shared-state contention:
finance, healthcare, software architecture, devops, legal.

Each task has:
  - A single shared shard all N agents write to (guaranteed contention)
  - Ground-truth consistency checks (not just code completion)
  - Known conflict scenarios where stale reads produce detectable errors

This experiment runs three conditions:
  A) S-Bus ORI-ON  : parallel agents, OCC enforced
  B) S-Bus ORI-OFF : parallel agents, LWW (no retry on 409)
  C) Sequential    : agents run one after another (no conflicts possible)

Primary metrics:
  - consistency_rate: fraction of runs where judge finds all GT checks pass
  - corruption_rate:  fraction of runs with detected contradictions
  - commit_rate:      fraction of commits that succeed on first attempt

This directly answers the reviewer concern:
  "Show that ORI prevents the corruptions it claims to prevent, on tasks
   where shared-state contention is the actual bottleneck."

REQUIRES
--------
  cargo run --release  (port 7000)
  pip install openai anthropic scipy

USAGE
-----
  # Quick (5 tasks × 10 runs, ~25min)
  OPENAI_API_KEY=sk-... python3 exp_shared_state.py \\
      --n-tasks 5 --n-runs 10 --workers 4

  # Paper-quality (10 tasks × 30 runs, ~2h)
  OPENAI_API_KEY=sk-... python3 exp_shared_state.py \\
      --n-tasks 10 --n-runs 30 --workers 8 \\
      --output results/shared_state_eval.csv
"""

import csv, json, os, sys, time, uuid, argparse, threading, socket
from concurrent.futures import ThreadPoolExecutor, as_completed
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

COND_ORI_ON  = "sbus_ori_on"
COND_ORI_OFF = "sbus_ori_off"
COND_SEQ     = "sequential"

TASKS_FILE = os.path.join(os.path.dirname(__file__), "shared_state_tasks.json")

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _opener():
    return build_opener(ProxyHandler({}))

def http_get(url, params=None):
    if params:
        url += "?" + urlencode(params)
    try:
        with _opener().open(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}

def http_post(url, body):
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener().open(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}

def health_check():
    try:
        s = socket.create_connection(("localhost", 7000), timeout=3); s.close()
    except Exception:
        return False
    st, _ = http_get(f"{SBUS_URL}/stats")
    return st == 200

# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, rpm=350):
        self._interval = 60.0 / rpm
        self._last = 0.0
        self._lock = threading.Lock()
    def acquire(self):
        with self._lock:
            now = time.monotonic()
            w = self._last + self._interval - now
            if w > 0: time.sleep(w)
            self._last = time.monotonic()

# ── ContribTracker ────────────────────────────────────────────────────────────

class ContribTracker:
    """Thread-safe per-trial log of all successful commits.
    Returns the full accumulated document for the judge — not just the last
    shard value, which only contains the final agent's delta."""
    def __init__(self):
        self._entries = []
        self._lock = threading.Lock()

    def record(self, role, step, delta):
        with self._lock:
            self._entries.append((step, role, delta))

    def document(self):
        """Full accumulated document sorted by step then role."""
        with self._lock:
            parts = [d for _, _, d in sorted(self._entries, key=lambda x: x[0])]
        return "\n\n".join(parts) if parts else ""

    def __len__(self):
        with self._lock:
            return len(self._entries)

# ── LLM helpers ───────────────────────────────────────────────────────────────

def _delta(oai, rl, task, current_content, agent_role, step, shard_key=""):
    """Generate agent contribution. Includes shard_key in prompt so the
    proxy can detect this reference as an R_hidden read."""
    shard_hint = f"Shared document key: {shard_key.rsplit('_',1)[0] if '_' in shard_key else shard_key}\n" if shard_key else ""
    rl.acquire()
    try:
        r = oai.chat.completions.create(
            model=BACKBONE, max_tokens=250, temperature=0.3,
            messages=[{"role": "user", "content": (
                f"You are the {agent_role} agent working on this task:\n"
                f"TASK: {task['description'][:300]}\n"
                f"{shard_hint}"
                f"Current shared document:\n{current_content[:400]}\n\n"
                f"Step {step}: Write 2-4 sentences of specific, concrete content "
                f"for your role ({agent_role}). Be precise with numbers, "
                f"names, and technical details. Your contribution must be "
                f"internally consistent with what is already written."
            )}])
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[{agent_role} s{step} ERR:{e}]"

# ── Consistency judge ─────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are evaluating a multi-agent collaborative document for INTERNAL CONSISTENCY.

TASK: {task_desc}

DOCUMENT CONTENT:
{content}

GROUND TRUTH CHECKS (evaluate each):
{checks}

INSTRUCTIONS:
- CONSISTENT: ALL ground truth checks pass (no contradictions found)
- CONTRADICTED: ONE OR MORE checks fail (contradictory statements found)
- INCOMPLETE: Too little content to evaluate any checks

Reply with EXACTLY one word: CONSISTENT | CONTRADICTED | INCOMPLETE
Then one newline, then list which checks failed (if any), max 2 sentences."""

def judge_consistency(task, content, oai, rl):
    checks_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(task["ground_truth_checks"]))
    # 4000 chars covers ~16 agent contributions at 250 chars each
    # (4 agents × 6 steps × 250 = 6000 total; 4000 gets ~65% coverage)
    prompt = JUDGE_PROMPT.format(
        task_desc=task["description"][:400],
        content=content[:4000] if content else "[empty]",
        checks=checks_text,
    )
    if HAS_ANTHROPIC:
        try:
            rl.acquire()
            msg = _anthropic.Anthropic().messages.create(
                model=JUDGE_MODEL, max_tokens=150, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            text = msg.content[0].text.strip()
        except Exception as e:
            return "INCOMPLETE", str(e)
    else:
        try:
            rl.acquire()
            r = oai.chat.completions.create(
                model=BACKBONE, max_tokens=150, temperature=0,
                messages=[
                    {"role": "system", "content": "Evaluate consistency. Reply CONSISTENT, CONTRADICTED, or INCOMPLETE."},
                    {"role": "user", "content": prompt}
                ])
            text = r.choices[0].message.content.strip()
        except Exception as e:
            return "INCOMPLETE", str(e)

    lines = text.strip().split("\n", 1)
    raw = lines[0].upper().strip()
    if "CONSISTENT" in raw and "IN" not in raw:
        verdict = "CONSISTENT"
    elif "CONTRADICT" in raw:
        verdict = "CONTRADICTED"
    else:
        verdict = "INCOMPLETE"
    reason = lines[1].strip() if len(lines) > 1 else ""
    return verdict, reason

# ── S-Bus helpers ─────────────────────────────────────────────────────────────

def _create_shard(run_id, task):
    shard = f"{task['shards'][0]}_{run_id}"
    st, _ = http_post(f"{SBUS_URL}/shard", {
        "key":      shard,
        "content":  f"Task: {task['description'][:100]}",
        "goal_tag": task["task_id"],
    })
    if st not in (200, 201):
        # Try admin endpoint if available
        http_post(f"{SBUS_URL}/admin/shard", {
            "key":      shard,
            "content":  f"Task: {task['description'][:100]}",
            "goal_tag": task["task_id"],
        })
    return shard

def _read_shard(shard, agent_id):
    st, d = http_get(f"{SBUS_URL}/shard/{shard}", {"agent_id": agent_id})
    if st == 200:
        return d.get("version", 0), d.get("content", "")
    return 0, ""

def _register(agent_id):
    http_post(f"{SBUS_URL}/session", {"agent_id": agent_id})

# ── Agent roles from task ─────────────────────────────────────────────────────

def _agent_roles(task, n_agents):
    """Extract agent role names from task description."""
    desc = task["description"]
    # Try to find "Agent N (Role):" patterns
    roles = []
    import re
    matches = re.findall(r'Agent\s+\d+\s+\(([^)]+)\)', desc)
    if matches:
        roles = matches[:n_agents]
    # Pad with generic roles if needed
    while len(roles) < n_agents:
        roles.append(f"specialist_{len(roles)+1}")
    return roles[:n_agents]

# ── Condition runners ─────────────────────────────────────────────────────────

def _agent_step_ori_on(oai, rl, shard, agent_id, role, task, step,
                        snap_ver, snap_content, tracker, max_retries=5):
    """ORI active: generate delta ONCE from snapshot, retry commit only on 409.
    No re-generation on retry — cuts LLM calls from ~7/step to ~4/step."""
    # Generate delta once from the snapshot content
    delta = _delta(oai, rl, task, snap_content, role, step, shard_key=shard)
    version = snap_ver

    for attempt in range(max_retries):
        st, resp = http_post(f"{SBUS_URL}/commit/v2", {
            "key": shard, "expected_version": version,
            "delta": delta, "agent_id": agent_id,
        })
        if st == 200 and "new_version" in resp:
            tracker.record(role, step, f"[{role}] {delta}")
            return True, attempt + 1
        if st == 409:
            # Update version only — reuse same delta (already generated)
            version, _ = _read_shard(shard, agent_id)
            time.sleep(0.03 * (attempt + 1))
            continue
        break   # DeltaTooLarge or other non-retryable
    return False, max_retries


def run_ori_on(oai, rl, task, n_agents, n_steps, run_id):
    """ORI active: concurrent reads → concurrent commits → 409 retried.
    ContribTracker accumulates ALL successful commits for the judge."""
    shard = _create_shard(run_id, task)
    # Tell the wrapper the actual UUID-suffixed key so promotion calls work
    if hasattr(oai, 'register_runtime_shard'):
        oai.register_runtime_shard(shard)
    roles = _agent_roles(task, n_agents)
    agents = [f"{task['task_id'][:8]}_{roles[i][:4]}_{run_id}" for i in range(n_agents)]
    for a in agents: _register(a)

    tracker = ContribTracker()
    ok = tot = 0
    for step in range(n_steps):
        snaps = {}
        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futs = {ex.submit(_read_shard, shard, agents[i]): i for i in range(n_agents)}
            for f in as_completed(futs):
                i = futs[f]
                snaps[i] = f.result()

        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futs = {
                ex.submit(_agent_step_ori_on, oai, rl, shard, agents[i],
                          roles[i], task, step, snaps[i][0], snaps[i][1], tracker): i
                for i in range(n_agents)
            }
            for f in as_completed(futs):
                committed, attempts = f.result()
                tot += attempts
                if committed: ok += 1

    return tracker.document(), ok, tot


def run_ori_off(oai, rl, task, n_agents, n_steps, run_id):
    """ORI off: concurrent reads → concurrent commits at STALE version → LWW.
    ContribTracker records only winning commits (others are silently lost)."""
    shard = _create_shard(run_id, task)
    if hasattr(oai, 'register_runtime_shard'):
        oai.register_runtime_shard(shard)
    roles = _agent_roles(task, n_agents)
    agents = [f"{task['task_id'][:8]}_{roles[i][:4]}_{run_id}" for i in range(n_agents)]

    tracker = ContribTracker()
    ok = tot = 0
    for step in range(n_steps):
        snaps = {}
        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futs = {ex.submit(_read_shard, shard, agents[i]): i for i in range(n_agents)}
            for f in as_completed(futs):
                i = futs[f]
                snaps[i] = f.result()

        # Capture loop variables BY VALUE via default args — classic closure fix
        def _commit_no_retry(i, _snaps=snaps, _step=step, _shard=shard):
            delta = _delta(oai, rl, task, _snaps[i][1], roles[i], _step, shard_key=_shard)
            st, resp = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard, "expected_version": snaps[i][0],
                "delta": delta, "agent_id": agents[i],
            })
            committed = (st == 200 and "new_version" in resp)
            if committed:
                tracker.record(roles[i], step, f"[{roles[i]}] {delta}")
            return committed

        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futs = {ex.submit(_commit_no_retry, i): i for i in range(n_agents)}
            for f in as_completed(futs):
                tot += 1
                if f.result(): ok += 1

    return tracker.document(), ok, tot


def run_sequential(oai, rl, task, n_agents, n_steps, run_id):
    """Sequential: agents run one after another — no conflicts possible."""
    shard = _create_shard(run_id, task)
    roles = _agent_roles(task, n_agents)
    agents = [f"{task['task_id'][:8]}_{roles[i][:4]}_{run_id}" for i in range(n_agents)]
    for a in agents: _register(a)

    tracker = ContribTracker()
    ok = tot = 0
    for a, role in zip(agents, roles):
        for step in range(n_steps):
            ver, content = _read_shard(shard, a)
            delta = _delta(oai, rl, task, content, role, step, shard_key=shard)
            st, resp = http_post(f"{SBUS_URL}/commit/v2", {
                "key": shard, "expected_version": ver,
                "delta": delta, "agent_id": a,
            })
            tot += 1
            if st == 200 and "new_version" in resp:
                ok += 1
                tracker.record(role, step, f"[{role}] {delta}")

    return tracker.document(), ok, tot

# ── Trial ─────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    run_id: str; task_id: str; domain: str; run_idx: int; condition: str
    n_agents: int; n_steps: int
    commits_ok: int; commits_total: int; commit_rate: float
    verdict: str
    is_consistent: bool; is_contradicted: bool
    judge_reason: str; surviving_content: str; wall_secs: float


def run_trial(oai, rl, task, run_idx, condition, n_agents, n_steps):
    run_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    try:
        if   condition == COND_ORI_ON:  content, ok, tot = run_ori_on(oai, rl, task, n_agents, n_steps, run_id)
        elif condition == COND_ORI_OFF: content, ok, tot = run_ori_off(oai, rl, task, n_agents, n_steps, run_id)
        else:                           content, ok, tot = run_sequential(oai, rl, task, n_agents, n_steps, run_id)
    except Exception as e:
        return None, str(e)

    wall = time.perf_counter() - t0
    verdict, reason = judge_consistency(task, content, oai, rl)
    return TrialResult(
        run_id=run_id, task_id=task["task_id"], domain=task["domain"],
        run_idx=run_idx, condition=condition, n_agents=n_agents, n_steps=n_steps,
        commits_ok=ok, commits_total=tot,
        commit_rate=round(ok / max(1, tot), 4),
        verdict=verdict,
        is_consistent=(verdict == "CONSISTENT"),
        is_contradicted=(verdict == "CONTRADICTED"),
        judge_reason=reason[:300], surviving_content=content[:800],
        wall_secs=round(wall, 1),
    ), None

# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(counts, conditions):
    from scipy import stats as scipy_stats

    labels = {
        COND_ORI_ON:  "S-Bus ORI-ON",
        COND_ORI_OFF: "S-Bus ORI-OFF (LWW)",
        COND_SEQ:     "Sequential",
    }

    print("=" * 68)
    print("RESULTS: Shared-State Multi-Agent Evaluation")
    print("=" * 68)

    for c in conditions:
        d = counts[c]; t = d["total"]
        if t == 0: continue
        cr = d["consistent"] / t * 100
        bar = "█" * int(cr / 100 * 30) + "░" * (30 - int(cr / 100 * 30))
        print(f"  {labels.get(c,c):<22}: consistency={cr:5.1f}% [{bar}]  "
              f"contradicted={d['contradicted']}  n={t}")

    print()
    pairs = [
        (COND_ORI_ON, COND_ORI_OFF, "KEY: ORI-ON vs ORI-OFF (same parallel arch)"),
        (COND_ORI_ON, COND_SEQ,     "S-Bus vs Sequential"),
    ]
    for c1, c2, label in pairs:
        if c1 not in conditions or c2 not in conditions: continue
        d1, d2 = counts[c1], counts[c2]
        n1, n2 = d1["total"], d2["total"]
        if n1 == 0 or n2 == 0: continue
        _, p = scipy_stats.fisher_exact(
            [[d1["consistent"], n1 - d1["consistent"]],
             [d2["consistent"], n2 - d2["consistent"]]],
            alternative="two-sided")
        diff = d1["consistent"]/n1 - d2["consistent"]/n2
        print(f"  {label}")
        print(f"    {d1['consistent']}/{n1}={d1['consistent']/n1*100:.1f}%  vs  "
              f"{d2['consistent']}/{n2}={d2['consistent']/n2*100:.1f}%"
              f"  Δ={diff*100:+.1f}pp  p={p:.4f}  "
              f"{'✅ sig' if p<0.05 else '– n.s.'}")
        print()

    print("DOMAIN BREAKDOWN:")
    domain_counts = {}
    for c in conditions:
        for domain, dc in counts[c].get("by_domain", {}).items():
            if domain not in domain_counts:
                domain_counts[domain] = {}
            domain_counts[domain][c] = dc
    for domain, dc in sorted(domain_counts.items()):
        parts = []
        for c in conditions:
            if c in dc:
                t = dc[c]["total"]
                cons = dc[c]["consistent"]
                parts.append(f"{labels.get(c,c)[:10]}:{cons/max(1,t)*100:.0f}%")
        print(f"  {domain:<20}: {' | '.join(parts)}")


# ── CSV writer ────────────────────────────────────────────────────────────────

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
def tprint(*a, **k):
    with _pl: print(*a, **k)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", default=TASKS_FILE)
    ap.add_argument("--n-tasks",   type=int, default=10)
    ap.add_argument("--n-runs",    type=int, default=30)
    ap.add_argument("--n-agents",  type=int, default=4)
    ap.add_argument("--n-steps",   type=int, default=15)
    ap.add_argument("--conditions", nargs="+",
                    default=[COND_ORI_ON, COND_ORI_OFF])
    ap.add_argument("--workers",   type=int, default=4)
    ap.add_argument("--rpm",       type=int, default=350)
    ap.add_argument("--output",    default="results/shared_state_eval.csv")
    ap.add_argument("--use-wrapper", action="store_true",
                    help="Enable PhiddenWrapper to measure R_hidden→R_obs promotion")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)
    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}"); sys.exit(1)

    # Load tasks
    if os.path.exists(args.tasks_file):
        with open(args.tasks_file) as f:
            all_tasks = json.load(f)
        tasks = all_tasks[:args.n_tasks]
    else:
        print(f"ERROR: tasks file not found: {args.tasks_file}")
        sys.exit(1)

    base_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Optionally wrap client to measure phidden promotion
    wrapper = None
    if args.use_wrapper:
        try:
            from phidden_wrapper import PhiddenWrapper
            wrapper = PhiddenWrapper(base_oai, sbus_url=SBUS_URL)
            oai = wrapper
            print(f"  PhiddenWrapper: ACTIVE — scanning completions for shard refs")
            # Pre-register all base shard names from task definitions
            for task in tasks:
                for shard in task.get("shards", []):
                    wrapper.register_shards([shard])
        except ImportError:
            print("  WARNING: phidden_wrapper.py not found — running without wrapper")
            oai = base_oai
    else:
        oai = base_oai

    rl  = RateLimiter(rpm=args.rpm)
    out = CSVWriter(args.output)

    work = [(t, r, c) for t in tasks for r in range(args.n_runs) for c in args.conditions]
    total = len(work)
    calls_per = args.n_agents * args.n_steps + 1
    est_h = (total * calls_per / args.rpm / 60) / args.workers

    print("=" * 68)
    print("exp_shared_state: Multi-Agent Shared-State Evaluation")
    print("=" * 68)
    print(f"  Tasks  : {len(tasks)}  Runs: {args.n_runs}  Agents: {args.n_agents}  Steps: {args.n_steps}")
    print(f"  Total  : {total} trials  Est: {est_h:.1f}h")
    print(f"  Domains: {list({t['domain'] for t in tasks})}")
    print()
    print("  These tasks have GENUINE shared-state contention.")
    print("  ORI-ON should show significantly higher consistency than ORI-OFF.")
    print()
    print("Running... (✓=consistent ✗=contradicted ·=incomplete)")

    counts = {c: {"consistent": 0, "contradicted": 0, "incomplete": 0, "total": 0,
                  "by_domain": {}}
              for c in args.conditions}
    done = [0]; lock = threading.Lock()
    t_start = time.time()

    def _run(unit):
        task, run_idx, condition = unit
        result, err = run_trial(oai, rl, task, run_idx, condition,
                                args.n_agents, args.n_steps)
        if result is None:
            tprint(f"\n  ERR: {err}"); return
        out.write(asdict(result))
        sym = "✓" if result.is_consistent else ("✗" if result.is_contradicted else "·")
        with lock:
            c = counts[condition]
            c["total"] += 1
            dom = result.domain
            if dom not in c["by_domain"]:
                c["by_domain"][dom] = {"consistent":0,"contradicted":0,"incomplete":0,"total":0}
            c["by_domain"][dom]["total"] += 1
            if result.is_consistent:
                c["consistent"] += 1; c["by_domain"][dom]["consistent"] += 1
            elif result.is_contradicted:
                c["contradicted"] += 1; c["by_domain"][dom]["contradicted"] += 1
            else:
                c["incomplete"] += 1; c["by_domain"][dom]["incomplete"] += 1
            done[0] += 1; n = done[0]
        with _pl:
            print(sym, end="", flush=True)
            if n % 30 == 0:
                el = time.time() - t_start
                eta = (total-n)/max(n,1)*el/60
                print(f"  [{n}/{total}] {el/60:.1f}m ETA {eta:.0f}m")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, u): u for u in work}
        try:
            for _ in as_completed(futures): pass
        except KeyboardInterrupt:
            print("\nInterrupted.")
            pool.shutdown(wait=False, cancel_futures=True)

    out.close()
    print("\n")
    print_stats(counts, args.conditions)
    if wrapper:
        wrapper.print_stats()
    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    main()