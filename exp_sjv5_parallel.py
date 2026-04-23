#!/usr/bin/env python3
"""
Exp SJ-v5 (PARALLEL): Stale Agent Fraction Dose-Response — Parallelised
=========================================================================
Runs all 1,000 trial units (10 tasks × 25 runs × 4 stale conditions)
concurrently using a ThreadPoolExecutor.

WHY THIS IS SAFE TO PARALLELISE
---------------------------------
Each trial (task × run_idx × n_stale) is completely independent:
  - Creates unique shard keys via uuid run_id → no S-Bus key collisions
  - Creates unique agent_ids with run_id suffix → no session collisions
  - OpenAI client is stateless/thread-safe → no API client collisions
  - CSV writes use a lock → no file corruption
  - admin/reset is NOT called per-run (would break concurrent runs)
    Instead each run creates its own isolated shards

WORKERS GUIDE
-------------
  --workers 4  : safe for OpenAI tier 1 (500 RPM), ~6 hours for full run
  --workers 8  : safe for OpenAI tier 2 (5,000 RPM), ~3 hours
  --workers 12 : aggressive, tier 2 only, ~2 hours
  --workers 16 : max practical (S-Bus connection pool limit), ~1.5 hours

USAGE
------
  # Quick test (5 tasks × 10 runs × 4 conditions = 200 trials, ~30 min at W=8)
  python3 exp_sjv5_parallel.py \
    --n-tasks 5 --n-runs 10 --n-steps 15 --workers 8

  # Full paper run (10 tasks × 25 runs × 4 conditions = 1000 trials, ~3h at W=8)
  python3 exp_sjv5_parallel.py \
    --n-tasks 10 --n-runs 25 --n-steps 20 --workers 8 \
    --output results/sjv5_parallel.csv

  # Replicate SJ-v4 (conditions 0 and 1 only, 20 tasks × 25 runs = 1000 trials)
  python3 exp_sjv5_parallel.py \
    --n-tasks 20 --n-runs 25 --stale-counts 0 1 --workers 8

REQUIRES
---------
  SBUS_ADMIN_ENABLED=1 cargo run --release
  pip install openai anthropic
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
N_AGENTS    = 4

# Tasks from SJ-v4 (cumulative-state sensitive)
SJV5_TASKS = [
    {"id": "sympy_algebraic",     "desc": "Fix SymPy solve() dropping solutions with positive assumptions. Agents coordinate changes to solve_engine, assumption_system, simplification, solution_filter.", "shards": ["solve_engine", "assumption_system", "simplification", "solution_filter"]},
    {"id": "django_migration",    "desc": "Fix Django migration squasher for circular RunSQL dependencies. Agents update migration_graph, squash_plan, dependency_resolver, rollback_state.", "shards": ["migration_graph", "squash_plan", "dependency_resolver", "rollback_state"]},
    {"id": "astropy_fits_header", "desc": "Fix Astropy FITS HIERARCH keyword parsing for non-standard dialects. Agents update header_parser, io_handler, keyword_registry, card_formatter.", "shards": ["header_parser", "io_handler", "keyword_registry", "card_formatter"]},
    {"id": "sklearn_clone",       "desc": "Fix scikit-learn estimator clone() for **kwargs in __init__. Agents update estimator_api, clone_logic, param_inspector, validation_utils.", "shards": ["estimator_api", "clone_logic", "param_inspector", "validation_utils"]},
    {"id": "django_queryset",     "desc": "Fix Django queryset ordering with select_related() and FK traversal. Agents coordinate models_state, orm_query, test_fixtures, migration_plan.", "shards": ["models_state", "orm_query", "test_fixtures", "migration_plan"]},
    {"id": "sympy_matrix",        "desc": "Fix SymPy eigenvals() for sparse matrices with Rational coefficients. Agents update matrix_engine, eigenvalue_solver, sparse_repr, rational_arithmetic.", "shards": ["matrix_engine", "eigenvalue_solver", "sparse_repr", "rational_arithmetic"]},
    {"id": "requests_redirect",   "desc": "Fix requests auth header stripping on cross-domain redirects. Agents update session_state, auth_handler, redirect_logic, header_policy.", "shards": ["session_state", "auth_handler", "redirect_logic", "header_policy"]},
    {"id": "astropy_wcs",         "desc": "Fix Astropy ZEA projection boundary checks near poles. Agents update wcs_transform, projection_math, coordinate_frame, bounds_check.", "shards": ["wcs_transform", "projection_math", "coordinate_frame", "bounds_check"]},
    {"id": "django_admin",        "desc": "Fix Django admin per-object permission checking for custom actions. Agents update admin_config, permission_check, action_registry, view_logic.", "shards": ["admin_config", "permission_check", "action_registry", "view_logic"]},
    {"id": "astropy_units",       "desc": "Fix Astropy unit conversion via non-SI intermediate representations. Agents update unit_registry, conversion_graph, quantity_repr, equivalency_map.", "shards": ["unit_registry", "conversion_graph", "quantity_repr", "equivalency_map"]},
    {"id": "sympy_integrate",     "desc": "Fix SymPy integrate() for piecewise functions with symbolic bounds. Agents coordinate integration_engine, piecewise_logic, bounds_check, simplification.", "shards": ["integration_engine", "piecewise_logic", "bounds_check", "simplification"]},
    {"id": "django_forms",        "desc": "Fix Django ModelForm field ordering with Meta.fields ordering override. Agents update form_metaclass, field_ordering, validation_logic, widget_registry.", "shards": ["form_metaclass", "field_ordering", "validation_logic", "widget_registry"]},
    {"id": "astropy_coordinates", "desc": "Fix Astropy SkyCoord frame transformation with intermediate frames. Agents update frame_registry, transform_graph, coordinate_repr, angle_utils.", "shards": ["frame_registry", "transform_graph", "coordinate_repr", "angle_utils"]},
    {"id": "requests_timeout",    "desc": "Fix requests connection vs read timeout not separating correctly. Agents update timeout_handler, connection_pool, socket_utils, retry_logic.", "shards": ["timeout_handler", "connection_pool", "socket_utils", "retry_logic"]},
    {"id": "sklearn_pipeline",    "desc": "Fix scikit-learn Pipeline feature_names_in_ propagation with ColumnTransformer. Agents update pipeline_meta, column_transformer, feature_names, validation.", "shards": ["pipeline_meta", "column_transformer", "feature_names", "validation"]},
    {"id": "django_signals",      "desc": "Fix Django signal dispatch with weak references and garbage collection. Agents update signal_registry, dispatch_logic, receiver_lookup, gc_handling.", "shards": ["signal_registry", "dispatch_logic", "receiver_lookup", "gc_handling"]},
    {"id": "sympy_latex",         "desc": "Fix SymPy LaTeX printer for nested fractions with assumptions. Agents update latex_printer, assumption_handler, fraction_repr, printer_settings.", "shards": ["latex_printer", "assumption_handler", "fraction_repr", "printer_settings"]},
    {"id": "astropy_time",        "desc": "Fix Astropy Time arithmetic with mixed time scales (UTC/TAI/TDB). Agents update time_scale, arithmetic_ops, leap_second_table, format_converter.", "shards": ["time_scale", "arithmetic_ops", "leap_second_table", "format_converter"]},
    {"id": "sklearn_cross_val",   "desc": "Fix scikit-learn cross_val_score with stratified group splits. Agents update cv_splitter, group_stratified, score_aggregation, random_state.", "shards": ["cv_splitter", "group_stratified", "score_aggregation", "random_state"]},
    {"id": "django_cache",        "desc": "Fix Django cache framework key generation for multi-database setups. Agents update cache_backend, key_generator, database_router, invalidation.", "shards": ["cache_backend", "key_generator", "database_router", "invalidation"]},
]

# ── Rate limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Token-bucket rate limiter to avoid OpenAI RPM limits.
    Default: 400 requests/min (conservative for tier 1: 500 RPM).
    For tier 2 (5,000 RPM), use rpm=4000.
    """
    def __init__(self, rpm: int = 400):
        self._interval = 60.0 / rpm
        self._last     = 0.0
        self._lock     = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now  = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# ── HTTP helpers (thread-safe: each call creates its own opener) ───────────────

def _make_opener():
    return build_opener(ProxyHandler({}))

def http_get(url: str, params: dict = None) -> tuple[int, dict]:
    if params:
        url += "?" + urlencode(params)
    opener = _make_opener()
    try:
        with opener.open(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}

def http_post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req  = Request(url, data=data, headers={"Content-Type": "application/json"})
    opener = _make_opener()
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}

def health_check() -> bool:
    try:
        s = socket.create_connection(("localhost", 7000), timeout=3); s.close()
    except Exception:
        return False
    st, _ = http_get(f"{SBUS_URL}/stats")
    return st == 200


# ── Judge ─────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """
You are evaluating MULTI-AGENT collaborative incremental contributions.
Shards contain agent deltas like "[agent_X sN] ..." — this format is EXPECTED and NORMAL.
Do NOT penalise for incomplete code, truncated lines, or delta format.

TASK: {task}
ACCUMULATED CONTRIBUTIONS: {content}

Evaluate only SEMANTIC CONSISTENCY:
  CORRECT   - agents converge on a consistent, relevant approach
  INCOMPLETE - too little content or entirely off-task
  CORRUPTED  - agents propose CONTRADICTORY strategies (e.g., one says PostgreSQL, another says DynamoDB)

Reply EXACTLY one of: CORRECT | INCOMPLETE | CORRUPTED
Then a newline, then one sentence on CONSISTENCY (not completeness)."""

def judge(task_desc: str, content: str, oai: OpenAI, rate_limiter: RateLimiter) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(
        task=task_desc[:400],
        content=content[:600] if content else "[empty]")

    if HAS_ANTHROPIC:
        try:
            rate_limiter.acquire()
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=JUDGE_MODEL, max_tokens=120, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            text = msg.content[0].text.strip()
        except Exception as e:
            text = f"INCOMPLETE\nJudge error: {e}"
    else:
        try:
            rate_limiter.acquire()
            r = oai.chat.completions.create(
                model=BACKBONE, max_tokens=120, temperature=0,
                messages=[
                    {"role": "system", "content": "Strict reviewer. Reply CORRECT, INCOMPLETE, or CORRUPTED then newline."},
                    {"role": "user",   "content": prompt},
                ])
            text = r.choices[0].message.content.strip()
        except Exception as e:
            text = f"INCOMPLETE\nJudge error: {e}"

    lines = text.strip().split("\n", 1)
    raw = lines[0].upper()
    if "CORRECT" in raw and "IN" not in raw:
        v = "CORRECT"
    elif "CORRUPT" in raw:
        v = "CORRUPTED"
    else:
        v = "INCOMPLETE"
    return v, lines[1].strip() if len(lines) > 1 else ""


# ── Core trial runner ─────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    run_id:            str
    task_id:           str
    run_idx:           int
    n_stale:           int
    stale_fraction:    float
    n_agents:          int
    n_steps:           int
    n_stale_steps:     int
    commits_ok:        int
    commits_total:     int
    verdict:           str
    is_correct:        bool
    is_corrupted:      bool
    judge_reason:      str
    surviving_content: str
    wall_secs:         float


def run_trial(
    oai:            OpenAI,
    rate_limiter:   RateLimiter,
    task:           dict,
    run_idx:        int,
    n_stale:        int,
    n_steps:        int,
    injection_step: int,
) -> TrialResult:
    """
    One independent trial: (task, run_idx, n_stale).
    Creates its own isolated shards via unique run_id.
    Safe to run concurrently with any other trial.
    """
    run_id = uuid.uuid4().hex[:8]
    shards = [f"{sk}_{run_id}" for sk in task["shards"]]
    agents = [f"agent_{i}_{run_id}" for i in range(N_AGENTS)]

    t0 = time.perf_counter()

    # Create shards (no reset — each run owns its own keys)
    for sk in shards:
        http_post(f"{SBUS_URL}/shard", {
            "key":      sk,
            "content":  f"Initial: {task['desc'][:80]}",
            "goal_tag": f"sjv5_{task['id']}",
        })
    for a in agents:
        http_post(f"{SBUS_URL}/session", {"agent_id": a, "session_ttl": 3600})

    # Capture stale snapshot at step 0
    stale_snapshots = {}
    for sk in shards:
        _, data = http_get(f"{SBUS_URL}/shard/{sk}", {"agent_id": "snap"})
        stale_snapshots[sk] = data.get("content", f"Initial: {task['desc'][:80]}")

    stale_agents  = set(agents[:n_stale])
    commits_ok    = 0
    commits_total = 0
    n_stale_steps = 0

    for step in range(n_steps):
        for agent in agents:
            target = shards[step % len(shards)]

            # Get current version for commit
            _, cur_data = http_get(f"{SBUS_URL}/shard/{target}", {"agent_id": agent})
            cur_ver = cur_data.get("version", 0) if cur_data else 0

            # Build context
            if agent in stale_agents and step >= injection_step:
                context = "\n".join(
                    f"  {sk}: {stale_snapshots[sk][:80]} [FROZEN step-0]"
                    for sk in shards)
                n_stale_steps += 1
            else:
                parts = []
                for sk in shards:
                    _, sdata = http_get(f"{SBUS_URL}/shard/{sk}", {"agent_id": agent})
                    if sdata:
                        parts.append(
                            f"  {sk}: v{sdata.get('version',0)} — "
                            f"{sdata.get('content','')[:80]}")
                context = "\n".join(parts)

            # LLM call (rate-limited)
            try:
                rate_limiter.acquire()
                resp = oai.chat.completions.create(
                    model=BACKBONE, max_tokens=100, temperature=0.3,
                    messages=[{"role": "user", "content": (
                        f"TASK: {task['desc'][:250]}\n"
                        f"Current state (step {step}):\n{context[:350]}\n"
                        f"Write ONE concrete technical improvement. Output ONLY the change."
                    )}])
                delta = (f"[{agent} s{step}] "
                         f"{resp.choices[0].message.content.strip()}")
            except Exception as e:
                delta = f"[{agent} s{step}] ERROR: {e}"

            # Commit at current version (stale context passes structural check)
            st, _ = http_post(f"{SBUS_URL}/commit/v2", {
                "key":              target,
                "expected_version": cur_ver,
                "delta":            delta,
                "agent_id":         agent,
                "read_set":         [{"key": target, "version_at_read": cur_ver}],
            })
            commits_total += 1
            if st == 200:
                commits_ok += 1

    # Collect final state
    final_parts = []
    for sk in shards:
        _, fdata = http_get(f"{SBUS_URL}/shard/{sk}", {"agent_id": "judge"})
        if fdata:
            final_parts.append(f"{sk}: {fdata.get('content','')[:250]}")
    final_content = "\n".join(final_parts)

    verdict, reason = judge(task["desc"], final_content, oai, rate_limiter)
    wall = time.perf_counter() - t0

    return TrialResult(
        run_id          = run_id,
        task_id         = task["id"],
        run_idx         = run_idx,
        n_stale         = n_stale,
        stale_fraction  = round(n_stale / N_AGENTS, 2),
        n_agents        = N_AGENTS,
        n_steps         = n_steps,
        n_stale_steps   = n_stale_steps,
        commits_ok      = commits_ok,
        commits_total   = commits_total,
        verdict         = verdict,
        is_correct      = (verdict == "CORRECT"),
        is_corrupted    = (verdict == "CORRUPTED"),
        judge_reason    = reason[:200],
        surviving_content = final_content[:300],
        wall_secs       = round(wall, 1),
    )


# ── Thread-safe progress printer ──────────────────────────────────────────────

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ── Thread-safe CSV writer ────────────────────────────────────────────────────

class CSVWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._path   = path
        self._lock   = threading.Lock()
        self._file   = open(path, "w", newline="")
        self._writer = None

    def write(self, row: dict) -> None:
        with self._lock:
            if self._writer is None:
                self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
                self._writer.writeheader()
            self._writer.writerow(row)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp SJ-v5 (PARALLEL): Stale fraction dose-response")
    parser.add_argument("--n-tasks",        type=int, default=10)
    parser.add_argument("--n-runs",         type=int, default=25)
    parser.add_argument("--n-steps",        type=int, default=20)
    parser.add_argument("--injection-step", type=int, default=5)
    parser.add_argument("--stale-counts",   type=int, nargs="+",
                        default=[0, 1, 2, 3])
    parser.add_argument("--workers",        type=int, default=4,
                        help="Parallel workers (4=tier1 safe, 8=tier2 safe, 12=aggressive)")
    parser.add_argument("--rpm",            type=int, default=400,
                        help="Max OpenAI requests/min (400=tier1, 4000=tier2)")
    parser.add_argument("--output",         default="results/sjv5_parallel.csv")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)

    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}")
        print("  Start: SBUS_ADMIN_ENABLED=1 cargo run --release")
        sys.exit(1)

    if not HAS_ANTHROPIC:
        print("NOTE: anthropic not installed — using GPT-4o-mini as judge")

    # Shared (thread-safe) objects
    oai          = OpenAI(api_key=api_key)
    rate_limiter = RateLimiter(rpm=args.rpm)
    csv_writer   = CSVWriter(args.output)

    tasks = SJV5_TASKS[:args.n_tasks]

    # Build ALL work units upfront
    work_units = [
        (task, run_idx, n_stale)
        for task    in tasks
        for run_idx in range(args.n_runs)
        for n_stale in args.stale_counts
    ]
    total = len(work_units)

    # Counters (thread-safe via lock)
    counts_lock = threading.Lock()
    counts = {k: {"correct": 0, "corrupted": 0, "incomplete": 0, "total": 0}
              for k in args.stale_counts}
    done_count = [0]   # mutable int via list

    print("=" * 70)
    print("Exp SJ-v5 PARALLEL: Stale Agent Fraction Dose-Response")
    print("=" * 70)
    print(f"Tasks        : {[t['id'] for t in tasks]}")
    print(f"Stale counts : {args.stale_counts} / {N_AGENTS} agents")
    print(f"Runs/cond    : {args.n_runs}  |  Total trials: {total}")
    print(f"Workers      : {args.workers}  |  Rate limit: {args.rpm} RPM")
    print(f"Steps/trial  : {args.n_steps}  |  Injection at step: {args.injection_step}")
    print()

    # Estimate time
    calls_per_trial = N_AGENTS * args.n_steps + 1  # agents + judge
    serial_secs     = total * calls_per_trial * (60 / args.rpm)
    parallel_secs   = serial_secs / args.workers
    print(f"Estimated time: {parallel_secs/3600:.1f} hours "
          f"(serial would be {serial_secs/3600:.1f} hours)")
    print()
    print("HYPOTHESIS:")
    print("  0 stale: baseline  (replicate SJ-v4 condition A)")
    print("  1 stale: diversity (replicate SJ-v4 condition B — expect LOWER corruption)")
    print("  2 stale: moderate  (expect similar or slightly lower)")
    print("  3 stale: saturated (expect HIGHER corruption — U-shape peak)")
    print()
    print("Running... (each dot = one completed trial)")
    print()

    t_start = time.time()

    def _run_unit(unit):
        task, run_idx, n_stale = unit
        try:
            result = run_trial(
                oai, rate_limiter, task, run_idx, n_stale,
                args.n_steps, args.injection_step)
            csv_writer.write(asdict(result))

            with counts_lock:
                k = result.n_stale
                counts[k]["total"] += 1
                if result.verdict == "CORRECT":
                    counts[k]["correct"] += 1
                elif result.verdict == "CORRUPTED":
                    counts[k]["corrupted"] += 1
                else:
                    counts[k]["incomplete"] += 1
                done_count[0] += 1
                n_done = done_count[0]

            # Progress dot with periodic summary line
            symbol = "✓" if result.is_correct else ("✗" if result.is_corrupted else "·")
            with _print_lock:
                print(symbol, end="", flush=True)
                if n_done % 50 == 0:
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed
                    eta  = (total - n_done) / rate if rate > 0 else 0
                    print(f"  [{n_done}/{total}] "
                          f"{elapsed/60:.1f}m elapsed, "
                          f"ETA {eta/60:.0f}m")
                    _print_live_summary(counts, args.stale_counts)

            return result
        except Exception as e:
            tprint(f"\n  ERROR in {unit}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_unit, unit): unit for unit in work_units}
        try:
            for _ in as_completed(futures):
                pass   # progress printed inside _run_unit
        except KeyboardInterrupt:
            print("\n\nInterrupted — saving partial results...")
            pool.shutdown(wait=False, cancel_futures=True)

    csv_writer.close()
    print()
    print()

    total_time = time.time() - t_start
    print(f"Completed {done_count[0]}/{total} trials in {total_time/60:.1f} minutes")
    print()
    _print_final_summary(counts, args)
    _run_stats(args.output, args.stale_counts)
    print(f"\nResults: {args.output}")


# ── Summary helpers ───────────────────────────────────────────────────────────

def _print_live_summary(counts: dict, stale_counts: list) -> None:
    parts = []
    for k in sorted(stale_counts):
        c = counts.get(k, {})
        t = c.get("total", 0)
        if t == 0:
            parts.append(f"k={k}: –")
        else:
            s50 = c["correct"] / t
            parts.append(f"k={k}: {s50*100:.0f}%")
    with _print_lock:
        print("  Current S@50: " + "  |  ".join(parts))


def _print_final_summary(counts: dict, args) -> None:
    print("=" * 70)
    print("SJ-V5 FINAL RESULTS")
    print("=" * 70)

    rates = {}
    for k in sorted(args.stale_counts):
        c = counts.get(k, {"correct": 0, "corrupted": 0, "incomplete": 0, "total": 0})
        t = c["total"]
        if t == 0:
            print(f"  {k}/{N_AGENTS} stale: no data")
            continue
        s50  = c["correct"] / t
        corr = c["corrupted"] / t
        c["incomplete"] / t
        rates[k] = s50
        bar = "█" * int(s50 * 30) + "░" * (30 - int(s50 * 30))
        print(f"  {k}/{N_AGENTS} stale ({k*25:3d}%): "
              f"n={t:4d} | S@50={s50*100:5.1f}% [{bar}]"
              f" | corrupt={corr*100:4.1f}%")

    if len(rates) >= 2:
        print()
        ks = sorted(rates.keys())
        base = rates.get(0, None)
        print("  Dose-response vs baseline (0 stale):")
        for k in ks:
            if k == 0 or base is None: continue
            diff = rates[k] - base
            arrow = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "≈")
            print(f"    k={k}: {diff*100:+.1f}pp  {arrow}")

        # U-shape check
        if all(k in rates for k in [0, 1, 2, 3]):
            r0, r1, r2, r3 = rates[0], rates[1], rates[2], rates[3]
            if r1 >= r0 and r3 < r1:
                shape = "U-SHAPE CONFIRMED (diversity peaks at k=1-2)"
            elif r0 >= r1 >= r2 >= r3:
                shape = "MONOTONE DECREASE (staleness uniformly hurts)"
            elif r3 >= r2 >= r1 >= r0:
                shape = "MONOTONE INCREASE (staleness uniformly helps)"
            else:
                shape = "NO CLEAR PATTERN"
            print()
            print(f"  Shape: {shape}")

    print()
    print("  PAPER TEXT (copy-paste ready for §7.8):")
    print()
    print("  \\textbf{Exp.~SJ-v5 (stale fraction dose-response)}.")
    print(f"  ({args.n_tasks} tasks, {args.n_runs} runs/condition,")
    print(f"  {N_AGENTS} agents, {args.n_steps} steps; stale injection at step {5}):")
    print()
    for k in sorted(args.stale_counts):
        c = counts.get(k, {"correct": 0, "total": 0})
        t = c["total"]
        if t == 0: continue
        s50 = c["correct"] / t
        print(f"  $k{k}/{N_AGENTS}$ stale: S@50~$= {s50*100:.1f}\\%$ ($n={t}$).")


def _run_stats(csv_path: str, stale_counts: list) -> None:
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("\nNOTE: pip install scipy for Fisher's exact tests")
        return

    rows = []
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return

    if not rows:
        return

    print()
    print("=" * 70)
    print("FISHER'S EXACT TESTS (all pairs vs k=0 baseline)")
    print("=" * 70)

    def get_counts(k):
        subset  = [r for r in rows if int(r["n_stale"]) == k]
        correct = sum(1 for r in subset if r["verdict"] == "CORRECT")
        return correct, len(subset) - correct, len(subset)

    c0, nc0, n0 = get_counts(0)
    if n0 == 0:
        print("  No data for k=0 baseline")
        return

    print(f"  Baseline k=0: {c0}/{n0} = {c0/n0*100:.1f}% correct")

    for k in sorted(stale_counts):
        if k == 0: continue
        ck, nck, nk = get_counts(k)
        if nk == 0: continue
        _, p = scipy_stats.fisher_exact([[c0, nc0], [ck, nck]], alternative="two-sided")
        d = ck/nk - c0/n0
        sig = "✅ p<0.05" if p < 0.05 else "– n.s."
        print(f"\n  k={k}/{N_AGENTS} vs k=0:")
        print(f"    k={k}: {ck}/{nk} = {ck/nk*100:.1f}%  |  "
              f"Δ={d*100:+.1f}pp  |  Fisher p={p:.4f}  {sig}")

    # Pairwise: 1 vs 3 (diversity peak vs high stale)
    c1, nc1, n1 = get_counts(1)
    c3, nc3, n3 = get_counts(3)
    if n1 > 0 and n3 > 0:
        _, p13 = scipy_stats.fisher_exact([[c1, nc1], [c3, nc3]], alternative="two-sided")
        print("\n  k=1 vs k=3 (diversity peak vs high stale):")
        print(f"    k=1: {c1/n1*100:.1f}%  k=3: {c3/n3*100:.1f}%  "
              f"p={p13:.4f}  {'✅ sig' if p13 < 0.05 else '– n.s.'}")


if __name__ == "__main__":
    main()
