import argparse
import csv
import json
import os
import sys
import time
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai"); sys.exit(1)

SBUS_URL = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE = "gpt-4o-mini"
MAX_DELTA = 1800

TASKS = [
    {
        "id": "django_orm",
        "desc": "Fix Django queryset ordering with select_related() FK traversal.",
        "shards": [
            {"key": "orm_core",     "role": "ORM Core Developer",
             "goal": "Diagnose the root cause in django/db/models/sql/compiler.py"},
            {"key": "query_comp",   "role": "Query Compiler Specialist",
             "goal": "Fix the get_order_by() method to handle FK traversal"},
            {"key": "test_writer",  "role": "Test Engineer",
             "goal": "Write regression tests covering the edge cases"},
            {"key": "reviewer",     "role": "Senior Reviewer",
             "goal": "Review all changes for correctness and backwards compatibility"},
        ]
    },
    {
        "id": "sympy_solve",
        "desc": "Fix SymPy solve() dropping solutions with positive assumptions.",
        "shards": [
            {"key": "solver_core",  "role": "Solver Core Developer",
             "goal": "Diagnose root cause in sympy/solvers/solvers.py _solve()"},
            {"key": "assumptions",  "role": "Assumption System Specialist",
             "goal": "Fix the assumption filtering logic"},
            {"key": "test_cases",   "role": "Test Engineer",
             "goal": "Write test cases with various assumption combinations"},
            {"key": "doc_writer",   "role": "Documentation Writer",
             "goal": "Document the fix and update docstrings"},
        ]
    },
    {
        "id": "astropy_fits",
        "desc": "Fix Astropy FITS HIERARCH keyword parsing non-standard dialects.",
        "shards": [
            {"key": "parser_core",  "role": "Parser Core Developer",
             "goal": "Fix HIERARCH keyword parsing in astropy/io/fits/card.py"},
            {"key": "io_handler",   "role": "IO Handler Specialist",
             "goal": "Update file reading to handle non-standard spacing"},
            {"key": "keyword_reg",  "role": "Keyword Registry Developer",
             "goal": "Update keyword registry to handle dialect variations"},
            {"key": "card_fmt",     "role": "Card Formatter",
             "goal": "Fix card formatting output for round-trip consistency"},
        ]
    },
    {
        "id": "django_migration",
        "desc": "Fix Django migration squasher circular RunSQL dependencies.",
        "shards": [
            {"key": "graph_res",    "role": "Graph Resolver",
             "goal": "Fix circular dependency detection in migration graph"},
            {"key": "squash_logic", "role": "Squash Logic Developer",
             "goal": "Fix squashmigrations command to handle RunSQL"},
            {"key": "dep_checker",  "role": "Dependency Checker",
             "goal": "Add validation to catch circular deps before squashing"},
            {"key": "state_rebuild","role": "State Rebuilder",
             "goal": "Ensure state is correctly rebuilt after squash"},
        ]
    },
    {
        "id": "requests_redirect",
        "desc": "Fix requests auth header stripping on cross-domain 301 redirects.",
        "shards": [
            {"key": "session_hdlr", "role": "Session Handler",
             "goal": "Fix session.rebuild_auth() in requests/sessions.py"},
            {"key": "auth_logic",   "role": "Auth Logic Developer",
             "goal": "Fix auth header preservation policy"},
            {"key": "redirect_ch",  "role": "Redirect Chain Specialist",
             "goal": "Implement correct cross-domain redirect handling"},
            {"key": "header_pol",   "role": "Header Policy Developer",
             "goal": "Define and implement header stripping policy"},
        ]
    },
    {
        "id": "sklearn_clone",
        "desc": "Fix scikit-learn clone() failing with **kwargs in __init__.",
        "shards": [
            {"key": "estimator_api","role": "Estimator API Developer",
             "goal": "Fix clone() in sklearn/base.py"},
            {"key": "clone_logic",  "role": "Clone Logic Specialist",
             "goal": "Handle **kwargs in constructor inspection"},
            {"key": "param_insp",   "role": "Parameter Inspector",
             "goal": "Fix get_params() to work with **kwargs"},
            {"key": "valid_utils",  "role": "Validation Utils Developer",
             "goal": "Add validation for **kwargs estimators"},
        ]
    },
    {
        "id": "sympy_matrix",
        "desc": "Fix SymPy eigenvals() for sparse Rational matrices.",
        "shards": [
            {"key": "matrix_eng",   "role": "Matrix Engine Developer",
             "goal": "Fix eigenvals() in sympy/matrices/matrices.py"},
            {"key": "eigen_solver", "role": "Eigenvalue Solver",
             "goal": "Fix the sparse matrix eigenvalue computation"},
            {"key": "sparse_back",  "role": "Sparse Backend Developer",
             "goal": "Optimize sparse Rational matrix operations"},
            {"key": "rat_arith",    "role": "Rational Arithmetic Specialist",
             "goal": "Fix Rational arithmetic edge cases in eigenvalue computation"},
        ]
    },
    {
        "id": "astropy_wcs",
        "desc": "Fix Astropy ZEA projection boundary errors near poles.",
        "shards": [
            {"key": "wcs_trans",    "role": "WCS Transform Developer",
             "goal": "Fix ZEA projection in astropy/wcs/utils.py"},
            {"key": "proj_math",    "role": "Projection Math Specialist",
             "goal": "Fix boundary condition math near poles"},
            {"key": "coord_frame",  "role": "Coordinate Frame Developer",
             "goal": "Fix coordinate frame handling at poles"},
            {"key": "bounds_chk",   "role": "Bounds Checker",
             "goal": "Add proper bounds checking for polar regions"},
        ]
    },
    {
        "id": "django_admin",
        "desc": "Fix Django admin bulk action per-object permission checks.",
        "shards": [
            {"key": "perm_layer",   "role": "Permission Layer Developer",
             "goal": "Fix per-object permission checks in admin"},
            {"key": "action_reg",   "role": "Action Registry Developer",
             "goal": "Fix bulk action registration with permissions"},
            {"key": "view_logic",   "role": "View Logic Developer",
             "goal": "Fix response_action() in django/contrib/admin/options.py"},
            {"key": "tmpl_render",  "role": "Template Renderer",
             "goal": "Update templates to show correct permission states"},
        ]
    },
    {
        "id": "django_forms",
        "desc": "Fix Django ModelForm field ordering with Meta.fields override.",
        "shards": [
            {"key": "form_meta",    "role": "Form Metaclass Developer",
             "goal": "Fix metaclass field ordering in ModelForm"},
            {"key": "field_order",  "role": "Field Ordering Specialist",
             "goal": "Fix fields_for_model() in django/forms/models.py"},
            {"key": "valid_logic",  "role": "Validation Logic Developer",
             "goal": "Ensure ordering is preserved through validation"},
            {"key": "widget_reg",   "role": "Widget Registry Developer",
             "goal": "Fix widget assignment to respect field ordering"},
        ]
    },
]

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

def health_check():
    import socket
    try: s = socket.create_connection(("localhost", 7000), timeout=3); s.close()
    except: return False
    st, _ = _req("GET", "/stats")
    return st == 200

_rl_lock = threading.Lock()
_rl_last = 0.0

def rl_acquire(rpm=200):
    global _rl_last
    interval = 60.0 / rpm
    with _rl_lock:
        now = time.monotonic()
        w = _rl_last + interval - now
        if w > 0: time.sleep(w)
        _rl_last = time.monotonic()

JUDGE_PROMPT = """You are evaluating multi-agent collaborative work on a software bug fix.

TASK: {task_desc}

AGENT ROLES AND THEIR CONTRIBUTIONS:
{contributions}

EVALUATION:
Rate whether the agents produced COMPLEMENTARY, NON-REDUNDANT contributions
that together form a coherent approach to fixing the bug.

Reply with EXACTLY one word: COHERENT | REDUNDANT | INCOMPLETE
Then one newline, then: "reason=<brief explanation>"

COHERENT = each agent contributed distinct, complementary work
REDUNDANT = multiple agents repeated similar work (wasted effort)
INCOMPLETE = too little content to evaluate"""

def judge_coherence(oai, task_desc, contributions_text):
    if not contributions_text or len(contributions_text.strip()) < 100:
        return "INCOMPLETE", "insufficient content"
    rl_acquire()
    try:
        r = oai.chat.completions.create(
            model=BACKBONE, max_tokens=60, temperature=0,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    task_desc=task_desc[:300],
                    contributions=contributions_text[:2000]
                )
            }])
        text = r.choices[0].message.content.strip()
        lines = text.split("\n", 1)
        raw = lines[0].upper().strip()
        reason = lines[1].replace("reason=","").strip() if len(lines)>1 else ""
        if "COHERENT" in raw: return "COHERENT", reason
        if "REDUNDANT" in raw: return "REDUNDANT", reason
        return "INCOMPLETE", reason
    except Exception as e:
        return "INCOMPLETE", str(e)

def run_trial(oai, task, n_steps, condition, run_id):
    shards = task["shards"]
    created_shards = {}
    for s in shards:
        key = f"{s['key']}_{run_id}"
        seed = f"Task: {task['desc']} Role: {s['role']}. Goal: {s['goal']}"
        _req("POST", "/shard", {"key": key, "content": seed[:MAX_DELTA],
                                "goal_tag": task["id"]})
        agent_id = f"agent_{s['key']}_{run_id}"
        _req("POST", "/session", {"agent_id": agent_id})
        created_shards[s['key']] = {
            "shard_key": key,
            "agent_id": agent_id,
            "role": s["role"],
            "goal": s["goal"],
            "frozen_context": seed,
        }

    commits_ok = 0
    all_contributions = []

    for step in range(n_steps):
        for shard_name, info in created_shards.items():
            shard_key = info["shard_key"]
            agent_id  = info["agent_id"]

            if condition == "fresh":
                st, d = _req("GET", f"/shard/{shard_key}",
                             params={"agent_id": agent_id})
                ver     = d.get("version", 0) if st == 200 else 0
                context = d.get("content", info["frozen_context"]) if st == 200 \
                          else info["frozen_context"]
            else:
                st, d = _req("GET", f"/shard/{shard_key}",
                             params={"agent_id": agent_id})
                ver = d.get("version", 0) if st == 200 else 0
                context = info["frozen_context"]

            rl_acquire()
            try:
                r = oai.chat.completions.create(
                    model=BACKBONE, max_tokens=150, temperature=0.3,
                    messages=[{"role": "user", "content":
                        f"TASK: {task['desc'][:200]}\n"
                        f"YOUR ROLE: {info['role']}\n"
                        f"YOUR GOAL: {info['goal']}\n"
                        f"CURRENT STATE OF YOUR WORK:\n{context[:500]}\n"
                        f"Add 3 sentences of concrete progress toward your goal. "
                        f"Be specific and non-redundant with step {step}."}])
                delta = f"[{info['role']} step{step}] " + \
                        r.choices[0].message.content.strip()
            except Exception:
                delta = f"[{info['role']} step{step}] ERR"

            delta = delta[:MAX_DELTA]

            st2, resp2 = _req("POST", "/commit/v2", {
                "key": shard_key,
                "expected_version": ver,
                "delta": delta,
                "agent_id": agent_id,
            })
            if st2 == 200 and "new_version" in resp2:
                commits_ok += 1
                all_contributions.append(f"[{info['role']}]: {delta[:200]}")

    final_text = "\n\n".join(all_contributions[-20:])  # last 20 contributions
    verdict, reason = judge_coherence(oai, task["desc"], final_text)

    return {
        "task_id": task["id"],
        "run_id": run_id,
        "condition": condition,
        "n_agents": len(shards),
        "n_steps": n_steps,
        "commits_ok": commits_ok,
        "commits_total": len(shards) * n_steps,
        "commit_rate": round(commits_ok / max(1, len(shards) * n_steps), 4),
        "verdict": verdict,
        "is_coherent": int(verdict == "COHERENT"),
        "is_redundant": int(verdict == "REDUNDANT"),
        "judge_reason": reason[:200],
        "wall_secs": 0.0,
    }

_csv_lock = threading.Lock()

def write_row(path, row):
    with _csv_lock:
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists: w.writeheader()
            w.writerow(row)

def load_done(path):
    done = set()
    if not os.path.exists(path): return done
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r.get("task_id",""), r.get("run_id",""),
                          r.get("condition","")))
    except: pass
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks",  type=int, default=10)
    ap.add_argument("--n-runs",   type=int, default=30)
    ap.add_argument("--n-steps",  type=int, default=8)
    ap.add_argument("--workers",  type=int, default=2)
    ap.add_argument("--output", default="results/dedicated_shard_semantic.csv")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)
    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}"); sys.exit(1)

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=30)
    tasks = TASKS[:args.n_tasks]
    conditions = ["fresh", "stale"]
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output)
                else ".", exist_ok=True)

    done = load_done(args.output)
    work = [(t, r, c) for t in tasks for r in range(args.n_runs)
            for c in conditions
            if (t["id"], str(r), c) not in done]

    total = len(tasks) * args.n_runs * 2
    print("Exp. DEDICATED-SHARD SEMANTIC")
    print(f"  {args.n_tasks} tasks × {args.n_runs} runs × 2 conditions = {total} trials")
    print(f"  Resuming: {total-len(work)} done, {len(work)} remaining")
    print()

    fresh_verdicts = []
    stale_verdicts = []
    lock = threading.Lock()
    completed = [total - len(work)]

    def _run(unit):
        task, rep, condition = unit
        rid = f"{task['id']}_{rep}"
        t0 = time.perf_counter()
        try:
            row = run_trial(oai, task, args.n_steps, condition, rid)
            row["wall_secs"] = round(time.perf_counter() - t0, 1)
            write_row(args.output, row)
            with lock:
                completed[0] += 1
                if condition == "fresh": fresh_verdicts.append(row["is_coherent"])
                else: stale_verdicts.append(row["is_coherent"])
            print(f"  [{completed[0]:4d}/{total}] {task['id'][:18]:18s} "
                  f"{condition:6s} → {row['verdict']:10s} "
                  f"commits={row['commits_ok']}/{row['commits_total']}")
        except Exception as e:
            print(f"  ERR {task['id']} {condition}: {e}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        try:
            for _ in pool.map(_run, work): pass
        except KeyboardInterrupt:
            print("\nInterrupted — progress saved.")

    if fresh_verdicts and stale_verdicts:
        import statistics
        fresh_rate = statistics.mean(fresh_verdicts)
        stale_rate = statistics.mean(stale_verdicts)
        print(f"\n{'='*55}")
        print("DEDICATED-SHARD SEMANTIC RESULT")
        print(f"{'='*55}")
        print(f"  FRESH (ORI active): COHERENT rate = {fresh_rate*100:.1f}% (n={len(fresh_verdicts)})")
        print(f"  STALE (R_hidden):   COHERENT rate = {stale_rate*100:.1f}% (n={len(stale_verdicts)})")
        lift = fresh_rate - stale_rate
        print(f"  Lift = {lift*100:+.1f} pp  ({'FRESH WINS ✓' if lift>0 else 'STALE WINS — bad'})")
        try:
            from scipy.stats import fisher_exact
            a = sum(fresh_verdicts); b = len(fresh_verdicts)-a
            c = sum(stale_verdicts); d = len(stale_verdicts)-c
            _, p = fisher_exact([[a,b],[c,d]], alternative='greater')
            print(f"  Fisher's exact p={p:.4f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'}")
        except: pass
        print(f"\n  Output → {args.output}")

if __name__ == "__main__":
    main()
