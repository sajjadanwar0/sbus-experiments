import argparse
import csv
import json
import os
import sys
import time
import threading
import uuid
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai"); sys.exit(1)

SBUS_URL = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE = "gpt-4o-mini"
CALL_TIMEOUT = 25
TRIAL_TIMEOUT = 240
MAX_DELTA = 1800

TASKS = [
    {"id": "django_queryset",
     "desc": "Fix Django queryset ordering with select_related() FK traversal. "
             "Agents: ORM core, query compiler, test fixtures, migration plan.",
     "seed": "Task: Fix Django queryset ordering bug in select_related() FK traversal. "
             "Root cause is in django/db/models/sql/compiler.py get_order_by()."},
    {"id": "django_migration",
     "desc": "Fix Django migration squasher circular RunSQL dependencies. "
             "Agents: graph resolver, squash logic, dependency checker, state rebuilder.",
     "seed": "Task: Fix Django migration squasher crash on circular RunSQL deps. "
             "Root cause in django/core/management/commands/squashmigrations.py."},
    {"id": "django_admin",
     "desc": "Fix Django admin bulk action per-object permission checks. "
             "Agents: permission layer, action registry, view logic, template renderer.",
     "seed": "Task: Fix Django admin bulk actions bypassing per-object permissions. "
             "Root cause in django/contrib/admin/options.py response_action()."},
    {"id": "sympy_algebraic",
     "desc": "Fix SymPy solve() dropping solutions with positive assumptions. "
             "Agents: solver core, assumption checker, filter logic, simplification.",
     "seed": "Task: Fix SymPy solve() silently dropping valid solutions under assumptions. "
             "Root cause in sympy/solvers/solvers.py _solve()."},
    {"id": "sympy_matrix",
     "desc": "Fix SymPy eigenvals() for sparse Rational matrices. "
             "Agents: matrix engine, eigenvalue solver, sparse backend, rational arithmetic.",
     "seed": "Task: Fix SymPy eigenvals() returning wrong results for sparse Rational matrices. "
             "Root cause in sympy/matrices/matrices.py."},
    {"id": "astropy_fits",
     "desc": "Fix Astropy FITS HIERARCH keyword parsing non-standard dialects. "
             "Agents: header parser, IO handler, keyword registry, card formatter.",
     "seed": "Task: Fix Astropy FITS HIERARCH keyword parsing failures. "
             "Root cause in astropy/io/fits/card.py."},
    {"id": "sklearn_clone",
     "desc": "Fix scikit-learn clone() failing with **kwargs in __init__. "
             "Agents: estimator API, clone logic, param inspector, validation utils.",
     "seed": "Task: Fix scikit-learn clone() crash with **kwargs constructors. "
             "Root cause in sklearn/base.py clone()."},
    {"id": "requests_redirect",
     "desc": "Fix requests auth header stripping on cross-domain 301 redirects. "
             "Agents: session handler, auth logic, redirect chain, header policy.",
     "seed": "Task: Fix requests library stripping auth headers on cross-domain redirects. "
             "Root cause in requests/sessions.py rebuild_auth()."},
    {"id": "astropy_wcs",
     "desc": "Fix Astropy ZEA projection boundary errors near poles. "
             "Agents: WCS transform, projection math, coordinate frame, bounds checker.",
     "seed": "Task: Fix Astropy ZEA sky projection ValueError near declination poles. "
             "Root cause in astropy/wcs/utils.py."},
    {"id": "django_forms",
     "desc": "Fix Django ModelForm field ordering with Meta.fields override. "
             "Agents: form metaclass, field ordering, validation logic, widget registry.",
     "seed": "Task: Fix Django ModelForm ignoring Meta.fields ordering with __all__. "
             "Root cause in django/forms/models.py fields_for_model()."},
]

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

def _req(method, path, body=None, params=None):
    url = SBUS_URL + path
    if params: url += "?" + urlencode(params)
    data = json.dumps(body).encode() if body else None
    hdrs = {"Content-Type": "application/json"} if data else {}
    req = Request(url, data=data, headers=hdrs, method=method)
    try:
        with urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try: return e.code, json.loads(e.read())
        except: return e.code, {}
    except Exception:
        return 0, {}

def create_shard(key, seed, task_id):
    _req("POST", "/shard", {"key": key, "content": seed[:MAX_DELTA],
                             "goal_tag": task_id})

def register_agent(agent_id):
    _req("POST", "/session", {"agent_id": agent_id})

def read_shard(key, agent_id):
    st, d = _req("GET", f"/shard/{key}", params={"agent_id": agent_id})
    return (d.get("version", 0), d.get("content", "")) if st == 200 else (0, "")

def do_commit(key, agent_id, ver, delta):
    delta = delta[:MAX_DELTA]
    st, resp = _req("POST", "/commit/v2", {
        "key": key, "expected_version": ver,
        "delta": delta, "agent_id": agent_id,
    })
    return st == 200 and "new_version" in resp

def health_check():
    import socket
    try:
        s = socket.create_connection(("localhost", 7000), timeout=3)
        s.close()
    except:
        return False
    st, _ = _req("GET", "/stats")
    return st == 200

def llm_call(oai, problem, context, agent_id, step):
    rl_acquire()
    result = [None]
    exc = [None]

    def _call():
        try:
            r = oai.chat.completions.create(
                model=BACKBONE, max_tokens=100, temperature=0.3,
                timeout=CALL_TIMEOUT,
                messages=[{"role": "user", "content":
                    f"TASK: {problem[:200]}\n"
                    f"STATE: {context[:350]}\n"
                    f"Add 2 sentences: one concrete technical fix step."}])
            result[0] = r.choices[0].message.content.strip()
        except Exception as e:
            exc[0] = str(e)

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=CALL_TIMEOUT + 5)
    if result[0] is None:
        return f"[{agent_id}_s{step}] TIMEOUT"
    return f"[{agent_id}_s{step}] {result[0]}"

def run_ori_on(oai, task, n_agents, n_steps, run_id):
    shard = f"doc_{run_id}"
    create_shard(shard, task["seed"], task["id"])
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    for a in agents: register_agent(a)

    commits_ok = commits_total = 0
    agent_ids_seen = set()

    for step in range(n_steps):
        snaps = {}
        for i, a in enumerate(agents):
            snaps[i] = read_shard(shard, a)

        deltas = {}
        for i, a in enumerate(agents):
            deltas[i] = llm_call(oai, task["desc"], snaps[i][1], a, step)

        for i, a in enumerate(agents):
            ver = snaps[i][0]
            delta = deltas[i]
            agent_ids_seen.add(a)
            for attempt in range(6):
                ok = do_commit(shard, a, ver, delta)
                commits_total += 1
                if ok:
                    commits_ok += 1
                    break
                ver, _ = read_shard(shard, a)
                time.sleep(0.05 * (attempt + 1))

    n_expected = n_agents * n_steps
    return commits_ok, n_expected, len(agent_ids_seen)


def run_ori_off(oai, task, n_agents, n_steps, run_id):
    shard = f"doc_{run_id}"
    create_shard(shard, task["seed"], task["id"])
    agents = [f"a{i}_{run_id}" for i in range(n_agents)]
    for a in agents: register_agent(a)

    commits_ok = commits_total = 0
    agent_ids_seen = set()

    for step in range(n_steps):
        ver, content = read_shard(shard, agents[0])

        deltas = {}
        for i, a in enumerate(agents):
            deltas[i] = llm_call(oai, task["desc"], content, a, step)

        for i, a in enumerate(agents):
            agent_ids_seen.add(a)
            ok = do_commit(shard, a, ver, deltas[i])
            commits_total += 1
            if ok: commits_ok += 1

    n_expected = n_agents * n_steps
    return commits_ok, n_expected, len(agent_ids_seen)

def load_completed(path):
    done = set()
    if not os.path.exists(path):
        return done
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r.get("task_id",""),
                          r.get("run_idx",""),
                          r.get("condition","")))
    except Exception:
        pass
    return done

_csv_lock = threading.Lock()

def write_row(path, row):
    with _csv_lock:
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists: w.writeheader()
            w.writerow(row)

def run_trial_safe(oai, task, n_agents, n_steps, run_idx, condition):
    run_id = uuid.uuid4().hex[:8]
    result = [None]

    def _run():
        try:
            t0 = time.perf_counter()
            if condition == "parallel_ori_on":
                ok, total, n_agents_seen = run_ori_on(
                    oai, task, n_agents, n_steps, run_id)
            else:
                ok, total, n_agents_seen = run_ori_off(
                    oai, task, n_agents, n_steps, run_id)
            wall = time.perf_counter() - t0
            commit_rate = ok / max(1, total)
            result[0] = {
                "task_id": task["id"],
                "run_idx": run_idx,
                "condition": condition,
                "n_agents": n_agents,
                "n_steps": n_steps,
                "commits_ok": ok,
                "commits_total": total,
                "commit_rate": round(commit_rate, 4),
                "n_agents_seen": n_agents_seen,
                "wall_secs": round(wall, 1),
            }
        except Exception as e:
            result[0] = {"error": str(e)}

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=TRIAL_TIMEOUT)

    if result[0] is None:
        print(f"    TIMEOUT after {TRIAL_TIMEOUT}s — skipping")
        return None
    if "error" in result[0]:
        print(f"    ERROR: {result[0]['error']} — skipping")
        return None
    return result[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks",   type=int, default=10)
    ap.add_argument("--n-runs",    type=int, default=50)
    ap.add_argument("--n-steps",   type=int, default=10)
    ap.add_argument("--n-agents",  type=int, default=4)
    ap.add_argument("--output",    default="results/ori_isolation_v4.csv")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)
    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}"); sys.exit(1)

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                 timeout=CALL_TIMEOUT + 5)
    tasks = TASKS[:args.n_tasks]
    conditions = ["parallel_ori_on", "parallel_ori_off"]
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output)
                else ".", exist_ok=True)

    done = load_completed(args.output)
    total_work = args.n_tasks * args.n_runs * len(conditions)
    skipped = sum(1 for t in tasks for r in range(args.n_runs)
                  for c in conditions
                  if (t["id"], str(r), c) in done)

    print(f"ORI Isolation v2 — {total_work} trials "
          f"({args.n_tasks} tasks × {args.n_runs} runs × 2 conditions)")
    print(f"Resuming: {skipped} already done, {total_work-skipped} remaining")
    print(f"Timeouts: call={CALL_TIMEOUT}s  trial={TRIAL_TIMEOUT}s\n")

    completed = skipped
    ori_on_rates = []
    ori_off_rates = []

    for task in tasks:
        for run_idx in range(args.n_runs):
            for condition in conditions:
                key = (task["id"], str(run_idx), condition)
                if key in done:
                    continue

                row = run_trial_safe(oai, task, args.n_agents,
                                     args.n_steps, run_idx, condition)
                if row:
                    write_row(args.output, row)
                    completed += 1
                    rate = row["commit_rate"]
                    label = "ON " if condition == "parallel_ori_on" else "OFF"
                    print(f"  [{completed:4d}/{total_work}] "
                          f"{task['id'][:18]:18s} run={run_idx:2d} "
                          f"ORI-{label} rate={rate:.3f} "
                          f"({row['commits_ok']}/{row['commits_total']}) "
                          f"{row['wall_secs']:.0f}s")
                    if condition == "parallel_ori_on":
                        ori_on_rates.append(rate)
                    else:
                        ori_off_rates.append(rate)

                    if completed % 20 == 0 and ori_on_rates and ori_off_rates:
                        import statistics
                        print(f"\n  --- Running summary ({completed} done) ---")
                        print(f"  ORI-ON  median rate: {statistics.median(ori_on_rates):.3f}")
                        print(f"  ORI-OFF median rate: {statistics.median(ori_off_rates):.3f}")
                        print()

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    if ori_on_rates and ori_off_rates:
        import statistics
        med_on  = statistics.median(ori_on_rates)
        med_off = statistics.median(ori_off_rates)
        print(f"ORI-ON  commit rate: {med_on:.3f}  (n={len(ori_on_rates)})")
        print(f"ORI-OFF commit rate: {med_off:.3f}  (n={len(ori_off_rates)})")
        print(f"Ratio: {med_on/max(0.001,med_off):.2f}x")
        try:
            from scipy.stats import mannwhitneyu
            _, p = mannwhitneyu(ori_on_rates, ori_off_rates, alternative="greater")
            print(f"Mann-Whitney p={p:.2e} "
                  f"{'***' if p<0.001 else '**' if p<0.01 else '*'}")
        except ImportError:
            pass
    print(f"\nOutput → {args.output}")

if __name__ == "__main__":
    main()
