import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
import tiktoken
from openai import OpenAI
from scipy.stats import mannwhitneyu

SBUS_URL  = "http://localhost:7000"
MODEL     = "gpt-4o-mini-2024-07-18"
TEMP      = 0.2
MAX_TOKS  = 300
ENC       = tiktoken.encoding_for_model("gpt-4o")
CLIENT    = OpenAI()

SWE_TASKS = [
    {
        "id": "django__django-11099",
        "repo": "django/django",
        "description": "HttpResponse doesn't handle memoryview objects. The HttpResponse class should be able to accept memoryview objects as content, similar to how it handles bytes.",
        "shards": [
            {"key": "bug_analysis",  "role": "Bug analyst", "initial": "No analysis yet."},
            {"key": "patch_plan",    "role": "Patch designer", "initial": "No patch plan yet."},
            {"key": "test_strategy", "role": "Test engineer", "initial": "No test strategy yet."},
        ],
        "min_steps": 6,
    },
    {
        "id": "django__django-11133",
        "repo": "django/django",
        "description": "Replacing call to django.utils.encoding.force_text with force_str. The force_text function is deprecated in favour of force_str and should be replaced throughout the codebase.",
        "shards": [
            {"key": "bug_analysis",  "role": "Deprecation analyst", "initial": "No analysis yet."},
            {"key": "patch_plan",    "role": "Migration planner", "initial": "No patch plan yet."},
            {"key": "test_strategy", "role": "Test engineer", "initial": "No test strategy yet."},
        ],
        "min_steps": 5,
    },
    {
        "id": "django__django-12856",
        "repo": "django/django",
        "description": "QuerySet.none() on combined queries returns all results. When a QuerySet has been combined via union(), intersection(), or difference(), calling .none() on the result should return an empty queryset but instead returns all results.",
        "shards": [
            {"key": "bug_analysis",  "role": "ORM analyst", "initial": "No analysis yet."},
            {"key": "patch_plan",    "role": "ORM patch designer", "initial": "No plan yet."},
            {"key": "test_strategy", "role": "Test engineer", "initial": "No strategy yet."},
        ],
        "min_steps": 7,
    },
    {
        "id": "matplotlib__matplotlib-23299",
        "repo": "matplotlib/matplotlib",
        "description": "get_backend() returns wrong backend after set_backend() is called. The get_backend() function should reflect the currently active backend after set_backend() has been called.",
        "shards": [
            {"key": "bug_analysis",  "role": "Backend analyst", "initial": "No analysis yet."},
            {"key": "patch_plan",    "role": "Patch designer", "initial": "No plan yet."},
            {"key": "test_strategy", "role": "Test engineer", "initial": "No strategy yet."},
        ],
        "min_steps": 5,
    },
    {
        "id": "pytest-dev__pytest-7168",
        "repo": "pytest-dev/pytest",
        "description": "pytest --collect-only gives a RecursionError on circular imports. When test collection encounters circular imports in conftest.py files, pytest raises a RecursionError rather than a clean error message.",
        "shards": [
            {"key": "bug_analysis",  "role": "Collection analyst", "initial": "No analysis yet."},
            {"key": "patch_plan",    "role": "Patch designer", "initial": "No plan yet."},
            {"key": "test_strategy", "role": "Test engineer", "initial": "No strategy yet."},
        ],
        "min_steps": 6,
    },
    {
        "id": "astropy__astropy-12907",
        "repo": "astropy/astropy",
        "description": "Modeling: Fit with weights gives wrong result when using compound models with fixed parameters. When fitting compound models with some parameters fixed, the weighted residuals are computed incorrectly.",
        "shards": [
            {"key": "bug_analysis",  "role": "Numerical analyst", "initial": "No analysis yet."},
            {"key": "patch_plan",    "role": "Fitting patch designer", "initial": "No plan yet."},
            {"key": "test_strategy", "role": "Test engineer", "initial": "No strategy yet."},
        ],
        "min_steps": 7,
    },
    {
        "id": "sympy__sympy-21379",
        "repo": "sympy/sympy",
        "description": "Unexpected TypeError when using subs with a Piecewise and Relational. Substituting into a Piecewise expression that contains Relational conditions raises a TypeError instead of returning the expected simplified expression.",
        "shards": [
            {"key": "bug_analysis",  "role": "CAS analyst", "initial": "No analysis yet."},
            {"key": "patch_plan",    "role": "Patch designer", "initial": "No plan yet."},
            {"key": "test_strategy", "role": "Test engineer", "initial": "No strategy yet."},
        ],
        "min_steps": 6,
    },
    {
        "id": "sphinx-doc__sphinx-8435",
        "repo": "sphinx-doc/sphinx",
        "description": "autodoc: Class attributes with no docstring are shown as 'None'. When autodoc processes class attributes that have no docstring, they appear as 'None' in the generated documentation rather than being omitted.",
        "shards": [
            {"key": "bug_analysis",  "role": "Autodoc analyst", "initial": "No analysis yet."},
            {"key": "patch_plan",    "role": "Patch designer", "initial": "No plan yet."},
            {"key": "test_strategy", "role": "Test engineer", "initial": "No strategy yet."},
        ],
        "min_steps": 5,
    },
    {
        "id": "scikit-learn__scikit-learn-13779",
        "repo": "scikit-learn/scikit-learn",
        "description": "GridSearchCV with refit=False raises AttributeError when accessing best_params_. After fitting GridSearchCV with refit=False, accessing .best_params_ raises AttributeError rather than returning the best parameter set found.",
        "shards": [
            {"key": "bug_analysis",  "role": "ML analyst", "initial": "No analysis yet."},
            {"key": "patch_plan",    "role": "Patch designer", "initial": "No plan yet."},
            {"key": "test_strategy", "role": "Test engineer", "initial": "No strategy yet."},
        ],
        "min_steps": 5,
    },
    {
        "id": "pallets__flask-4045",
        "repo": "pallets/flask",
        "description": "Flask doesn't handle HEAD requests properly for routes with only GET defined. When a route defines only GET, Flask should automatically support HEAD requests by returning the GET response with an empty body, but currently returns 405 Method Not Allowed.",
        "shards": [
            {"key": "bug_analysis",  "role": "HTTP analyst", "initial": "No analysis yet."},
            {"key": "patch_plan",    "role": "Routing patch designer", "initial": "No plan yet."},
            {"key": "test_strategy", "role": "Test engineer", "initial": "No strategy yet."},
        ],
        "min_steps": 5,
    },
]

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def llm_call(system: str, user: str) -> tuple[str, int, int]:
    resp = CLIENT.chat.completions.create(
        model=MODEL, temperature=TEMP, max_tokens=MAX_TOKS,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
    )
    text = resp.choices[0].message.content or ""
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


def sbus_create(key: str, content: str, goal_tag: str):
    httpx.post(f"{SBUS_URL}/shard",
               json={"key": key, "content": content, "goal_tag": goal_tag},
               timeout=10).raise_for_status()


def sbus_read(key: str) -> dict:
    return httpx.get(f"{SBUS_URL}/shard/{key}", timeout=10).json()


def sbus_commit(key: str, version: int, delta: str, agent_id: str) -> bool:
    r = httpx.post(f"{SBUS_URL}/commit",
                   json={"key": key, "expected_version": version,
                         "delta": delta, "agent_id": agent_id},
                   timeout=15)
    return r.status_code == 200

@dataclass
class RunResult:
    run_id:       str
    system:       str
    task_id:      str
    n_agents:     int
    n_steps:      int
    coord_tokens: int
    work_tokens:  int
    cwr:          float
    wall_time_s:  float

def run_sbus(task: dict, n_agents: int, n_steps: int) -> RunResult:
    suffix       = f"{task['id']}_{n_agents}a_{int(time.time())}"
    shards       = task["shards"][:n_agents]
    coord_tokens = 0
    work_tokens  = 0

    for s in shards:
        sbus_create(f"{suffix}_{s['key']}", s["initial"], s["key"])

    def agent_step(idx: int, step: int):
        nonlocal coord_tokens, work_tokens
        shard_def = shards[idx % len(shards)]
        key = f"{suffix}_{shard_def['key']}"

        data = sbus_read(key)
        content = data["content"]
        version = data["version"]
        c_read  = count_tokens(content)

        delta, pt, ct = llm_call(
            f"You are a {shard_def['role']} working on a bug fix.",
            f"Issue: {task['description']}\n\n"
            f"Current {shard_def['key']}:\n{content}\n\n"
            f"Step {step}: Advance this. Be specific. 150 words max.",
        )

        for _ in range(3):
            if sbus_commit(key, version, delta, f"agent-{idx}"):
                break
            data    = sbus_read(key)
            version = data["version"]
            c_read += count_tokens(data["content"])

        coord_tokens += c_read
        work_tokens  += pt + ct

    t0 = time.time()
    step_budget = max(n_steps, task["min_steps"] + 2)
    for step in range(step_budget):
        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            list(pool.map(lambda i: agent_step(i, step), range(n_agents)))
    wall = time.time() - t0

    cwr = coord_tokens / max(work_tokens, 1)
    return RunResult(
        run_id=suffix, system="sbus", task_id=task["id"],
        n_agents=n_agents, n_steps=step_budget,
        coord_tokens=coord_tokens, work_tokens=work_tokens,
        cwr=round(cwr, 4), wall_time_s=round(wall, 1),
    )

def run_coordinator_worker(task: dict, n_agents: int, n_steps: int) -> RunResult:
    suffix       = f"cw_{task['id']}_{n_agents}a_{int(time.time())}"
    coord_tokens = 0
    work_tokens  = 0
    shared_ctx   = f"Task: {task['description']}"
    step_budget  = max(n_steps, task["min_steps"] + 2)

    t0 = time.time()
    for step in range(step_budget):
        worker_outputs = []
        for i in range(n_agents):
            ctx_read = count_tokens(shared_ctx)
            coord_tokens += ctx_read

            output, pt, ct = llm_call(
                f"You are specialist {i} on a software bug fix.",
                f"Context:\n{shared_ctx}\n\nStep {step}: Your contribution.",
            )
            work_tokens  += pt + ct
            worker_outputs.append(output)

        all_outputs  = "\n\n".join(f"Agent {i}: {o}" for i, o in enumerate(worker_outputs))
        summary, pt, ct = llm_call(
            "You are a coordinator. Summarise the agents' contributions concisely.",
            f"Task: {task['description']}\n\nOutputs:\n{all_outputs}",
        )
        coord_tokens += pt + ct
        shared_ctx    = summary

    wall = time.time() - t0
    cwr  = coord_tokens / max(work_tokens, 1)
    return RunResult(
        run_id=suffix, system="coordinator_worker",
        task_id=task["id"], n_agents=n_agents, n_steps=step_budget,
        coord_tokens=coord_tokens, work_tokens=work_tokens,
        cwr=round(cwr, 4), wall_time_s=round(wall, 1),
    )

def analyse(results: list[RunResult]):
    sbus_cwr = [r.cwr for r in results if r.system == "sbus"]
    cw_cwr   = [r.cwr for r in results if r.system == "coordinator_worker"]

    sbus_mean = sum(sbus_cwr) / len(sbus_cwr) if sbus_cwr else 0
    cw_mean   = sum(cw_cwr)   / len(cw_cwr)   if cw_cwr   else 0
    reduction = (cw_mean - sbus_mean) / cw_mean if cw_mean > 0 else 0

    print("\n" + "="*60)
    print("SWE-bench Lite CWR Results")
    print("="*60)
    print(f"  S-Bus mean CWR              : {sbus_mean:.3f}  (n={len(sbus_cwr)})")
    print(f"  Coordinator-Worker mean CWR : {cw_mean:.3f}  (n={len(cw_cwr)})")
    print(f"  CWR reduction               : {reduction:.1%}")

    if sbus_cwr and cw_cwr:
        stat, p = mannwhitneyu(sbus_cwr, cw_cwr, alternative="less")
        n1, n2  = len(sbus_cwr), len(cw_cwr)
        r       = 1 - (2 * stat) / (n1 * n2)
        sig     = "***" if p < 0.0001 else "**" if p < 0.001 else "*" if p < 0.05 else "ns"
        print(f"\n  Mann-Whitney U={stat:.0f}  p={p:.4f}  r={r:.3f}  {sig}")
        print("  (r=1.0 = complete separation; every S-Bus obs < every CW obs)")

    print("\n  Per-task breakdown:")
    print(f"  {'Task ID':<40} {'S-Bus CWR':>10} {'CW CWR':>10} {'Reduction':>10}")
    print("  " + "-"*72)
    task_ids = sorted(set(r.task_id for r in results))
    for tid in task_ids:
        s = next((r.cwr for r in results if r.task_id == tid and r.system == "sbus"), None)
        c = next((r.cwr for r in results if r.task_id == tid and r.system == "coordinator_worker"), None)
        if s and c:
            red = (c - s) / c
            print(f"  {tid:<40} {s:>10.3f} {c:>10.3f} {red:>9.1%}")

def write_csv(results: list[RunResult], path: str):
    fields = ["run_id", "system", "task_id", "n_agents", "n_steps",
              "coord_tokens", "work_tokens", "cwr", "wall_time_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "run_id": r.run_id, "system": r.system, "task_id": r.task_id,
                "n_agents": r.n_agents, "n_steps": r.n_steps,
                "coord_tokens": r.coord_tokens, "work_tokens": r.work_tokens,
                "cwr": r.cwr, "wall_time_s": r.wall_time_s,
            })
    print(f"\nCSV written to {path}")

def main():
    ap = argparse.ArgumentParser(description="SWE-bench Lite CWR experiment")
    ap.add_argument("--agents", type=int, nargs="+", default=[4])
    ap.add_argument("--tasks",  type=int, default=10, help="Number of SWE tasks (max 10)")
    ap.add_argument("--steps",  type=int, default=8,  help="Steps per agent per task")
    ap.add_argument("--out",    default="results/swebenches.csv")
    args = ap.parse_args()

    try:
        httpx.get(f"{SBUS_URL}/stats", timeout=3).raise_for_status()
    except Exception:
        print(f"ERROR: Cannot reach S-Bus server at {SBUS_URL}")
        sys.exit(1)

    tasks = SWE_TASKS[:args.tasks]
    os.makedirs("results", exist_ok=True)
    results: list[RunResult] = []

    for n in args.agents:
        for task in tasks:
            print(f"\nTask: {task['id']} | N={n}")

            print("  Running S-Bus...")
            r_sbus = run_sbus(task, n, args.steps)
            results.append(r_sbus)
            print(f"  S-Bus CWR={r_sbus.cwr:.3f}  wall={r_sbus.wall_time_s}s")

            print("  Running Coordinator-Worker baseline...")
            r_cw = run_coordinator_worker(task, n, args.steps)
            results.append(r_cw)
            print(f"  CW CWR={r_cw.cwr:.3f}  wall={r_cw.wall_time_s}s")

    analyse(results)
    write_csv(results, args.out)


if __name__ == "__main__":
    main()
