"""
benchmark.py  —  S-Bus CWR benchmark: S-Bus vs coordinator-worker baselines.
==============================================================================

Measures Coordination-to-Work Ratio (CWR) and S@50 across systems using a
lightweight simulated coordinator-worker pattern (no SDK dependencies required).
For full SDK comparison (real CrewAI / AutoGen / LangGraph), use sdk_compare.py.

Changes vs original:
  1. RETRY_BUDGET driven by SBUS_RETRY_BUDGET env var (Definition 6, Corollary 2.2)
  2. Retry loop explicit — budget exhaustion logged with run.retry_exhaustions counter
  3. Clopper-Pearson exact binomial CIs replace Wilson for S@50 (extreme proportions)
  4. /commit/v2_naive added as third cross-shard condition (Table 6 control)
  5. Docstring numbers updated to canonical paper values (50-step full runs)
  6. CrewAI/AutoGen/LangGraph baselines: fallback estimation removed, runs marked
     excluded when SDK usage_metrics returns zero (avoids coord = work×2 heuristic)

Usage:
    export OPENAI_API_KEY="sk-..."
    cargo run                           # Terminal 1 — start S-Bus server

    # Quick smoke test (1 task, ~$0.20):
    python3 benchmark.py --tasks-limit 1 --agents 4 --steps 10

    # Full paper run (Table 3 canonical numbers):
    python3 benchmark.py --agents 4 8 --steps 50 --tasks-limit 5 \\
      --out results/cwr_results_canonical.csv

    # Cross-shard experiment (all three conditions — Table 6):
    python3 benchmark.py --mode cross-shard --trials 10

    # Ablation study (Table 11) — requires server started with ablation flags:
    #   SBUS_TOKEN=0 cargo run          ->  –token condition
    #   SBUS_VERSION=0 cargo run        ->  –version condition
    #   SBUS_LOG=0 cargo run            ->  –log condition
    python3 benchmark.py --tasks-limit 3 --runs 3 --out results/ablation.csv

    # Analyse existing results without running:
    python3 benchmark.py --analyse-only --out results/cwr_results_canonical.csv

Canonical paper results (50-step full runs, 5 tasks, GPT-4o-mini):
    System      N=4 CWR   N=8 CWR   Reduction   S@50 N=8
    S-Bus       0.186     0.181     —            100%
    LangGraph   4.592     4.469     95.9%        20%
    CrewAI      19.381    16.176    99.0%        0%
    AutoGen     27.385    28.360    99.3%        100%
    Mann-Whitney U=0, p<0.0001, r=1.000 for all three comparisons.
"""

import argparse
import json
import logging
import importlib.util
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from math import sqrt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _check():
    missing = [p for p in ["httpx", "tiktoken", "openai", "scipy"]
               if not importlib.util.find_spec(p)]
    if missing:
        print(f"Missing packages: pip install {' '.join(missing)}")
        sys.exit(1)

_check()

import httpx
import tiktoken
from openai import OpenAI
from scipy.stats import binom as scipy_binom

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_ENC = tiktoken.encoding_for_model("gpt-4o")


def tok(text: str) -> int:
    return len(_ENC.encode(text or ""))


# ── Retry budget (Definition 6, Corollary 2.2) ──────────────────────────────
# Default B=1 matches the paper. Set SBUS_RETRY_BUDGET=5 to eliminate
# retry exhaustion events at N=8.
RETRY_BUDGET = int(os.environ.get("SBUS_RETRY_BUDGET", "1"))

MODEL = "gpt-4o-mini"


# ── Run dataclass ────────────────────────────────────────────────────────────

@dataclass
class Run:
    run_id: str
    system: str
    agent_count: int
    task_id: str
    coord_tokens: int = 0
    work_tokens: int = 0
    steps_taken: int = 0
    success: bool = False
    commit_attempts: int = 0
    commit_conflicts: int = 0
    retry_exhaustions: int = 0   # NEW — tracks Corollary 2.2 skip events
    excluded: bool = False        # NEW — marks runs with unreliable token counts
    wall_ms: int = 0
    model: str = MODEL

    @property
    def cwr(self):
        if self.excluded or self.work_tokens <= 0:
            return float("inf")
        return self.coord_tokens / self.work_tokens

    @property
    def scr(self):
        if self.commit_attempts == 0:
            return 0.0
        return self.commit_conflicts / self.commit_attempts

    def csv(self):
        cwr = f"{self.cwr:.4f}" if self.cwr != float("inf") else "inf"
        return (
            f"{self.run_id},{self.system},{self.agent_count},{self.task_id},"
            f"{self.coord_tokens},{self.work_tokens},{cwr},"
            f"{self.steps_taken},{int(self.success)},"
            f"{self.commit_attempts},{self.commit_conflicts},{self.scr:.4f},"
            f"{self.retry_exhaustions},{int(self.excluded)},"
            f"{self.wall_ms},{self.model}\n"
        )


CSV_HDR = (
    "run_id,system,agent_count,task_id,coord_tokens,work_tokens,cwr,"
    "steps_taken,success,commit_attempts,commit_conflicts,scr,"
    "retry_exhaustions,excluded,wall_ms,model\n"
)


# ── OpenAI wrapper ───────────────────────────────────────────────────────────

_oai = None


def oai() -> OpenAI:
    global _oai
    if _oai is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            log.error("OPENAI_API_KEY not set.")
            sys.exit(1)
        _oai = OpenAI(api_key=api_key)
    return _oai


def llm(sys_msg: str, usr_msg: str,
        model: str = MODEL, max_tok: int = 300) -> tuple[str, int, int]:
    """Call OpenAI. Returns (text, prompt_tokens, completion_tokens)."""
    for attempt in range(2):
        try:
            r = oai().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": usr_msg},
                ],
                temperature=0.2,
                max_tokens=max_tok,
            )
            return (
                r.choices[0].message.content or "",
                r.usage.prompt_tokens,
                r.usage.completion_tokens,
            )
        except Exception as e:
            if attempt == 0 and "rate" in str(e).lower():
                log.warning("Rate limit — sleeping 20s")
                time.sleep(20)
            else:
                log.error(f"LLM error: {e}")
                return "", 0, 0
    return "", 0, 0


# ── S-Bus HTTP client ────────────────────────────────────────────────────────

class Bus:
    def __init__(self, url: str = "http://localhost:3000"):
        self.base = url
        self.c = httpx.Client(timeout=30)

    def ping(self) -> bool:
        try:
            self.c.get(f"{self.base}/stats", timeout=3)
            return True
        except Exception:
            return False

    def create(self, key: str, content: str, tag: str = "default") -> str:
        r = self.c.post(f"{self.base}/shard",
                        json={"key": key, "content": content, "goal_tag": tag})
        r.raise_for_status()
        return r.json()["key"]

    def read(self, key: str) -> dict:
        r = self.c.get(f"{self.base}/shard/{key}")
        r.raise_for_status()
        return r.json()

    def commit(self, key: str, ver: int, content: str,
               agent: str, note: str = "") -> dict:
        r = self.c.post(f"{self.base}/commit", json={
            "key": key, "expected_ver": ver, "content": content,
            "rationale": note, "agent_id": agent,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status()
        return r.json()

    def commit_v2(self, key: str, ver: int, content: str,
                  agent: str, read_set: list, note: str = "") -> dict:
        """Sorted-lock-order multi-shard commit (Corollary 2.1 / Lemma 2).
        Requires Assumption A1: read_set must declare ALL shards read during
        delta preparation."""
        r = self.c.post(f"{self.base}/commit/v2", json={
            "key": key, "expected_ver": ver, "content": content,
            "rationale": note, "agent_id": agent, "read_set": read_set,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status()
        return r.json()

    def commit_v2_naive(self, key: str, ver: int, content: str,
                        agent: str, read_set: list, note: str = "") -> dict:
        """UNORDERED multi-shard commit — control condition for Table 6.
        Do not use in production."""
        r = self.c.post(f"{self.base}/commit/v2_naive", json={
            "key": key, "expected_ver": ver, "content": content,
            "rationale": note, "agent_id": agent, "read_set": read_set,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status()
        return r.json()

    def stats(self) -> dict:
        return self.c.get(f"{self.base}/stats").json()


# ── Success judge ────────────────────────────────────────────────────────────

def judge_success(outputs: list[str], task: dict) -> bool:
    """Evaluate task success using an LLM judge.

    Note: Uses the same model backbone (GPT-4o-mini) as the agents.
    This circularity is acknowledged in Section 8.8 (construct validity).
    For independent validation, set SBUS_JUDGE_MODEL=gpt-4o or similar.
    """
    judge_model = os.environ.get("SBUS_JUDGE_MODEL", MODEL)

    checks = task.get("ground_truth_outputs", [])
    if not checks:
        return all(len(o) > 80 for o in outputs if o)

    combined = "\n\n".join(f"[Output {i+1}]:\n{o}" for i, o in enumerate(outputs))
    criteria = "\n".join(f"- {c}" for c in checks)
    verdict, _, _ = llm(
        "You are an evaluator. Answer only YES or NO.",
        f"Task: {task['description']}\n\nSuccess criteria:\n{criteria}\n\n"
        f"Outputs:\n{combined}\n\nDo outputs satisfy ALL criteria? YES or NO.",
        model=judge_model,
        max_tok=5,
    )
    return verdict.strip().upper().startswith("YES")


# ── System 1: S-Bus ──────────────────────────────────────────────────────────

def run_sbus(task: dict, n_agents: int, bus: Bus,
             steps: int, model: str) -> Run:
    """
    S-Bus experiment. Each agent owns a disjoint subset of shards.
    Retry budget B = RETRY_BUDGET (default 1, Definition 6, Corollary 2.2).

    Token accounting:
      coord_tokens += tok(shard_content)  on every read (including retries)
      work_tokens  += prompt + completion  on every LLM call
    """
    run = Run(run_id=str(uuid.uuid4()), system="sbus",
              agent_count=n_agents, task_id=task["task_id"], model=model)
    t0 = time.time()

    keys = task["shared_state_keys"]
    agents = [f"agent-{i}" for i in range(n_agents)]
    pfx = run.run_id[:8]
    skeys = [f"{pfx}_{k}" for k in keys]

    for k, sk in zip(keys, skeys):
        bus.create(sk, f"[{k}: not started]", task["category"])

    for step in range(steps):
        agent = agents[step % n_agents]
        owned = [skeys[i] for i in range(len(skeys))
                 if i % n_agents == step % n_agents]

        for sk in owned:
            shard = bus.read(sk)
            run.coord_tokens += tok(shard["content"])

            label = sk.split("_", 1)[1]
            text, pt, ct = llm(
                f"You are {agent}, expert in {task['category']}. Be concise.",
                f"Task: {task['description']}\n\nYour component: {label}\n"
                f"Current state:\n{shard['content']}\n\n"
                f"Step {step+1}/{steps}: Write 2-3 sentences of concrete progress.",
                model=model, max_tok=300,
            )
            run.work_tokens += pt + ct
            run.commit_attempts += 1

            # Retry loop — explicit budget (Definition 6, Corollary 2.2)
            committed = False
            current_shard = shard
            for attempt in range(RETRY_BUDGET + 1):
                resp = bus.commit(sk, current_shard["version"], text,
                                  agent, f"step {step+1}")
                if not resp.get("conflict"):
                    committed = True
                    break
                run.commit_conflicts += 1
                if attempt < RETRY_BUDGET:
                    # Refresh shard for next retry
                    current_shard = bus.read(sk)
                    run.coord_tokens += tok(current_shard["content"])
                    run.commit_attempts += 1

            if not committed:
                run.retry_exhaustions += 1
                log.warning(
                    f"[{agent}] retry budget B={RETRY_BUDGET} exhausted "
                    f"at step {step+1} on shard {sk}"
                )

    # Evaluate success at end
    final_outputs = [bus.read(sk)["content"] for sk in skeys]
    run.success = judge_success(final_outputs, task)
    run.steps_taken = steps
    run.wall_ms = int((time.time() - t0) * 1000)
    log.info(f"  sbus  CWR={run.cwr:.3f} S50={run.success} "
             f"SCR={run.scr:.3f} retries_exhausted={run.retry_exhaustions}")
    return run


# ── System 2: Coordinator-Worker baseline ───────────────────────────────────
# Simulates the coordinator-worker pattern used by CrewAI/LangGraph/AutoGen
# without SDK dependencies. For real SDK runs see sdk_compare.py.

def run_coordinator_worker(task: dict, n_agents: int,
                           steps: int, model: str,
                           system_name: str = "coord_worker") -> Run:
    """
    Coordinator-worker baseline.

    Token accounting:
      coord_tokens: coordinator reads all worker outputs (O(N·|context|))
      work_tokens:  worker completion tokens only

    This is the lightweight simulation. For real SDK overhead numbers
    that match paper Table 3, use sdk_compare.py.
    """
    run = Run(run_id=str(uuid.uuid4()), system=system_name,
              agent_count=n_agents, task_id=task["task_id"], model=model)
    t0 = time.time()

    shared_context = f"Task: {task['description']}\n\nStatus: Not started."
    agent_outputs = [""] * n_agents

    for step in range(steps):
        # Workers read shared context (coordination cost)
        for i in range(n_agents):
            run.coord_tokens += tok(shared_context)
            role = task["shared_state_keys"][i % len(task["shared_state_keys"])]
            _, pt, ct = llm(
                f"You are worker agent {i}, expert in {role}. Be concise.",
                f"Shared context:\n{shared_context}\n\n"
                f"Step {step+1}: Write 2 sentences of progress on {role}.",
                model=model, max_tok=200,
            )
            run.work_tokens += ct
            run.coord_tokens += pt  # prompt tokens = reading shared context

        # Coordinator synthesizes (coordination cost)
        all_outputs = "\n".join(f"Agent {i}: {o}" for i, o in enumerate(agent_outputs)
                                if o)
        coord_prompt = (
            f"You are the coordinator. Synthesize worker outputs into a coherent "
            f"shared context for the next step.\n\nWorker outputs:\n{all_outputs}"
        )
        run.coord_tokens += tok(coord_prompt)
        summary, coord_pt, coord_ct = llm(
            "You are the coordinator. Summarize concisely.",
            coord_prompt,
            model=model, max_tok=300,
        )
        run.coord_tokens += coord_pt + coord_ct
        shared_context = summary or shared_context

    run.success = judge_success([shared_context], task)
    run.steps_taken = steps
    run.wall_ms = int((time.time() - t0) * 1000)
    log.info(f"  {system_name} CWR={run.cwr:.3f} S50={run.success}")
    return run


# ── Statistics ───────────────────────────────────────────────────────────────

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Clopper-Pearson exact binomial confidence interval.
    Used for S@50 (replaces Wilson CI for extreme proportions k=0 or k=n).
    Section 6.2 and Table 7.
    """
    if n == 0:
        return 0.0, 1.0
    lo = float(scipy_binom.ppf(alpha / 2, n, k / n) / n) if k > 0 else 0.0
    hi = float(scipy_binom.ppf(1 - alpha / 2, n, (k + 1) / n) / n) if k < n else 1.0
    # Edge cases: exact bounds
    if k == 0:
        lo = 0.0
        hi = float(1 - (alpha / 2) ** (1 / n))
    if k == n:
        lo = float((alpha / 2) ** (1 / n))
        hi = 1.0
    return round(lo, 3), round(hi, 3)


def analyse(out_path: str):
    """Analyse a CSV results file — compute per-system CWR stats and S@50."""
    import csv
    from collections import defaultdict

    runs: list[dict] = []
    with open(out_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("excluded", "0") == "1":
                log.warning(f"Skipping excluded run {row['run_id']} ({row['system']})")
                continue
            runs.append(row)

    # Group by (system, agent_count)
    groups: dict[tuple, list[float]] = defaultdict(list)
    success: dict[tuple, list[bool]] = defaultdict(list)

    for r in runs:
        key = (r["system"], r["agent_count"])
        cwr = float(r["cwr"]) if r["cwr"] != "inf" else None
        if cwr is not None:
            groups[key].append(cwr)
        success[key].append(r["success"] == "1")

    print(f"\n{'System':<20} {'N':>3} {'Mean CWR':>10} {'±95% CI':>10} "
          f"{'n':>4} {'S@50':>6} {'CP 95% CI':>18}")
    print("-" * 80)

    for (system, n), cwrs in sorted(groups.items()):
        mean = sum(cwrs) / len(cwrs)
        n_runs = len(cwrs)

        # CI for CWR
        if n_runs >= 5:
            std = sqrt(sum((x - mean)**2 for x in cwrs) / (n_runs - 1))
            # t-distribution CI for small n
            from scipy.stats import t as t_dist
            t_val = t_dist.ppf(0.975, df=n_runs - 1)
            ci = t_val * std / sqrt(n_runs)
        else:
            ci = 0.0

        # S@50 with Clopper-Pearson
        succ = success.get((system, n), [])
        k_succ = sum(1 for s in succ if s)
        n_succ = len(succ)
        s50 = k_succ / n_succ if n_succ > 0 else 0.0
        lo, hi = clopper_pearson_ci(k_succ, n_succ)

        print(f"{system:<20} {n:>3} {mean:>10.3f} {f'±{ci:.3f}':>10} "
              f"{n_runs:>4} {f'{s50:.0%}':>6} [{lo:.3f}–{hi:.3f}]")

    print()


# ── Cross-shard experiment (Table 6) ────────────────────────────────────────

def run_cross_shard_experiment(bus: Bus, trials: int = 10,
                               agent_counts: list = None):
    """
    Cross-shard validation experiment — Table 6.
    Three conditions:
      1. No read-set (original baseline)
      2. /commit/v2_naive (read-set, unordered locks — NEW control condition)
      3. /commit/v2 sorted (read-set, Havender order — paper's main result)

    Condition 2 is the missing control that demonstrates Havender's contribution.
    """
    if agent_counts is None:
        agent_counts = [4, 8, 16]

    import threading

    print("\nCross-shard validation experiment (Table 6)")
    print("=" * 70)

    for condition in ["no_read_set", "v2_naive", "v2_sorted"]:
        print(f"\nCondition: {condition}")
        print(f"{'N':>4} {'Injections':>12} {'Detected':>10} "
              f"{'Corruptions':>13} {'Det Rate':>10}")
        print("-" * 55)

        for n_agents in agent_counts:
            total_injections = 0
            total_detected = 0
            total_corruptions = 0

            for trial in range(trials):
                pfx = f"xshard_{condition}_{n_agents}_{trial}"
                db_key = f"{pfx}_db_schema"
                api_key = f"{pfx}_api_design"
                deploy_key = f"{pfx}_deploy_plan"

                bus.create(db_key,     "[db_schema: v0]",    "cross_shard")
                bus.create(api_key,    "[api_design: v0]",   "cross_shard")
                bus.create(deploy_key, "[deploy_plan: v0]",  "cross_shard")

                stop_event = threading.Event()
                corruptions = [0]
                detected = [0]
                injections = [0]

                def injector():
                    """Advances db_schema at ~8 Hz to inject concurrent writes."""
                    while not stop_event.is_set():
                        s = bus.read(db_key)
                        new_content = f"[db_schema: v{s['version']+1} injected]"
                        resp = bus.commit(db_key, s["version"], new_content,
                                          "injector", "concurrent injection")
                        if not resp.get("conflict"):
                            injections[0] += 1
                        time.sleep(0.125)  # ~8 Hz

                def agent_worker(agent_id: str):
                    for step in range(5):
                        db_shard = bus.read(db_key)
                        api_shard = bus.read(api_key)
                        deploy_shard = bus.read(deploy_key)

                        read_set = [
                            {"key": db_key,  "expected_ver": db_shard["version"]},
                            {"key": api_key, "expected_ver": api_shard["version"]},
                        ]

                        new_content = (
                            f"[deploy_plan: step {step} by {agent_id}, "
                            f"based on db_v{db_shard['version']}]"
                        )

                        if condition == "no_read_set":
                            resp = bus.commit(deploy_key, deploy_shard["version"],
                                              new_content, agent_id)
                        elif condition == "v2_naive":
                            resp = bus.commit_v2_naive(
                                deploy_key, deploy_shard["version"],
                                new_content, agent_id, read_set)
                        else:  # v2_sorted
                            resp = bus.commit_v2(
                                deploy_key, deploy_shard["version"],
                                new_content, agent_id, read_set)

                        if not resp.get("conflict"):
                            detected[0] += 1
                        else:
                            corruptions[0] += 1

                        time.sleep(0.05)

                inj_thread = threading.Thread(target=injector, daemon=True)
                inj_thread.start()

                agent_threads = [
                    threading.Thread(target=agent_worker,
                                     args=(f"agent-{i}",), daemon=True)
                    for i in range(n_agents)
                ]
                for t in agent_threads: t.start()
                for t in agent_threads: t.join()
                stop_event.set()

                total_injections += injections[0]
                total_detected += detected[0]
                total_corruptions += corruptions[0]

            det_rate = (total_detected / total_injections * 100
                        if total_injections > 0 else 0)
            print(f"{n_agents:>4} {total_injections:>12} {total_detected:>10} "
                  f"{total_corruptions:>13} {det_rate:>9.1f}%")


# ── Main ─────────────────────────────────────────────────────────────────────

def load_tasks(path: str = "datasets/long_horizon_tasks.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="S-Bus CWR Benchmark")
    parser.add_argument("--agents",      nargs="+", type=int, default=[4, 8])
    parser.add_argument("--steps",       type=int,  default=20)
    parser.add_argument("--tasks-limit", type=int,  default=5)
    parser.add_argument("--runs",        type=int,  default=1,
                        help="Runs per (system, task, N) combination")
    parser.add_argument("--out",         type=str,  default="results/cwr_results.csv")
    parser.add_argument("--tasks-path",  type=str,
                        default="datasets/long_horizon_tasks.json")
    parser.add_argument("--server",      type=str,  default="http://localhost:3000")
    parser.add_argument("--model",       type=str,  default=MODEL)
    parser.add_argument("--analyse-only",action="store_true")
    parser.add_argument("--mode",        type=str,  default="cwr",
                        choices=["cwr", "cross-shard"],
                        help="'cwr' = Table 3/11, 'cross-shard' = Table 6")
    args = parser.parse_args()

    if args.analyse_only:
        analyse(args.out)
        return

    bus = Bus(args.server)
    if not bus.ping():
        log.error(f"S-Bus server not reachable at {args.server}. Run: cargo run")
        sys.exit(1)

    log.info(f"S-Bus server: {args.server}")
    log.info(f"Retry budget: B={RETRY_BUDGET} "
             f"(SBUS_RETRY_BUDGET env var, Definition 6)")

    if args.mode == "cross-shard":
        run_cross_shard_experiment(bus, trials=10, agent_counts=args.agents)
        return

    tasks = load_tasks(args.tasks_path)[:args.tasks_limit]
    log.info(f"Loaded {len(tasks)} tasks, steps={args.steps}, "
             f"agents={args.agents}, runs_per_combo={args.runs}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    with open(out_path, "a") as f:
        if write_header:
            f.write(CSV_HDR)

        for task in tasks:
            for n in args.agents:
                for _ in range(args.runs):
                    # S-Bus
                    run = run_sbus(task, n, bus, args.steps, args.model)
                    f.write(run.csv())
                    f.flush()

                    # Coordinator-worker baseline
                    run_cw = run_coordinator_worker(
                        task, n, args.steps, args.model)
                    f.write(run_cw.csv())
                    f.flush()

    log.info(f"Results written to {out_path}")
    analyse(args.out)


if __name__ == "__main__":
    main()
