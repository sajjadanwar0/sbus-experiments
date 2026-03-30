"""
benchmark.py
==============
CWR benchmark: S-Bus vs coordinator-worker baselines.

Measures Coordination-to-Work Ratio (CWR) and S@50 across systems.
All baselines model the coordinator-worker pattern used by CrewAI,
AutoGen, and LangGraph.

Usage:
    export OPENAI_API_KEY=""
    cargo run                          # Terminal 1 — start server

    # Quick test (1 task, 4 agents, ~$0.20):
    python3 harness/exp.py --tasks-limit 1 --agents 4 --steps 10

    # Full paper run (5 tasks, 4+8 agents, ~$5):
    python3 harness/exp.py --all --agents 4 8 --steps 20

    # Analyze existing results:
    python3 benchmark.py --analyze-only --out results/cwr_results.csv
"""

import argparse
import json
import logging
import importlib.util
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────
def _check():
    missing = [p for p in ["httpx", "tiktoken", "openai"]
               if not importlib.util.find_spec(p)]
    if missing:
        print(f"Missing packages: pip install {' '.join(missing)}")
        sys.exit(1)

_check()

import httpx
import tiktoken
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_ENC = tiktoken.encoding_for_model("gpt-4o")


def tok(text: str) -> int:
    return len(_ENC.encode(text or ""))


# ── Run data class ────────────────────────────────────────────────────────────

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
    wall_ms: int = 0
    model: str = "gpt-4o-mini"

    @property
    def cwr(self):
        return self.coord_tokens / self.work_tokens if self.work_tokens else float("inf")

    @property
    def scr(self):
        return self.commit_conflicts / self.commit_attempts if self.commit_attempts else 0.0

    def csv(self):
        cwr = f"{self.cwr:.4f}" if self.cwr != float("inf") else "inf"
        return (
            f"{self.run_id},{self.system},{self.agent_count},{self.task_id},"
            f"{self.coord_tokens},{self.work_tokens},{cwr},"
            f"{self.steps_taken},{int(self.success)},"
            f"{self.commit_attempts},{self.commit_conflicts},{self.scr:.4f},"
            f"{self.wall_ms},{self.model}\n"
        )


CSV_HDR = (
    "run_id,system,agent_count,task_id,coord_tokens,work_tokens,cwr,"
    "steps_taken,success,commit_attempts,commit_conflicts,scr,wall_ms,model\n"
)


# ── OpenAI wrapper ────────────────────────────────────────────────────────────

_oai = None


def oai() -> OpenAI:
    global _oai
    if _oai is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            log.error("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='sk-...'")
            sys.exit(1)
        _oai = OpenAI(api_key=api_key)
    return _oai


def llm(sys_msg: str, usr_msg: str,
        model: str = "gpt-4o-mini",
        max_tok: int = 400) -> tuple[str, int, int]:
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


# ── S-Bus HTTP client ─────────────────────────────────────────────────────────

class Bus:
    def __init__(self, url: str = "http://localhost:3000"):
        self.base = url
        self.c    = httpx.Client(timeout=30)

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
            "key":          key,
            "expected_ver": ver,
            "content":      content,
            "rationale":    note,
            "agent_id":     agent,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status()
        return r.json()


# ── SYSTEM 1: S-Bus ───────────────────────────────────────────────────────────

def run_sbus(task: dict, n_agents: int, bus: Bus,
             steps: int, model: str) -> Run:
    """
    S-Bus: each agent owns dedicated shards.
    Coordination cost = shard reads only — no central summarizer.
    Conflicts detected and resolved by the ACP.
    """
    run = Run(
        run_id=str(uuid.uuid4()), system="sbus",
        agent_count=n_agents, task_id=task["task_id"], model=model,
    )
    t0     = time.time()
    keys   = task["shared_state_keys"]
    agents = [f"agent-{i}" for i in range(n_agents)]
    pfx    = run.run_id[:8]
    skeys  = [f"{pfx}_{k}" for k in keys]

    for k, sk in zip(keys, skeys):
        bus.create(sk, f"[{k}: not started]", task["category"])

    for step in range(steps):
        agent = agents[step % n_agents]
        owned = [skeys[i] for i in range(len(skeys))
                 if i % n_agents == step % n_agents]

        for sk in owned:
            shard = bus.read(sk)
            # Reading current shard = coordination cost (state distribution)
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
            resp = bus.commit(sk, shard["version"], text, agent, f"step {step+1}")
            if resp.get("conflict"):
                run.commit_conflicts += 1
                shard = bus.read(sk)
                run.coord_tokens += tok(shard["content"])
                run.commit_attempts += 1
                resp = bus.commit(sk, shard["version"], text, agent,
                                  f"step {step+1} retry")
                if resp.get("conflict"):
                    run.commit_conflicts += 1

        if step + 1 >= task.get("min_steps", 5):
            if all(bus.read(sk)["version"] >= 2 for sk in skeys):
                run.success     = True
                run.steps_taken = step + 1
                break

    run.steps_taken = run.steps_taken or steps
    run.wall_ms     = int((time.time() - t0) * 1000)
    log.info(f"  sbus   {task['task_id']} a={n_agents} "
             f"CWR={run.cwr:.3f} S@50={run.success} SCR={run.scr:.3f}")
    return run


# ── SYSTEM 2: Coordinator-Worker ──────────────────────────────────────────────

def run_coord_worker(task: dict, n_agents: int, system_name: str,
                     steps: int, model: str) -> Run:
    """
    Coordinator-worker: models the pattern used by CrewAI, AutoGen, LangGraph.

    Coordination tax = cost of:
      1. Each worker reading the full shared context (N × context_size tokens)
      2. The coordinator reading all worker outputs and writing a summary

    CWR = (context distribution + summarization) / worker reasoning
    """
    run = Run(
        run_id=str(uuid.uuid4()), system=system_name,
        agent_count=n_agents, task_id=task["task_id"], model=model,
    )
    t0      = time.time()
    context = f"Task: {task['description']}\n\nStatus: Not started."

    for step in range(steps):
        worker_outputs = []

        # Worker phase: each reads full context (coordination cost)
        for i in range(n_agents):
            run.coord_tokens += tok(context)

            text, pt, ct = llm(
                f"You are worker {i+1} of {n_agents}, "
                f"expert in {task['category']}.",
                f"Shared context:\n{context}\n\n"
                f"Step {step+1}/{steps}: As worker {i+1}, add your contribution "
                f"(2-3 sentences).",
                model=model, max_tok=300,
            )
            run.work_tokens += pt + ct
            if text:
                worker_outputs.append(f"[Worker {i+1}]: {text}")

        # Coordinator phase: reads all outputs + writes summary (pure overhead)
        if worker_outputs:
            all_out = "\n\n".join(worker_outputs)
            run.coord_tokens += tok(all_out)

            summary, pt, ct = llm(
                "You are the coordinator. Synthesise all worker outputs into "
                "one coherent status update (3-5 sentences).",
                f"Task: {task['description']}\n\nWorker outputs:\n{all_out}",
                model=model, max_tok=400,
            )
            run.coord_tokens += pt + ct
            if summary:
                context = summary

        if step + 1 >= task.get("min_steps", 5):
            run.success     = True
            run.steps_taken = step + 1
            break

    run.steps_taken = run.steps_taken or steps
    run.wall_ms     = int((time.time() - t0) * 1000)
    log.info(f"  {system_name:<10} {task['task_id']} a={n_agents} "
             f"CWR={run.cwr:.3f} S@50={run.success}")
    return run


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(csv_path: Path):
    try:
        import pandas as pd
        from scipy import stats
    except ImportError:
        print("pip install pandas scipy  to run analysis")
        return

    df = pd.read_csv(csv_path)
    df = df[df["cwr"] != float("inf")]
    if df.empty:
        print("No finite CWR values to analyse yet.")
        return

    print(f"\n{'='*62}")
    print("CWR by (system, agent_count)")
    print(f"{'='*62}")
    print(f"{'System':<12} {'Agents':>7}  {'Mean CWR':>10}  {'±95%CI':>9}  {'n':>4}")
    print("-" * 48)
    for ac in sorted(df["agent_count"].unique()):
        for sys in sorted(df["system"].unique()):
            sub = df[(df["system"] == sys) & (df["agent_count"] == ac)]["cwr"]
            if len(sub) == 0:
                continue
            ci = 1.96 * sub.std() / len(sub)**0.5 if len(sub) > 1 else 0
            print(f"{sys:<12} {ac:>7}  {sub.mean():>10.3f}  "
                  f"±{ci:>8.3f}  {len(sub):>4}")

    print(f"\n{'='*62}")
    print("Mann-Whitney U: S-Bus CWR < each baseline (one-sided)")
    print(f"{'='*62}")
    sbus_cwr = df[df["system"] == "sbus"]["cwr"]
    for sys in [s for s in df["system"].unique() if s != "sbus"]:
        base = df[df["system"] == sys]["cwr"]
        if len(base) < 2 or len(sbus_cwr) < 2:
            print(f"  sbus < {sys:<12}: n too small "
                  f"({len(sbus_cwr)},{len(base)})")
            continue
        u, p = stats.mannwhitneyu(sbus_cwr, base, alternative="less")
        r    = 1 - (2 * u) / (len(sbus_cwr) * len(base))
        sig  = ("***" if p < 0.001 else
                "**"  if p < 0.01  else
                "*"   if p < 0.05  else "ns")
        print(f"  sbus < {sys:<12}: U={u:.0f}  p={p:.4f}  r={r:.3f}  {sig}")

    print(f"\n{'='*62}")
    print("S@50 success rate")
    print(f"{'='*62}")
    s50 = df.groupby(["system", "agent_count"])["success"].mean().reset_index()
    s50["success"] = (s50["success"] * 100).round(1).astype(str) + "%"
    print(s50.to_string(index=False))

    print(f"\n{'='*62}")
    print("CWR reduction vs best baseline")
    print(f"{'='*62}")
    sbus_m = df[df["system"] == "sbus"]["cwr"].mean()
    others = [s for s in df["system"].unique() if s != "sbus"]
    if others:
        best_n = min(others, key=lambda s: df[df["system"] == s]["cwr"].mean())
        best_m = df[df["system"] == best_n]["cwr"].mean()
        print(f"  S-Bus mean CWR : {sbus_m:.3f}")
        print(f"  Best baseline  : {best_n} ({best_m:.3f})")
        print(f"  Reduction      : {(best_m - sbus_m) / best_m * 100:.1f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="S-Bus CWR benchmark")
    p.add_argument("--system",
                   choices=["sbus", "crewai", "autogen", "langgraph", "all"],
                   default="sbus")
    p.add_argument("--all",       dest="run_all", action="store_true")
    p.add_argument("--agents",    type=int, nargs="+", default=[4])
    p.add_argument("--steps",     type=int, default=20,
                   help="Max steps per task (paper uses 50)")
    p.add_argument("--tasks",     default="datasets/long_horizon_tasks.json")
    p.add_argument("--tasks-limit", type=int, default=None,
                   help="Only run first N tasks (quick test)")
    p.add_argument("--model",     default="gpt-4o-mini")
    p.add_argument("--sbus-url",  default="http://localhost:3000")
    p.add_argument("--out",       default="results/cwr_results.csv")
    p.add_argument("--analyse-only", action="store_true")
    args = p.parse_args()

    if args.analyse_only:
        path = Path(args.out)
        if not path.exists():
            print(f"Not found: {path}")
            sys.exit(1)
        analyse(path)
        return

    systems = (
        ["sbus", "crewai", "autogen", "langgraph"]
        if (args.run_all or args.system == "all")
        else [args.system]
    )

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        log.error(f"Tasks file not found: {tasks_path}")
        sys.exit(1)
    tasks = json.loads(tasks_path.read_text())
    if args.tasks_limit:
        tasks = tasks[: args.tasks_limit]

    log.info(f"Loaded {len(tasks)} tasks | systems={systems} | "
             f"agents={args.agents} | steps={args.steps}")

    bus = Bus(args.sbus_url)
    if "sbus" in systems and not bus.ping():
        log.error(f"S-Bus server not reachable at {args.sbus_url}. "
                  f"Run 'cargo run' first.")
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    need_header = not out.exists()

    total = len(systems) * len(args.agents) * len(tasks)
    log.info(f"Total runs planned: {total}")

    done, runs = 0, []
    try:
        for sys_name in systems:
            for n in args.agents:
                for task in tasks:
                    done += 1
                    log.info(f"[{done}/{total}] {sys_name} "
                             f"agents={n} task={task['task_id']}")
                    try:
                        run = (
                            run_sbus(task, n, bus, args.steps, args.model)
                            if sys_name == "sbus"
                            else run_coord_worker(
                                task, n, sys_name, args.steps, args.model)
                        )
                        runs.append(run)
                        with open(out, "a") as f:
                            if need_header:
                                f.write(CSV_HDR)
                                need_header = False
                            f.write(run.csv())
                    except Exception as e:
                        log.error(f"Run failed: {e}")
    except KeyboardInterrupt:
        log.info("Interrupted — progress saved.")

    log.info(f"\n{len(runs)} runs written to {out}")
    if runs:
        analyse(out)


if __name__ == "__main__":
    main()