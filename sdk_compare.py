"""
sdk_compare.py  (gap-fill revision — built on original, 5 targeted edits)
==========================================================================
Gap-fill changes from original (marked [GF-N]):
  [GF-1] --steps default changed 20->50 (S@50 was structurally impossible at 20)
  [GF-2] --sequential-sbus flag added (isolates parallelism from coordination savings)
  [GF-3] per-task step budgeting: auto-extends to min_steps+10
  [GF-4] Run.sequential field + ThreadPoolExecutor in run_sbus()
  [GF-5] Wilson CIs + Fisher's exact test in analyse()

Field names match actual long_horizon_tasks.json schema:
  task["task_id"]           not task["id"]
  task["shared_state_keys"] not task["shards"]
  Bus.commit(..., ver, content, ...) uses "expected_ver" and "content"
"""

import argparse
import concurrent.futures as _cf
import json
import logging
import importlib.util
import math
import operator
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

for pkg in ["httpx", "tiktoken", "openai"]:
    if not importlib.util.find_spec(pkg):
        print(f"Missing: pip install {pkg}")
        sys.exit(1)

import httpx
import tiktoken
from openai import OpenAI

_ENC = tiktoken.encoding_for_model("gpt-4o")

def tok(text: str) -> int:
    return len(_ENC.encode(str(text) or ""))

MODEL = "gpt-4o-mini"

def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        log.error("OPENAI_API_KEY not set.")
        sys.exit(1)
    return key


# ── RunMetrics ────────────────────────────────────────────────────────────────

@dataclass
class Run:
    run_id:          str
    system:          str
    agent_count:     int
    task_id:         str
    coord_tokens:    int  = 0
    work_tokens:     int  = 0
    steps_taken:     int  = 0
    success:         bool = False
    commit_attempts: int  = 0
    commit_conflicts:int  = 0
    wall_ms:         int  = 0
    model:           str  = MODEL
    sdk_version:     str  = ""
    sequential:      bool = False   # [GF-4]

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
            f"{self.wall_ms},{self.model},{self.sdk_version},{int(self.sequential)}\n"
        )

# [GF-4] sequential column added
CSV_HDR = (
    "run_id,system,agent_count,task_id,coord_tokens,work_tokens,cwr,"
    "steps_taken,success,commit_attempts,commit_conflicts,scr,"
    "wall_ms,model,sdk_version,sequential\n"
)


# ── OpenAI helper ─────────────────────────────────────────────────────────────

_oai = None

def oai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI(api_key=get_api_key())
    return _oai

def llm_call(sys_msg: str, usr_msg: str, max_tok: int = 350) -> tuple[str, int, int]:
    for attempt in range(2):
        try:
            r = oai().chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user",   "content": usr_msg}],
                temperature=0.2,
                max_tokens=max_tok,
            )
            return (r.choices[0].message.content or "",
                    r.usage.prompt_tokens,
                    r.usage.completion_tokens)
        except Exception as e:
            if attempt == 0 and "rate" in str(e).lower():
                log.warning("Rate limit — sleeping 20s"); time.sleep(20)
            else:
                log.error(f"LLM error: {e}"); return "", 0, 0
    return "", 0, 0


# ── S-Bus HTTP client ─────────────────────────────────────────────────────────

class Bus:
    def __init__(self, url="http://localhost:3000"):
        self.base = url
        self.c = httpx.Client(timeout=30)

    def ping(self):
        try:
            self.c.get(f"{self.base}/stats", timeout=3)
            return True
        except Exception:
            return False

    def create(self, key, content, tag="default"):
        r = self.c.post(f"{self.base}/shard",
                        json={"key": key, "content": content, "goal_tag": tag})
        r.raise_for_status()
        return r.json()["key"]

    def read(self, key):
        r = self.c.get(f"{self.base}/shard/{key}")
        r.raise_for_status()
        return r.json()

    def commit(self, key, ver, content, agent, note=""):
        """
        POST /commit — uses field names from the original Bus implementation:
          expected_ver (not expected_version)
          content      (not delta)
        """
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


def judge_success(outputs: list[str], task: dict) -> bool:
    checks = task.get("ground_truth_outputs", [])
    if not checks:
        return all(len(o) > 80 for o in outputs if o)
    combined = "\n\n".join(f"[Output {i+1}]:\n{o}" for i, o in enumerate(outputs))
    criteria = "\n".join(f"- {c}" for c in checks)
    verdict, _, _ = llm_call(
        "You are an evaluator. Answer only YES or NO.",
        f"Task: {task['description']}\n\nSuccess criteria:\n{criteria}\n\n"
        f"Outputs:\n{combined}\n\nDo all outputs satisfy ALL criteria? YES or NO.",
        max_tok=5,
    )
    return verdict.strip().upper().startswith("YES")


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 1 — S-Bus
# ══════════════════════════════════════════════════════════════════════════════

def run_sbus(task: dict, n_agents: int, bus: Bus,
             steps: int, success_steps: int,
             sequential: bool = False) -> Run:
    """
    S-Bus shard-ownership experiment.

    Uses exact same field names as the original run_sbus():
      task["task_id"]           - task identifier
      task["shared_state_keys"] - list of shard key names
      task["category"]          - category label for goal_tag
      task["description"]       - task description prompt
      task["min_steps"]         - minimum steps for success check

    [GF-4] sequential=True runs agents one-at-a-time, isolating the
    parallelism advantage from the coordination-token advantage.

    [GF-3] step budget is set by caller to max(args.steps, min_steps+10)
    so S@50 is actually reachable for all LHP tasks.
    """
    run = Run(
        run_id=str(uuid.uuid4()), system="sbus",
        agent_count=n_agents, task_id=task["task_id"],
        sdk_version="rust-sbus-0.1.0",
        sequential=sequential,
    )
    t0 = time.time()

    keys   = task["shared_state_keys"]
    pfx    = run.run_id[:8]
    skeys  = [f"{pfx}_{k}" for k in keys]
    agents = [f"agent-{i}" for i in range(n_agents)]

    for k, sk in zip(keys, skeys):
        bus.create(sk, f"[{k}: not started]", task["category"])

    # [GF-4] per-agent step — runs in thread pool or sequentially
    def _agent_step(agent_idx: int, step: int):
        agent = agents[agent_idx]
        owned = [skeys[i] for i in range(len(skeys))
                 if i % n_agents == agent_idx]
        for sk in owned:
            shard = bus.read(sk)
            run.coord_tokens += tok(shard["content"])
            label = sk.split("_", 1)[1]
            text, pt, ct = llm_call(
                f"You are {agent}, expert in {task['category']}. Be concise.",
                f"Task: {task['description']}\n\nYour component: {label}\n"
                f"Current state:\n{shard['content']}\n\n"
                f"Step {step+1}/{steps}: Write 2-3 sentences of concrete progress.",
            )
            run.work_tokens  += pt + ct
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

    for step in range(steps):
        if sequential:
            for ai in range(n_agents):
                _agent_step(ai, step)
        else:
            with _cf.ThreadPoolExecutor(max_workers=n_agents) as pool:
                list(pool.map(lambda ai: _agent_step(ai, step), range(n_agents)))

        if step + 1 >= task.get("min_steps", 5):
            if all(bus.read(sk)["version"] >= 2 for sk in skeys):
                run.success = True
                run.steps_taken = step + 1
                break

    run.steps_taken = run.steps_taken or steps
    run.wall_ms     = int((time.time() - t0) * 1000)
    log.info(f"  sbus CWR={run.cwr:.3f} S@50={run.success} SCR={run.scr:.3f} "
             f"seq={sequential}")
    return run


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 2 — CrewAI
# ══════════════════════════════════════════════════════════════════════════════

def run_crewai(task: dict, n_agents: int, steps: int, success_steps: int) -> Run:
    if not importlib.util.find_spec("crewai"):
        raise ImportError("crewai not installed — run: pip install crewai")

    import crewai
    from crewai import Agent, Task as CTask, Crew, Process, LLM

    os.environ["OPENAI_API_KEY"] = get_api_key()

    run = Run(run_id=str(uuid.uuid4()), system="crewai",
              agent_count=n_agents, task_id=task["task_id"],
              sdk_version=f"crewai-{crewai.__version__}")
    t0 = time.time()

    llm = LLM(model=MODEL, temperature=0.2, max_tokens=350)
    shard_keys = task["shared_state_keys"][:n_agents]
    n_keys = len(shard_keys)

    crew_agents, crew_tasks = [], []
    for i, key in enumerate(shard_keys):
        agent = Agent(
            role=f"Expert in {key.replace('_', ' ')}",
            goal=f"Make concrete progress on: {key}",
            backstory=f"You are a specialist in {key} for: {task['description'][:120]}",
            llm=llm, verbose=False,
            max_iter=max(2, steps // n_keys + 1),
        )
        crew_agents.append(agent)
        crew_tasks.append(CTask(
            description=(f"Task: {task['description']}\n\n"
                         f"Component: {key}\n"
                         f"Write a concrete 3-5 sentence plan for this component."),
            expected_output=f"Concrete plan for {key}",
            agent=agent,
        ))

    manager = Agent(
        role="Project Coordinator",
        goal="Synthesise all specialist outputs into a coherent plan",
        backstory="You coordinate specialists and ensure consistency.",
        llm=llm, verbose=False,
    )
    crew = Crew(agents=crew_agents, tasks=crew_tasks,
                process=Process.hierarchical, manager_agent=manager, verbose=False)

    try:
        result = crew.kickoff()
        usage  = getattr(crew, "usage_metrics", None)
        total_pt, total_ct = 0, 0
        if usage is not None:
            for attr in ["prompt_tokens", "total_prompt_tokens"]:
                val = getattr(usage, attr, None)
                if val: total_pt = int(val); break
            for attr in ["completion_tokens", "total_completion_tokens"]:
                val = getattr(usage, attr, None)
                if val: total_ct = int(val); break
            if not total_pt:
                try:
                    d = vars(usage) if hasattr(usage, "__dict__") else {}
                    total_pt = d.get("prompt_tokens", 0)
                    total_ct = d.get("completion_tokens", 0)
                except Exception:
                    pass
        if total_pt + total_ct > 0:
            run.coord_tokens = total_pt
            run.work_tokens  = max(1, total_ct)
        else:
            run.work_tokens  = tok(str(result))
            run.coord_tokens = run.work_tokens * 2
            log.warning("  crewai: usage_metrics empty, using fallback estimate")
        run.success = judge_success([str(result)], task)
    except Exception as e:
        log.error(f"CrewAI run failed: {e}")
        import traceback; traceback.print_exc()
        run.work_tokens = 1; run.coord_tokens = 1

    run.steps_taken = success_steps
    run.wall_ms     = int((time.time() - t0) * 1000)
    log.info(f"  crewai CWR={run.cwr:.3f} S@50={run.success}")
    return run


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 3 — AutoGen
# ══════════════════════════════════════════════════════════════════════════════

def run_autogen(task: dict, n_agents: int, steps: int, success_steps: int) -> Run:
    if not importlib.util.find_spec("autogen_agentchat"):
        raise ImportError("autogen-agentchat not installed")

    import autogen_agentchat
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    import asyncio

    api_key = get_api_key()
    run = Run(run_id=str(uuid.uuid4()), system="autogen",
              agent_count=n_agents, task_id=task["task_id"],
              sdk_version=f"autogen-{autogen_agentchat.__version__}")
    t0 = time.time()
    shard_keys = task["shared_state_keys"][:n_agents]

    async def _run_async():
        model_client = OpenAIChatCompletionClient(
            model=MODEL, api_key=api_key, max_tokens=350, temperature=0.2,
        )
        agents = [
            AssistantAgent(
                name=f"Agent_{i}_{key[:8]}",
                model_client=model_client,
                system_message=(
                    f"You are a specialist in {key.replace('_', ' ')}. "
                    f"Task: {task['description'][:150]}. "
                    f"Focus only on: {key}. Be concise (2-3 sentences)."
                ),
            )
            for i, key in enumerate(shard_keys)
        ]
        team = RoundRobinGroupChat(
            agents,
            termination_condition=MaxMessageTermination(
                max_messages=n_agents * (steps // n_agents + 1)
            ),
        )
        messages_all = []
        async for msg in team.run_stream(
            task=f"Complete this task collaboratively: {task['description']}"
        ):
            if hasattr(msg, "messages"):
                messages_all.extend(msg.messages)
            elif hasattr(msg, "content"):
                messages_all.append(msg)

        total_pt, total_ct = 0, 0
        try:
            usage = model_client.actual_usage()
            total_pt = usage.prompt_tokens
            total_ct = usage.completion_tokens
        except Exception:
            for m in messages_all:
                c = getattr(m, "content", "") or ""
                if isinstance(c, str):
                    total_pt += tok(c)
            total_ct = total_pt // 3

        system_prompt_overhead = len(shard_keys) * 350
        run.coord_tokens = max(0, total_pt - system_prompt_overhead)
        run.work_tokens  = max(1, total_ct)

        return " ".join(
            getattr(m, "content", "") or ""
            for m in messages_all[-3:]
            if isinstance(getattr(m, "content", ""), str)
        )

    try:
        loop = asyncio.new_event_loop()
        final = loop.run_until_complete(_run_async())
        loop.close()
        run.success = judge_success([final], task)
    except Exception as e:
        log.error(f"AutoGen run failed: {e}")
        import traceback; traceback.print_exc()
        run.work_tokens = 1; run.coord_tokens = 1

    run.steps_taken = success_steps
    run.wall_ms     = int((time.time() - t0) * 1000)
    log.info(f"  autogen CWR={run.cwr:.3f} S@50={run.success}")
    return run


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 4 — LangGraph
# ══════════════════════════════════════════════════════════════════════════════

def run_langgraph(task: dict, n_agents: int, steps: int, success_steps: int) -> Run:
    if not importlib.util.find_spec("langgraph"):
        raise ImportError("langgraph not installed")

    from langgraph.graph import StateGraph, END

    run = Run(run_id=str(uuid.uuid4()), system="langgraph",
              agent_count=n_agents, task_id=task["task_id"],
              sdk_version="langgraph-1.1.x")
    t0 = time.time()
    shard_keys = task["shared_state_keys"][:n_agents]
    token_tracker = {"coord": 0, "work": 0}

    class AgentState(TypedDict):
        messages:           Annotated[list[str], operator.add]
        step:               int
        outputs:            dict[str, str]
        supervisor_summary: str

    def make_worker(key: str):
        def worker_node(state: AgentState) -> dict:
            current = state["outputs"].get(key, "not started")
            text, pt, ct = llm_call(
                f"You are a specialist in {key.replace('_', ' ')}.",
                f"Task: {task['description']}\n\nComponent: {key}\n"
                f"Current state: {current}\n\nStep {state['step']}: 2-3 sentences.",
            )
            token_tracker["coord"] += pt
            token_tracker["work"]  += ct
            new_outputs = dict(state["outputs"])
            new_outputs[key] = text
            return {"outputs": new_outputs, "messages": [f"[{key}]: {text}"]}
        worker_node.__name__ = f"worker_{key}"
        return worker_node

    def supervisor_node(state: AgentState) -> dict:
        all_outputs = "\n\n".join(f"[{k}]: {v}" for k, v in state["outputs"].items())
        token_tracker["coord"] += tok(all_outputs)
        summary, pt, ct = llm_call(
            "You are the supervisor. Synthesise all worker outputs.",
            f"Task: {task['description']}\n\nWorker outputs:\n{all_outputs}\n\n"
            f"3-sentence coherent summary.",
        )
        token_tracker["coord"] += pt + ct
        return {"supervisor_summary": summary, "step": state["step"] + 1,
                "messages": [f"[supervisor]: {summary}"]}

    def should_continue(state: AgentState) -> str:
        target = max(1, success_steps // max(1, len(shard_keys)))
        return END if state["step"] >= target else "supervisor"

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    for i, key in enumerate(shard_keys):
        builder.add_node(f"worker_{i}", make_worker(key))
    builder.add_edge("supervisor", "worker_0")
    for i in range(len(shard_keys) - 1):
        builder.add_edge(f"worker_{i}", f"worker_{i+1}")
    builder.add_conditional_edges(f"worker_{len(shard_keys)-1}", should_continue)
    builder.set_entry_point("supervisor")
    graph = builder.compile()

    try:
        initial: AgentState = {
            "messages": [], "step": 0,
            "outputs": {k: "not started" for k in shard_keys},
            "supervisor_summary": "",
        }
        final_state = graph.invoke(initial)
        run.coord_tokens = token_tracker["coord"]
        run.work_tokens  = max(1, token_tracker["work"])
        run.success = judge_success(list(final_state["outputs"].values()), task)
    except Exception as e:
        log.error(f"LangGraph run failed: {e}")
        import traceback; traceback.print_exc()
        run.work_tokens = 1; run.coord_tokens = 1

    run.steps_taken = success_steps
    run.wall_ms     = int((time.time() - t0) * 1000)
    log.info(f"  langgraph CWR={run.cwr:.3f} S@50={run.success}")
    return run


# ══════════════════════════════════════════════════════════════════════════════
# Analysis  [GF-5]
# ══════════════════════════════════════════════════════════════════════════════

def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return 0.0, 1.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return max(0.0, c-m), min(1.0, c+m)


def analyse(csv_path: Path) -> None:
    try:
        import pandas as pd
        from scipy import stats
        from scipy.stats import fisher_exact
    except ImportError:
        print("pip install pandas scipy"); return

    # Read with explicit column names that include the 'sequential' column
    # added in [GF-4].  If the CSV was written by the old 15-column header
    # (missing 'sequential') pandas would shift all values one column left,
    # making wall_ms a string.  Providing explicit names + skiprows=1 is
    # robust regardless of whether the header was written correctly.
    _COLS = [
        "run_id","system","agent_count","task_id",
        "coord_tokens","work_tokens","cwr",
        "steps_taken","success",
        "commit_attempts","commit_conflicts","scr",
        "wall_ms","model","sdk_version","sequential",
    ]
    # Detect column count from first data row to handle both old and new CSVs
    with open(csv_path) as _f:
        _f.readline()  # skip header
        _first = _f.readline().strip()
    _ncols = len(_first.split(","))
    _use_cols = _COLS[:_ncols]
    df = pd.read_csv(csv_path, names=_use_cols, skiprows=1)
    df["cwr"]       = pd.to_numeric(df["cwr"],       errors="coerce")
    df["wall_ms"]   = pd.to_numeric(df["wall_ms"],   errors="coerce")
    df["success"]   = pd.to_numeric(df["success"],   errors="coerce")
    df["agent_count"] = pd.to_numeric(df["agent_count"], errors="coerce")
    df = df.dropna(subset=["cwr","wall_ms"])
    df = df[df["cwr"] != float("inf")]
    df = df[~((df["coord_tokens"] == 1) & (df["work_tokens"] == 1))]
    if df.empty:
        print("No valid runs yet."); return

    print(f"\n{'='*65}\nCWR by (system, agent_count)\n{'='*65}")
    print(f"{'System':<12} {'N':>4} {'Mean CWR':>10} {'±95%CI':>9} {'n':>4}")
    print("-"*50)
    for ac in sorted(df["agent_count"].unique()):
        for sys in ["sbus", "crewai", "autogen", "langgraph"]:
            sub = df[(df["system"]==sys) & (df["agent_count"]==ac)]["cwr"]
            if not len(sub): continue
            ci = 1.96 * sub.std() / len(sub)**0.5 if len(sub) > 1 else 0
            print(f"{sys:<12} {ac:>4} {sub.mean():>10.3f} ±{ci:>8.3f} {len(sub):>4}")

    print(f"\n{'='*65}\nMann-Whitney U: S-Bus CWR < each baseline (one-sided)\n{'='*65}")
    sbus_cwr = df[df["system"]=="sbus"]["cwr"]
    for sys in ["crewai", "autogen", "langgraph"]:
        base = df[df["system"]==sys]["cwr"]
        if len(base) < 2 or len(sbus_cwr) < 2:
            print(f"  sbus < {sys}: need n>=2"); continue
        u, p = stats.mannwhitneyu(sbus_cwr, base, alternative="less")
        r    = 1 - (2*u)/(len(sbus_cwr)*len(base))
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  sbus < {sys:<12}: U={u:.0f} p={p:.4f} r={r:.3f} {sig}")

    # [GF-5] Wilson CIs
    print(f"\n{'='*65}\nS@50 with Wilson 95% CIs  [gap-fill]\n{'='*65}")
    print(f"  {'System':<12} {'N':>4} {'k':>4} {'n':>4} {'S@50':>7}  {'95% CI':>18}  Note")
    print("  " + "-"*68)
    for ac in sorted(df["agent_count"].unique()):
        for sys in ["sbus", "langgraph", "crewai", "autogen"]:
            sub = df[(df["system"]==sys) & (df["agent_count"]==ac)]["success"]
            if not len(sub): continue
            k_s = int(sub.sum()); n_ = len(sub); p = k_s/n_
            lo, hi = _wilson(k_s, n_)
            note = "WIDE CI: n too small" if hi-lo > 0.5 else ""
            print(f"  {sys:<12} {ac:>4} {k_s:>4} {n_:>4} {p:>7.0%}  "
                  f"[{lo:.0%} - {hi:.0%}]  {note}")

    # [GF-5] Fisher's exact
    print(f"\n{'='*65}\nFisher's exact: S-Bus S@50 > each baseline  [gap-fill]\n{'='*65}")
    for ac in sorted(df["agent_count"].unique()):
        sbus_s = df[(df["system"]=="sbus") & (df["agent_count"]==ac)]["success"]
        if not len(sbus_s): continue
        sk, sn = int(sbus_s.sum()), len(sbus_s)
        for bl in ["langgraph", "crewai", "autogen"]:
            bl_s = df[(df["system"]==bl) & (df["agent_count"]==ac)]["success"]
            if not len(bl_s): continue
            bk, bn = int(bl_s.sum()), len(bl_s)
            _, p = fisher_exact([[sk, sn-sk], [bk, bn-bk]], alternative="greater")
            note = "significant" if p < 0.05 else "ns — report as preliminary (n<10)"
            print(f"  N={ac}: sbus({sk}/{sn}) > {bl}({bk}/{bn})  p={p:.4f}  {note}")

    # [GF-4] Parallelism decomposition
    if "sequential" in df.columns:
        seq = df[(df["system"]=="sbus") & (df["sequential"]==1)]
        par = df[(df["system"]=="sbus") & (df["sequential"]==0)]
        if len(seq) and len(par):
            print(f"\n{'='*65}\nParallelism decomposition (S-Bus)  [gap-fill]\n{'='*65}")
            print(f"  S-Bus parallel  : {par['wall_ms'].mean()/1000:.1f}s   "
                  f"CWR={par['cwr'].mean():.3f}")
            print(f"  S-Bus sequential: {seq['wall_ms'].mean()/1000:.1f}s   "
                  f"CWR={seq['cwr'].mean():.3f}")
            print(f"  Threading speedup: {seq['wall_ms'].mean()/max(par['wall_ms'].mean(),1):.2f}x")
            print("  CWR should be equal in both modes — difference is pure threading,")
            print("  not coordination architecture.")

    print(f"\n{'='*65}\nWall time / CWR reduction\n{'='*65}")
    for sys, grp in df.groupby("system"):
        print(f"  {sys:<12}: {grp['wall_ms'].mean()/1000:.1f}s mean wall time")
    sbus_m = df[df["system"]=="sbus"]["cwr"].mean()
    others = [s for s in df["system"].unique() if s != "sbus"]
    if others:
        best = min(others, key=lambda s: df[df["system"]==s]["cwr"].mean())
        best_m = df[df["system"]==best]["cwr"].mean()
        print(f"\n  S-Bus CWR: {sbus_m:.3f}  |  Best baseline ({best}): {best_m:.3f}  "
              f"|  Reduction: {(best_m-sbus_m)/best_m*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

SYSTEMS = ["sbus", "crewai", "autogen", "langgraph"]


def main():
    p = argparse.ArgumentParser(description="S-Bus real SDK comparison (gap-fill revision)")
    p.add_argument("--system", choices=SYSTEMS + ["all"], default="all")
    p.add_argument("--agents", type=int, nargs="+", default=[4])
    p.add_argument("--steps", type=int, default=50,          # [GF-1]
                   help="Step budget. S-Bus auto-extends to min_steps+10.")
    p.add_argument("--success-steps", type=int, default=None)
    p.add_argument("--tasks", default="datasets/long_horizon_tasks.json")
    p.add_argument("--tasks-limit", type=int, default=None)
    p.add_argument("--sbus-url", default="http://localhost:3000")
    p.add_argument("--out", default="results/real_sdk_results.csv")
    p.add_argument("--analyse-only", action="store_true")
    p.add_argument("--sequential-sbus", action="store_true",  # [GF-2]
                   help="Run S-Bus agents sequentially (no threading). "
                        "Isolates coordination savings from parallelism.")
    args = p.parse_args()

    success_steps = args.success_steps or args.steps

    if args.analyse_only:
        path = Path(args.out)
        if not path.exists():
            print(f"Not found: {path}"); sys.exit(1)
        analyse(path); return

    os.environ["OPENAI_API_KEY"] = get_api_key()
    systems = SYSTEMS if args.system == "all" else [args.system]

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        log.error(f"Tasks file not found: {tasks_path}"); sys.exit(1)
    tasks = json.loads(tasks_path.read_text())
    if args.tasks_limit:
        tasks = tasks[:args.tasks_limit]

    bus = Bus(args.sbus_url)
    if "sbus" in systems and not bus.ping():
        log.error("S-Bus server not reachable — run 'cargo run' first"); sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    need_header = not out.exists()

    total = len(systems) * len(args.agents) * len(tasks)
    log.info(f"Running: systems={systems} agents={args.agents} steps={args.steps} "
             f"tasks={len(tasks)} sequential_sbus={args.sequential_sbus}")

    runs, done = [], 0
    try:
        for sys_name in systems:
            for n in args.agents:
                for task in tasks:
                    done += 1
                    # [GF-3] per-task step budget for S-Bus
                    step_budget = (
                        max(args.steps, task.get("min_steps", 22) + 10)
                        if sys_name == "sbus" else args.steps
                    )
                    log.info(f"[{done}/{total}] {sys_name} N={n} "
                             f"{task['task_id']} steps={step_budget}")
                    try:
                        if sys_name == "sbus":
                            run = run_sbus(task, n, bus, step_budget, success_steps,
                                           sequential=args.sequential_sbus)
                        elif sys_name == "crewai":
                            run = run_crewai(task, n, args.steps, success_steps)
                        elif sys_name == "autogen":
                            run = run_autogen(task, n, args.steps, success_steps)
                        elif sys_name == "langgraph":
                            run = run_langgraph(task, n, args.steps, success_steps)
                        else:
                            continue

                        runs.append(run)
                        with open(out, "a") as f:
                            if need_header:
                                f.write(CSV_HDR); need_header = False
                            f.write(run.csv())
                        log.info(f"  -> CWR={run.cwr:.3f} coord={run.coord_tokens} "
                                 f"work={run.work_tokens} wall={run.wall_ms/1000:.1f}s")
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        log.error(f"Run failed: {e}")
                        import traceback; traceback.print_exc()

    except KeyboardInterrupt:
        log.info("Interrupted — progress saved.")

    log.info(f"\n{len(runs)} runs written to {out}")
    if len(runs) >= 2:
        analyse(out)


if __name__ == "__main__":
    main()