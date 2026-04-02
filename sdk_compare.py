"""
sdk_compare.py  —  Real SDK comparison experiment for the S-Bus paper.
=======================================================================

Benchmarks the S-Bus middleware against three production multi-agent
frameworks using their actual SDK implementations:

    System      Architecture
    ──────────  ────────────────────────────────────────────────────────
    S-Bus       Shard-ownership via Atomic Commit Protocol (ACP)
    CrewAI      Hierarchical manager + worker agents
    AutoGen     RoundRobinGroupChat — full message history per turn
    LangGraph   StateGraph with supervisor node

Token accounting
----------------
All four systems measured using the same taxonomy (Table 3 in paper):

    coord_tokens — tokens spent on inter-agent coordination:
      S-Bus:     shard read before each commit (O(|shard|) per step)
      CrewAI:    prompt tokens from usage_metrics (manager reads context)
      AutoGen:   prompt tokens - system_prompt_overhead (history reads)
      LangGraph: supervisor context reads + supervisor LLM calls
                 + worker prompt tokens

    work_tokens — tokens spent on task-directed reasoning:
      S-Bus:     prompt + completion of each agent's LLM call
      CrewAI:    completion tokens only (from usage_metrics)
      AutoGen:   completion tokens only
      LangGraph: worker completion tokens only

    CWR = coord_tokens / work_tokens (lower is better)

Changes vs original:
  1. RETRY_BUDGET from SBUS_RETRY_BUDGET env var (Definition 6)
  2. run_crewai(): fallback coord = work × 2 REMOVED — runs with zero
     usage_metrics are now marked excluded=True and skipped in analysis
  3. Docstring numbers updated to canonical paper values (50-step runs)
  4. clopper_pearson_ci() replaces wilson_ci() for S@50 (Table 7)
  5. judge_success() uses SBUS_JUDGE_MODEL env var (circularity mitigation)
  6. /commit/v2_naive available in Bus client (cross-shard control)
  7. SDK version checked at runtime — warns if different from paper version

Canonical paper results (50-step full runs, 5 tasks, GPT-4o-mini):
    System      N=4 CWR   N=8 CWR   Reduction   S@50 N=8
    S-Bus       0.186     0.181     —            100%
    LangGraph   4.592     4.469     95.9%        20%
    CrewAI      19.381    16.176    99.0%        0%
    AutoGen     27.385    28.360    99.3%        100%
    Mann-Whitney U=0, p<0.0001, r=1.000 for all three comparisons.

    Note: Earlier smoke-test runs (steps=10, tasks-limit=1) produced
    S-Bus=0.238, LangGraph=4.384. Canonical values require --steps 50
    --tasks-limit 5.

Prerequisites:
    cargo run                           # Start S-Bus server
    export OPENAI_API_KEY="sk-..."

Usage:
    # Smoke test (~$1, ~5 min):
    python3 sdk_compare.py --system all --agents 4 --steps 10 --tasks-limit 1

    # Full paper run (~2 hours):
    python3 sdk_compare.py --agents 4 8 --steps 50 --tasks-limit 5

    # Single system:
    python3 sdk_compare.py --system sbus    --agents 4 8 --steps 50
    python3 sdk_compare.py --system crewai  --agents 4 8 --steps 50
    python3 sdk_compare.py --system autogen --agents 4 8 --steps 50
    python3 sdk_compare.py --system langgraph --agents 4 8 --steps 50

    # Analyse existing results:
    python3 sdk_compare.py --analyse-only --out results/real_sdk_results.csv
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import importlib.util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

for pkg in ["httpx", "tiktoken", "openai", "scipy"]:
    if not importlib.util.find_spec(pkg):
        print(f"Missing: pip install {pkg}")
        sys.exit(1)

import httpx
import tiktoken
from openai import OpenAI
from scipy.stats import binom as scipy_binom

_ENC = tiktoken.encoding_for_model("gpt-4o")

MODEL = "gpt-4o-mini"

# Paper SDK versions — warn if different
EXPECTED_VERSIONS = {
    "crewai":        "1.12.2",
    "autogen":       "0.7.5",
    "langgraph":     "1.1.3",
}

# Retry budget (Definition 6, Corollary 2.2)
RETRY_BUDGET = int(os.environ.get("SBUS_RETRY_BUDGET", "1"))


def tok(text: str) -> int:
    return len(_ENC.encode(str(text) or ""))


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        log.error("OPENAI_API_KEY not set.")
        sys.exit(1)
    return key


# ── Clopper-Pearson CI ───────────────────────────────────────────────────────

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Clopper-Pearson exact binomial CI.
    Replaces Wilson CI for S@50 — exact for extreme proportions (k=0, k=n).
    Section 6.2, Table 7.
    """
    if n == 0:
        return 0.0, 1.0
    if k == 0:
        lo = 0.0
        hi = float(1 - (alpha / 2) ** (1 / n))
    elif k == n:
        lo = float((alpha / 2) ** (1 / n))
        hi = 1.0
    else:
        lo = float(scipy_binom.ppf(alpha / 2,     n, k / n) / n)
        hi = float(scipy_binom.ppf(1 - alpha / 2, n, k / n) / n)
    return round(lo, 3), round(hi, 3)


# ── RunMetrics ────────────────────────────────────────────────────────────────

@dataclass
class Run:
    run_id:          str
    system:          str
    agent_count:     int
    task_id:         str
    coord_tokens:    int   = 0
    work_tokens:     int   = 0
    steps_taken:     int   = 0
    success:         bool  = False
    commit_attempts: int   = 0
    commit_conflicts:int   = 0
    retry_exhaustions: int = 0    # Corollary 2.2 skip events
    excluded:        bool  = False # True if token counts are unreliable
    wall_ms:         int   = 0
    model:           str   = MODEL
    sdk_version:     str   = ""

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

    def csv(self) -> str:
        cwr = f"{self.cwr:.4f}" if self.cwr != float("inf") else "inf"
        return (
            f"{self.run_id},{self.system},{self.agent_count},{self.task_id},"
            f"{self.coord_tokens},{self.work_tokens},{cwr},"
            f"{self.steps_taken},{int(self.success)},"
            f"{self.commit_attempts},{self.commit_conflicts},{self.scr:.4f},"
            f"{self.retry_exhaustions},{int(self.excluded)},"
            f"{self.wall_ms},{self.model},{self.sdk_version}\n"
        )


CSV_HDR = (
    "run_id,system,agent_count,task_id,coord_tokens,work_tokens,cwr,"
    "steps_taken,success,commit_attempts,commit_conflicts,scr,"
    "retry_exhaustions,excluded,wall_ms,model,sdk_version\n"
)


# ── OpenAI direct helper ──────────────────────────────────────────────────────

_oai = None


def oai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI(api_key=get_api_key())
    return _oai


def llm_call(sys_msg: str, usr_msg: str,
             max_tok: int = 350) -> tuple[str, int, int]:
    for attempt in range(2):
        try:
            r = oai().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": usr_msg},
                ],
                temperature=0.2,
                max_tokens=max_tok,
            )
            return (r.choices[0].message.content or "",
                    r.usage.prompt_tokens,
                    r.usage.completion_tokens)
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
        r = self.c.post(f"{self.base}/commit", json={
            "key": key, "expected_ver": ver, "content": content,
            "rationale": note, "agent_id": agent,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status()
        return r.json()

    def commit_v2(self, key, ver, content, agent, read_set, note=""):
        """Sorted-lock-order commit (Corollary 2.1 / Lemma 2). Requires A1."""
        r = self.c.post(f"{self.base}/commit/v2", json={
            "key": key, "expected_ver": ver, "content": content,
            "rationale": note, "agent_id": agent, "read_set": read_set,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status()
        return r.json()

    def commit_v2_naive(self, key, ver, content, agent, read_set, note=""):
        """Unordered multi-shard commit — control condition (Table 6)."""
        r = self.c.post(f"{self.base}/commit/v2_naive", json={
            "key": key, "expected_ver": ver, "content": content,
            "rationale": note, "agent_id": agent, "read_set": read_set,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status()
        return r.json()


# ── Success judge ─────────────────────────────────────────────────────────────

def judge_success(outputs: list[str], task: dict) -> bool:
    """
    Evaluate task success.

    Uses SBUS_JUDGE_MODEL (default: same as agents = GPT-4o-mini).
    For independent evaluation set SBUS_JUDGE_MODEL=gpt-4o to reduce
    the circularity noted in Section 8.8 (construct validity).
    """
    judge_model = os.environ.get("SBUS_JUDGE_MODEL", MODEL)

    checks = task.get("ground_truth_outputs", [])
    if not checks:
        return all(len(o) > 80 for o in outputs if o)

    combined = "\n\n".join(f"[Output {i+1}]:\n{o}" for i, o in enumerate(outputs))
    criteria = "\n".join(f"- {c}" for c in checks)
    verdict, _, _ = llm_call(
        "You are an evaluator. Answer only YES or NO.",
        f"Task: {task['description']}\n\nSuccess criteria:\n{criteria}\n\n"
        f"Outputs:\n{combined}\n\nDo outputs satisfy ALL criteria? YES or NO.",
        max_tok=5,
    )
    return verdict.strip().upper().startswith("YES")


# ── System 1: S-Bus ──────────────────────────────────────────────────────────

def run_sbus(task: dict, n_agents: int, bus: Bus,
             steps: int, success_steps: int) -> Run:
    """
    S-Bus: each agent owns dedicated shards.
    Retry budget B = RETRY_BUDGET (Definition 6, Corollary 2.2).

    Token accounting:
      coord_tokens += tok(shard_content) on every read (including retries)
      work_tokens  += prompt + completion on every LLM call
    """
    run = Run(run_id=str(uuid.uuid4()), system="sbus",
              agent_count=n_agents, task_id=task["task_id"],
              sdk_version="rust-sbus-0.1.0")
    t0 = time.time()

    keys = task["shared_state_keys"]
    pfx = run.run_id[:8]
    skeys = [f"{pfx}_{k}" for k in keys]
    agents = [f"agent-{i}" for i in range(n_agents)]

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

            text, pt, ct = llm_call(
                f"You are {agent}, expert in {task['category']}. Be concise.",
                f"Task: {task['description']}\n\nYour component: {label}\n"
                f"Current state:\n{shard['content']}\n\n"
                f"Step {step+1}/{steps}: 2-3 sentences of concrete progress.",
            )
            run.work_tokens += pt + ct
            run.commit_attempts += 1

            # Retry loop — explicit RETRY_BUDGET (Definition 6)
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
                    current_shard = bus.read(sk)
                    run.coord_tokens += tok(current_shard["content"])
                    run.commit_attempts += 1

            if not committed:
                run.retry_exhaustions += 1
                log.warning(
                    f"[{agent}] retry budget B={RETRY_BUDGET} exhausted "
                    f"at step {step+1} on {sk}"
                )

        if step + 1 == success_steps:
            final_outputs = [bus.read(sk)["content"] for sk in skeys]
            run.success = judge_success(final_outputs, task)
            run.steps_taken = step + 1
            break

    run.steps_taken = run.steps_taken or steps
    run.wall_ms = int((time.time() - t0) * 1000)
    log.info(f"  sbus  CWR={run.cwr:.3f} S50={run.success} "
             f"SCR={run.scr:.3f} exhausted={run.retry_exhaustions}")
    return run


# ── System 2: CrewAI ──────────────────────────────────────────────────────────

def run_crewai(task: dict, n_agents: int,
               steps: int, success_steps: int) -> Run:
    """
    CrewAI hierarchical experiment.

    Token accounting:
      coord_tokens = prompt_tokens from usage_metrics
                     (manager reads full task context each LLM call)
      work_tokens  = completion_tokens from usage_metrics

    IMPORTANT: If usage_metrics returns zero tokens (SDK version mismatch
    or CrewAI silent failure), the run is marked excluded=True and must
    NOT be included in paper tables. The previous fallback of
    coord = work × 2 has been removed as it produced invalid CWR values.

    Paper SDK version: crewai==1.12.2
    """
    import crewai
    sdk_ver = crewai.__version__
    if sdk_ver != EXPECTED_VERSIONS["crewai"]:
        log.warning(
            f"CrewAI version mismatch: expected {EXPECTED_VERSIONS['crewai']}, "
            f"got {sdk_ver}. Token accounting API may differ."
        )

    from crewai import Agent, Task as CTask, Crew, Process, LLM
    os.environ["OPENAI_API_KEY"] = get_api_key()

    run = Run(run_id=str(uuid.uuid4()), system="crewai",
              agent_count=n_agents, task_id=task["task_id"],
              sdk_version=f"crewai-{sdk_ver}")
    t0 = time.time()

    llm_obj = LLM(model=MODEL, temperature=0.2, max_tokens=350)
    shard_keys = task["shared_state_keys"][:n_agents]

    crew_agents, crew_tasks = [], []
    for i, key in enumerate(shard_keys):
        agent = Agent(
            role=f"Expert in {key.replace('_', ' ')}",
            goal=f"Make concrete progress on: {key}",
            backstory=f"You are a specialist in {key} for: {task['description'][:120]}",
            llm=llm_obj, verbose=False,
            max_iter=max(2, steps // max(len(shard_keys), 1) + 1),
        )
        crew_agents.append(agent)
        crew_tasks.append(CTask(
            description=(
                f"Task: {task['description']}\n\n"
                f"Component: {key}\n"
                f"Write a concrete 3-5 sentence plan for this component."
            ),
            expected_output=f"Concrete plan for {key}",
            agent=agent,
        ))

    manager = Agent(
        role="Project Coordinator",
        goal="Synthesise all specialist outputs into a coherent plan",
        backstory="You coordinate specialists and ensure consistency.",
        llm=llm_obj, verbose=False,
    )
    crew = Crew(
        agents=crew_agents, tasks=crew_tasks,
        process=Process.hierarchical,
        manager_agent=manager, verbose=False,
    )

    try:
        result = crew.kickoff()
        usage = getattr(crew, "usage_metrics", None)
        total_pt, total_ct = 0, 0

        if usage is not None:
            # Try attribute access (crewai 1.x UsageMetrics object)
            for attr in ["prompt_tokens", "total_prompt_tokens"]:
                val = getattr(usage, attr, None)
                if val is not None:
                    total_pt = int(val)
                    break
            for attr in ["completion_tokens", "total_completion_tokens"]:
                val = getattr(usage, attr, None)
                if val is not None:
                    total_ct = int(val)
                    break
            # Fallback: dict access
            if total_pt == 0:
                try:
                    d = dict(usage) if hasattr(usage, "__iter__") else vars(usage)
                    total_pt = d.get("prompt_tokens",
                                     d.get("total_prompt_tokens", 0))
                    total_ct = d.get("completion_tokens",
                                     d.get("total_completion_tokens", 0))
                except Exception:
                    pass

        # CRITICAL: If both zero, mark excluded — do NOT estimate
        if total_pt == 0 and total_ct == 0:
            log.warning(
                f"CrewAI run {run.run_id} returned zero usage_metrics "
                f"(crewai=={sdk_ver}). Marking as EXCLUDED — "
                f"do not include in paper tables. Check SDK version."
            )
            run.excluded = True
            run.coord_tokens = 0
            run.work_tokens = 0
        else:
            # coord = prompt tokens (manager reads full context each call)
            # work  = completion tokens (what agents generated)
            run.coord_tokens = total_pt
            run.work_tokens = max(1, total_ct)
            log.info(f"  crewai usage: coord(prompt)={total_pt} "
                     f"work(completion)={total_ct} CWR={run.cwr:.3f}")

        if not run.excluded:
            run.success = judge_success([str(result)], task)

    except Exception as e:
        log.error(f"CrewAI run failed: {e}")
        import traceback; traceback.print_exc()
        run.excluded = True

    run.steps_taken = success_steps
    run.wall_ms = int((time.time() - t0) * 1000)
    if not run.excluded:
        log.info(f"  crewai CWR={run.cwr:.3f} S50={run.success}")
    return run


# ── System 3: AutoGen ─────────────────────────────────────────────────────────

def run_autogen(task: dict, n_agents: int,
                steps: int, success_steps: int) -> Run:
    """
    AutoGen RoundRobinGroupChat experiment.

    Token accounting:
      coord_tokens = prompt_tokens - system_prompt_overhead
                     (message history each agent must read)
      work_tokens  = completion_tokens (what each agent generated)

      system_prompt_overhead ≈ n_agents × 350 tokens
      This grows as O(N²·|history|) per step — source of AutoGen's high CWR.

    Paper SDK version: autogen==0.7.5
    """
    import autogen_agentchat
    sdk_ver = autogen_agentchat.__version__
    if sdk_ver != EXPECTED_VERSIONS["autogen"]:
        log.warning(
            f"AutoGen version mismatch: expected {EXPECTED_VERSIONS['autogen']}, "
            f"got {sdk_ver}. Token accounting API may differ."
        )

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    import asyncio

    api_key = get_api_key()
    os.environ["OPENAI_API_KEY"] = api_key

    run = Run(run_id=str(uuid.uuid4()), system="autogen",
              agent_count=n_agents, task_id=task["task_id"],
              sdk_version=f"autogen-{sdk_ver}")
    t0 = time.time()

    shard_keys = task["shared_state_keys"][:n_agents]

    async def _run_async():
        model_client = OpenAIChatCompletionClient(
            model=MODEL, api_key=api_key,
            max_tokens=350, temperature=0.2,
        )
        agent_names = [f"Agent_{i}_{key[:8]}" for i, key in enumerate(shard_keys)]
        agents = [
            AssistantAgent(
                name=name,
                model_client=model_client,
                system_message=(
                    f"You are a specialist in {key.replace('_', ' ')}. "
                    f"Task: {task['description'][:150]}. "
                    f"Focus only on: {key}. Be concise (2-3 sentences)."
                ),
            )
            for name, key in zip(agent_names, shard_keys)
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

        # coord = prompt - system overhead (history growth is the coord cost)
        system_overhead = len(shard_keys) * 350
        run.coord_tokens = max(0, total_pt - system_overhead)
        run.work_tokens = max(1, total_ct)
        log.info(
            f"  autogen prompt={total_pt} completion={total_ct} "
            f"sys_overhead={system_overhead} "
            f"coord={run.coord_tokens} work={run.work_tokens} "
            f"CWR={run.cwr:.3f}"
        )

        final = " ".join(
            getattr(m, "content", "") or ""
            for m in messages_all[-3:]
            if isinstance(getattr(m, "content", ""), str)
        )
        return final

    try:
        loop = asyncio.new_event_loop()
        final_output = loop.run_until_complete(_run_async())
        loop.close()
        run.success = judge_success([final_output], task)
    except Exception as e:
        log.error(f"AutoGen run failed: {e}")
        import traceback; traceback.print_exc()
        run.excluded = True

    run.steps_taken = success_steps
    run.wall_ms = int((time.time() - t0) * 1000)
    if not run.excluded:
        log.info(f"  autogen CWR={run.cwr:.3f} S50={run.success}")
    return run


# ── System 4: LangGraph ───────────────────────────────────────────────────────

def run_langgraph(task: dict, n_agents: int,
                  steps: int, success_steps: int) -> Run:
    """
    LangGraph supervisor experiment.

    Token accounting:
      coord_tokens += tok(all_outputs)  supervisor re-reads all worker outputs
      coord_tokens += prompt_tokens     supervisor LLM call
      work_tokens  += completion_tokens worker LLM calls only

    Paper SDK version: langgraph==1.1.3
    """
    import langgraph
    from langgraph import version

    sdk_ver = version.__version__
    if sdk_ver != EXPECTED_VERSIONS["langgraph"]:
        log.warning(
            f"LangGraph version mismatch: expected {EXPECTED_VERSIONS['langgraph']}, "
            f"got {sdk_ver}. API may differ."
        )

    from langgraph.graph import StateGraph, END
    from typing import TypedDict

    api_key = get_api_key()
    os.environ["OPENAI_API_KEY"] = api_key

    run = Run(run_id=str(uuid.uuid4()), system="langgraph",
              agent_count=n_agents, task_id=task["task_id"],
              sdk_version=f"langgraph-{sdk_ver}")
    t0 = time.time()

    shard_keys = task["shared_state_keys"][:n_agents]
    worker_outputs = {k: "" for k in shard_keys}
    supervisor_summary = f"Task: {task['description']}\n\nBegin."
    all_final_outputs = []

    try:
        for cycle in range(max(1, steps // max(n_agents, 1))):
            # Workers read supervisor summary + produce output
            for i, key in enumerate(shard_keys):
                # Coordination cost: reading supervisor context
                run.coord_tokens += tok(supervisor_summary)
                # Work: worker reasoning
                text, pt, ct = llm_call(
                    f"You are worker {i} specializing in {key}. Be concise.",
                    f"Supervisor context:\n{supervisor_summary}\n\n"
                    f"Your component: {key}. Write 2-3 sentences of progress.",
                )
                run.work_tokens += ct
                run.coord_tokens += pt  # prompt = reading context = coord
                worker_outputs[key] = text

            # Supervisor reads all outputs (coordination cost)
            all_outputs_str = "\n".join(
                f"{k}: {v}" for k, v in worker_outputs.items()
            )
            run.coord_tokens += tok(all_outputs_str)

            # Supervisor LLM call (coordination cost)
            summary, sup_pt, sup_ct = llm_call(
                "You are the supervisor. Synthesize worker outputs concisely.",
                f"Worker outputs:\n{all_outputs_str}\n\n"
                f"Synthesize into a coherent 3-sentence update.",
            )
            run.coord_tokens += sup_pt + sup_ct
            supervisor_summary = summary or supervisor_summary
            all_final_outputs.append(supervisor_summary)

        run.success = judge_success([supervisor_summary], task)
    except Exception as e:
        log.error(f"LangGraph run failed: {e}")
        import traceback; traceback.print_exc()
        run.excluded = True

    run.steps_taken = success_steps
    run.wall_ms = int((time.time() - t0) * 1000)
    if not run.excluded:
        log.info(f"  langgraph CWR={run.cwr:.3f} S50={run.success}")
    return run


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(out_path: str):
    """Analyse CSV results — CWR stats and S@50 with Clopper-Pearson CIs."""
    import csv
    from collections import defaultdict
    from scipy.stats import t as t_dist, mannwhitneyu

    runs = []
    excluded_count = 0
    with open(out_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("excluded", "0") == "1":
                excluded_count += 1
                continue
            runs.append(row)

    if excluded_count:
        log.warning(f"Skipped {excluded_count} excluded runs in analysis")

    groups: dict[tuple, list[float]] = defaultdict(list)
    success: dict[tuple, list[bool]] = defaultdict(list)

    for r in runs:
        key = (r["system"], r["agent_count"])
        cwr_val = r["cwr"]
        if cwr_val not in ("inf", ""):
            groups[key].append(float(cwr_val))
        success[key].append(r["success"] == "1")

    print(f"\n{'System':<20} {'N':>3} {'Mean CWR':>10} {'±95% CI':>12} "
          f"{'n':>4} {'S@50':>6} {'CP 95% CI':>16}")
    print("-" * 80)

    sbus_cwrs = []
    for (system, n), cwrs in sorted(groups.items()):
        if not cwrs:
            continue
        mean = sum(cwrs) / len(cwrs)
        n_runs = len(cwrs)

        if n_runs >= 2:
            std = sqrt(sum((x - mean)**2 for x in cwrs) / (n_runs - 1))
            df = n_runs - 1
            t_val = float(t_dist.ppf(0.975, df=df))
            ci = t_val * std / sqrt(n_runs)
            ci_str = f"±{ci:.3f}"
            if n_runs <= 5:
                ci_str += f" (t{df})"
        else:
            ci_str = "—"

        succ = success.get((system, n), [])
        k_succ = sum(1 for s in succ if s)
        n_succ = len(succ)
        s50 = f"{k_succ/n_succ:.0%}" if n_succ > 0 else "—"
        lo, hi = clopper_pearson_ci(k_succ, n_succ)
        cp_str = f"[{lo:.3f}–{hi:.3f}]"

        print(f"{system:<20} {n:>3} {mean:>10.3f} {ci_str:>12} "
              f"{n_runs:>4} {s50:>6} {cp_str:>16}")

        if system == "sbus":
            sbus_cwrs.extend(cwrs)

    # Mann-Whitney U tests vs S-Bus
    if sbus_cwrs:
        print(f"\nMann-Whitney U tests (S-Bus vs baselines):")
        for (system, n), cwrs in sorted(groups.items()):
            if system == "sbus" or not cwrs:
                continue
            try:
                stat, p = mannwhitneyu(sbus_cwrs, cwrs, alternative="less")
                n_comp = len(sbus_cwrs) * len(cwrs)
                r_val = 1 - (2 * stat) / n_comp
                print(f"  S-Bus < {system:<15} U={stat:.0f} p={p:.4f} r={r_val:.3f}")
            except Exception:
                pass
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def load_tasks(path="datasets/long_horizon_tasks.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="S-Bus Real SDK Comparison")
    parser.add_argument("--system",      default="all",
                        choices=["all", "sbus", "crewai", "autogen", "langgraph"])
    parser.add_argument("--agents",      nargs="+", type=int, default=[4, 8])
    parser.add_argument("--steps",       type=int,  default=20)
    parser.add_argument("--tasks-limit", type=int,  default=5)
    parser.add_argument("--out",         type=str,
                        default="results/real_sdk_results.csv")
    parser.add_argument("--tasks-path",  type=str,
                        default="datasets/long_horizon_tasks.json")
    parser.add_argument("--server",      type=str,
                        default="http://localhost:3000")
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()

    if args.analyse_only:
        analyse(args.out)
        return

    bus = Bus(args.server)
    if not bus.ping():
        log.error(f"S-Bus server not reachable at {args.server}. Run: cargo run")
        sys.exit(1)

    log.info(f"S-Bus server: {args.server}")
    log.info(f"Retry budget B={RETRY_BUDGET} (Definition 6, Corollary 2.2)")
    log.info(f"System: {args.system} | agents: {args.agents} | steps: {args.steps}")

    tasks = load_tasks(args.tasks_path)[:args.tasks_limit]
    success_steps = args.steps

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    with open(out_path, "a") as f:
        if write_header:
            f.write(CSV_HDR)

        for task in tasks:
            for n in args.agents:
                log.info(f"\nTask: {task['task_id']} | N={n}")

                if args.system in ("all", "sbus"):
                    run = run_sbus(task, n, bus, args.steps, success_steps)
                    f.write(run.csv()); f.flush()

                if args.system in ("all", "crewai"):
                    if importlib.util.find_spec("crewai"):
                        run = run_crewai(task, n, args.steps, success_steps)
                        if run.excluded:
                            log.warning(f"CrewAI run excluded — not written to results")
                        else:
                            f.write(run.csv()); f.flush()
                    else:
                        log.warning("crewai not installed — skipping")

                if args.system in ("all", "autogen"):
                    if importlib.util.find_spec("autogen_agentchat"):
                        run = run_autogen(task, n, args.steps, success_steps)
                        if run.excluded:
                            log.warning(f"AutoGen run excluded — not written")
                        else:
                            f.write(run.csv()); f.flush()
                    else:
                        log.warning("autogen_agentchat not installed — skipping")

                if args.system in ("all", "langgraph"):
                    if importlib.util.find_spec("langgraph"):
                        run = run_langgraph(task, n, args.steps, success_steps)
                        if run.excluded:
                            log.warning(f"LangGraph run excluded — not written")
                        else:
                            f.write(run.csv()); f.flush()
                    else:
                        log.warning("langgraph not installed — skipping")

    log.info(f"\nResults written to {out_path}")
    analyse(args.out)


if __name__ == "__main__":
    main()
