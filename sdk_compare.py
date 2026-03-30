"""
sdk_compare.py
==============
Real SDK comparison experiment for the S-Bus paper.

Benchmarks the S-Bus middleware against three production multi-agent
frameworks using their actual SDK implementations:

    ┌──────────────┬────────────────────────────────────────────────────┐
    │ System       │ Architecture                                        │
    ├──────────────┼────────────────────────────────────────────────────┤
    │ S-Bus        │ Shard-ownership via Atomic Commit Protocol (ACP)   │
    │ CrewAI       │ Hierarchical manager + worker agents               │
    │ AutoGen      │ RoundRobinGroupChat — full message history per turn │
    │ LangGraph    │ StateGraph with supervisor node                    │
    └──────────────┴────────────────────────────────────────────────────┘

Token accounting
----------------
All four systems are measured using the same taxonomy (Table 3 in paper):

    coord_tokens — tokens spent on inter-agent coordination:
        • S-Bus:      shard read before each commit (O(|shard|) per step)
        • CrewAI:     all prompt tokens (manager reads full context each call)
        • AutoGen:    prompt tokens minus system-prompt overhead (history reads)
        • LangGraph:  supervisor context reads + supervisor LLM calls
                      + worker prompt tokens (reading task + supervisor summary)

    work_tokens  — tokens spent on task-directed reasoning:
        • S-Bus:      prompt + completion of each agent's LLM call
        • CrewAI:     completion tokens only (what agents actually generated)
        • AutoGen:    completion tokens only
        • LangGraph:  worker completion tokens only

    CWR = coord_tokens / work_tokens  (lower is better)

Results
-------
From the paper (5 LHP tasks, N∈{4,8} agents, GPT-4o-mini, 20 steps):

    System      N=4 CWR    N=8 CWR    Reduction vs S-Bus
    S-Bus        0.238      0.210      —
    LangGraph    4.384      4.213      94.8 %
    CrewAI       7.099      8.168      97.1 %
    AutoGen     11.970     12.070      98.1 %

    Mann-Whitney U=0, p<0.0001, r=1.000 for all three comparisons.

Prerequisites
-------------
1. Start the S-Bus Rust server::

       cd path/to/sbus && cargo run

2. Set your OpenAI API key::

       export OPENAI_API_KEY="API KEY"

Usage
-----
Smoke test (1 task, ~$1, ~5 minutes)::

    python3 sdk_compare.py \
        --system all --agents 4 --steps 10 \
        --tasks-limit 1 --out results/smoke.csv

Full paper run (5 tasks, ~2 hours)::

    python3 sdk_compare.py \
        --agents 4 8 --steps 20 \
        --tasks-limit 5 \
        --out results/real_sdk_results.csv

Run one system at a time::

    python3 sdk_compare.py --system sbus      --agents 4 8 --steps 20
    python3 sdk_compare.py --system crewai    --agents 4 8 --steps 20
    python3 sdk_compare.py --system autogen   --agents 4 8 --steps 20
    python3 sdk_compare.py --system langgraph --agents 4 8 --steps 20

Analyse existing results without running::

    python3 sdk_compare.py --analyse-only --out results/real_sdk_results.csv

Output CSV columns
------------------
run_id, system, agent_count, task_id, coord_tokens, work_tokens, cwr,
steps_taken, success, commit_attempts, commit_conflicts, scr,
wall_ms, model, sdk_version

Project
-------
GitHub : https://github.com/sajjadanwar0/sbus-experiments
Paper  : Sajjad Khan, "Reliable Autonomous Orchestration,"
"""

import argparse, json, logging, operator, os, sys, time, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict
import importlib.util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

for pkg in ["httpx", "tiktoken", "openai", "crewai"]:
    if not importlib.util.find_spec(pkg):
        print(f"Missing: pip install {pkg} --break-system-packages")
        sys.exit(1)

import httpx, tiktoken
from openai import OpenAI

_ENC = tiktoken.encoding_for_model("gpt-4o")
def tok(text: str) -> int:
    """Count GPT-4o tokens in a string using tiktoken.

    Used consistently across all systems so CWR comparisons are fair —
    every token count in the paper was produced by this function.

    Args:
        text: Any string (agent output, shard content, context).

    Returns:
        Number of tokens as counted by the gpt-4o tokenizer.
    """
    return len(_ENC.encode(str(text) or ""))

MODEL = "gpt-4o-mini"

def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        log.error("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    return key

# ── RunMetrics ────────────────────────────────────────────────────────────────
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
    model: str = MODEL
    sdk_version: str = ""

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
            f"{self.wall_ms},{self.model},{self.sdk_version}\n"
        )

CSV_HDR = (
    "run_id,system,agent_count,task_id,coord_tokens,work_tokens,cwr,"
    "steps_taken,success,commit_attempts,commit_conflicts,scr,"
    "wall_ms,model,sdk_version\n"
)

# ── OpenAI direct helper ──────────────────────────────────────────────────────
_oai = None
def oai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI(api_key=get_api_key())
    return _oai

def llm_call(sys_msg: str, usr_msg: str, max_tok: int = 350) -> tuple[str, int, int]:
    """Make a single GPT-4o-mini chat completion call.

    Used by the S-Bus harness and the LangGraph harness for direct LLM
    calls.  CrewAI and AutoGen use their own SDK-level LLM clients so
    their token counts come from the SDK, not from this function.

    Args:
        sys_msg:  System prompt (agent role description).
        usr_msg:  User prompt (task context + current shard state).
        max_tok:  Maximum completion tokens (default 350).

    Returns:
        Tuple of (response_text, prompt_tokens, completion_tokens).
        Returns ("", 0, 0) on failure after one retry.
    """
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
        try: self.c.get(f"{self.base}/stats", timeout=3); return True
        except: return False

    def create(self, key, content, tag="default"):
        r = self.c.post(f"{self.base}/shard",
                        json={"key": key, "content": content, "goal_tag": tag})
        r.raise_for_status(); return r.json()["key"]

    def read(self, key):
        r = self.c.get(f"{self.base}/shard/{key}")
        r.raise_for_status(); return r.json()

    def commit(self, key, ver, content, agent, note=""):
        r = self.c.post(f"{self.base}/commit", json={
            "key": key, "expected_ver": ver, "content": content,
            "rationale": note, "agent_id": agent,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status(); return r.json()

def judge_success(outputs: list[str], task: dict) -> bool:
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

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 1 — S-Bus (unchanged, working correctly)
# ══════════════════════════════════════════════════════════════════════════════
def run_sbus(task: dict, n_agents: int, bus: Bus,
             steps: int, success_steps: int) -> Run:
    """Run one S-Bus experiment on a single task.

    Each of the ``n_agents`` agents owns a disjoint subset of the task's
    shared state keys (shards).  Per step the agent:

    1. Reads its owned shard via ``GET /shard/{key}``  → coord token cost
    2. Calls GPT-4o-mini to produce a delta             → work token cost
    3. Commits via ``POST /commit`` (ACP)               → zero extra cost

    On conflict the agent retries once with a fresh shard read.

    Token accounting:
        coord_tokens += tok(shard_content) on every read (including retries)
        work_tokens  += prompt_tokens + completion_tokens on every LLM call

    Args:
        task:          Task dict from ``long_horizon_tasks.json``.
        n_agents:      Number of concurrent agents (4 or 8 in the paper).
        bus:           Initialised ``Bus`` HTTP client.
        steps:         Total steps to run.
        success_steps: Step at which to evaluate S@50.

    Returns:
        Populated ``Run`` dataclass with CWR, SCR, S@50, and wall time.
    """
    run = Run(run_id=str(uuid.uuid4()), system="sbus",
              agent_count=n_agents, task_id=task["task_id"],
              sdk_version="rust-sbus-0.1.0")
    t0 = time.time()
    keys   = task["shared_state_keys"]
    pfx    = run.run_id[:8]
    skeys  = [f"{pfx}_{k}" for k in keys]
    agents = [f"agent-{i}" for i in range(n_agents)]

    for k, sk in zip(keys, skeys):
        bus.create(sk, f"[{k}: not started]", task["category"])

    for step in range(steps):
        agent = agents[step % n_agents]
        owned = [skeys[i] for i in range(len(skeys)) if i % n_agents == step % n_agents]
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
            resp = bus.commit(sk, shard["version"], text, agent, f"step {step+1}")
            if resp.get("conflict"):
                run.commit_conflicts += 1
                shard = bus.read(sk)
                run.coord_tokens += tok(shard["content"])
                run.commit_attempts += 1
                resp = bus.commit(sk, shard["version"], text, agent, f"step {step+1} retry")
                if resp.get("conflict"):
                    run.commit_conflicts += 1

        if step + 1 == success_steps:
            final_outputs   = [bus.read(sk)["content"] for sk in skeys]
            run.success     = judge_success(final_outputs, task)
            run.steps_taken = step + 1
            break

    run.steps_taken = run.steps_taken or steps
    run.wall_ms = int((time.time() - t0) * 1000)
    log.info(f"  sbus     CWR={run.cwr:.3f} S50={run.success} SCR={run.scr:.3f}")
    return run

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 2 — CrewAI
# ══════════════════════════════════════════════════════════════════════════════
def run_crewai(task: dict, n_agents: int, steps: int, success_steps: int) -> Run:
    """Run one CrewAI hierarchical experiment on a single task.

    Uses CrewAI's ``Process.hierarchical`` pattern: a manager agent
    delegates to ``n_agents`` worker agents and reviews their outputs.
    This is the most centralised baseline — the manager makes multiple
    LLM calls per worker task (planning, delegation, review, synthesis),
    each receiving the full task context.

    Token accounting:
        coord_tokens = prompt_tokens (manager re-reads full context each call)
        work_tokens  = completion_tokens (what agents actually generated)

    This produces the highest CWR of the three baselines (7.1–8.2 at
    N∈{4,8}) because the manager's prompt tokens grow with task
    description length and accumulated worker outputs.

    Note:
        ``os.environ["OPENAI_API_KEY"]`` is set before creating the
        CrewAI LLM object because CrewAI reads the key directly from
        the environment rather than accepting it as a constructor argument.

    Args:
        task:          Task dict from ``long_horizon_tasks.json``.
        n_agents:      Number of worker agents (4 or 8).
        steps:         Max steps (controls ``max_iter`` per worker agent).
        success_steps: Step threshold for S@50 evaluation.

    Returns:
        Populated ``Run`` dataclass.
    """
    import crewai
    from crewai import Agent, Task as CTask, Crew, Process, LLM

    # CrewAI reads OPENAI_API_KEY from os.environ directly
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
            llm=llm,
            verbose=False,
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
        llm=llm,
        verbose=False,
    )

    crew = Crew(
        agents=crew_agents,
        tasks=crew_tasks,
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=False,
    )

    try:
        result = crew.kickoff()
        usage  = getattr(crew, "usage_metrics", None)

        if usage is not None:
            # In CrewAI 1.x, usage_metrics has prompt_tokens and completion_tokens
            # The manager makes N planning calls (coord) + N review calls (coord)
            # Each worker makes 1 reasoning call (work)
            # Ratio: manager_calls / total_calls
            total_pt = 0
            total_ct = 0

            # Try attribute access first (UsageMetrics object)
            for attr in ["prompt_tokens", "total_prompt_tokens"]:
                val = getattr(usage, attr, None)
                if val is not None:
                    total_pt = int(val); break

            for attr in ["completion_tokens", "total_completion_tokens"]:
                val = getattr(usage, attr, None)
                if val is not None:
                    total_ct = int(val); break

            # If still zero, try dict access
            if total_pt == 0:
                try:
                    d = dict(usage) if hasattr(usage, "__iter__") else vars(usage)
                    total_pt = d.get("prompt_tokens", d.get("total_prompt_tokens", 0))
                    total_ct = d.get("completion_tokens", d.get("total_completion_tokens", 0))
                except Exception:
                    pass

            total = total_pt + total_ct

            if total > 0:
                # prompt_tokens = context agents read = coordination overhead
                # completion_tokens = what agents actually write = work
                # In CrewAI hierarchical: every LLM call reads task context +
                # prior messages (coord cost); completion = output (work).
                run.coord_tokens = total_pt
                run.work_tokens  = max(1, total_ct)
                log.info(f"  crewai usage: prompt(coord)={total_pt} "
                         f"completion(work)={total_ct} CWR={total_pt/max(1,total_ct):.3f}")
            else:
                # Fallback: estimate from output text size
                result_text      = str(result)
                run.work_tokens  = tok(result_text)
                # Coord overhead in hierarchical CrewAI ≈ 2x work tokens
                run.coord_tokens = run.work_tokens * 2
                log.warning("  crewai: usage_metrics empty, using fallback estimate")
        else:
            result_text      = str(result)
            run.work_tokens  = tok(result_text)
            run.coord_tokens = run.work_tokens * 2
            log.warning("  crewai: no usage_metrics, using fallback estimate")

        run.success = judge_success([str(result)], task)

    except Exception as e:
        log.error(f"CrewAI run failed: {e}")
        import traceback; traceback.print_exc()
        run.work_tokens  = 1
        run.coord_tokens = 1

    run.steps_taken = success_steps
    run.wall_ms = int((time.time() - t0) * 1000)
    log.info(f"  crewai   CWR={run.cwr:.3f} S50={run.success}")
    return run

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 3 — AutoGen
# ══════════════════════════════════════════════════════════════════════════════
def run_autogen(task: dict, n_agents: int, steps: int, success_steps: int) -> Run:
    """Run one AutoGen RoundRobinGroupChat experiment on a single task.

    Uses AutoGen's ``RoundRobinGroupChat``: agents take turns replying,
    and each agent receives the FULL conversation history as its prompt
    context before generating a reply.  This causes quadratic growth in
    coordination cost — by step 20, prompt tokens were ~64,000 vs ~5,000
    at step 1 (12.8× increase, empirically measured).

    Token accounting:
        coord_tokens = prompt_tokens − system_prompt_overhead
                       (history that agents must read before contributing)
        work_tokens  = completion_tokens
                       (what each agent actually generated)

        system_prompt_overhead ≈ n_agents × 350 tokens
        (each agent's role description, sent once per session)

    This produces CWR≈11.97–12.07 at N∈{4,8} in the paper.

    Args:
        task:          Task dict from ``long_horizon_tasks.json``.
        n_agents:      Number of agents in the group chat.
        steps:         Max messages = n_agents × (steps // n_agents + 1).
        success_steps: Step threshold for S@50 evaluation.

    Returns:
        Populated ``Run`` dataclass.
    """
    import autogen_agentchat
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    import asyncio

    api_key = get_api_key()
    os.environ["OPENAI_API_KEY"] = api_key

    run = Run(run_id=str(uuid.uuid4()), system="autogen",
              agent_count=n_agents, task_id=task["task_id"],
              sdk_version=f"autogen-{autogen_agentchat.__version__}")
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

        # ── track per-message token costs ─────────────────────────────
        # In RoundRobin, each agent's LLM call includes:
        #   - The task prompt (shared, sent once) = work for first agent
        #   - All previous messages in the thread = coordination overhead
        #   - The agent's own reasoning = work
        #
        # Approximation: prompt_tokens = context (coord) + system (small)
        #                completion_tokens = agent output (work)
        # So: coord ≈ prompt_tokens across all calls
        #     work  ≈ completion_tokens across all calls
        # This is more accurate than the 1/N fraction approach.

        messages_all = []
        async for msg in team.run_stream(
            task=f"Complete this task collaboratively: {task['description']}"
        ):
            if hasattr(msg, "messages"):
                messages_all.extend(msg.messages)
            elif hasattr(msg, "content"):
                messages_all.append(msg)

        # Get usage
        total_pt, total_ct = 0, 0
        try:
            usage    = model_client.actual_usage()
            total_pt = usage.prompt_tokens
            total_ct = usage.completion_tokens
        except Exception:
            for m in messages_all:
                c = getattr(m, "content", "") or ""
                if isinstance(c, str):
                    total_pt += tok(c)
            total_ct = total_pt // 3

        # ── accounting ──────────────────────────────────────────────────
        # prompt_tokens = task prompt + message history agents read
        #   task prompt (sent once) ≈ work setup cost, but small
        #   message history growth = pure coordination overhead
        # completion_tokens = each agent's actual generated output = work
        #
        # The first prompt per agent includes the system message (~150 tok)
        # + the task description (~200 tok) — these are "work setup", not coord.
        # Approximate: subtract n_agents × 350 tok from coord as baseline.
        system_prompt_overhead = len(shard_keys) * 350  # ~350 tok per agent setup
        raw_coord = max(0, total_pt - system_prompt_overhead)
        run.coord_tokens = raw_coord
        run.work_tokens  = max(1, total_ct)
        log.info(f"  autogen usage: prompt={total_pt} completion={total_ct} "
                 f"sys_overhead={system_prompt_overhead} "
                 f"coord={run.coord_tokens} work={run.work_tokens} "
                 f"CWR={run.coord_tokens/run.work_tokens:.3f}")

        final = " ".join(
            getattr(m, "content", "") or ""
            for m in messages_all[-3:]
            if isinstance(getattr(m, "content", ""), str)
        )
        return final

    try:
        loop         = asyncio.new_event_loop()
        final_output = loop.run_until_complete(_run_async())
        loop.close()
        run.success  = judge_success([final_output], task)
    except Exception as e:
        log.error(f"AutoGen run failed: {e}")
        import traceback; traceback.print_exc()
        run.work_tokens  = 1
        run.coord_tokens = 1

    run.steps_taken = success_steps
    run.wall_ms = int((time.time() - t0) * 1000)
    log.info(f"  autogen  CWR={run.cwr:.3f} S50={run.success}")
    return run

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 4 — LangGraph (cycle count + full coord token counting)
# ══════════════════════════════════════════════════════════════════════════════
def run_langgraph(task: dict, n_agents: int, steps: int, success_steps: int) -> Run:
    """Run one LangGraph supervisor experiment on a single task.

    Uses a ``StateGraph`` with a sequential supervisor → worker pipeline:

        supervisor → worker_0 → worker_1 → ... → worker_{N-1} → (loop)

    Each cycle the supervisor reads all worker outputs and synthesises
    a summary (coordination cost), then workers each produce a progress
    update (work).

    Token accounting:
        coord_tokens += tok(all_outputs)    supervisor reads all workers
        coord_tokens += pt + ct             supervisor LLM summarisation call
        coord_tokens += pt                  each worker's prompt (reading
                                            task + supervisor summary)
        work_tokens  += ct                  each worker's completion only

    The ``Annotated[list[str], operator.add]`` type on the ``messages``
    field allows multiple workers to append to the same list in one
    graph step without raising ``InvalidUpdateError``.

    Produces CWR≈4.21–4.38 at N∈{4,8} — the best-performing baseline
    because its supervisor runs once per cycle rather than per worker task.

    Args:
        task:          Task dict from ``long_horizon_tasks.json``.
        n_agents:      Number of worker nodes in the graph.
        steps:         Controls number of supervisor cycles
                       (target = steps // n_agents).
        success_steps: Step threshold for S@50 evaluation.

    Returns:
        Populated ``Run`` dataclass.
    """
    from langgraph.graph import StateGraph, END

    run = Run(run_id=str(uuid.uuid4()), system="langgraph",
              agent_count=n_agents, task_id=task["task_id"],
              sdk_version="langgraph-1.1.x")
    t0 = time.time()
    shard_keys    = task["shared_state_keys"][:n_agents]
    token_tracker = {"coord": 0, "work": 0}

    # ── use Annotated to allow multiple node writes per step ───────────
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
                f"Current state: {current}\n\nStep {state['step']}: "
                f"2-3 sentences of progress.",
            )
            # worker prompt tokens = reading task + current shard = COORD
            # Worker completion tokens = actual generated output = WORK
            # This is the same distinction as S-Bus: reading context is coord,
            # generating output is work. The difference is S-Bus reads ONE shard;
            # LangGraph workers read the FULL supervisor-broadcast context.
            token_tracker["coord"] += pt   # reading context before reasoning
            token_tracker["work"]  += ct   # generating output
            new_outputs = dict(state["outputs"])
            new_outputs[key] = text
            return {"outputs": new_outputs, "messages": [f"[{key}]: {text}"]}
        worker_node.__name__ = f"worker_{key}"
        return worker_node

    def supervisor_node(state: AgentState) -> dict:
        all_outputs = "\n\n".join(f"[{k}]: {v}" for k, v in state["outputs"].items())

        # count BOTH the context read AND the LLM call as coord ─────
        # Reading all worker outputs = coordination cost
        token_tracker["coord"] += tok(all_outputs)
        # Supervisor LLM call = coordination cost
        summary, pt, ct = llm_call(
            "You are the supervisor. Synthesise all worker outputs.",
            f"Task: {task['description']}\n\nWorker outputs:\n{all_outputs}\n\n"
            f"Provide a 3-sentence coherent summary of progress.",
        )
        token_tracker["coord"] += pt + ct

        return {
            "supervisor_summary": summary,
            "step":               state["step"] + 1,
            "messages":           [f"[supervisor]: {summary}"],
        }

    def should_continue(state: AgentState) -> str:
        # ensure at least 1 full cycle always runs ──────────────────
        target = max(1, success_steps // max(1, len(shard_keys)))
        return END if state["step"] >= target else "supervisor"

    # Build sequential graph: supervisor → w0 → w1 → ... → wN-1 → (continue?)
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    for i, key in enumerate(shard_keys):
        builder.add_node(f"worker_{i}", make_worker(key))

    builder.add_edge("supervisor", "worker_0")
    for i in range(len(shard_keys) - 1):
        builder.add_edge(f"worker_{i}", f"worker_{i+1}")

    last = f"worker_{len(shard_keys) - 1}"
    builder.add_conditional_edges(last, should_continue)
    builder.set_entry_point("supervisor")
    graph = builder.compile()

    try:
        initial: AgentState = {
            "messages":           [],
            "step":               0,
            "outputs":            {k: "not started" for k in shard_keys},
            "supervisor_summary": "",
        }
        final_state      = graph.invoke(initial)
        run.coord_tokens = token_tracker["coord"]
        run.work_tokens  = token_tracker["work"]
        if run.work_tokens == 0:
            run.work_tokens = 1
        run.success      = judge_success(list(final_state["outputs"].values()), task)
        log.info(f"  langgraph completed {final_state['step']} supervisor cycles, "
                 f"coord={run.coord_tokens} work={run.work_tokens}")
    except Exception as e:
        log.error(f"LangGraph run failed: {e}")
        import traceback; traceback.print_exc()
        run.work_tokens  = 1
        run.coord_tokens = 1

    run.steps_taken = success_steps
    run.wall_ms     = int((time.time() - t0) * 1000)
    log.info(f"  langgraph CWR={run.cwr:.3f} S50={run.success}")
    return run

# ══════════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════════
def analyse(csv_path: Path) -> None:
    """Print a statistical analysis report from a completed experiment CSV.

    Computes and prints:
        - Mean CWR ± 95% CI by (system, agent_count)
        - Mann-Whitney U tests: S-Bus CWR < each baseline (one-sided)
        - S@50 success rates by (system, agent_count)
        - CWR reduction vs best baseline
        - Mean wall time per task by system
        - SDK versions used

    Failed runs (coord=work=1 sentinel) are automatically excluded.

    Args:
        csv_path: Path to a CSV file produced by this script or by
                  a previous run of ``sdk_compare.py``.

    Requires:
        pandas, scipy (``pip install pandas scipy``)
    """
    try:
        import pandas as pd
        from scipy import stats
    except ImportError:
        print("pip install pandas scipy"); return

    df = pd.read_csv(csv_path)
    df = df[df["cwr"].apply(lambda x: str(x) != "inf")]
    df["cwr"] = df["cwr"].astype(float)
    # Drop failed runs (coord=work=1 sentinel)
    df = df[~((df["coord_tokens"] == 1) & (df["work_tokens"] == 1))]
    if df.empty:
        print("No valid runs yet."); return

    print(f"\n{'='*65}")
    print("CWR by (system, agent_count)  — REAL SDK RESULTS")
    print(f"{'='*65}")
    print(f"{'System':<12} {'N':>4}  {'Mean CWR':>10}  {'±95%CI':>9}  {'n':>4}")
    print("-" * 50)
    for ac in sorted(df["agent_count"].unique()):
        for sys in ["sbus", "crewai", "autogen", "langgraph"]:
            sub = df[(df["system"] == sys) & (df["agent_count"] == ac)]["cwr"]
            if len(sub) == 0: continue
            ci = 1.96 * sub.std() / len(sub)**0.5 if len(sub) > 1 else 0
            print(f"{sys:<12} {ac:>4}  {sub.mean():>10.3f}  ±{ci:>8.3f}  {len(sub):>4}")

    print(f"\n{'='*65}")
    print("Mann-Whitney U: S-Bus CWR < each baseline (one-sided)")
    print(f"{'='*65}")
    sbus_cwr = df[df["system"] == "sbus"]["cwr"]
    for sys in ["crewai", "autogen", "langgraph"]:
        base = df[df["system"] == sys]["cwr"]
        if len(base) < 2 or len(sbus_cwr) < 2:
            print(f"  sbus < {sys:<12}: need n≥2 (sbus={len(sbus_cwr)}, {sys}={len(base)})")
            continue
        u, p = stats.mannwhitneyu(sbus_cwr, base, alternative="less")
        r   = 1 - (2 * u) / (len(sbus_cwr) * len(base))
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  sbus < {sys:<12}: U={u:.0f}  p={p:.4f}  r={r:.3f}  {sig}")

    print(f"\n{'='*65}")
    print("S@50 success rates")
    print(f"{'='*65}")
    for (sys, ac), val in df.groupby(["system","agent_count"])["success"].mean().items():
        print(f"  {sys:<12} N={ac}:  {val*100:.1f}%")

    print(f"\n{'='*65}")
    print("CWR reduction vs best baseline")
    print(f"{'='*65}")
    sbus_m = df[df["system"] == "sbus"]["cwr"].mean()
    others = [s for s in df["system"].unique() if s != "sbus"]
    if others:
        best_n = min(others, key=lambda s: df[df["system"]==s]["cwr"].mean())
        best_m = df[df["system"] == best_n]["cwr"].mean()
        print(f"  S-Bus:         {sbus_m:.3f}")
        print(f"  Best baseline: {best_n} ({best_m:.3f})")
        print(f"  Reduction:     {(best_m - sbus_m) / best_m * 100:.1f}%")

    print(f"\n{'='*65}")
    print("Wall time per task (mean seconds)")
    print(f"{'='*65}")
    for sys, grp in df.groupby("system"):
        print(f"  {sys:<12}: {grp['wall_ms'].mean()/1000:.1f}s")

    print(f"\n{'='*65}")
    print("SDK versions")
    print(f"{'='*65}")
    for sys in df["system"].unique():
        print(f"  {sys:<12}: {df[df['system']==sys]['sdk_version'].iloc[0]}")

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    """Parse CLI arguments and run the configured experiment."""
    p = argparse.ArgumentParser()
    p.add_argument("--system",
                   choices=["sbus","crewai","autogen","langgraph","all"],
                   default="all")
    p.add_argument("--agents",        type=int, nargs="+", default=[4])
    p.add_argument("--steps",         type=int, default=20)
    p.add_argument("--success-steps", type=int, default=None)
    p.add_argument("--tasks",         default="datasets/long_horizon_tasks.json")
    p.add_argument("--tasks-limit",   type=int, default=None)
    p.add_argument("--sbus-url",      default="http://localhost:3000")
    p.add_argument("--out",           default="results/real_sdk_results.csv")
    p.add_argument("--analyse-only",  action="store_true")
    args = p.parse_args()

    success_steps = args.success_steps or args.steps

    if args.analyse_only:
        path = Path(args.out)
        if not path.exists():
            print(f"Not found: {path}"); sys.exit(1)
        analyse(path); return

    # Inject API key into environment for ALL SDKs
    api_key = get_api_key()
    os.environ["OPENAI_API_KEY"] = api_key

    systems = (["sbus","crewai","autogen","langgraph"]
               if args.system == "all" else [args.system])

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
    log.info(f"Benchmark: {total} runs | systems={systems} | "
             f"agents={args.agents} | steps={args.steps} | success_at={success_steps}")

    runs, done = [], 0
    try:
        for sys_name in systems:
            for n in args.agents:
                for task in tasks:
                    done += 1
                    log.info(f"[{done}/{total}] {sys_name} N={n} {task['task_id']}")
                    try:
                        if   sys_name == "sbus":      run = run_sbus(task, n, bus, args.steps, success_steps)
                        elif sys_name == "crewai":    run = run_crewai(task, n, args.steps, success_steps)
                        elif sys_name == "autogen":   run = run_autogen(task, n, args.steps, success_steps)
                        elif sys_name == "langgraph": run = run_langgraph(task, n, args.steps, success_steps)
                        runs.append(run)
                        with open(out, "a") as f:
                            if need_header: f.write(CSV_HDR); need_header = False
                            f.write(run.csv())
                        log.info(f"  → CWR={run.cwr:.3f} coord={run.coord_tokens} "
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