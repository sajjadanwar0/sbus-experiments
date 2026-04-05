"""
sdk_compare.py  —  S-Bus paper SDK comparison experiment.
==========================================================

Supports three LLM providers for both agent backbone and judge:

    Provider      Models                    Env var required
    ──────────    ──────────────────────    ──────────────────────────────
    openai        gpt-4o-mini (default)     OPENAI_API_KEY
    anthropic     claude-haiku-3            ANTHROPIC_API_KEY
    groq          llama-3.1-8b-instant      GROQ_API_KEY
                  llama-3.3-70b-versatile
    ollama        llama3.2 (local, free)    none (server at localhost:11434)

Usage:
    # Default (GPT-4o-mini backbone + judge):
    python sdk_compare.py --steps 50 --tasks-limit 5

    # Claude Haiku backbone, GPT-4o-mini judge:
    python sdk_compare.py --model claude-haiku-3 --judge gpt-4o-mini

    # Llama via Groq backbone, Claude judge:
    python sdk_compare.py --model llama-3.1-8b-instant --judge claude-haiku-3

    # Llama local via Ollama backbone, GPT judge:
    python sdk_compare.py --model ollama/llama3.2 --judge gpt-4o-mini

    # Full multi-model paper run (all three backbones):
    python sdk_compare.py --multi-model --steps 50 --tasks-limit 30 --runs 3

    # Analyse only:
    python sdk_compare.py --analyse-only --out results/real_sdk_results.csv

Prerequisites:
    cargo run                        # S-Bus server on :7000
    pip install openai anthropic httpx tiktoken scipy

    For Groq Llama:
        pip install groq
        export GROQ_API_KEY="gsk_..."

    For local Ollama Llama:
        curl https://ollama.ai/install.sh | sh
        ollama pull llama3.2
        ollama serve                 # runs on localhost:11434
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
import importlib.util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Dependency check ─────────────────────────────────────────────────────────

for pkg in ["httpx", "tiktoken", "openai", "scipy"]:
    if not importlib.util.find_spec(pkg):
        print(f"Missing: pip install {pkg}")
        sys.exit(1)

import httpx
import tiktoken
from openai import OpenAI
from scipy.stats import binom as scipy_binom

_ENC = tiktoken.encoding_for_model("gpt-4o")

# ── Paper SDK versions ────────────────────────────────────────────────────────

EXPECTED_VERSIONS = {
    "crewai":    "1.12.2",
    "autogen":   "0.7.5",
    "langgraph": "1.1.3",
}

RETRY_BUDGET = int(os.environ.get("SBUS_RETRY_BUDGET", "1"))


def tok(text: str) -> int:
    return len(_ENC.encode(str(text) or ""))


# ── Provider router ───────────────────────────────────────────────────────────
#
#  All LLM calls go through llm_call(). The provider is selected by
#  the model name prefix or an explicit flag:
#
#    gpt-*            → OpenAI
#    claude-*         → Anthropic
#    llama-* / groq/* → Groq (OpenAI-compatible)
#    ollama/*         → local Ollama (OpenAI-compatible on :11434)
#
#  Returns: (text, prompt_tokens, completion_tokens)

_openai_client: OpenAI | None = None
_anthropic_client = None     # lazy import
_groq_client = None          # lazy import

def _get_openai() -> OpenAI | None:
    global _openai_client
    if _openai_client is None:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            log.error("OPENAI_API_KEY not set.")
            sys.exit(1)
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        if not importlib.util.find_spec("anthropic"):
            log.error("anthropic package not installed. Run: pip install anthropic")
            sys.exit(1)
        import anthropic  # type: ignore
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            log.error("ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


def _get_groq():
    global _groq_client
    if _groq_client is None:
        if not importlib.util.find_spec("groq"):
            log.error("groq package not installed. Run: pip install groq")
            sys.exit(1)
        from groq import Groq  # type: ignore
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            log.error("GROQ_API_KEY not set.")
            sys.exit(1)
        _groq_client = Groq(api_key=key)
    return _groq_client


def _get_ollama() -> OpenAI:
    """
    Ollama exposes an OpenAI-compatible endpoint on :11434.
    No API key required. Model name must be stripped of 'ollama/' prefix.
    """
    return OpenAI(
        api_key="ollama",           # placeholder — Ollama ignores the key
        base_url="http://localhost:11434/v1",
    )


def _provider_for(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("ollama/"):
        return "ollama"
    if m.startswith("llama") or m.startswith("groq/") or m.startswith("mixtral"):
        return "groq"
    # Fallback: assume OpenAI-compatible
    return "openai"


def llm_call(
    sys_msg: str,
    usr_msg: str,
    model: str = "gpt-4o-mini",
    max_tok: int = 350,
) -> tuple[str, int, int]:
    """
    Single unified LLM call. Returns (text, prompt_tokens, completion_tokens).
    Provider selected automatically from model name.
    """
    provider = _provider_for(model)

    for attempt in range(3):
        try:
            if provider == "anthropic":
                return _call_anthropic(sys_msg, usr_msg, model, max_tok)
            elif provider == "ollama":
                actual_model = model.replace("ollama/", "")
                return _call_openai_compat(
                    _get_ollama(), sys_msg, usr_msg, actual_model, max_tok
                )
            elif provider == "groq":
                actual_model = model.replace("groq/", "")
                return _call_groq(sys_msg, usr_msg, actual_model, max_tok)
            else:  # openai
                return _call_openai_compat(
                    _get_openai(), sys_msg, usr_msg, model, max_tok
                )

        except Exception as e:
            err_str = str(e).lower()
            if attempt < 2 and ("rate" in err_str or "429" in err_str or "overloaded" in err_str):
                wait = 20 * (attempt + 1)
                log.warning(f"Rate limit ({provider}/{model}) — sleeping {wait}s")
                time.sleep(wait)
            else:
                log.error(f"LLM error [{provider}/{model}] attempt {attempt+1}: {e}")
                if attempt == 2:
                    return "", 0, 0

    return "", 0, 0


def _call_openai_compat(
    client: OpenAI,
    sys_msg: str,
    usr_msg: str,
    model: str,
    max_tok: int,
) -> tuple[str, int, int]:
    r = client.chat.completions.create(
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
        r.usage.prompt_tokens if r.usage else tok(sys_msg + usr_msg),
        r.usage.completion_tokens if r.usage else tok(r.choices[0].message.content or ""),
    )


def _call_anthropic(
    sys_msg: str,
    usr_msg: str,
    model: str,
    max_tok: int,
) -> tuple[str, int, int]:
    """
    Anthropic SDK call.

    Model mapping:
        claude-haiku-3        → claude-3-haiku-20240307
        claude-haiku-3.5      → claude-3-5-haiku-20241022
        claude-sonnet-3.5     → claude-3-5-sonnet-20241022
    """
    MODEL_MAP = {
        "claude-haiku-3":    "claude-3-haiku-20240307",
        "claude-haiku-3.5":  "claude-3-5-haiku-20241022",
        "claude-sonnet-3.5": "claude-3-5-sonnet-20241022",
        "claude-opus-3":     "claude-3-opus-20240229",
    }
    api_model = MODEL_MAP.get(model, model)
    client = _get_anthropic()
    r = client.messages.create(
        model=api_model,
        max_tokens=max_tok,
        system=sys_msg,
        messages=[{"role": "user", "content": usr_msg}],
        temperature=0.2,
    )
    text = r.content[0].text if r.content else ""
    pt = r.usage.input_tokens if r.usage else tok(sys_msg + usr_msg)
    ct = r.usage.output_tokens if r.usage else tok(text)
    return text, pt, ct


def _call_groq(
    sys_msg: str,
    usr_msg: str,
    model: str,
    max_tok: int,
) -> tuple[str, int, int]:
    """
    Groq API call (Llama, Mixtral, etc.)

    Recommended fast models:
        llama-3.1-8b-instant     (~500 tok/s, cheapest)
        llama-3.3-70b-versatile  (best quality)
        mixtral-8x7b-32768       (good balance)
    """
    time.sleep(2.5)
    client = _get_groq()
    r = client.chat.completions.create(
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
        r.usage.prompt_tokens if r.usage else tok(sys_msg + usr_msg),
        r.usage.completion_tokens if r.usage else tok(r.choices[0].message.content or ""),
    )


# ── Clopper-Pearson CI ────────────────────────────────────────────────────────

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    if k == 0:
        lo, hi = 0.0, float(1 - (alpha / 2) ** (1 / n))
    elif k == n:
        lo, hi = float((alpha / 2) ** (1 / n)), 1.0
    else:
        lo = float(scipy_binom.ppf(alpha / 2,     n, k / n) / n)
        hi = float(scipy_binom.ppf(1 - alpha / 2, n, k / n) / n)
    return round(lo, 3), round(hi, 3)


# ── RunMetrics ────────────────────────────────────────────────────────────────

@dataclass
class Run:
    run_id:           str
    system:           str
    agent_count:      int
    task_id:          str
    backbone_model:   str   = "gpt-4o-mini"
    judge_model:      str   = "gpt-4o-mini"
    coord_tokens:     int   = 0
    work_tokens:      int   = 0
    steps_taken:      int   = 0
    success:          bool  = False
    commit_attempts:  int   = 0
    commit_conflicts: int   = 0
    retry_exhaustions:int   = 0
    excluded:         bool  = False
    wall_ms:          int   = 0
    sdk_version:      str   = ""

    @property
    def cwr(self) -> float:
        if self.excluded or self.work_tokens <= 0:
            return float("inf")
        return self.coord_tokens / self.work_tokens

    @property
    def scr(self) -> float:
        if self.commit_attempts == 0:
            return 0.0
        return self.commit_conflicts / self.commit_attempts

    def csv(self) -> str:
        cwr = f"{self.cwr:.4f}" if self.cwr != float("inf") else "inf"
        return (
            f"{self.run_id},{self.system},{self.agent_count},{self.task_id},"
            f"{self.backbone_model},{self.judge_model},"
            f"{self.coord_tokens},{self.work_tokens},{cwr},"
            f"{self.steps_taken},{int(self.success)},"
            f"{self.commit_attempts},{self.commit_conflicts},{self.scr:.4f},"
            f"{self.retry_exhaustions},{int(self.excluded)},"
            f"{self.wall_ms},{self.sdk_version}\n"
        )


CSV_HDR = (
    "run_id,system,agent_count,task_id,backbone_model,judge_model,"
    "coord_tokens,work_tokens,cwr,"
    "steps_taken,success,"
    "commit_attempts,commit_conflicts,scr,"
    "retry_exhaustions,excluded,"
    "wall_ms,sdk_version\n"
)


# ── S-Bus HTTP client ─────────────────────────────────────────────────────────

class Bus:
    def __init__(self, url: str = "http://localhost:7000"):
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

    def commit(self, key: str, ver: int, content: str, agent: str, note: str = "") -> dict:
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
        r = self.c.post(f"{self.base}/commit/v2", json={
            "key": key, "expected_ver": ver, "content": content,
            "rationale": note, "agent_id": agent, "read_set": read_set,
        })
        if r.status_code in (409, 423):
            return {"conflict": True, "error": r.json().get("error", "")}
        r.raise_for_status()
        return r.json()


# ── Success judge ─────────────────────────────────────────────────────────────

def judge_success(outputs: list[str], task: dict, judge_model: str) -> bool:
    """
    Evaluate task success using the specified judge model.

    To eliminate same-family circularity (Section 8 limitation),
    use a different model family for judge vs backbone:
        backbone=gpt-4o-mini  → judge=claude-haiku-3
        backbone=claude-haiku-3 → judge=gpt-4o-mini
        backbone=llama-3.1-8b  → judge=gpt-4o-mini

    The judge model is passed explicitly from CLI --judge argument.
    """
    checks = task.get("ground_truth_outputs", [])
    if not checks:
        return all(len(o) > 80 for o in outputs if o)

    combined  = "\n\n".join(f"[Output {i+1}]:\n{o}" for i, o in enumerate(outputs))
    criteria  = "\n".join(f"- {c}" for c in checks)
    verdict, _, _ = llm_call(
        "You are an evaluator. Answer only YES or NO.",
        f"Task: {task['description']}\n\nSuccess criteria:\n{criteria}\n\n"
        f"Outputs:\n{combined}\n\nDo outputs satisfy ALL criteria? YES or NO.",
        model=judge_model,
        max_tok=5,
    )
    return verdict.strip().upper().startswith("YES")


# ── System 1: S-Bus ───────────────────────────────────────────────────────────

def run_sbus(
    task: dict,
    n_agents: int,
    bus: Bus,
    steps: int,
    backbone_model: str,
    judge_model: str,
) -> Run:
    """
    S-Bus decentralised shard-ownership experiment.

    Token accounting:
        coord_tokens += tok(shard_content) on each read (inc. retries)
        work_tokens  += prompt + completion on each LLM call
    """
    run = Run(
        run_id=str(uuid.uuid4()), system="sbus",
        agent_count=n_agents, task_id=task["task_id"],
        backbone_model=backbone_model, judge_model=judge_model,
        sdk_version="rust-sbus-0.1.0",
    )
    t0 = time.time()

    keys  = task["shared_state_keys"]
    pfx   = run.run_id[:8]
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
                model=backbone_model,
            )
            run.work_tokens += pt + ct
            run.commit_attempts += 1

            committed = False
            cur = shard
            for attempt in range(RETRY_BUDGET + 1):
                resp = bus.commit(sk, cur["version"], text, agent, f"step {step+1}")
                if not resp.get("conflict"):
                    committed = True
                    break
                run.commit_conflicts += 1
                if attempt < RETRY_BUDGET:
                    cur = bus.read(sk)
                    run.coord_tokens += tok(cur["content"])
                    run.commit_attempts += 1

            if not committed:
                run.retry_exhaustions += 1
                log.warning(
                    f"[{agent}] retry budget B={RETRY_BUDGET} exhausted "
                    f"at step {step+1} on {sk}"
                )

    final_outputs = [bus.read(sk)["content"] for sk in skeys]
    run.success    = judge_success(final_outputs, task, judge_model)
    run.steps_taken = steps
    run.wall_ms    = int((time.time() - t0) * 1000)
    log.info(
        f"  sbus [{backbone_model}] CWR={run.cwr:.3f} S50={run.success} "
        f"SCR={run.scr:.3f}"
    )
    return run


# ── System 2: CrewAI ──────────────────────────────────────────────────────────

def run_crewai(
    task: dict,
    n_agents: int,
    steps: int,
    backbone_model: str,
    judge_model: str,
) -> Run:
    """
    CrewAI hierarchical experiment.

    backbone_model support:
        gpt-*         → OpenAI LLM directly
        claude-*      → Anthropic via LiteLLM prefix "anthropic/"
        llama-*/groq/ → Groq via LiteLLM prefix "groq/"
        ollama/*      → Ollama via LiteLLM prefix "ollama/"

    Token accounting:
        coord_tokens = prompt_tokens (manager reads full task context)
        work_tokens  = completion_tokens
    """
    import crewai  # type: ignore
    sdk_ver = crewai.__version__
    if sdk_ver != EXPECTED_VERSIONS["crewai"]:
        log.warning(
            f"CrewAI version mismatch: expected {EXPECTED_VERSIONS['crewai']}, "
            f"got {sdk_ver}."
        )

    from crewai import Agent, Task as CTask, Crew, Process, LLM  # type: ignore

    run = Run(
        run_id=str(uuid.uuid4()), system="crewai",
        agent_count=n_agents, task_id=task["task_id"],
        backbone_model=backbone_model, judge_model=judge_model,
        sdk_version=f"crewai-{sdk_ver}",
    )
    t0 = time.time()

    # Build LiteLLM model string that CrewAI understands
    litellm_model = _to_litellm(backbone_model)
    llm_obj = LLM(model=litellm_model, temperature=0.2, max_tokens=350)

    shard_keys = task["shared_state_keys"][:n_agents]
    crew_agents, crew_tasks = [], []
    for i, key in enumerate(shard_keys):
        agent = Agent(
            role=f"Expert in {key.replace('_', ' ')}",
            goal=f"Make concrete progress on: {key}",
            backstory=(
                f"You are a specialist in {key} for: "
                f"{task['description'][:120]}"
            ),
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
        result  = crew.kickoff()
        usage   = getattr(crew, "usage_metrics", None)
        total_pt, total_ct = _extract_crew_usage(usage, sdk_ver)

        if total_pt == 0 and total_ct == 0:
            log.warning(
                f"CrewAI run {run.run_id} returned zero usage_metrics "
                f"(crewai=={sdk_ver}). Marking EXCLUDED."
            )
            run.excluded = True
        else:
            run.coord_tokens = total_pt
            run.work_tokens  = max(1, total_ct)
            run.success = judge_success([str(result)], task, judge_model)
            log.info(
                f"  crewai [{backbone_model}] CWR={run.cwr:.3f} "
                f"S50={run.success}"
            )
    except Exception as e:
        log.error(f"CrewAI run failed: {e}")
        import traceback; traceback.print_exc()
        run.excluded = True

    run.steps_taken = steps
    run.wall_ms     = int((time.time() - t0) * 1000)
    return run


def _extract_crew_usage(usage, sdk_ver: str) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    pt, ct = 0, 0
    for attr in ["prompt_tokens", "total_prompt_tokens"]:
        val = getattr(usage, attr, None)
        if val is not None:
            pt = int(val); break
    for attr in ["completion_tokens", "total_completion_tokens"]:
        val = getattr(usage, attr, None)
        if val is not None:
            ct = int(val); break
    if pt == 0:
        try:
            d  = dict(usage) if hasattr(usage, "__iter__") else vars(usage)
            pt = d.get("prompt_tokens", d.get("total_prompt_tokens", 0))
            ct = d.get("completion_tokens", d.get("total_completion_tokens", 0))
        except Exception:
            pass
    return int(pt), int(ct)


def _to_litellm(model: str) -> str:
    """
    Convert our model name to a LiteLLM prefix string that CrewAI/AutoGen accept.

    LiteLLM routing:
        gpt-*          → "gpt-..."           (no prefix, OpenAI default)
        claude-*       → "anthropic/claude-3-haiku-20240307" etc.
        llama-*/groq/* → "groq/llama-..."
        ollama/*       → "ollama/llama3.2" etc.
    """
    ANTHROPIC_MAP = {
        "claude-haiku-3":    "claude-3-haiku-20240307",
        "claude-haiku-3.5":  "claude-3-5-haiku-20241022",
        "claude-sonnet-3.5": "claude-3-5-sonnet-20241022",
    }
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return model
    if model.startswith("claude-"):
        api = ANTHROPIC_MAP.get(model, model)
        return f"anthropic/{api}"
    if model.startswith("ollama/"):
        return model  # LiteLLM handles "ollama/..." natively
    if model.startswith("groq/"):
        return model
    # Bare llama/mixtral → assume groq
    if model.startswith("llama") or model.startswith("mixtral"):
        return f"groq/{model}"
    return model


# ── System 3: AutoGen ─────────────────────────────────────────────────────────

def run_autogen(
    task: dict,
    n_agents: int,
    steps: int,
    backbone_model: str,
    judge_model: str,
) -> Run:
    """
    AutoGen RoundRobinGroupChat experiment.

    backbone_model support:
        gpt-*    → OpenAI (native)
        claude-* → OpenAI-compat via LiteLLM proxy OR direct Anthropic client
        llama-*  → Groq OpenAI-compatible endpoint

    Token accounting:
        coord_tokens = prompt_tokens - system_prompt_overhead
        work_tokens  = completion_tokens
    """
    import autogen_agentchat  # type: ignore
    sdk_ver = autogen_agentchat.__version__
    if sdk_ver != EXPECTED_VERSIONS["autogen"]:
        log.warning(
            f"AutoGen version mismatch: expected {EXPECTED_VERSIONS['autogen']}, "
            f"got {sdk_ver}."
        )

    from autogen_agentchat.agents import AssistantAgent  # type: ignore
    from autogen_agentchat.teams import RoundRobinGroupChat  # type: ignore
    from autogen_agentchat.conditions import MaxMessageTermination  # type: ignore
    import asyncio

    run = Run(
        run_id=str(uuid.uuid4()), system="autogen",
        agent_count=n_agents, task_id=task["task_id"],
        backbone_model=backbone_model, judge_model=judge_model,
        sdk_version=f"autogen-{sdk_ver}",
    )
    t0 = time.time()

    shard_keys = task["shared_state_keys"][:n_agents]

    async def _run_async():
        model_client = _make_autogen_client(backbone_model)
        agent_names  = [f"Agent_{i}_{key[:8]}" for i, key in enumerate(shard_keys)]
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
            usage   = model_client.actual_usage()
            total_pt = usage.prompt_tokens
            total_ct = usage.completion_tokens
        except Exception:
            for m in messages_all:
                c = getattr(m, "content", "") or ""
                if isinstance(c, str):
                    total_pt += tok(c)
            total_ct = total_pt // 3

        system_overhead      = len(shard_keys) * 350
        run.coord_tokens     = max(0, total_pt - system_overhead)
        run.work_tokens      = max(1, total_ct)

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
        run.success = judge_success([final_output], task, judge_model)
        log.info(
            f"  autogen [{backbone_model}] CWR={run.cwr:.3f} "
            f"S50={run.success}"
        )
    except Exception as e:
        log.error(f"AutoGen run failed: {e}")
        import traceback; traceback.print_exc()
        run.excluded = True

    run.steps_taken = steps
    run.wall_ms     = int((time.time() - t0) * 1000)
    return run


def _make_autogen_client(model: str):
    """Build the right AutoGen model client for the given model string."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient  # type: ignore

    provider = _provider_for(model)
    if provider == "openai":
        return OpenAIChatCompletionClient(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            max_tokens=350, temperature=0.2,
        )
    elif provider == "anthropic":
        # AutoGen does not have a native Anthropic client yet in 0.7.x.
        # Use the OpenAI-compat shim via LiteLLM proxy if available,
        # otherwise fall back to our llm_call wrapper inside a custom agent.
        # For simplicity: use litellm's OpenAI-compat wrapper if litellm installed.
        if importlib.util.find_spec("litellm"):
            import litellm  # type: ignore
            litellm.set_verbose = False
            ANTHROPIC_MAP = {
                "claude-haiku-3":   "claude-3-haiku-20240307",
                "claude-haiku-3.5": "claude-3-5-haiku-20241022",
            }
            api_model = ANTHROPIC_MAP.get(model, model)
            return OpenAIChatCompletionClient(
                model=f"anthropic/{api_model}",
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                max_tokens=350, temperature=0.2,
            )
        else:
            log.warning(
                f"litellm not installed; AutoGen + Anthropic requires it. "
                f"Run: pip install litellm  Falling back to gpt-4o-mini."
            )
            return OpenAIChatCompletionClient(
                model="gpt-4o-mini",
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                max_tokens=350, temperature=0.2,
            )
    elif provider == "groq":
        # Groq is OpenAI-compatible
        actual = model.replace("groq/", "")
        return OpenAIChatCompletionClient(
            model=actual,
            api_key=os.environ.get("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
            max_tokens=350, temperature=0.2,
        )
    elif provider == "ollama":
        actual = model.replace("ollama/", "")
        return OpenAIChatCompletionClient(
            model=actual,
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            max_tokens=350, temperature=0.2,
        )
    # Default
    return OpenAIChatCompletionClient(
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        max_tokens=350, temperature=0.2,
    )


# ── System 4: LangGraph ───────────────────────────────────────────────────────

def run_langgraph(
    task: dict,
    n_agents: int,
    steps: int,
    backbone_model: str,
    judge_model: str,
) -> Run:
    """
    LangGraph supervisor experiment.

    backbone_model support:
        gpt-*    → ChatOpenAI
        claude-* → ChatAnthropic (requires langchain-anthropic)
        llama-*  → ChatOpenAI pointed at Groq endpoint

    Token accounting:
        coord_tokens += tok(all_outputs) + supervisor prompt + supervisor completion
        work_tokens  += worker completion tokens
    """
    import langgraph  # type: ignore
    from langgraph import version as lgv  # type: ignore
    sdk_ver = lgv.__version__
    if sdk_ver != EXPECTED_VERSIONS["langgraph"]:
        log.warning(
            f"LangGraph version mismatch: expected "
            f"{EXPECTED_VERSIONS['langgraph']}, got {sdk_ver}."
        )

    run = Run(
        run_id=str(uuid.uuid4()), system="langgraph",
        agent_count=n_agents, task_id=task["task_id"],
        backbone_model=backbone_model, judge_model=judge_model,
        sdk_version=f"langgraph-{sdk_ver}",
    )
    t0 = time.time()

    shard_keys       = task["shared_state_keys"][:n_agents]
    worker_outputs   = {k: "" for k in shard_keys}
    supervisor_summary = f"Task: {task['description']}\n\nBegin."

    try:
        for _cycle in range(max(1, steps // max(n_agents, 1))):
            # Workers
            for i, key in enumerate(shard_keys):
                run.coord_tokens += tok(supervisor_summary)
                text, pt, ct = llm_call(
                    f"You are worker {i} specializing in {key}. Be concise.",
                    f"Supervisor context:\n{supervisor_summary}\n\n"
                    f"Your component: {key}. Write 2-3 sentences of progress.",
                    model=backbone_model,
                )
                run.work_tokens  += ct
                run.coord_tokens += pt
                worker_outputs[key] = text

            # Supervisor
            all_out = "\n".join(f"{k}: {v}" for k, v in worker_outputs.items())
            run.coord_tokens += tok(all_out)
            summary, sp, sc = llm_call(
                "You are the supervisor. Synthesize worker outputs concisely.",
                f"Worker outputs:\n{all_out}\n\n"
                f"Synthesize into a coherent 3-sentence update.",
                model=backbone_model,
            )
            run.coord_tokens   += sp + sc
            supervisor_summary  = summary or supervisor_summary

        run.success = judge_success([supervisor_summary], task, judge_model)
        log.info(
            f"  langgraph [{backbone_model}] CWR={run.cwr:.3f} "
            f"S50={run.success}"
        )
    except Exception as e:
        log.error(f"LangGraph run failed: {e}")
        import traceback; traceback.print_exc()
        run.excluded = True

    run.steps_taken = steps
    run.wall_ms     = int((time.time() - t0) * 1000)
    return run


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(out_path: str):
    import csv
    from collections import defaultdict
    from scipy.stats import t as t_dist, mannwhitneyu

    runs, excluded_count = [], 0
    with open(out_path) as f:
        for row in csv.DictReader(f):
            if row.get("excluded", "0") == "1":
                excluded_count += 1
            else:
                runs.append(row)

    if excluded_count:
        log.warning(f"Skipped {excluded_count} excluded runs in analysis")

    # Group by (system, agent_count, backbone_model)
    groups:  dict[tuple, list[float]] = defaultdict(list)
    success: dict[tuple, list[bool]]  = defaultdict(list)

    for r in runs:
        key = (r["system"], r["agent_count"], r.get("backbone_model", "?"))
        if r["cwr"] not in ("inf", ""):
            groups[key].append(float(r["cwr"]))
        success[key].append(r["success"] == "1")

    print(f"\n{'System':<14} {'N':>3} {'Model':<25} {'Mean CWR':>10} "
          f"{'±95%CI':>10} {'n':>4} {'S@50':>6} {'CP-CI':>16}")
    print("-" * 95)

    sbus_cwrs: list[float] = []
    for (system, n, model), cwrs in sorted(groups.items()):
        if not cwrs:
            continue
        mean   = sum(cwrs) / len(cwrs)
        n_runs = len(cwrs)
        ci_str = "—"
        if n_runs >= 2:
            from math import sqrt as _sqrt
            std   = _sqrt(sum((x - mean)**2 for x in cwrs) / (n_runs - 1))
            t_val = float(t_dist.ppf(0.975, df=n_runs - 1))
            ci    = t_val * std / _sqrt(n_runs)
            ci_str = f"±{ci:.3f}"

        succ   = success.get((system, n, model), [])
        k_s    = sum(1 for s in succ if s)
        s50    = f"{k_s/len(succ):.0%}" if succ else "—"
        lo, hi = clopper_pearson_ci(k_s, len(succ))
        cp_str = f"[{lo:.2f}–{hi:.2f}]"

        print(f"{system:<14} {n:>3} {model:<25} {mean:>10.3f} "
              f"{ci_str:>10} {n_runs:>4} {s50:>6} {cp_str:>16}")

        if system == "sbus":
            sbus_cwrs.extend(cwrs)

    if sbus_cwrs:
        print("\nMann-Whitney U (S-Bus < baseline, all models combined):")
        all_groups = defaultdict(list)
        for (sys, n, _model), cwrs in groups.items():
            all_groups[sys].extend(cwrs)
        for sys, cwrs in sorted(all_groups.items()):
            if sys == "sbus" or not cwrs:
                continue
            try:
                from scipy.stats import mannwhitneyu
                stat, p = mannwhitneyu(sbus_cwrs, cwrs, alternative="less")
                n_c = len(sbus_cwrs) * len(cwrs)
                r   = 1 - (2 * stat) / n_c
                print(f"  S-Bus < {sys:<15} U={stat:.0f} p={p:.4f} r={r:.3f}")
            except Exception:
                pass
    print()


# ── Task loader ───────────────────────────────────────────────────────────────

def load_tasks(path: str = "datasets/long_horizon_tasks.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="S-Bus Real SDK Comparison — multi-model edition"
    )
    parser.add_argument("--system",      default="all",
                        choices=["all", "sbus", "crewai", "autogen", "langgraph"])
    parser.add_argument("--agents",      nargs="+", type=int, default=[4, 8])
    parser.add_argument("--steps",       type=int, default=20)
    parser.add_argument("--tasks-limit", type=int, default=5)
    parser.add_argument("--runs",        type=int, default=1,
                        help="Repetitions per (task × N) for statistical validity")
    parser.add_argument("--out",         type=str,
                        default="results/real_sdk_results.csv")
    parser.add_argument("--tasks-path",  type=str,
                        default="datasets/long_horizon_tasks.json")
    parser.add_argument("--server",      type=str,
                        default="http://localhost:7000")
    parser.add_argument("--analyse-only", action="store_true")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help=(
            "Backbone LLM for all agents. Examples:\n"
            "  gpt-4o-mini             (default, OpenAI)\n"
            "  claude-haiku-3          (Anthropic, needs ANTHROPIC_API_KEY)\n"
            "  claude-haiku-3.5        (Anthropic)\n"
            "  llama-3.1-8b-instant    (Groq, needs GROQ_API_KEY)\n"
            "  llama-3.3-70b-versatile (Groq)\n"
            "  ollama/llama3.2         (local Ollama, free, needs ollama serve)\n"
        ),
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help=(
            "Judge model for S@50 evaluation. Defaults to --model.\n"
            "Use a DIFFERENT family to eliminate same-family circularity:\n"
            "  --model gpt-4o-mini --judge claude-haiku-3\n"
            "  --model claude-haiku-3 --judge gpt-4o-mini\n"
            "  --model llama-3.1-8b-instant --judge gpt-4o-mini\n"
        ),
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help=(
            "Run full experiment across all three backbone models "
            "(gpt-4o-mini, claude-haiku-3, llama-3.1-8b-instant). "
            "Requires OPENAI_API_KEY + ANTHROPIC_API_KEY + GROQ_API_KEY."
        ),
    )

    args = parser.parse_args()

    # Resolve judge model — default to cross-family if not specified
    if args.judge is None:
        provider = _provider_for(args.model)
        if provider == "openai":
            args.judge = "claude-haiku-3"   # cross-family default
        elif provider == "anthropic":
            args.judge = "gpt-4o-mini"
        else:
            args.judge = "gpt-4o-mini"
        log.info(
            f"Judge model not specified. Defaulting to {args.judge} "
            f"(cross-family from backbone {args.model})."
        )

    if args.analyse_only:
        analyse(args.out)
        return

    # Multi-model mode: run all backbones sequentially
    if args.multi_model:
        models = [
            ("gpt-4o-mini",           "claude-haiku-3"),
            ("claude-haiku-3",        "gpt-4o-mini"),
            ("llama-3.1-8b-instant",  "gpt-4o-mini"),
        ]
        log.info("Multi-model mode: running all three backbones")
        for backbone, judge in models:
            _run_experiment(args, backbone, judge)
        analyse(args.out)
        return

    _run_experiment(args, args.model, args.judge)
    analyse(args.out)


def _run_experiment(args, backbone_model: str, judge_model: str):
    bus = Bus(args.server)
    if not bus.ping():
        log.error(f"S-Bus server not reachable at {args.server}. Run: cargo run")
        sys.exit(1)

    log.info(
        f"Backbone: {backbone_model} | Judge: {judge_model} | "
        f"System: {args.system} | N: {args.agents} | "
        f"steps: {args.steps} | tasks: {args.tasks_limit} | runs: {args.runs}"
    )

    tasks    = load_tasks(args.tasks_path)[:args.tasks_limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    with open(out_path, "a") as f:
        if write_header:
            f.write(CSV_HDR)

        for task in tasks:
            for n in args.agents:
                for run_idx in range(args.runs):
                    log.info(
                        f"\nTask: {task['task_id']} | N={n} | "
                        f"run {run_idx+1}/{args.runs} | "
                        f"backbone={backbone_model}"
                    )

                    if args.system in ("all", "sbus"):
                        run = run_sbus(task, n, bus, args.steps,
                                       backbone_model, judge_model)
                        f.write(run.csv()); f.flush()

                    if args.system in ("all", "crewai"):
                        if importlib.util.find_spec("crewai"):
                            run = run_crewai(task, n, args.steps,
                                             backbone_model, judge_model)
                            if run.excluded:
                                log.warning("CrewAI run excluded — not written")
                            else:
                                f.write(run.csv()); f.flush()
                        else:
                            log.warning("crewai not installed — skipping")

                    if args.system in ("all", "autogen"):
                        if importlib.util.find_spec("autogen_agentchat"):
                            run = run_autogen(task, n, args.steps,
                                              backbone_model, judge_model)
                            if run.excluded:
                                log.warning("AutoGen run excluded — not written")
                            else:
                                f.write(run.csv()); f.flush()
                        else:
                            log.warning("autogen_agentchat not installed — skipping")

                    if args.system in ("all", "langgraph"):
                        if importlib.util.find_spec("langgraph"):
                            run = run_langgraph(task, n, args.steps,
                                                backbone_model, judge_model)
                            if run.excluded:
                                log.warning("LangGraph run excluded — not written")
                            else:
                                f.write(run.csv()); f.flush()
                        else:
                            log.warning("langgraph not installed — skipping")


if __name__ == "__main__":
    main()