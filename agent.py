import json
import time
import asyncio
import logging
from typing import Dict, Optional, Tuple, Any

import httpx
from openai import AsyncOpenAI

from domains import SHARD_KEYS

log = logging.getLogger(__name__)

MAX_RETRIES = 20
RETRY_BASE_MS = 30
LLM_TIMEOUT_S = 60.0
HTTP_TIMEOUT_S = 10.0

SYSTEM_PROMPT = """You are a senior data engineer designing one component of \
a data pipeline. You are part of a team of four engineers — each owning a \
different component. Your job is to produce a concrete, production-grade \
design for YOUR component that is consistent with what your teammates have \
written.

You receive:
- The domain description (what the pipeline is for)
- The current content of all four components (some may still be TBD)
- Which component is yours

You produce STRICT JSON with these fields:
- design: the full text of YOUR component's design (replace any TBD content)
- referenced_components: list of OTHER component names you explicitly \
referenced or depended on in your design (e.g., ["ingestion", "storage"])
- referenced_entities: list of specific things from other components your \
design depends on (e.g., ["Kafka", "S3 bucket structure", "schema registry"])

The referenced_components and referenced_entities lists are how we verify \
cross-component coherence. Reference siblings honestly. If your design \
genuinely depends on a teammate's choice (e.g., your transformation logic \
needs to know if ingestion uses Kafka or Kinesis), reference it explicitly. \
If your design is independent of a sibling, leave that sibling out of the \
referenced_components list.

Design must be concrete: name specific technologies (e.g., "Apache Kafka 3.6 \
with 32 partitions"), not abstract concepts (e.g., "a message broker"). \
Aim for 100-200 words of design text."""


def _user_prompt(domain: Dict[str, Any], my_shard: str,
                 current_state: Dict[str, str], step: int) -> str:
    others = [k for k in SHARD_KEYS if k != my_shard]
    state_lines = ["CURRENT PIPELINE STATE:"]
    for k in SHARD_KEYS:
        marker = "  >>> YOUR COMPONENT <<<" if k == my_shard else ""
        state_lines.append(f"\n--- {k.upper()} ---{marker}\n{current_state[k]}")
    state_block = "\n".join(state_lines)
    return (
        f"DOMAIN: {domain['name']}\n"
        f"DESCRIPTION:\n{domain['description']}\n\n"
        f"ARCHITECTURAL PRESSURES: {', '.join(domain['pressures'])}\n\n"
        f"YOUR COMPONENT: {my_shard}\n"
        f"SIBLING COMPONENTS: {', '.join(others)}\n\n"
        f"COORDINATION STEP: {step + 1}\n\n"
        f"{state_block}\n\n"
        f"Produce strict JSON: {{'design': str, 'referenced_components': [str], "
        f"'referenced_entities': [str]}}. Replace the TBD content for "
        f"'{my_shard}' with your concrete design."
    )


async def _call_llm(openai_client: AsyncOpenAI, model: str,
                    system: str, user: str) -> Dict[str, Any]:
    try:
        resp = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            ),
            timeout=LLM_TIMEOUT_S,
        )
        raw = resp.choices[0].message.content
        return json.loads(raw)
    except asyncio.TimeoutError:
        return {"_error": "LLM:Timeout"}
    except json.JSONDecodeError as e:
        return {"_error": f"LLM:JSONDecode:{e}"}
    except Exception as e:
        return {"_error": f"LLM:{type(e).__name__}:{e}"}


async def _http_get_shard(client: httpx.AsyncClient, base_url: str,
                           agent_id: str, key: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        r = await client.get(
            f"{base_url}/shard/{key}",
            params={"agent_id": agent_id},
            timeout=HTTP_TIMEOUT_S,
        )
        if r.status_code == 200:
            j = r.json()
            return j["version"], j["content"]
        log.warning(f"GET /shard/{key} -> {r.status_code}")
        return None, None
    except Exception as e:
        log.warning(f"GET /shard/{key} failed: {e}")
        return None, None


async def _http_commit(client: httpx.AsyncClient, base_url: str,
                        agent_id: str, key: str, expected_version: int,
                        delta: str) -> Dict[str, Any]:
    payload = {
        "key": key,
        "expected_version": expected_version,
        "delta": delta,
        "agent_id": agent_id,
        "additive_hint": False,
    }
    try:
        r = await client.post(
            f"{base_url}/commit/v2",
            json=payload,
            timeout=HTTP_TIMEOUT_S,
        )
        return {"status": r.status_code, **(r.json() if r.content else {})}
    except Exception as e:
        return {"status": None, "_error": f"HTTP:{type(e).__name__}:{e}"}


async def run_agent(
    *,
    base_url: str,
    agent_id: str,
    my_shard: str,
    domain: Dict[str, Any],
    n_steps: int,
    openai_client: AsyncOpenAI,
    model: str,
) -> Dict[str, Any]:
    records = []
    async with httpx.AsyncClient() as http_client:
        for step in range(n_steps):
            t0 = time.time()
            current_state = {}
            for key in SHARD_KEYS:
                _, content = await _http_get_shard(
                    http_client, base_url, agent_id, key
                )
                current_state[key] = content if content is not None else "(unreadable)"

            llm_resp = await _call_llm(
                openai_client, model,
                SYSTEM_PROMPT,
                _user_prompt(domain, my_shard, current_state, step),
            )
            if "_error" in llm_resp:
                records.append({
                    "step": step,
                    "llm_status": "error",
                    "llm_error": llm_resp["_error"],
                    "commit_status": None,
                    "elapsed_s": round(time.time() - t0, 2),
                })
                continue

            design = llm_resp.get("design", "")
            ref_comps = llm_resp.get("referenced_components", [])
            ref_ents = llm_resp.get("referenced_entities", [])

            v, _ = await _http_get_shard(http_client, base_url, agent_id, my_shard)
            if v is None:
                records.append({
                    "step": step,
                    "llm_status": "ok",
                    "commit_status": None,
                    "commit_error": "couldnt_fetch_my_shard_version",
                    "elapsed_s": round(time.time() - t0, 2),
                })
                continue
            expected_version = v
            commit_attempts = 0
            commit_resp = None
            for retry in range(MAX_RETRIES):
                commit_attempts += 1
                commit_resp = await _http_commit(
                    http_client, base_url, agent_id, my_shard,
                    expected_version, design,
                )
                if commit_resp.get("status") == 200:
                    break
                if commit_resp.get("status") in (409, 410):
                    if retry < 5:
                        backoff_ms = RETRY_BASE_MS * (2 ** retry)
                    else:
                        backoff_ms = RETRY_BASE_MS * 32  # 960ms cap
                    await asyncio.sleep(backoff_ms / 1000.0)
                    for k in SHARD_KEYS:
                        await _http_get_shard(http_client, base_url, agent_id, k)
                    v, _ = await _http_get_shard(
                        http_client, base_url, agent_id, my_shard
                    )
                    if v is None:
                        break
                    expected_version = v
                    continue
                break

            records.append({
                "step": step,
                "llm_status": "ok",
                "llm_design_excerpt": design[:200],
                "referenced_components": ref_comps,
                "referenced_entities": ref_ents,
                "commit_status": commit_resp.get("status") if commit_resp else None,
                "commit_attempts": commit_attempts,
                "version_after_commit": commit_resp.get("new_version") if commit_resp else None,
                "elapsed_s": round(time.time() - t0, 2),
            })

    return {
        "agent_id": agent_id,
        "my_shard": my_shard,
        "domain": domain["name"],
        "n_steps": n_steps,
        "records": records,
    }