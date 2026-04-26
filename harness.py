import asyncio
import logging
import time
from typing import Dict, Any
import httpx
from openai import AsyncOpenAI
from agent import run_agent, _http_get_shard
from domains import DOMAINS, INITIAL_SHARDS, SHARD_KEYS

log = logging.getLogger(__name__)


async def _server_reset(client: httpx.AsyncClient, base_url: str) -> None:
    r = await client.post(f"{base_url}/admin/reset", timeout=10.0)
    if r.status_code == 403:
        raise SystemExit(
            "ERROR: S-Bus server returned 403 on /admin/reset.\n"
            "       Admin endpoints are disabled. Restart the server with:\n"
            "         SBUS_ADMIN_ENABLED=1 cargo run --release --bin sbus-server\n"
            "       (or set SBUS_ADMIN_ENABLED=1 in the environment that starts it)."
        )
    r.raise_for_status()


async def _server_stats(client: httpx.AsyncClient, base_url: str) -> Dict[str, Any]:
    try:
        r = await client.get(f"{base_url}/stats", timeout=10.0)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.warning(f"GET /stats failed: {e}")
    return {}


async def _server_create_shard(client: httpx.AsyncClient, base_url: str,
                                key: str, content: str, goal_tag: str) -> None:
    payload = {"key": key, "content": content, "goal_tag": goal_tag}
    r = await client.post(f"{base_url}/admin/shard", json=payload, timeout=10.0)
    r.raise_for_status()


async def _server_set_mode(client: httpx.AsyncClient, base_url: str,
                            condition: str) -> None:
    ori_enabled = condition == "ori_on"
    try:
        r = await client.post(
            f"{base_url}/admin/config",
            json={"ori_enabled": ori_enabled},
            timeout=10.0,
        )
        if r.status_code == 404:
            log.warning(
                "Server has no /admin/config endpoint. ori_off will not "
                "disable cross-shard checking; results will be invalid. "
                "Either implement /admin/config on the server or run two "
                "servers and override base_url per condition."
            )
        else:
            r.raise_for_status()
    except httpx.RequestError as e:
        log.warning(f"set_mode failed: {e}")


async def run_trial(
    *,
    base_url: str,
    domain_name: str,
    condition: str,
    trial_idx: int,
    n_steps: int,
    openai_client: AsyncOpenAI,
    model: str,
) -> Dict[str, Any]:
    domain = next(d for d in DOMAINS if d["name"] == domain_name)
    trial_id = f"{domain_name}|{condition}|t{trial_idx}"
    log.info(f"=== trial {trial_id} ===")
    t0 = time.time()

    async with httpx.AsyncClient() as http_client:
        await _server_reset(http_client, base_url)
        await _server_set_mode(http_client, base_url, condition)
        for shard_key, content in INITIAL_SHARDS.items():
            await _server_create_shard(
                http_client, base_url, shard_key, content,
                goal_tag=f"{domain_name}_{shard_key}",
            )

        agent_tasks = []
        for shard_key in SHARD_KEYS:
            agent_id = f"agent_{shard_key}"
            agent_tasks.append(asyncio.create_task(run_agent(
                base_url=base_url,
                agent_id=agent_id,
                my_shard=shard_key,
                domain=domain,
                n_steps=n_steps,
                openai_client=openai_client,
                model=model,
            )))
        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        final_state = {}
        for shard_key in SHARD_KEYS:
            v, content = await _http_get_shard(
                http_client, base_url, "harness_observer", shard_key
            )
            final_state[shard_key] = {"version": v, "content": content}

        server_stats = await _server_stats(http_client, base_url)

    elapsed = round(time.time() - t0, 2)

    n_409 = 0
    n_410 = 0
    n_200 = 0
    n_other = 0
    n_llm_errors = 0
    total_commit_attempts = 0
    n_rejections_total = 0
    for ar in agent_results:
        if isinstance(ar, Exception):
            log.error(f"agent crashed: {ar}")
            continue
        for rec in ar["records"]:
            cs = rec.get("commit_status")
            attempts = rec.get("commit_attempts", 0) or 0
            if cs == 200:
                n_200 += 1
                n_rejections_total += max(0, attempts - 1)
            elif cs == 409:
                n_409 += 1
                n_rejections_total += attempts
            elif cs == 410:
                n_410 += 1
                n_rejections_total += attempts
            elif cs is not None:
                n_other += 1
            if rec.get("llm_status") == "error":
                n_llm_errors += 1
            total_commit_attempts += attempts

    return {
        "trial_id": trial_id,
        "domain": domain_name,
        "condition": condition,
        "trial_idx": trial_idx,
        "n_steps": n_steps,
        "elapsed_s": elapsed,
        "metrics": {
            "n_commit_200": n_200,
            "n_commit_409": n_409,
            "n_commit_410": n_410,
            "n_commit_other": n_other,
            "n_rejections_total": n_rejections_total,

            "n_llm_errors": n_llm_errors,
            "total_commit_attempts": total_commit_attempts,
            "view_divergent_commits": server_stats.get("view_divergent_commits", 0),
            "view_checked_commits": server_stats.get("view_checked_commits", 0),
            "view_divergence_rate": server_stats.get("view_divergence_rate", 0.0),
            "server_ori_enabled": server_stats.get("ori_enabled"),
        },
        "final_state": final_state,
        "agent_results": [
            ar if not isinstance(ar, Exception) else {"error": str(ar)}
            for ar in agent_results
        ],
    }
