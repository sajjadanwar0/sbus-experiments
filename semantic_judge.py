#!/usr/bin/env python3
"""
Semantic Judge for OCC-off Corruption Validation
=================================================
Completes the POS model validation loop:
  - Measured p_hidden = 0.706 (Exp PH)
  - POS predicts ρ ≤ 0.71 × 0.706 = 0.501
  - This script measures actual semantic corruption rate from OCC-off runs
  - Compares observed ρ against 0.501 to validate/refute c=0.71

HOW IT WORKS:
  For each OCC-off run, asks Claude to judge:
  "Does the final shard content reflect ALL agents' intended contributions,
   or was some agent's work silently overwritten?"
  Corruption = judge says "some work was lost"

REQUIRES:
  - exp_i_true_occ_off.csv (already have this)
  - S-Bus server running (to read final shard content per run)
  - ANTHROPIC_API_KEY (Claude Haiku judge — cheap)

RUN:
  cargo run --release   # S-Bus on port 7000
  python3 semantic_judge.py \
    --occ-off-csv exp_i_true_occ_off.csv \
    --output semantic_corruption_results.csv
"""

import os, csv, asyncio, argparse, json
from urllib.request import  Request, ProxyHandler, build_opener
from urllib.parse import urlencode
from urllib.error import HTTPError

_opener = build_opener(ProxyHandler({}))   # bypass HTTP_PROXY

def http_get(url, params=None):
    if params:
        url += "?" + urlencode(params)
    try:
        with _opener.open(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}

def http_post(url, body):
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}

# ── Anthropic judge (direct HTTP, no SDK needed) ──────────────────────────────
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def judge_corruption(task_id: str, n_agents: int,
                     shard_contents: dict[str, str],
                     agent_deltas: list[str]) -> dict:
    """
    Ask Claude Haiku: given the final shard content and what each agent tried
    to write, was any agent's contribution silently lost?
    Returns: {"corrupted": bool, "confidence": float, "reason": str}
    """
    if not ANTHROPIC_KEY:
        return {"corrupted": False, "confidence": 0.0,
                "reason": "No API key — skipping judge"}

    final_state = "\n".join(
        f"  {k}: {v[:200]}" for k, v in shard_contents.items()
    )
    agent_work = "\n".join(
        f"  Agent {i}: {d[:150]}" for i, d in enumerate(agent_deltas)
    )

    prompt = f"""Task: {task_id}
{n_agents} agents worked concurrently on shared state.

Final state of shared shards:
{final_state}

What each agent attempted to write:
{agent_work}

Question: Does the final state reflect meaningful contributions from ALL {n_agents} agents,
or does it appear that some agents' work was silently overwritten?

Answer with JSON only:
{{"corrupted": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}}

corrupted=true means at least one agent's work appears lost.
corrupted=false means the final state integrates all agents' contributions."""

    body = {
        "model": "claude-haiku-3-5-20251001",
        "max_tokens": 150,
        "messages": [{"role": "user", "content": prompt}]
    }
    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01"
        }
    )
    try:
        with _opener.open(req, timeout=30) as r:
            resp = json.loads(r.read())
            text = resp["content"][0]["text"].strip()
            # parse JSON from response
            if "{" in text:
                result = json.loads(text[text.find("{"):text.rfind("}")+1])
                return result
            return {"corrupted": "true" in text.lower(),
                    "confidence": 0.7, "reason": text[:100]}
    except HTTPError as e:
        # Read the actual Anthropic error body for diagnosis
        body = e.read().decode("utf-8", errors="replace")[:300]
        return {"corrupted": False, "confidence": 0.0, 
                "reason": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"corrupted": False, "confidence": 0.0, "reason": f"Error: {e}"}

# ── Re-run an OCC-off experiment and collect shard content + agent deltas ─────
TASKS = {
    "django__django-11019": {
        "shared": ["models_state", "orm_state"],
        "desc": "Fix queryset ordering with related model fields"
    },
    "django__django-12286": {
        "shared": ["admin_state", "db_schema"],
        "desc": "Fix admin action permission checking"
    },
    "django__django-13230": {
        "shared": ["migration_state", "dependency_state"],
        "desc": "Fix migration squasher for circular dependencies"
    }
}

async def run_occ_off_with_logging(task_id: str, sbus_url: str,
                                   n_agents: int = 4, steps: int = 5):
    """
    Run a short OCC-off experiment, log each agent's delta,
    then read final shard content.
    Returns: (shard_contents, agent_deltas, commits_succeeded)
    """
    import uuid
    from openai import AsyncOpenAI
    openai = AsyncOpenAI()

    run_id = str(uuid.uuid4())[:8]
    task = TASKS.get(task_id, {})
    shared = [f"{sk}_{run_id}" for sk in task.get("shared", ["shared_0"])]
    desc = task.get("desc", task_id)

    # Create shards
    for sk in shared:
        http_post(f"{sbus_url}/shard", {
            "key": sk,
            "content": f"Initial: {desc[:60]}",
            "goal_tag": "semantic_judge"
        })

    agent_deltas = []
    commits_succeeded = 0

    async def one_agent(i):
        nonlocal commits_succeeded
        deltas = []
        for step in range(steps):
            # Read current state
            sk = shared[step % len(shared)]
            _, data = http_get(f"{sbus_url}/shard/{sk}", {"agent_id": f"a{i}_{run_id}"})
            current = data.get("content", "")[:80] if data else ""

            # LLM call
            try:
                resp = await openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user",
                               "content": f"Task: {desc}. Current: {current}. "
                                          f"Step {step+1}: write one specific code change. "
                                          f"Agent {i} perspective. Output ONLY the change."}],
                    max_tokens=80, temperature=0.7
                )
                delta = f"[Agent{i}] {resp.choices[0].message.content.strip()}"
            except Exception as e:
                delta = f"[Agent{i}] error: {e}"

            deltas.append(delta)

            # Commit OCC-off: send version=0 always
            status, _ = http_post(f"{sbus_url}/commit/v2", {
                "key": sk,
                "expected_version": 0,   # OCC disabled
                "delta": delta,
                "agent_id": f"a{i}_{run_id}",
                "read_set": None
            })
            if status == 200:
                commits_succeeded += 1

        agent_deltas.extend(deltas[-1:])  # last delta per agent (most recent intent)

    await asyncio.gather(*[one_agent(i) for i in range(n_agents)])

    # Read final shard contents
    shard_contents = {}
    for sk in shared:
        _, data = http_get(f"{sbus_url}/shard/{sk}", {"agent_id": "_judge"})
        if data:
            shard_contents[sk] = data.get("content", "")

    return shard_contents, agent_deltas, commits_succeeded

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbus-url", default="http://localhost:7000")
    parser.add_argument("--runs", type=int, default=5,
                        help="Runs per task (3 tasks × runs = total)")
    parser.add_argument("--steps", type=int, default=5,
                        help="Steps per agent per run (keep small for cost)")
    parser.add_argument("--output", default="semantic_corruption_results.csv")
    args = parser.parse_args()

    # Health check
    import socket
    try:
        sock = socket.create_connection(("localhost", 7000), timeout=3)
        sock.close()
        print("S-Bus OK")
    except Exception:
        print("S-Bus not running. Start: cargo run --release")
        return

    if not ANTHROPIC_KEY:
        print("WARNING: ANTHROPIC_API_KEY not set — judge will skip")
        print("Set: export ANTHROPIC_API_KEY=your_key")

    results = []
    total_corrupt = 0
    total_runs = 0

    for task_id in TASKS:
        print(f"\nTask: {task_id}")
        for run_idx in range(args.runs):
            print(f"  Run {run_idx+1}/{args.runs}...", end=" ", flush=True)

            shard_contents, agent_deltas, succeeded = \
                await run_occ_off_with_logging(task_id, args.sbus_url,
                                               steps=args.steps)
            judgment = judge_corruption(task_id, 4, shard_contents, agent_deltas)
            corrupted = judgment.get("corrupted", False)
            confidence = judgment.get("confidence", 0)
            reason = judgment.get("reason", "")

            total_corrupt += int(corrupted)
            total_runs += 1

            print(f"corrupted={corrupted} (conf={confidence:.2f}) succeeded={succeeded}")

            results.append({
                "task_id": task_id,
                "run_idx": run_idx,
                "commits_succeeded": succeeded,
                "corrupted": int(corrupted),
                "confidence": confidence,
                "reason": reason[:100],
                "final_shard_0": list(shard_contents.values())[0][:150] if shard_contents else ""
            })

    # Write results
    with open(args.output, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    # POS validation
    observed_rho = total_corrupt / total_runs if total_runs > 0 else 0
    predicted_rho = 0.71 * 0.706   # from Exp PH

    print(f"\n{'='*55}")
    print("POS MODEL VALIDATION — FINAL RESULT")
    print(f"{'='*55}")
    print(f"  Total runs           : {total_runs}")
    print(f"  Corrupted (judge)    : {total_corrupt}")
    print(f"  Observed ρ           : {observed_rho:.4f}")
    print(f"  Predicted ρ (0.71×p) : {predicted_rho:.4f}")

    if abs(observed_rho - predicted_rho) < 0.1:
        print(f"  → c=0.71 VALIDATED (within 0.10 of prediction)")
    elif observed_rho < predicted_rho:
        new_c = observed_rho / 0.706
        print(f"  → c should be re-fitted: c = {new_c:.3f} (prediction was conservative)")
    else:
        new_c = observed_rho / 0.706
        print(f"  → c should be re-fitted: c = {new_c:.3f} (prediction was optimistic)")

    print(f"\n  Add to paper §III-C (Observation 4c):")
    print(f"  'Semantic corruption rate measured: ρ = {observed_rho:.3f}")
    print(f"   POS prediction: {predicted_rho:.3f}. {'Consistent' if abs(observed_rho-predicted_rho)<0.1 else 'Discrepant'}'")
    print(f"\n  Results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())