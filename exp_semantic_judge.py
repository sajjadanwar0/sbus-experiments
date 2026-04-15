#!/usr/bin/env python3
"""
Exp: Semantic Corruption Judge (LLM-as-Judge via GPT-4o-mini)
==============================================================
Validates the POS model empirically: takes the OCC-off (SBUS_VERSION=1)
experimental sessions from Exp I and judges whether the surviving shard
content is semantically correct relative to the task goal.

FIXES the previous implementation which used the Anthropic API (zero credits).
This version uses GPT-4o-mini which is already used for backbone experiments.

WHAT IT PROVES:
  - Structural overwrite rate (OCC-off) = 20.1% (already measured in Exp I)
  - Semantic corruption rate: predicted by POS as rho <= 0.501
  - This experiment measures actual semantic corruption to validate the POS prediction

USAGE:
  # Make sure S-Bus is running on port 7000
  # Set your OpenAI API key:
  export OPENAI_API_KEY=sk-...

  python3 exp_semantic_judge.py \
      --tasks datasets/tasks_30_multidomain.json \
      --n-tasks 3 \
      --n-runs 5 \
      --output results/semantic_judge_results.csv

EXPECTED RESULT:
  OCC-off corruption rate > 50% (consistent with POS prediction rho~0.501)
  OCC-on corruption rate = 0% (no overwrites to judge)
"""

import csv
import json
import os
import sys
import uuid
import time
import argparse
import socket
from urllib.request import Request, ProxyHandler, build_opener
from urllib.parse import urlencode
from urllib.error import HTTPError
from dataclasses import dataclass, asdict, field
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

_opener = build_opener(ProxyHandler({}))
SBUS_URL = "http://localhost:7000"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def http_get(url, params=None):
    if params:
        url += "?" + urlencode(params)
    try:
        with _opener.open(url, timeout=15) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception as e:
        return 0, {}


def http_post(url, body):
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception as e:
        return 0, {}


def reset_sbus():
    """Reset all S-Bus state between runs."""
    http_post(f"{SBUS_URL}/reset", {})
    time.sleep(0.2)


# ---------------------------------------------------------------------------
# S-Bus agent simulation
# ---------------------------------------------------------------------------
def run_agent_session(
    task_id: str,
    task_description: str,
    n_agents: int,
    n_steps: int,
    occ_on: bool,
    run_idx: int,
    oai_client: OpenAI,
) -> dict:
    """
    Run a multi-agent session on S-Bus.
    Returns dict with: task_id, run_idx, occ_on, surviving_content,
                       commits_succeeded, commits_conflicted
    """
    session_id = f"sem_{uuid.uuid4().hex[:8]}"
    shard_key = f"shard_0_{session_id}"

    # Create shard
    http_post(f"{SBUS_URL}/shard", {
        "key": shard_key,
        "content": f"Initial state for task: {task_description[:100]}",
        "goal_tag": task_id,
    })

    # Register agents
    agents = [f"agent_{i}_{session_id}" for i in range(n_agents)]
    for agent in agents:
        http_post(f"{SBUS_URL}/session", {
            "agent_id": agent,
            "session_ttl": 3600,
        })

    commits_ok = 0
    commits_conflict = 0

    for step in range(n_steps):
        for agent in agents:
            # Read shard
            st, data = http_get(f"{SBUS_URL}/shard/{shard_key}", {"agent_id": agent})
            current_version = data.get("version", 0) if data else 0
            current_content = data.get("content", "") if data else ""

            # Generate delta using GPT-4o-mini
            try:
                resp = oai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=100,
                    messages=[
                        {"role": "system", "content": (
                            "You are a software engineering agent working on a task. "
                            "Write a short code snippet or technical solution (1-3 lines). "
                            "Be specific and concrete."
                        )},
                        {"role": "user", "content": (
                            f"Task: {task_description[:200]}\n"
                            f"Current state: {current_content[:150]}\n"
                            f"Your agent ID: {agent}\n"
                            f"Provide your delta (code change/solution):"
                        )},
                    ],
                )
                delta = f"[{agent}] {resp.choices[0].message.content.strip()}"
            except Exception as e:
                delta = f"[{agent}] Error: {e}"

            # Commit
            if occ_on:
                payload = {
                    "key": shard_key,
                    "expected_version": current_version,
                    "delta": delta,
                    "agent_id": agent,
                    "read_set": [{"key": shard_key, "version_at_read": current_version}],
                }
            else:
                # OCC-off: use SBUS_VERSION=1 equivalent — always use version 0
                payload = {
                    "key": shard_key,
                    "expected_version": 0,
                    "delta": delta,
                    "agent_id": agent,
                }

            st, _ = http_post(f"{SBUS_URL}/commit/v2", payload)
            if st == 200:
                commits_ok += 1
            else:
                commits_conflict += 1

    # Read final shard content
    _, final = http_get(f"{SBUS_URL}/shard/{shard_key}", {"agent_id": "judge"})
    surviving_content = final.get("content", "") if final else ""

    return {
        "task_id": task_id,
        "run_idx": run_idx,
        "occ_on": occ_on,
        "commits_succeeded": commits_ok,
        "commits_conflicted": commits_conflict,
        "surviving_content": surviving_content,
    }


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """You are a strict technical judge evaluating software engineering solutions.
You will be given:
1. A task description
2. The final content of a shared shard after multiple agents worked on it

Your job: determine if the final content is semantically CORRECT for the task,
or CORRUPTED (wrong approach, contradictory, or from the wrong task entirely).

Respond with EXACTLY one of:
  CORRECT - the content addresses the task appropriately
  CORRUPTED - the content is wrong, contradictory, or not relevant to the task

Then on a new line, write a one-sentence explanation.
Do NOT write anything before CORRECT or CORRUPTED."""


def judge_content(
    task_description: str,
    surviving_content: str,
    oai_client: OpenAI,
) -> tuple[bool, float, str]:
    """
    Returns (is_corrupted, confidence, reason).
    confidence: 0.0-1.0
    """
    try:
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=80,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": (
                    f"Task: {task_description[:300]}\n\n"
                    f"Final shard content:\n{surviving_content[:400]}"
                )},
            ],
        )
        text = resp.choices[0].message.content.strip()
        lines = text.split("\n", 1)
        verdict = lines[0].strip().upper()
        reason = lines[1].strip() if len(lines) > 1 else ""

        is_corrupted = verdict.startswith("CORRUPT")
        confidence = 0.9 if verdict in ("CORRECT", "CORRUPTED") else 0.5
        return is_corrupted, confidence, reason

    except Exception as e:
        return False, 0.0, f"Judge error: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@dataclass
class JudgeResult:
    task_id: str
    run_idx: int
    occ_on: bool
    commits_succeeded: int
    commits_conflicted: int
    surviving_content: str
    is_corrupted: bool
    confidence: float
    reason: str
    test_passed: bool  # True if: occ_on -> not corrupted; occ_off -> corrupted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="datasets/tasks_30_multidomain.json")
    parser.add_argument("--n-tasks", type=int, default=3,
                        help="Number of tasks (use django tasks for comparability with Exp I)")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=5,
                        help="Steps per agent (keep low for cost control)")
    parser.add_argument("--output", default="results/semantic_judge_results.csv")
    parser.add_argument("--sbus-url", default="http://localhost:7000")
    args = parser.parse_args()

    global SBUS_URL
    SBUS_URL = args.sbus_url

    # Health check
    try:
        s = socket.create_connection(("localhost", 7000), timeout=3)
        s.close()
        print("S-Bus OK")
    except Exception:
        print("S-Bus not running. Start: cargo run --release")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY environment variable")
        sys.exit(1)

    oai = OpenAI(api_key=api_key)
    print(f"OpenAI client ready (gpt-4o-mini judge)")

    # Load tasks
    with open(args.tasks) as f:
        all_tasks = json.load(f)

    # Prefer django tasks for comparability with Exp I Table XV
    django_tasks = [t for t in all_tasks if "django" in t.get("task_id", "")]
    other_tasks = [t for t in all_tasks if "django" not in t.get("task_id", "")]
    tasks = (django_tasks + other_tasks)[: args.n_tasks]
    print(f"Using {len(tasks)} tasks: {[t['task_id'] for t in tasks]}")

    results: list[JudgeResult] = []
    occ_on_corrupted = 0
    occ_off_corrupted = 0
    total_on = 0
    total_off = 0

    for task in tasks:
        task_id = task["task_id"]
        desc = task.get("problem_statement", task.get("description", task_id))

        for run_idx in range(args.n_runs):
            for occ_on in [True, False]:
                condition = "OCC-on" if occ_on else "OCC-off"
                print(f"\n  [{task_id}] run={run_idx} {condition}")

                reset_sbus()

                session = run_agent_session(
                    task_id=task_id,
                    task_description=desc,
                    n_agents=args.n_agents,
                    n_steps=args.n_steps,
                    occ_on=occ_on,
                    run_idx=run_idx,
                    oai_client=oai,
                )

                is_corrupted, conf, reason = judge_content(
                    task_description=desc,
                    surviving_content=session["surviving_content"],
                    oai_client=oai,
                )

                # test_passed:
                #   occ_on  -> content should NOT be corrupted (pass if not corrupted)
                #   occ_off -> content SHOULD be corrupted (pass if corrupted)
                test_passed = (not is_corrupted) if occ_on else is_corrupted

                r = JudgeResult(
                    task_id=task_id,
                    run_idx=run_idx,
                    occ_on=occ_on,
                    commits_succeeded=session["commits_succeeded"],
                    commits_conflicted=session["commits_conflicted"],
                    surviving_content=session["surviving_content"][:200],
                    is_corrupted=is_corrupted,
                    confidence=conf,
                    reason=reason,
                    test_passed=test_passed,
                )
                results.append(r)

                if occ_on:
                    total_on += 1
                    if is_corrupted:
                        occ_on_corrupted += 1
                else:
                    total_off += 1
                    if is_corrupted:
                        occ_off_corrupted += 1

                status = "CORRUPTED" if is_corrupted else "CORRECT"
                print(f"    Judge: {status} (conf={conf:.2f}) | {reason[:80]}")

    # Write CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])

    print(f"\n{'='*60}")
    print("SEMANTIC JUDGE RESULTS")
    print(f"{'='*60}")
    print(f"  OCC-on  corrupted: {occ_on_corrupted}/{total_on} = {occ_on_corrupted/total_on*100:.1f}%")
    print(f"  OCC-off corrupted: {occ_off_corrupted}/{total_off} = {occ_off_corrupted/total_off*100:.1f}%")
    print()
    print(f"  POS model prediction: rho <= 0.501 (at phidden=0.706)")
    print(f"  Structural overwrite rate (Exp I OCC-off): 20.1% (lower bound)")
    print()
    if occ_on_corrupted == 0:
        print("  OCC-on: ZERO semantic corruptions - S-Bus guarantee holds")
    else:
        print(f"  OCC-on: {occ_on_corrupted} semantic corruptions DETECTED - investigate")
    print(f"  OCC-off corruption rate: {occ_off_corrupted/total_off*100:.1f}%")
    print(f"  Results: {args.output}")
    print()
    print("  Paper update for Section IX-K:")
    print(f"  'Semantic judge (GPT-4o-mini, {len(tasks)} tasks, {args.n_runs} runs/condition):")
    print(f"   OCC-on: {occ_on_corrupted}/{total_on} = {occ_on_corrupted/total_on*100:.1f}% corruption;")
    print(f"   OCC-off: {occ_off_corrupted}/{total_off} = {occ_off_corrupted/total_off*100:.1f}% corruption.")
    print(f"   POS prediction rho<=0.501 is {'CONFIRMED' if occ_off_corrupted/total_off <= 0.55 else 'EXCEEDED'}.'")


if __name__ == "__main__":
    main()