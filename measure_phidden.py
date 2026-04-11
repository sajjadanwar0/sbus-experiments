#!/usr/bin/env python3
"""
L4 — Measure Real p_hidden (proxy-safe version)
Uses ProxyHandler({}) to bypass any HTTP_PROXY env vars.
Run: python3 measure_phidden.py --tasks 3 --output phidden_results.csv
"""

import os, re, csv, time, uuid, json, asyncio, argparse, statistics, sys, socket
from urllib.request import urlopen, Request, ProxyHandler, build_opener
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI

openai_client = AsyncOpenAI()

SBUS_URL   = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE   = "gpt-4o-mini"
N_AGENTS   = 4
STEPS      = 10
SHARD_KEYS = ["models_state", "orm_state", "admin_state", "db_schema",
              "migration_state", "dependency_state"]

_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in SHARD_KEYS) + r')\b')

# Build opener that bypasses HTTP_PROXY / http_proxy env vars entirely
_opener = build_opener(ProxyHandler({}))


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def http_get(url: str, params: dict = None) -> tuple[int, dict]:
    if params:
        url = url + "?" + urlencode(params)
    try:
        with _opener.open(url, timeout=10) as r:
            try:
                return r.status, json.loads(r.read())
            except json.JSONDecodeError:
                return r.status, {}
    except HTTPError as e:
        return e.code, {}
    except Exception as e:
        print(f"  [http_get] {type(e).__name__}: {e}")
        return 0, {}


def http_post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req  = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=15) as r:
            try:
                return r.status, json.loads(r.read())
            except json.JSONDecodeError:
                return r.status, {}
    except HTTPError as e:
        return e.code, {}
    except Exception as e:
        print(f"  [http_post] {type(e).__name__}: {e}")
        return 0, {}


# ── Health check ──────────────────────────────────────────────────────────────

def health_check(base_url: str) -> bool:
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host   = parsed.hostname or "localhost"
    port   = parsed.port or 7000

    # Check for proxy env vars — common cause of urllib failures
    proxy_vars = {k: v for k, v in os.environ.items()
                  if k.lower() in ("http_proxy", "https_proxy", "all_proxy")}
    if proxy_vars:
        print(f"  NOTE: proxy env vars detected (bypassed): {proxy_vars}")

    print(f"  TCP connect to {host}:{port} ...", end=" ", flush=True)
    try:
        sock = socket.create_connection((host, port), timeout=3)
        sock.close()
        print("OK")
    except ConnectionRefusedError:
        print("FAILED")
        print(f"\n  Port {port} is not open. Start the server:")
        print(f"    cd ~/sbus && cargo run --release")
        print(f"  Wait for: 'Listening on 0.0.0.0:{port}'")
        return False
    except Exception as e:
        print(f"FAILED — {e}")
        return False

    print(f"  HTTP GET {base_url}/stats ...", end=" ", flush=True)
    status, body = http_get(f"{base_url}/stats")
    if status == 200:
        commits = body.get("total_commits", "?")
        shards  = body.get("total_shards", "?")
        print(f"OK  (shards={shards}, commits={commits})")
        return True
    else:
        print(f"FAILED (status={status})")
        return False


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class StepLog:
    run_id:         str
    task_id:        str
    agent_id:       str
    step:           int
    r_obs_count:    int   = 0
    r_hidden_count: int   = 0
    p_hidden:       float = 0.0
    commit_status:  str   = ""


def count_hidden_reads(messages: list) -> int:
    return sum(
        len(_pattern.findall(msg.get("content", "") if isinstance(msg.get("content"), str) else ""))
        for msg in messages
    )


# ── Tasks ─────────────────────────────────────────────────────────────────────

TASKS = [
    {"id": "django__django-11019",
     "desc": "Fix queryset ordering with related model fields. "
             "Agents coordinate changes to models_state and orm_state.",
     "shared": ["models_state", "orm_state"]},
    {"id": "django__django-12286",
     "desc": "Fix admin action permission checking. "
             "Agents update admin_state and db_schema.",
     "shared": ["admin_state", "db_schema"]},
    {"id": "django__django-13230",
     "desc": "Fix migration squasher for circular deps. "
             "Agents update migration_state and dependency_state.",
     "shared": ["migration_state", "dependency_state"]},
]


# ── Single agent ──────────────────────────────────────────────────────────────

async def run_agent(agent_id: str, shared_shards: list, private_shard: str,
                    base_url: str, task_desc: str,
                    run_id: str, task_id: str) -> list[StepLog]:
    logs = []
    history = []

    for step in range(STEPS):
        shard_data = {}
        read_set   = []
        r_obs      = 0

        for sk in shared_shards + [private_shard]:
            status, data = http_get(f"{base_url}/shard/{sk}",
                                    {"agent_id": agent_id})
            if status == 200:
                shard_data[sk] = data
                read_set.append({"key": sk,
                                  "version_at_read": data.get("version", 0)})
                r_obs += 1

        context = "\n".join(
            f"  {k}: v{v.get('version',0)} — {v.get('content','')[:80]}"
            for k, v in shard_data.items()
        )
        user_msg = (
            f"Task: {task_desc} Step {step+1}/{STEPS}.\n"
            f"Current state:\n{context}\n"
            f"Write one concrete technical change (1 sentence). Output ONLY the change."
        )
        messages = [
            {"role": "system", "content": "You are a software engineering agent."}
        ] + history[-4:] + [{"role": "user", "content": user_msg}]

        r_hidden = count_hidden_reads(messages)

        try:
            resp = await openai_client.chat.completions.create(
                model=BACKBONE, messages=messages, max_tokens=100, temperature=0.7
            )
            delta = resp.choices[0].message.content.strip()
        except Exception as e:
            delta = f"[error: {e}]"

        history += [{"role": "user", "content": user_msg},
                    {"role": "assistant", "content": delta}]

        target = shared_shards[step % len(shared_shards)]
        ev = shard_data.get(target, {}).get("version", 0)
        status, _ = http_post(f"{base_url}/commit/v2", {
            "key":              target,
            "expected_version": ev,
            "delta":            delta,
            "agent_id":         agent_id,
            "read_set":         read_set,
        })

        p_h = r_hidden / (r_hidden + r_obs) if (r_hidden + r_obs) > 0 else 0.0
        logs.append(StepLog(
            run_id=run_id, task_id=task_id, agent_id=agent_id, step=step,
            r_obs_count=r_obs, r_hidden_count=r_hidden,
            p_hidden=round(p_h, 4),
            commit_status="ok" if status == 200 else f"conflict_{status}",
        ))

    return logs


# ── Single run ────────────────────────────────────────────────────────────────

async def run_one(task: dict, base_url: str) -> list[StepLog]:
    run_id  = str(uuid.uuid4())[:8]
    shared  = [f"{sk}_{run_id}" for sk in task["shared"]]
    private = [f"priv_{i}_{run_id}" for i in range(N_AGENTS)]

    for sk in shared + private:
        http_post(f"{base_url}/shard", {
            "key":      sk,
            "content":  f"Initial: {task['desc'][:80]}",
            "goal_tag": f"phidden_{task['id']}",
        })

    results = await asyncio.gather(*[
        run_agent(f"agent_{i}_{run_id}", shared, private[i],
                  base_url, task["desc"], run_id, task["id"])
        for i in range(N_AGENTS)
    ], return_exceptions=True)

    return [log for r in results if isinstance(r, list) for log in r]


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbus-url", default=SBUS_URL)
    parser.add_argument("--tasks",  type=int, default=3)
    parser.add_argument("--output", default="phidden_results.csv")
    args = parser.parse_args()

    print(f"Checking S-Bus at {args.sbus_url} ...")
    if not health_check(args.sbus_url):
        sys.exit(1)
    print()

    all_logs: list[StepLog] = []
    for task in TASKS[:max(1, min(3, args.tasks))]:
        print(f"Running task: {task['id']} ...", end=" ", flush=True)
        logs = await run_one(task, args.sbus_url)
        all_logs.extend(logs)
        mean_ph = statistics.mean(l.p_hidden for l in logs) if logs else 0
        print(f"steps={len(logs)}, mean_p_hidden={mean_ph:.3f}")

    if not all_logs:
        print("No data collected.")
        sys.exit(1)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_logs[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(l) for l in all_logs])

    all_ph    = [l.p_hidden for l in all_logs]
    mean_ph   = statistics.mean(all_ph)
    predicted = 0.71 * mean_ph

    print(f"\n{'='*55}")
    print("POS MODEL VALIDATION RESULTS")
    print(f"{'='*55}")
    print(f"  Steps measured       : {len(all_logs)}")
    print(f"  Mean p_hidden        : {mean_ph:.4f}")
    print(f"  POS predicted rho    : {predicted:.4f}  (0.71 x p_hidden)")
    print()
    print("  p_hidden growth by step (context accumulation effect):")
    for step in range(STEPS):
        vals = [l.p_hidden for l in all_logs if l.step == step]
        if vals:
            m   = statistics.mean(vals)
            bar = "█" * int(m * 30)
            print(f"    Step {step+1:2d}: {m:.3f}  {bar}")
    print()
    print(f"  Results written to : {args.output}")
    print(f"  Add to paper §3.3  : measured p_hidden = {mean_ph:.4f}")

if __name__ == "__main__":
    asyncio.run(main())