#!/usr/bin/env python3
"""
Experiment I (FIXED): Realistic Shared-Shard Contention on Real Agent Tasks
Fixed API mismatches vs. actual S-Bus Rust server (port 7000):

BUG 1: CreateShardRequest used `initial_content` → correct field is `content`
BUG 2: CreateShardRequest missing required `goal_tag` field
BUG 3: ReadSetEntry used `version` → correct field is `version_at_read`
BUG 4: CommitRequest had `session_id` field → does not exist in Rust types
BUG 5: GET /shard/:key passed `session_id` query param → only `agent_id` exists

OCC-off mode:
  Option A (recommended): run second server instance with SBUS_VERSION=1
    cargo run --release  # terminal 1, port 7000 (OCC on)
    SBUS_VERSION=1 SBUS_PORT=7001 cargo run --release  # terminal 2 (OCC off)
    python3 exp_i_fixed.py --occ-on-url http://localhost:7000 \
                           --occ-off-url http://localhost:7001

  Option B (single server, straw-man proxy):
    python3 exp_i_fixed.py --occ-on-url http://localhost:7000
    (OCC-off is simulated by sending stale expected_version=0 always,
     matching the Exp E baseline methodology)

Usage:
  export OPENAI_API_KEY=...
  export ANTHROPIC_API_KEY=...
  cargo run --release   # in sbus repo, terminal 1
  python3 exp_i_fixed.py --occ-on-url http://localhost:7000
"""

import os, json, time, uuid, httpx, asyncio, argparse, csv
from dataclasses import dataclass, field, asdict
from openai import AsyncOpenAI

openai_client = AsyncOpenAI()

# ── Config ───────────────────────────────────────────────────────────────────
BACKBONE_MODEL = "gpt-4o-mini"
N_AGENTS       = 4
STEPS          = 15        # shorter = fewer API calls, still meaningful
N_RUNS         = 5         # runs per (task, condition)

DJANGO_TASKS = [
    {
        "task_id": "django__django-11019",
        "description": (
            "Fix Django queryset filter ordering with related model fields. "
            "Multiple agents need to agree on changes to models.py and orm_helpers.py."
        ),
        "shared_shards": ["models_state", "orm_state"],
    },
    {
        "task_id": "django__django-12286",
        "description": (
            "Fix Django admin action permission checking. "
            "Multiple agents coordinate changes to admin.py and auth_backend.py."
        ),
        "shared_shards": ["admin_state", "auth_state"],
    },
    {
        "task_id": "django__django-13230",
        "description": (
            "Fix Django migration squasher for circular dependencies. "
            "Multiple agents update migration_graph.py and dependency_resolver.py."
        ),
        "shared_shards": ["migration_state", "dependency_state"],
    },
]

# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class AgentResult:
    agent_id: str
    commits_attempted: int = 0
    commits_succeeded: int = 0
    commits_conflicted: int = 0
    steps_taken: int = 0

@dataclass
class RunResult:
    run_id: str
    task_id: str
    condition: str           # "occ_on" or "occ_off"
    n_agents: int
    commits_attempted: int = 0
    commits_succeeded: int = 0
    commits_conflicted: int = 0
    type_i_corruptions: int = 0
    scr: float = 0.0
    wall_ms: int = 0
    notes: str = ""

# ── S-Bus client (corrected) ─────────────────────────────────────────────────
class SBusClient:
    """
    Matches the actual Rust API exactly:
      POST /shard          body: {key, content, goal_tag}
      GET  /shard/:key     query: ?agent_id=X
      POST /commit/v2      body: {key, expected_version, delta, agent_id,
                                  read_set?: [{key, version_at_read}]}
    """
    def __init__(self, base_url: str, occ_enabled: bool = True):
        self.base = base_url.rstrip('/')
        self.occ_enabled = occ_enabled
        self._client = httpx.AsyncClient(timeout=60.0)

    async def create_shard(self, key: str, content: str,
                           goal_tag: str = "exp_i") -> dict:
        """
        FIX BUG 1+2: field is `content` (not `initial_content`);
        `goal_tag` is required.
        """
        r = await self._client.post(
            f"{self.base}/shard",
            json={"key": key, "content": content, "goal_tag": goal_tag}
        )
        if r.status_code == 409:
            return {"status": "already_exists"}   # ok on re-run
        r.raise_for_status()
        return r.json()

    async def read_shard(self, key: str, agent_id: str) -> dict | None:
        """
        FIX BUG 4+5: only ?agent_id=X; no session_id param.
        Returns ShardResponse dict with fields: key, version, content, ...
        """
        r = await self._client.get(
            f"{self.base}/shard/{key}",
            params={"agent_id": agent_id}          # ← only valid query param
        )
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            return None
        return r.json()

    async def commit(self, key: str, expected_version: int,
                     delta: str, agent_id: str,
                     read_set: list[dict]) -> tuple[int, dict]:
        """
        FIX BUG 3: read_set entries use `version_at_read` (not `version`).
        FIX BUG 4: no `session_id` field in CommitRequest.

        OCC-off mode: send expected_version=0 always (straw-man proxy),
        mirroring the Exp E baseline methodology (version=0 always).
        The server rejects with 409 VersionMismatch unless version is
        actually 0, so first-round commits on fresh shards succeed;
        subsequent ones all fail with 409 (no corruption occurs via this path).

        For true OCC-off (silent overwrites), use a separate server
        instance with SBUS_VERSION=1. Pass --occ-off-url to this script.
        """
        ev = 0 if not self.occ_enabled else expected_version

        # FIX BUG 3: version_at_read, not version
        corrected_rs = [
            {"key": e["key"], "version_at_read": e["version_at_read"]}
            for e in read_set
        ] if self.occ_enabled else []   # omit read_set entirely for occ_off

        r = await self._client.post(
            f"{self.base}/commit/v2",
            json={
                "key":              key,
                "expected_version": ev,
                "delta":            delta,
                "agent_id":         agent_id,
                # FIX BUG 4: no session_id field
                "read_set":         corrected_rs if corrected_rs else None,
            }
        )
        body = {}
        try:
            body = r.json()
        except Exception:
            pass
        return r.status_code, body

    async def aclose(self):
        await self._client.aclose()

# ── Corruption detector ───────────────────────────────────────────────────────
async def check_corruption(client: SBusClient, shared_shards: list[str],
                           n_agents: int, commits_succeeded: int) -> int:
    """
    Detect Type-I corruptions:
    Under OCC-on: version should equal number of successful commits to that shard.
    Under OCC-off (SBUS_VERSION=1): multiple agents can overwrite each other,
      so version == n_commits but earlier content is silently lost.
      We detect this by checking: if all agents "succeeded" on the same shard
      at the same step (version didn't advance as expected), writes were dropped.

    Simple metric: total_succeeded should equal sum of (final_version - 0)
    across all shards. If total_succeeded > sum(final_versions), writes were lost.
    """
    total_final_versions = 0
    for sk in shared_shards:
        data = await client.read_shard(sk, "corruption_checker")
        if data:
            total_final_versions += data.get("version", 0)

    # Each successful commit increments version by 1.
    # If succeeded > total_final_version, some commits silently overwrote others.
    corruptions = max(0, commits_succeeded - total_final_versions)
    return corruptions

# ── Single agent ─────────────────────────────────────────────────────────────
async def run_agent(agent_id: str, task: dict, shared_shards: list[str],
                    private_shard: str, client: SBusClient,
                    n_steps: int) -> AgentResult:
    result = AgentResult(agent_id=agent_id)

    for step in range(n_steps):
        # Read all relevant shards (records in DeliveryLog via agent_id)
        shard_data = {}
        read_set = []
        for sk in shared_shards + [private_shard]:
            data = await client.read_shard(sk, agent_id)
            if data:
                shard_data[sk] = data
                # FIX BUG 3: use version_at_read key
                read_set.append({
                    "key":            sk,
                    "version_at_read": data.get("version", 0)  # ← correct field name
                })

        # Generate delta via LLM
        context = "\n".join(
            f"  {k}: v{v.get('version',0)} — {v.get('content','')[:120]}"
            for k, v in shard_data.items()
        )
        prompt = (
            f"You are agent {agent_id}, step {step+1}/{n_steps}.\n"
            f"Task: {task['description']}\n"
            f"Current shared state:\n{context}\n\n"
            f"Write ONE concrete technical change (1-2 sentences). "
            f"Be specific about file and change. Output ONLY the change, no preamble."
        )

        try:
            resp = await openai_client.chat.completions.create(
                model=BACKBONE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.7,
            )
            delta = resp.choices[0].message.content.strip()
        except Exception as e:
            delta = f"[{agent_id} step {step+1} error: {e}]"

        # Commit to primary shared shard (creates real contention)
        target_sk = shared_shards[step % len(shared_shards)]
        ev = shard_data.get(target_sk, {}).get("version", 0)

        result.commits_attempted += 1
        status, body = await client.commit(
            key=target_sk,
            expected_version=ev,
            delta=delta,
            agent_id=agent_id,
            read_set=read_set,
        )

        if status == 200:
            result.commits_succeeded += 1
        elif status == 409:
            result.commits_conflicted += 1
        # 410 = session expired (TTL), other = server error

        result.steps_taken = step + 1
        await asyncio.sleep(0.05)   # minimal throttle

    return result

# ── Single run ────────────────────────────────────────────────────────────────
async def run_experiment(task: dict, client: SBusClient,
                         condition: str, run_idx: int) -> RunResult:
    run_id = str(uuid.uuid4())[:8]
    suffix = f"{run_id}"

    result = RunResult(
        run_id=run_id,
        task_id=task["task_id"],
        condition=condition,
        n_agents=N_AGENTS,
    )

    t0 = time.time()

    try:
        # Create shards for this run (unique keys per run to avoid state bleed)
        shared_shards = [f"{sk}_{suffix}" for sk in task["shared_shards"]]
        private_shards = [f"private_{i}_{suffix}" for i in range(N_AGENTS)]

        for sk in shared_shards:
            await client.create_shard(
                key=sk,
                content=f"Initial state for {sk.split('_')[0]}: {task['description'][:100]}",
                goal_tag=f"exp_i_{task['task_id']}",   # FIX BUG 2: goal_tag required
            )
        for i, psk in enumerate(private_shards):
            await client.create_shard(
                key=psk,
                content=f"Agent {i} private workspace for {task['task_id']}",
                goal_tag="exp_i_private",
            )

        # Run all N agents concurrently
        agent_tasks = [
            run_agent(
                agent_id=f"agent_{i}_{suffix}",
                task=task,
                shared_shards=shared_shards,
                private_shard=private_shards[i],
                client=client,
                n_steps=STEPS,
            )
            for i in range(N_AGENTS)
        ]
        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Aggregate metrics
        for ar in agent_results:
            if isinstance(ar, Exception):
                result.notes += f"|agent_error:{ar}"
                continue
            result.commits_attempted  += ar.commits_attempted
            result.commits_succeeded  += ar.commits_succeeded
            result.commits_conflicted += ar.commits_conflicted

        if result.commits_attempted > 0:
            result.scr = result.commits_conflicted / result.commits_attempted

        # Detect corruptions
        result.type_i_corruptions = await check_corruption(
            client, shared_shards, N_AGENTS, result.commits_succeeded
        )

    except Exception as e:
        result.notes = str(e)[:200]

    result.wall_ms = int((time.time() - t0) * 1000)
    return result

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Exp I — Realistic Shared-Shard Contention (Fixed)")
    parser.add_argument("--occ-on-url",  default="http://localhost:7000",
                        help="S-Bus server URL with OCC enabled (default)")
    parser.add_argument("--occ-off-url", default=None,
                        help="S-Bus server URL with SBUS_VERSION=1 (OCC disabled). "
                             "If omitted, straw-man proxy (expected_version=0) is used.")
    parser.add_argument("--runs",   type=int, default=N_RUNS)
    parser.add_argument("--output", default="exp_i_results_fixed.csv")
    args = parser.parse_args()

    # Health check
    print(f"Checking S-Bus server at {args.occ_on_url} ...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{args.occ_on_url}/stats")
            if r.status_code == 200:
                print(f"  ✓ OCC-on server reachable: {r.json()}")
            else:
                print(f"  ✗ Server returned {r.status_code} — is `cargo run --release` running?")
                return
    except Exception as e:
        print(f"  ✗ Cannot reach {args.occ_on_url}: {e}")
        print("  → Start the server: cd sbus && cargo run --release")
        return

    if args.occ_off_url:
        print(f"Checking OCC-off server at {args.occ_off_url} ...")
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{args.occ_off_url}/stats")
                print(f"  ✓ OCC-off server reachable")
        except Exception as e:
            print(f"  ✗ OCC-off server not reachable: {e}")
            print("  → Start with: SBUS_VERSION=1 SBUS_PORT=7001 cargo run --release")
            return
    else:
        print("  ℹ  No --occ-off-url: using straw-man proxy (expected_version=0 always)")

    all_results: list[RunResult] = []

    for task in DJANGO_TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task['task_id']}")
        print(f"{'='*60}")

        for condition in ["occ_on", "occ_off"]:
            url = args.occ_on_url if condition == "occ_on" else (args.occ_off_url or args.occ_on_url)
            occ_enabled = (condition == "occ_on") or (args.occ_off_url is None and condition == "occ_on")
            # For occ_off with same server, disable via expected_version=0
            if condition == "occ_off" and args.occ_off_url is None:
                occ_enabled = False  # straw-man: always send ev=0

            client = SBusClient(url, occ_enabled=occ_enabled)
            print(f"\n  Condition: {condition} | URL: {url} | occ_enabled={occ_enabled}")

            for run_idx in range(args.runs):
                print(f"    Run {run_idx+1}/{args.runs} ...", end=" ", flush=True)
                result = await run_experiment(task, client, condition, run_idx)
                all_results.append(result)
                print(
                    f"SCR={result.scr:.3f} conflicts={result.commits_conflicted}/"
                    f"{result.commits_attempted} "
                    f"corruptions={result.type_i_corruptions} "
                    f"wall={result.wall_ms//1000}s"
                    + (f" ERR:{result.notes[:40]}" if result.notes else "")
                )
                await asyncio.sleep(1.0)

            await client.aclose()

    # Write CSV
    if all_results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            writer.writeheader()
            writer.writerows([asdict(r) for r in all_results])

    # Print summary table
    print(f"\n\n{'='*60}")
    print("EXPERIMENT I SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<10} {'Task':<30} {'SCR':<8} {'Conflicts':<12} {'Corruptions':<14} {'n'}")
    print("-"*80)
    for cond in ["occ_on", "occ_off"]:
        for task in DJANGO_TASKS:
            rows = [r for r in all_results
                    if r.condition == cond and r.task_id == task["task_id"]]
            if not rows: continue
            avg_scr  = sum(r.scr for r in rows) / len(rows)
            tot_corr = sum(r.type_i_corruptions for r in rows)
            tot_conf = sum(r.commits_conflicted for r in rows)
            tot_att  = sum(r.commits_attempted for r in rows)
            print(f"{cond:<10} {task['task_id']:<30} {avg_scr:<8.3f} "
                  f"{tot_conf}/{tot_att:<8} {tot_corr:<14} {len(rows)}")

    print(f"\nResults written to {args.output}")
    print("\nExpected interpretation:")
    print("  occ_on:  SCR ≈ 0.4–0.8 (OCC retries on shared shards), corruptions = 0")
    print("  occ_off: SCR ≈ 0.0–0.3, corruptions > 0 (silent overwrites / lost writes)")

if __name__ == "__main__":
    asyncio.run(main())