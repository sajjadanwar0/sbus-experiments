#!/usr/bin/env python3
"""
pg_sbus_server.py — PostgreSQL SERIALIZABLE baseline for S-Bus paper (v3)

FIX v3: Set SERIALIZABLE isolation via PostgreSQL connection option, not SQL command.
  - v1 bug: `conn.autocommit = False` is read-only in psycopg3 async (HTTP 500)
  - v2 bug: `SET TRANSACTION ISOLATION LEVEL SERIALIZABLE` fails because
            psycopg3 with autocommit=False implicitly opens a transaction BEFORE
            the SET command — SET TRANSACTION must be the first statement.
  - v3 fix: Pass `options="-c default_transaction_isolation=serializable"` at
            connect time. This sets the isolation level for ALL transactions
            on this connection without needing a SET command.

Install:
    pip install psycopg[binary] fastapi uvicorn

Run:
    PG_DSN="host=localhost dbname=sbus_baseline user=sbus_user password=sbus_pass" \
    PG_PORT=7001 python3 pg_sbus_server.py
"""

import os, hashlib, asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

PG_DSN = os.getenv("PG_DSN",
                   "host=localhost dbname=sbus_baseline user=sbus_user password=sbus_pass")
PORT = int(os.getenv("PG_PORT", "7001"))
# Connection string that forces SERIALIZABLE for all transactions
PG_DSN_SER = PG_DSN + " options=-c\\ default_transaction_isolation=serializable"

_delivery_log: dict = defaultdict(dict)
_commit_count = 0
_conflict_count = 0

SCHEMA = """
         CREATE TABLE IF NOT EXISTS shards \
         ( \
             key \
             TEXT \
             PRIMARY \
             KEY, \
             version \
             BIGINT \
             NOT \
             NULL \
             DEFAULT \
             0, \
             content \
             TEXT \
             NOT \
             NULL \
             DEFAULT \
             '', \
             goal_tag \
             TEXT \
             NOT \
             NULL \
             DEFAULT \
             '', \
             created_at \
             TIMESTAMPTZ \
             DEFAULT \
             NOW \
         ( \
         ),
             updated_at TIMESTAMPTZ DEFAULT NOW \
         ( \
         )
             );
         CREATE TABLE IF NOT EXISTS shard_log \
         ( \
             id \
             BIGSERIAL \
             PRIMARY \
             KEY, \
             key \
             TEXT \
             NOT \
             NULL, \
             version \
             BIGINT \
             NOT \
             NULL, \
             agent_id \
             TEXT \
             NOT \
             NULL, \
             delta \
             TEXT \
             NOT \
             NULL, \
             committed_at \
             TIMESTAMPTZ \
             DEFAULT \
             NOW \
         ( \
         )
             ); \
         """


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with await psycopg.AsyncConnection.connect(
            PG_DSN, autocommit=True
    ) as conn:
        await conn.execute(SCHEMA)
    print(f"✅ PostgreSQL schema ready")
    print(f"   DSN: {PG_DSN}")
    print(f"   All transactions: SERIALIZABLE (via connection option)")
    yield


app = FastAPI(lifespan=lifespan, title="S-Bus PG Baseline v3")


class CreateShardReq(BaseModel):
    key: str
    content: str = ""
    goal_tag: str = ""


class CommitReq(BaseModel):
    key: str
    expected_version: int
    delta: str
    agent_id: str
    read_set: Optional[list] = None


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


async def pg_connect_ser():
    """Connect with SERIALIZABLE isolation set at the session level."""
    return await psycopg.AsyncConnection.connect(
        PG_DSN_SER,
        autocommit=False,
        row_factory=dict_row
    )


async def pg_connect_ro():
    """Connect for read-only / DDL operations (autocommit)."""
    return await psycopg.AsyncConnection.connect(
        PG_DSN, autocommit=True, row_factory=dict_row
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/shard", status_code=200)
async def create_shard(req: CreateShardReq):
    async with await pg_connect_ro() as conn:
        existing = await (await conn.execute(
            "SELECT key FROM shards WHERE key=%s", (req.key,)
        )).fetchone()
        if existing:
            raise HTTPException(409, {"error": "ShardAlreadyExists"})
        await conn.execute(
            "INSERT INTO shards(key,content,goal_tag) VALUES(%s,%s,%s)",
            (req.key, req.content, req.goal_tag)
        )
    return {"key": req.key, "version": 0, "content": req.content,
            "goal_tag": req.goal_tag}


@app.get("/shard/{key}")
async def read_shard(key: str, agent_id: str = ""):
    async with await pg_connect_ro() as conn:
        row = await (await conn.execute(
            "SELECT key,version,content,goal_tag,created_at,updated_at "
            "FROM shards WHERE key=%s", (key,)
        )).fetchone()
    if not row:
        raise HTTPException(404, {"error": "ShardNotFound", "key": key})
    if agent_id:
        _delivery_log[agent_id][key] = row["version"]
    return dict(row)


@app.get("/shards")
async def list_shards():
    async with await pg_connect_ro() as conn:
        rows = await (await conn.execute(
            "SELECT key FROM shards ORDER BY key"
        )).fetchall()
    return [r["key"] for r in rows]


@app.post("/commit/v2")
async def commit_v2(req: CommitReq):
    """
    OCC commit using SERIALIZABLE isolation.
    Connection has default_transaction_isolation=serializable set at connect time,
    so every transaction on this connection is automatically SERIALIZABLE.
    """
    global _commit_count, _conflict_count
    max_retries = 5

    for attempt in range(max_retries):
        try:
            async with await pg_connect_ser() as conn:
                async with conn.transaction():
                    # Read + lock primary shard
                    row = await (await conn.execute(
                        "SELECT version, content FROM shards "
                        "WHERE key=%s FOR UPDATE", (req.key,)
                    )).fetchone()
                    if not row:
                        raise HTTPException(404,
                                            {"error": "ShardNotFound", "key": req.key})

                    # OCC version check
                    if row["version"] != req.expected_version:
                        _conflict_count += 1
                        raise HTTPException(409, {
                            "error": "VersionMismatch",
                            "key": req.key,
                            "expected": req.expected_version,
                            "found": row["version"],
                        })

                    # Cross-shard read-set validation
                    read_set = list(req.read_set or [])
                    for k, v in _delivery_log.get(req.agent_id, {}).items():
                        if k != req.key and not any(
                                (r.get("key") if isinstance(r, dict) else r[0]) == k
                                for r in read_set
                        ):
                            read_set.append({"key": k, "version_at_read": v})

                    for rs in read_set:
                        k = rs.get("key") if isinstance(rs, dict) else rs[0]
                        v = rs.get("version_at_read") if isinstance(rs, dict) else rs[1]
                        if k == req.key:
                            continue
                        rs_row = await (await conn.execute(
                            "SELECT version FROM shards "
                            "WHERE key=%s FOR SHARE", (k,)
                        )).fetchone()
                        if rs_row and rs_row["version"] != v:
                            _conflict_count += 1
                            raise HTTPException(409, {
                                "error": "CrossShardStale",
                                "key": k,
                            })

                    # Apply delta
                    new_ver = row["version"] + 1
                    await conn.execute(
                        "UPDATE shards SET version=%s, content=%s, "
                        "updated_at=NOW() WHERE key=%s",
                        (new_ver, req.delta, req.key)
                    )
                    await conn.execute(
                        "INSERT INTO shard_log(key,version,agent_id,delta) "
                        "VALUES(%s,%s,%s,%s)",
                        (req.key, new_ver, req.agent_id, req.delta)
                    )

                _commit_count += 1
                return {"new_version": new_ver, "shard_id": sha(req.delta)}

        except HTTPException:
            raise
        except psycopg.errors.SerializationFailure:
            _conflict_count += 1
            await asyncio.sleep(0.01 * (attempt + 1))
            continue
        except Exception as e:
            raise HTTPException(500, {"error": "Internal", "message": str(e)})

    _conflict_count += 1
    raise HTTPException(409, {
        "error": "VersionMismatch",
        "key": req.key,
        "message": f"Serialization failed after {max_retries} retries",
    })


@app.post("/commit/v2_rc")
async def commit_v2_rc(req: CommitReq):
    """
    OCC commit using READ COMMITTED + SELECT FOR UPDATE (fair comparison).

    READ COMMITTED isolation: no predicate locks, no table-scan serialization.
    Row-level locks via SELECT FOR UPDATE on exact keys only.
    This is semantically equivalent to S-Bus per-key OCC.

    Expected SCR: 0.000 on distinct shards (same as S-Bus),
    confirming per-key OCC is the common mechanism.
    """
    global _commit_count, _conflict_count
    max_retries = 5

    for attempt in range(max_retries):
        try:
            async with await psycopg.AsyncConnection.connect(
                    PG_DSN,
                    autocommit=False,
                    row_factory=dict_row
            ) as conn:
                # READ COMMITTED (PostgreSQL default) — no predicate locks
                # SELECT FOR UPDATE acquires row-level lock on exact key only
                async with conn.transaction():
                    row = await (await conn.execute(
                        "SELECT version, content FROM shards "
                        "WHERE key=%s FOR UPDATE",
                        (req.key,)
                    )).fetchone()
                    if not row:
                        raise HTTPException(404, {"error": "ShardNotFound"})

                    if row["version"] != req.expected_version:
                        _conflict_count += 1
                        raise HTTPException(409, {
                            "error": "VersionMismatch",
                            "expected": req.expected_version,
                            "found": row["version"],
                        })

                    new_ver = row["version"] + 1
                    await conn.execute(
                        "UPDATE shards SET version=%s, content=%s, updated_at=NOW() WHERE key=%s",
                        (new_ver, req.delta, req.key)
                    )
                    await conn.execute(
                        "INSERT INTO shard_log(key,version,agent_id,delta) VALUES(%s,%s,%s,%s)",
                        (req.key, new_ver, req.agent_id, req.delta)
                    )

                _commit_count += 1
                return {"new_version": new_ver, "shard_id": sha(req.delta)}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, {"error": "Internal", "message": str(e)})

    _conflict_count += 1
    raise HTTPException(409, {"error": "VersionMismatch", "message": "Max retries"})


@app.delete("/shard/{key}")
async def delete_shard(key: str):
    async with await psycopg.AsyncConnection.connect(
            PG_DSN, autocommit=True
    ) as conn:
        await conn.execute("DELETE FROM shards WHERE key=%s", (key,))
    return {"deleted": key}


@app.get("/stats")
async def stats():
    async with await pg_connect_ro() as conn:
        n = (await (await conn.execute(
            "SELECT COUNT(*) AS n FROM shards"
        )).fetchone())["n"]
    ta = _commit_count + _conflict_count
    return {
        "system": "postgresql_serializable",
        "total_shards": n,
        "total_commits": _commit_count,
        "total_conflicts": _conflict_count,
        "total_attempts": ta,
        "scr": _conflict_count / ta if ta > 0 else 0.0,
        "wal_enabled": True,
        "wal_path": "postgresql_wal",
        "delivery_log": {
            "tracked_agents": len(_delivery_log),
            "total_deliveries": sum(len(v) for v in _delivery_log.values()),
        },
    }


if __name__ == "__main__":
    print(f"PostgreSQL S-Bus baseline v3 — port {PORT}")
    print(f"Fix: SERIALIZABLE set via options=-c default_transaction_isolation=serializable")
    uvicorn.run(app, host="0.0.0.0", port=PORT)