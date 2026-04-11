#!/usr/bin/env python3
"""
test_endpoints.py — Quick API sanity check for S-Bus server
Run this before any experiment to verify all endpoints work.

Usage:
    python3 test_endpoints.py
    python3 test_endpoints.py --url http://localhost:7001   # custom port
"""
import sys, json, httpx, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:7000")
args = parser.parse_args()

URL = args.url
c   = httpx.Client(timeout=10)

ok = lambda msg: print(f"  ✅ {msg}")
fail = lambda msg: (print(f"  ❌ {msg}"), sys.exit(1))

print(f"Testing S-Bus at {URL}\n")

# 1. Stats
r = c.get(f"{URL}/stats")
if r.status_code != 200: fail(f"GET /stats → {r.status_code}")
stats = r.json()
ok(f"GET /stats → {r.status_code}  (WAL: {stats.get('wal_path', 'disabled')})")

# 2. Create shard
r = c.post(f"{URL}/shard", json={"key": "_test_", "content": "hello", "goal_tag": "endpoint_test"})
if r.status_code not in (200, 201, 409):
    fail(f"POST /shard → {r.status_code}: {r.text[:100]}")
ok(f"POST /shard → {r.status_code}")

# 3. Read shard
r = c.get(f"{URL}/shard/_test_")
if r.status_code != 200: fail(f"GET /shard/:key → {r.status_code}")
ver = r.json().get("version", 0)
ok(f"GET /shard/:key → 200  (version={ver})")

# 4. Commit
r = c.post(f"{URL}/commit/v2", json={
    "key": "_test_", "expected_version": ver,
    "delta": "world", "agent_id": "test"
})
if r.status_code != 200: fail(f"POST /commit/v2 → {r.status_code}: {r.text[:100]}")
new_ver = r.json().get("new_version", ver+1)
ok(f"POST /commit/v2 → 200  (new_version={new_ver})")

# 5. Delete (optional)
r = c.delete(f"{URL}/shard/_test_")
if r.status_code in (200, 204, 404):
    ok(f"DELETE /shard/:key → {r.status_code}")
else:
    print(f"  ⚠  DELETE /shard/:key → {r.status_code} (endpoint may not exist, that\'s OK)")

print()
print("All core endpoints working. Ready to run experiments.")
print(f"WAL: {stats.get('wal_path', 'disabled')}")
print(f"TTL: {stats.get('lease_timeout_secs', 'unknown')}s")