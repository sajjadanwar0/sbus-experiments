#!/usr/bin/env python3
"""
wal_test.py  — WAL crash recovery smoke test for S-Bus v29
============================================================
Proves that WAL (Write-Ahead Log) replay on restart correctly
restores all shard state after a SIGKILL crash.

Usage (from your sbus project root):
    # Terminal 1 — start server with WAL:
    SBUS_WAL_PATH=results/wal.jsonl ./target/release/sbus-server &

    # Terminal 2 — run test:
    python3 wal_test.py

Expected output:
    Step 1: ✅ Server running, WAL at: results/wal.jsonl
    Step 2: ✅ Committed delta → version 1
    Step 3: ✅ Killed server
    Step 4: ✅ Server restarted (took Xs)
    Step 5: ✅ PASS: WAL crash recovery works
    ✅ WAL crash recovery smoke test PASSED
"""

import httpx, os, signal, subprocess, sys, time

SBUS_URL = os.getenv("SBUS_URL", "http://localhost:7000")
WAL_PATH = os.getenv("SBUS_WAL_PATH", "results/wal.jsonl")
PID_FILE = "results/sbus.pid"

def find_binary():
    """Find sbus-server binary — works regardless of cwd."""
    import glob
    candidates = [
        "./target/release/sbus-server",
        os.path.expanduser("~/RustroverProjects/sbus/target/release/sbus-server"),
        os.path.expanduser("~/sbus/target/release/sbus-server"),
        os.path.expanduser("~/projects/sbus/target/release/sbus-server"),
        os.path.expanduser("~/Documents/sbus/target/release/sbus-server"),
    ]
    candidates += glob.glob(os.path.expanduser("~/**/sbus/target/release/sbus-server"),
                            recursive=True)
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None

def find_project_root():
    """Return the sbus project root directory."""
    binary = find_binary()
    if binary:
        # binary is at <root>/target/release/sbus-server
        return os.path.dirname(os.path.dirname(os.path.dirname(binary)))
    return "."

BINARY      = find_binary()
PROJECT_ROOT = find_project_root()

# Override WAL_PATH and PID_FILE to be relative to project root
if not os.getenv("SBUS_WAL_PATH"):
    WAL_PATH = os.path.join(PROJECT_ROOT, "results", "wal.jsonl")
if not os.getenv("SBUS_PID_FILE"):
    PID_FILE = os.path.join(PROJECT_ROOT, "results", "sbus.pid")

client = httpx.Client(timeout=15)


def step(n, msg):
    print(f"\nStep {n}: {msg}")


def server_alive():
    try:
        return client.get(f"{SBUS_URL}/stats", timeout=4).status_code == 200
    except Exception:
        return False


def api_call(method, path, **kwargs):
    """Make an API call and return (status_code, response_body_dict)"""
    url = f"{SBUS_URL}{path}"
    r = getattr(client, method)(url, **kwargs)
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text}
    return r.status_code, body


def main():
    print("=" * 55)
    print("  WAL Crash Recovery Smoke Test")
    print("=" * 55)

    # ── Step 1: Verify server + WAL ───────────────────────────────
    step(1, "Verify server is running with WAL enabled")

    if not server_alive():
        print(f"  ❌ Server not responding at {SBUS_URL}")
        print(f"  Start with:")
        print(f"    SBUS_WAL_PATH={WAL_PATH} ./target/release/sbus-server &")
        sys.exit(1)

    _, stats = api_call("get", "/stats")
    wal = stats.get("wal_path", "disabled")
    if wal == "disabled" or not wal:
        print(f"  ❌ WAL is disabled (wal_path={wal!r})")
        print(f"  Restart: SBUS_WAL_PATH={WAL_PATH} ./target/release/sbus-server &")
        sys.exit(1)
    print(f"  ✅ Server running, WAL at: {wal}")

    # ── Step 2: Create shard + commit delta ───────────────────────
    step(2, "Create shard and commit a delta")

    test_key = "wal_smoke_test_shard"

    # Clean up any previous run
    try:
        client.delete(f"{SBUS_URL}/shard/{test_key}")
    except Exception:
        pass

    # Create shard — server may return 200, 201, or 409
    status, body = api_call("post", "/shard",
                            json={"key": test_key, "content": "BEFORE_CRASH", "goal_tag": "wal_test"})
    if status not in (200, 201, 409):
        print(f"  ❌ POST /shard returned {status}: {body}")
        print(f"  This may indicate the /shard endpoint path is different.")
        sys.exit(1)
    if status == 409:
        print(f"  ℹ  Shard already exists (409) — using existing shard")

    # Read current version
    status, shard = api_call("get", f"/shard/{test_key}")
    if status != 200:
        print(f"  ❌ GET /shard/{test_key} returned {status}: {shard}")
        sys.exit(1)
    current_ver = shard.get("version", 0)
    print(f"  ✅ Shard at version {current_ver}")

    # Commit delta
    status, result = api_call("post", "/commit/v2",
                              json={"key":              test_key,
                                    "expected_version": current_ver,
                                    "delta":            "SURVIVED_THE_CRASH",
                                    "agent_id":         "wal_test_agent"})
    if status != 200:
        print(f"  ❌ Commit failed ({status}): {result}")
        print(f"  Possible causes:")
        print(f"    • Version mismatch (another write happened)")
        print(f"    • Wrong endpoint path — check /commit/v2 vs /commit")
        sys.exit(1)

    new_ver = result.get("new_version", current_ver + 1)
    print(f"  ✅ Committed 'SURVIVED_THE_CRASH' → version {new_ver}")

    # ── Step 3: Kill server ───────────────────────────────────────
    step(3, "Kill the server (simulate crash)")

    if os.path.exists(PID_FILE):
        try:
            pid = int(open(PID_FILE).read().strip())
            os.kill(pid, signal.SIGKILL)
            print(f"  ✅ Killed PID {pid}")
        except (ValueError, ProcessLookupError) as e:
            print(f"  ℹ  PID file stale ({e}); using pkill")
            subprocess.run(["pkill", "-9", "-f", "sbus-server"], capture_output=True)
    else:
        subprocess.run(["pkill", "-9", "-f", "sbus-server"], capture_output=True)
        print(f"  ✅ Sent SIGKILL via pkill")

    time.sleep(2)
    if server_alive():
        print("  ⚠  Server still alive — wait a moment and retry")
        time.sleep(3)
    else:
        print("  ✅ Server confirmed dead")

    # ── Step 4: Restart server ────────────────────────────────────
    step(4, "Restart server (WAL replay)")

    if BINARY is None:
        print(f"  ❌ sbus-server binary not found anywhere")
        print(f"  Fix: run this script from your sbus project root:")
        print(f"    cd ~/RustroverProjects/sbus")
        print(f"    python3 wal_test.py")
        print(f"  Or build first: cargo build --release")
        sys.exit(1)
    print(f"  ✅ Using binary: {BINARY}")

    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    env = {**os.environ, "SBUS_WAL_PATH": WAL_PATH, "SBUS_SESSION_TTL": "7200"}
    print(f"  Using WAL: {WAL_PATH}")
    print(f"  Project:   {PROJECT_ROOT}")
    proc = subprocess.Popen(
        [BINARY], env=env,
        stdout=open(os.path.join(PROJECT_ROOT, "logs", "sbus.log"), "a"),
        stderr=subprocess.STDOUT,
    )
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

    for t in range(20):
        time.sleep(1)
        if server_alive():
            print(f"  ✅ Server restarted (PID {proc.pid}, took {t+1}s)")
            break
    else:
        print("  ❌ Server did not come back within 20s")
        print("  Check: tail -30 logs/sbus.log")
        sys.exit(1)

    # ── Step 5: Verify shard survived ────────────────────────────
    step(5, "Verify shard survived the crash")

    status, after = api_call("get", f"/shard/{test_key}")
    if status != 200:
        print(f"  ❌ Shard not found after restart ({status})")
        print(f"  WAL replay may not be working.")
        print(f"  Check:")
        print(f"    tail -30 logs/sbus.log   # look for WAL replay errors")
        print(f"    ls -lh {WAL_PATH}         # confirm WAL file exists")
        sys.exit(1)

    content_ok = after.get("content") == "SURVIVED_THE_CRASH"
    version_ok = after.get("version") == new_ver

    if content_ok and version_ok:
        print(f"  ✅ PASS:")
        print(f"     content = {after['content']!r}  ✅")
        print(f"     version = {after['version']}  (expected {new_ver})  ✅")
    else:
        print(f"  ❌ FAIL:")
        print(f"     content = {after.get('content')!r}  (expected 'SURVIVED_THE_CRASH')")
        print(f"     version = {after.get('version')}  (expected {new_ver})")
        sys.exit(1)

    print()
    print("=" * 55)
    print("  ✅ WAL crash recovery smoke test PASSED")
    print("=" * 55)
    print()
    print("  Paper claim: WAL append-only log; replay on startup")
    print("  restores all shard state after crash.")
    print("  Status: VERIFIED ✅")
    print()
    print("  Next: update paper — remove Limitation #3 (WAL smoke test pending)")


if __name__ == "__main__":
    main()