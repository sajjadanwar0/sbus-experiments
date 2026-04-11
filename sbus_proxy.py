"""
L1 — S-Bus LLM API Proxy (mitmproxy addon)
===========================================
Intercepts every /chat/completions call, scans the messages array for shard
key references, and registers those reads in S-Bus's DeliveryLog via HTTP GET.
This moves prompt-context reads from R_hidden into R_obs, closing the gap.

SETUP (one-time):
    pip install mitmproxy httpx

RUN:
    # Terminal 1 — your S-Bus server
    cargo run --release   # listens on port 7000

    # Terminal 2 — this proxy
    mitmdump -s sbus_proxy.py --listen-port 8080 --ssl-insecure

    # In your experiment environment
    export HTTPS_PROXY=http://localhost:8080
    export HTTP_PROXY=http://localhost:8080
    python3 your_experiment.py

HOW IT WORKS:
    1. Agent sends a /chat/completions request through the proxy
    2. Proxy scans all messages for shard key patterns
    3. Any shard key found → proxy calls GET /shard/:key?agent_id=X on S-Bus
    4. S-Bus records that read in the DeliveryLog (now part of R_obs)
    5. At commit time, WSI validation covers those reads too
    6. R_hidden gap shrinks to near zero for well-named shards

CONFIGURATION:
    Edit SBUS_URL and SHARD_KEYS below to match your deployment.
"""

import json
import re
import threading
import httpx
from mitmproxy import http

# ── Configuration ─────────────────────────────────────────────────────────────
SBUS_URL = "http://localhost:7000"

# List ALL shard keys your agents might reference in prompts.
# Add any key you pass to POST /shard during shard creation.
SHARD_KEYS = [
    "db_schema",
    "models_state",
    "orm_state",
    "admin_state",
    "auth_state",
    "migration_state",
    "dependency_state",
    "architecture_doc",
    "api_design",
    "test_plan",
]

# Header your agents use to identify themselves (optional).
# If absent, agent_id is extracted from the request body or defaults to "proxy_agent".
AGENT_ID_HEADER = "X-Agent-ID"

# ── Compiled pattern ───────────────────────────────────────────────────────────
_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(k) for k in SHARD_KEYS) + r')\b')

# ── Synchronous S-Bus registration (runs in a background thread) ───────────────
_sbus_client = httpx.Client(timeout=5.0)

def _register_read(shard_key: str, agent_id: str) -> None:
    """Call GET /shard/:key?agent_id=X to register read in DeliveryLog."""
    try:
        r = _sbus_client.get(
            f"{SBUS_URL}/shard/{shard_key}",
            params={"agent_id": agent_id}
        )
        if r.status_code == 200:
            data = r.json()
            print(f"[proxy] R_hidden→R_obs: agent={agent_id} shard={shard_key} v={data.get('version','?')}")
        elif r.status_code == 404:
            pass  # shard not created yet — skip silently
        else:
            print(f"[proxy] WARN: shard={shard_key} returned {r.status_code}")
    except Exception as e:
        print(f"[proxy] ERROR registering {shard_key}: {e}")


# ── mitmproxy addon ────────────────────────────────────────────────────────────
class SBusProxyAddon:

    def request(self, flow: http.HTTPFlow) -> None:
        """Intercept outgoing LLM API requests."""
        # Only intercept OpenAI/Anthropic chat completions
        path = flow.request.path
        if not any(p in path for p in ["/chat/completions", "/messages"]):
            return

        # Extract agent_id from custom header or default
        agent_id = flow.request.headers.get(AGENT_ID_HEADER, "proxy_agent")

        # Parse request body
        try:
            body = json.loads(flow.request.content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        # Collect all messages content (system + user + assistant history)
        all_text = ""
        for msg in body.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                all_text += " " + content
            elif isinstance(content, list):
                # Anthropic multi-part format
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        all_text += " " + part.get("text", "")

        # Find all shard key references
        found_keys = set(_PATTERN.findall(all_text))
        if not found_keys:
            return

        # Register each found key in S-Bus DeliveryLog (non-blocking)
        for key in found_keys:
            t = threading.Thread(
                target=_register_read,
                args=(key, agent_id),
                daemon=True
            )
            t.start()

        if found_keys:
            print(f"[proxy] Intercepted {len(found_keys)} R_hidden reads for agent={agent_id}: {found_keys}")


# Register the addon with mitmproxy
addons = [SBusProxyAddon()]