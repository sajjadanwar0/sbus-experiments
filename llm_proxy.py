#!/usr/bin/env python3
"""
llm_proxy.py — LLM API Proxy for S-Bus Rhidden → Robs Promotion
================================================================
Sits between agents and the OpenAI API as a transparent HTTP proxy.

For every LLM completion response, the proxy:
  1. Forwards the request to OpenAI unmodified
  2. Scans the response text for registered S-Bus shard keys
  3. For each shard key found in the response, issues
       GET /shard/{key}?agent_id={agent_id}
     to register the hidden read in the S-Bus DeliveryLog
  4. Returns the response to the agent unmodified

Effect: reads that previously were Rhidden (agent saw shard content via LLM
context, never issued an explicit HTTP GET) are now recorded in the
DeliveryLog as Robs. When the agent later commits, the ACP validates
these formerly-hidden reads — phidden drops toward 0.

Architecture:
  Agent → localhost:8080 (proxy) → api.openai.com
  Proxy side-channel: GET /shard/{key}?agent_id=X → S-Bus DeliveryLog

Usage:
  # Terminal 1: start S-Bus
  cargo run --release

  # Terminal 2: start proxy
  python3 llm_proxy.py --sbus http://localhost:7000 --port 8080

  # Terminal 3: run experiment with proxy
  OPENAI_BASE_URL=http://localhost:8080 python3 exp_shared_state.py ...

  # Or set env var for OpenAI SDK:
  OPENAI_BASE_URL=http://localhost:8080/v1 python3 your_agent.py

Measurement:
  GET http://localhost:8080/proxy/stats
  → {"total_requests": N, "hidden_reads_promoted": M, "promotion_rate": R}
"""

import http.server
import urllib.request
import urllib.error
import json
import os
import sys
import re
import threading
import time
import argparse
from urllib.parse import urlparse, urljoin

# ── Config ────────────────────────────────────────────────────────────────────

SBUS_URL      = os.getenv("SBUS_URL", "http://localhost:7000")
OPENAI_TARGET = "https://api.openai.com"
PROXY_PORT    = int(os.getenv("PROXY_PORT", "8080"))

# ── Shard key registry (populated at startup from S-Bus /shards) ──────────────

class ShardRegistry:
    """Maintains the list of known shard keys for pattern matching.
    Tracks both full keys (portfolio_state_a1b2c3d4) and base names
    (portfolio_state) so LLM responses that reference the concept
    without the UUID suffix are still detected."""
    def __init__(self, sbus_url):
        self._sbus_url = sbus_url
        self._keys = set()          # full keys: portfolio_state_a1b2c3d4
        self._base_names = set()    # base names: portfolio_state
        self._lock = threading.Lock()
        self._refresh()

    def _refresh(self):
        try:
            req = urllib.request.Request(f"{self._sbus_url}/shards")
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            keys = data.get("shards", [])
            with self._lock:
                self._keys = set(keys)
                # Extract base names: strip trailing _hex8 UUID suffix
                self._base_names = set()
                for k in keys:
                    # e.g. portfolio_state_a1b2c3d4 -> portfolio_state
                    parts = k.rsplit("_", 1)
                    if len(parts) == 2 and len(parts[1]) == 8:
                        self._base_names.add(parts[0])
                    else:
                        self._base_names.add(k)
        except Exception:
            pass

    def refresh(self):
        self._refresh()

    def scan(self, text):
        """Return set of shard BASE NAMES that appear in text.
        Checks both full keys and base names so UUID-free references
        (e.g. 'portfolio_state' in LLM output) are detected."""
        with self._lock:
            full_hits  = {k for k in self._keys       if k in text}
            base_hits  = {b for b in self._base_names  if b in text}
            return full_hits | base_hits

    def add(self, key):
        with self._lock:
            self._keys.add(key)
            parts = key.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) == 8:
                self._base_names.add(parts[0])
            else:
                self._base_names.add(key)


# ── DeliveryLog promoter ─────────────────────────────────────────────────────

class DeliveryLogPromoter:
    """Issues S-Bus GET /shard/{key}?agent_id=X to register hidden reads."""
    def __init__(self, sbus_url):
        self._sbus_url = sbus_url
        self._total_promoted = 0
        self._lock = threading.Lock()

    def promote(self, shard_key, agent_id):
        """
        Register a hidden shard reference as an observable read.
        Issues GET /shard/{key}?agent_id={agent_id} — S-Bus records this
        in the DeliveryLog, making it part of the effective read set on
        next commit.
        """
        url = f"{self._sbus_url}/shard/{shard_key}?agent_id={agent_id}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as r:
                r.read()  # discard response — we just need the side effect
            with self._lock:
                self._total_promoted += 1
            return True
        except Exception:
            return False

    @property
    def total_promoted(self):
        with self._lock:
            return self._total_promoted


# ── Agent ID extractor ────────────────────────────────────────────────────────

def _extract_agent_id(request_body):
    """
    Extract agent_id from the LLM request.
    Agents should pass agent_id in the 'user' field of the OpenAI request,
    or as a custom header X-Agent-Id. Falls back to 'unknown'.
    """
    try:
        data = json.loads(request_body)
        # Check 'user' field (OpenAI standard field for tracking)
        if "user" in data:
            return data["user"]
        # Check first system message for agent_id pattern
        for msg in data.get("messages", []):
            if msg.get("role") == "system":
                m = re.search(r"agent[_-]id[:\s]+([a-zA-Z0-9_\-]+)", msg.get("content", ""))
                if m:
                    return m.group(1)
    except Exception:
        pass
    return "unknown"


# ── Stats ─────────────────────────────────────────────────────────────────────

class ProxyStats:
    def __init__(self):
        self.total_requests = 0
        self.total_completions = 0
        self.hidden_reads_promoted = 0
        self.promotion_by_agent = {}
        self.promotion_by_shard = {}
        self._lock = threading.Lock()

    def record_request(self):
        with self._lock:
            self.total_requests += 1

    def record_completion(self, agent_id, promoted_keys):
        with self._lock:
            self.total_completions += 1
            self.hidden_reads_promoted += len(promoted_keys)
            if agent_id not in self.promotion_by_agent:
                self.promotion_by_agent[agent_id] = 0
            self.promotion_by_agent[agent_id] += len(promoted_keys)
            for k in promoted_keys:
                if k not in self.promotion_by_shard:
                    self.promotion_by_shard[k] = 0
                self.promotion_by_shard[k] += 1

    def to_dict(self):
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "total_completions": self.total_completions,
                "hidden_reads_promoted": self.hidden_reads_promoted,
                "promotion_rate": (
                    self.hidden_reads_promoted / max(1, self.total_completions)
                ),
                "by_agent": dict(self.promotion_by_agent),
                "by_shard": dict(self.promotion_by_shard),
            }


# ── HTTP Proxy Handler ────────────────────────────────────────────────────────

_registry: ShardRegistry = None
_promoter: DeliveryLogPromoter = None
_stats: ProxyStats = None


class ProxyHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress default access log noise

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _forward_to_openai(self, body):
        """Forward request to OpenAI using http.client directly.
        Explicit header allowlist avoids ALL header duplication issues."""
        import http.client, ssl

        # Build exact header set — explicit allowlist, no dict-overwrite risk
        # Pull Authorization using case-insensitive lookup from self.headers
        auth   = self.headers.get("Authorization") or self.headers.get("authorization", "")
        ctype  = self.headers.get("Content-Type")  or self.headers.get("content-type", "")
        org    = self.headers.get("OpenAI-Organization") or self.headers.get("openai-organization", "")
        proj   = self.headers.get("OpenAI-Project") or self.headers.get("openai-project", "")

        fwd = {}
        if auth:   fwd["Authorization"]       = auth
        if ctype:  fwd["Content-Type"]        = ctype
        if org:    fwd["OpenAI-Organization"] = org
        if proj:   fwd["OpenAI-Project"]      = proj
        if body:   fwd["Content-Length"]      = str(len(body))

        # Forward via http.client — bypasses urllib header magic entirely
        ctx = ssl.create_default_context()
        try:
            conn = http.client.HTTPSConnection("api.openai.com", timeout=120, context=ctx)
            conn.request(self.command, self.path, body=body or None, headers=fwd)
            resp = conn.getresponse()
            status      = resp.status
            resp_headers = dict(resp.getheaders())
            resp_body   = resp.read()
            conn.close()
            return status, resp_headers, resp_body
        except Exception as e:
            return 502, {}, json.dumps({"error": str(e)}).encode()

    def _handle_stats(self):
        """Serve proxy statistics at GET /proxy/stats."""
        body = json.dumps(_stats.to_dict(), indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_shard_register(self, body):
        """
        Allow agents to register new shards with the proxy at runtime.
        POST /proxy/register-shard {"key": "my_shard_key"}
        """
        try:
            data = json.loads(body)
            key = data.get("key", "")
            if key:
                _registry.add(key)
                # Also refresh from S-Bus
                _registry.refresh()
        except Exception:
            pass
        resp = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self):
        _stats.record_request()
        if self.path == "/proxy/stats":
            self._handle_stats()
            return
        # Refresh shard registry on every GET /shards (keep it current)
        if self.path.endswith("/shards"):
            _registry.refresh()
        # Forward to OpenAI
        status, headers, body = self._forward_to_openai(b"")
        self._send_response(status, headers, body)

    def do_POST(self):
        _stats.record_request()
        body = self._read_body()

        # Proxy internal endpoints
        if self.path == "/proxy/register-shard":
            self._handle_shard_register(body)
            return

        # Forward to OpenAI
        status, resp_headers, resp_body = self._forward_to_openai(body)

        # Intercept completions responses to scan for shard key references.
        # resp_headers from urllib are lowercase — use case-insensitive lookup.
        content_type = next(
            (v for k, v in resp_headers.items() if k.lower() == "content-type"),
            ""
        )
        is_completion = (
            "/chat/completions" in self.path
            and status == 200
            and "application/json" in content_type
        )
        if is_completion:
            try:
                agent_id = self._extract_agent_id_from_request(body)
                completion_text = self._extract_completion_text(resp_body)
                if completion_text:
                    # Scan for shard keys and promote to DeliveryLog
                    found_keys = _registry.scan(completion_text)
                    if found_keys:
                        promoted = set()
                        for key in found_keys:
                            if _promoter.promote(key, agent_id):
                                promoted.add(key)
                        if promoted:
                            _stats.record_completion(agent_id, promoted)
                    else:
                        _stats.record_completion(agent_id, set())
            except Exception:
                pass  # Never fail the proxied response

        self._send_response(status, resp_headers, resp_body)

    def _extract_agent_id_from_request(self, body):
        return _extract_agent_id(body)

    def _extract_completion_text(self, resp_body):
        """Extract text from OpenAI chat completion response."""
        try:
            data = json.loads(resp_body)
            # Standard chat completion
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                return msg.get("content", "")
        except Exception:
            pass
        return ""

    def _send_response(self, status, headers, body):
        self.send_response(status)
        skip = {"transfer-encoding", "connection", "content-length"}
        for k, v in headers.items():
            if k.lower() not in skip:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Shard key auto-refresh thread ─────────────────────────────────────────────

def _refresh_loop(registry, interval=30):
    """Periodically refresh shard keys from S-Bus so new shards are picked up."""
    while True:
        time.sleep(interval)
        registry.refresh()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _registry, _promoter, _stats, OPENAI_TARGET

    ap = argparse.ArgumentParser(description="S-Bus LLM API proxy")
    ap.add_argument("--port",  type=int, default=PROXY_PORT)
    ap.add_argument("--sbus",  default=SBUS_URL)
    ap.add_argument("--openai-target", default=OPENAI_TARGET,
                    help="OpenAI base URL to forward to")
    args = ap.parse_args()

    OPENAI_TARGET = args.openai_target

    _registry = ShardRegistry(args.sbus)
    _promoter = DeliveryLogPromoter(args.sbus)
    _stats    = ProxyStats()

    # ── Startup self-test: verify we can reach OpenAI with the key ────────────
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            test_req = urllib.request.Request(
                f"{OPENAI_TARGET}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(test_req, timeout=10) as r:
                print(f"  Auth check    : OK (HTTP {r.status})")
        except urllib.error.HTTPError as e:
            print(f"  Auth check    : FAILED (HTTP {e.code})")
            if e.code == 401:
                print("  ERROR: OpenAI API key is invalid or not set.")
                print("  Set OPENAI_API_KEY before starting the proxy.")
                sys.exit(1)
    else:
        print("  Auth check    : SKIPPED (OPENAI_API_KEY not set)")
        print("  WARNING: Agents using this proxy need their own API key in headers.")

    # Start shard key refresh thread
    t = threading.Thread(target=_refresh_loop, args=(_registry,), daemon=True)
    t.start()

    print(f"S-Bus LLM Proxy starting on port {args.port}")
    print(f"  Forwarding to : {OPENAI_TARGET}")
    print(f"  S-Bus         : {args.sbus}")
    print(f"  Shard keys    : {len(_registry._keys)} loaded")
    print(f"  Stats         : GET http://localhost:{args.port}/proxy/stats")
    print(f"  Register shard: POST /proxy/register-shard {{\"key\": \"...\"}}")
    print()
    print("Agents should use:")
    print(f"  OPENAI_BASE_URL=http://localhost:{args.port}/v1")
    print("  # and set 'user' field in API calls to their agent_id")
    print()

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nProxy stopped.")
        final = _stats.to_dict()
        print(f"Final stats: {json.dumps(final, indent=2)}")


if __name__ == "__main__":
    main()