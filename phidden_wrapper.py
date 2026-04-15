#!/usr/bin/env python3
"""
phidden_wrapper.py — Python-level R_hidden → R_obs Promotion Wrapper
=====================================================================
Wraps the OpenAI client to intercept every completion response,
scan it for shard key references, and register those references
with the S-Bus DeliveryLog — promoting R_hidden reads to R_obs.

This replaces the HTTP proxy approach and avoids all API key routing
issues. The wrapper operates at the Python object level, not the
HTTP level.

PAPER CLAIM (§8):
  "The production path to phidden → 0 is an LLM API proxy: a
  transparent layer between agents and the LLM API that intercepts
  completion responses, scans for shard-key references, and promotes
  R_hidden reads to R_obs without any agent code change."

This wrapper implements that claim at the SDK level — identical
semantics, simpler deployment.

USAGE:
  from phidden_wrapper import PhiddenWrapper

  # Wrap once — use exactly like a normal OpenAI client
  client = PhiddenWrapper(
      openai_client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
      sbus_url="http://localhost:7000",
  )

  # Register shard keys (call once per trial)
  client.register_shards(["portfolio_state", "patient_record"])

  # Use exactly like oai.chat.completions.create(...)
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[...],
      user="agent_id_here",    # used to attribute hidden reads
  )

  # Get promotion stats
  stats = client.stats()
  # {"total_completions": N, "hidden_reads_promoted": M, ...}
"""

import re
import json
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError


class _CompletionsProxy:
    """Intercepts chat.completions.create() calls."""

    def __init__(self, real_completions, wrapper):
        self._real = real_completions
        self._wrapper = wrapper

    def create(self, *args, **kwargs):
        # Extract agent_id from the 'user' field if present
        agent_id = kwargs.get("user", "unknown")

        # Make the actual OpenAI call unchanged
        response = self._real.create(*args, **kwargs)

        # Scan the response text for shard key references
        try:
            text = ""
            for choice in response.choices:
                msg = choice.message
                if hasattr(msg, "content") and msg.content:
                    text += msg.content
            if text:
                self._wrapper._on_completion(text, agent_id)
        except Exception:
            pass  # Never fail the caller

        return response


class _ChatProxy:
    def __init__(self, real_chat, wrapper):
        self.completions = _CompletionsProxy(real_chat.completions, wrapper)


class PhiddenWrapper:
    """
    Wraps an OpenAI client to intercept completions and promote
    R_hidden shard-key reads to R_obs via S-Bus DeliveryLog.

    Drop-in replacement: use `wrapper.chat.completions.create()`
    instead of `oai.chat.completions.create()`.
    """

    # Semantic keywords per shard base name.
    # LLM responses never contain "portfolio_state" but always contain "portfolio".
    SHARD_KEYWORDS = {
        "portfolio_state":  ["portfolio", "allocation", "equity", "VaR", "rebalance"],
        "patient_record":   ["patient", "medical record", "diagnosis", "medication"],
        "api_schema":       ["API", "endpoint", "schema", "api_schema"],
        "k8s_config":       ["kubernetes", "k8s", "deployment", "k8s_config"],
        "paper_outline":    ["paper outline", "contribution", "methodology", "paper_outline"],
        "financial_report": ["financial report", "revenue", "EBITDA", "earnings"],
        "runbook":          ["runbook", "incident", "rollback", "escalation"],
        "service_contract": ["service contract", "SLA", "service_contract"],
        "pipeline_config":  ["pipeline", "training", "feature vector", "pipeline_config"],
        "contract_draft":   ["contract", "liability", "indemnification", "contract_draft"],
    }

    def __init__(self, openai_client, sbus_url: str = "http://localhost:7000"):
        self._client = openai_client
        self._sbus_url = sbus_url
        self._lock = threading.Lock()

        self._shard_keys: set = set()
        self._base_names: set = set()
        # keyword → shard base name mapping for this run
        self._keyword_map: dict = {}

        self._total_completions = 0
        self._total_promoted = 0
        self._promoted_by_key: dict = {}
        self._promoted_by_agent: dict = {}

        self.chat = _ChatProxy(openai_client.chat, self)

    # ── Shard registration ────────────────────────────────────────────────────

    def register_shards(self, keys: list):
        """Register shard base names for keyword scanning."""
        with self._lock:
            for k in keys:
                self._shard_keys.add(k)
                parts = k.rsplit("_", 1)
                base = parts[0] if (len(parts)==2 and len(parts[1])==8 and parts[1].isalnum()) else k
                self._base_names.add(base)
                for keyword in self.SHARD_KEYWORDS.get(base, [base]):
                    # Map keyword → base name (full key updated later by register_runtime_shard)
                    if keyword.lower() not in self._keyword_map:
                        self._keyword_map[keyword.lower()] = base

    def register_runtime_shard(self, full_key: str):
        """Register the actual UUID-suffixed shard key created at trial start.
        Call this immediately after _create_shard() so _promote uses the real key.
        e.g. full_key = 'portfolio_state_a1b2c3d4'"""
        with self._lock:
            self._shard_keys.add(full_key)
            parts = full_key.rsplit("_", 1)
            base = parts[0] if (len(parts)==2 and len(parts[1])==8 and parts[1].isalnum()) else full_key
            self._base_names.add(base)
            # Update keyword_map to point to the full key for _promote
            for keyword in self.SHARD_KEYWORDS.get(base, [base]):
                self._keyword_map[keyword.lower()] = full_key

    def clear_shards(self):
        """Clear shard registry between trials."""
        with self._lock:
            self._shard_keys.clear()
            self._base_names.clear()
            self._keyword_map.clear()

    # ── Interception ──────────────────────────────────────────────────────────

    def _on_completion(self, text: str, agent_id: str):
        """Scans completion text for shard keywords, promotes matches to R_obs."""
        with self._lock:
            self._total_completions += 1
            keyword_map = dict(self._keyword_map)  # keyword → full_key

        text_lower = text.lower()
        # Find which full shard keys are referenced (via any keyword)
        matched_keys = set()
        for keyword, full_key in keyword_map.items():
            if keyword in text_lower:
                matched_keys.add(full_key)

        for full_key in matched_keys:
            promoted = self._promote(full_key, agent_id)
            if promoted:
                base = full_key.rsplit("_", 1)[0] if "_" in full_key else full_key
                with self._lock:
                    self._total_promoted += 1
                    self._promoted_by_key[base] = self._promoted_by_key.get(base, 0) + 1
                    self._promoted_by_agent[agent_id] = self._promoted_by_agent.get(agent_id, 0) + 1

    def _promote(self, shard_key: str, agent_id: str) -> bool:
        """
        Issue GET /shard/{key}?agent_id={id} to register the hidden
        read in the S-Bus DeliveryLog → promotes R_hidden to R_obs.
        """
        url = f"{self._sbus_url}/shard/{shard_key}?agent_id={agent_id}"
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=5) as r:
                r.read()
            return True
        except Exception:
            return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_completions":   self._total_completions,
                "hidden_reads_promoted": self._total_promoted,
                "promotion_rate": (
                    self._total_promoted / max(1, self._total_completions)
                ),
                "by_key":   dict(self._promoted_by_key),
                "by_agent": dict(self._promoted_by_agent),
            }

    def print_stats(self):
        s = self.stats()
        print(f"\n=== PhiddenWrapper Stats ===")
        print(f"  Completions intercepted : {s['total_completions']}")
        print(f"  Hidden reads promoted   : {s['hidden_reads_promoted']}")
        print(f"  Promotion rate          : {s['promotion_rate']*100:.1f}%")
        if s["by_key"]:
            print(f"  By shard key:")
            for k, v in sorted(s["by_key"].items()):
                print(f"    {k}: {v}")
        print()