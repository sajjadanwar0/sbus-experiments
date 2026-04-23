import threading
from urllib.request import Request, urlopen

class _CompletionsProxy:
    def __init__(self, real_completions, wrapper):
        self._real = real_completions
        self._wrapper = wrapper

    def create(self, *args, **kwargs):
        agent_id = kwargs.get("user", "unknown")

        response = self._real.create(*args, **kwargs)
        try:
            text = ""
            for choice in response.choices:
                msg = choice.message
                if hasattr(msg, "content") and msg.content:
                    text += msg.content
            if text:
                self._wrapper._on_completion(text, agent_id)
        except Exception:
            pass

        return response


class _ChatProxy:
    def __init__(self, real_chat, wrapper):
        self.completions = _CompletionsProxy(real_chat.completions, wrapper)


class PhiddenWrapper:
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
        self._keyword_map: dict = {}

        self._total_completions = 0
        self._total_promoted = 0
        self._promoted_by_key: dict = {}
        self._promoted_by_agent: dict = {}

        self.chat = _ChatProxy(openai_client.chat, self)

    def register_shards(self, keys: list):
        with self._lock:
            for k in keys:
                self._shard_keys.add(k)
                parts = k.rsplit("_", 1)
                base = parts[0] if (len(parts)==2 and len(parts[1])==8 and parts[1].isalnum()) else k
                self._base_names.add(base)
                for keyword in self.SHARD_KEYWORDS.get(base, [base]):
                    if keyword.lower() not in self._keyword_map:
                        self._keyword_map[keyword.lower()] = base

    def register_runtime_shard(self, full_key: str):
        with self._lock:
            self._shard_keys.add(full_key)
            parts = full_key.rsplit("_", 1)
            base = parts[0] if (len(parts)==2 and len(parts[1])==8 and parts[1].isalnum()) else full_key
            self._base_names.add(base)
            for keyword in self.SHARD_KEYWORDS.get(base, [base]):
                self._keyword_map[keyword.lower()] = full_key

    def clear_shards(self):
        with self._lock:
            self._shard_keys.clear()
            self._base_names.clear()
            self._keyword_map.clear()

    def _on_completion(self, text: str, agent_id: str):
        with self._lock:
            self._total_completions += 1
            keyword_map = dict(self._keyword_map)  # keyword → full_key

        text_lower = text.lower()
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
        url = f"{self._sbus_url}/shard/{shard_key}?agent_id={agent_id}"
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=5) as r:
                r.read()
            return True
        except Exception:
            return False

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
        print("\n=== PhiddenWrapper Stats ===")
        print(f"  Completions intercepted : {s['total_completions']}")
        print(f"  Hidden reads promoted   : {s['hidden_reads_promoted']}")
        print(f"  Promotion rate          : {s['promotion_rate']*100:.1f}%")
        if s["by_key"]:
            print("  By shard key:")
            for k, v in sorted(s["by_key"].items()):
                print(f"    {k}: {v}")
        print()
