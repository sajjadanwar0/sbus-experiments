import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from urllib.parse import urlencode
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import HTTPError

try:
    from anthropic import AsyncAnthropic
except ImportError:
    print("ERROR: uv add anthropic"); sys.exit(1)

SBUS_URL  = os.getenv("SBUS_URL",  "http://localhost:7000")
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:9000")
BACKBONE  = os.getenv("BACKBONE",  "claude-haiku-4-5-20251001")
TEMP      = float(os.getenv("TEMP", "0.3"))

_opener = build_opener(ProxyHandler({}))

TASK_DOMAINS = {
    "django_queryset": {
        "desc":   "Fix Django queryset ordering with select_related() on FK traversal.",
        "shards": ["models_state", "orm_query", "test_fixtures", "migration_plan"],
        "tasks": [
            {"id": "django-11019", "goal": "Fix queryset ordering with select_related fields"},
            {"id": "django-11820", "goal": "Fix ordering on annotated querysets"},
            {"id": "django-14155", "goal": "Fix reverse FK ordering in admin views"},
        ],
    },
    "django_admin": {
        "desc":   "Fix Django admin list_display rendering with computed fields.",
        "shards": ["admin_config", "model_fields", "template_render", "test_client"],
        "tasks": [
            {"id": "django-14034", "goal": "Fix list_display with @admin.display decorator"},
            {"id": "django-15127", "goal": "Fix admin inline formset validation"},
        ],
    },
    "django_migration": {
        "desc":   "Fix Django migration dependency resolution on multi-app state.",
        "shards": ["migration_graph", "app_state", "schema_editor", "test_db"],
        "tasks": [
            {"id": "django-15098", "goal": "Fix migration squash with RunPython"},
            {"id": "django-16105", "goal": "Fix migration dependency detection"},
        ],
    },
    "astropy_fits": {
        "desc":   "Fix FITS header parsing with CONTINUE cards and BLANK keyword.",
        "shards": ["header_state", "hdu_structure", "card_formatter", "io_path"],
        "tasks": [
            {"id": "astropy-13969", "goal": "Fix FITS CONTINUE card round-tripping"},
            {"id": "astropy-14328", "goal": "Fix FITS BLANK/BZERO interaction"},
        ],
    },
    "astropy_wcs": {
        "desc":   "Fix WCS celestial-coordinate transformations with SIP distortion.",
        "shards": ["wcs_transform", "sip_coeffs", "coord_frame", "pixel_map"],
        "tasks": [
            {"id": "astropy-14539", "goal": "Fix WCS with SIP distortion on ref pixel"},
            {"id": "astropy-12907", "goal": "Fix celestial frame conversion"},
        ],
    },
    "astropy_units": {
        "desc":   "Fix astropy.units composite-unit equivalencies.",
        "shards": ["unit_registry", "equivalency_map", "quantity_state", "unit_parser"],
        "tasks": [
            {"id": "astropy-13745", "goal": "Fix composite-unit equivalency"},
            {"id": "astropy-14096", "goal": "Fix quantity dimensional analysis"},
        ],
    },
    "sympy_solver": {
        "desc":   "Fix SymPy solveset for transcendental equations with piecewise branches.",
        "shards": ["expr_tree", "assumption_ctx", "solver_strategy", "test_cases"],
        "tasks": [
            {"id": "sympy-22714", "goal": "Fix solveset for piecewise transcendentals"},
            {"id": "sympy-23117", "goal": "Fix Intersection(FiniteSet, ImageSet)"},
        ],
    },
    "sympy_matrix": {
        "desc":   "Fix SymPy matrix-decomposition handling of symbolic entries.",
        "shards": ["matrix_repr", "decomp_method", "symbolic_engine", "numerical_fallback"],
        "tasks": [
            {"id": "sympy-22456", "goal": "Fix matrix LU with symbolic pivots"},
            {"id": "sympy-23824", "goal": "Fix eigen for block-diagonal symbolic"},
        ],
    },
    "requests_session": {
        "desc":   "Fix Requests session cookie-jar merging on redirect chain.",
        "shards": ["cookie_jar", "session_state", "redirect_chain", "adapter_pool"],
        "tasks": [
            {"id": "requests-6028", "goal": "Fix cookie-jar merge on 3xx redirect"},
            {"id": "requests-6358", "goal": "Fix HTTPAdapter pool with proxy"},
        ],
    },
    "scikit_estimator": {
        "desc":   "Fix scikit-learn estimator __sklearn_is_fitted__ with partial fit.",
        "shards": ["estimator_state", "fit_params", "validation_hook", "test_grid"],
        "tasks": [
            {"id": "sklearn-25443", "goal": "Fix is_fitted on partial_fit path"},
            {"id": "sklearn-26242", "goal": "Fix GridSearchCV with refit=False"},
        ],
    },
}

N_AGENTS = 4
DOMAIN_VOCAB = sorted({s for d in TASK_DOMAINS.values() for s in d["shards"]})

def report_change_tool(vocab_names):
    return {
        "name": "report_change",
        "description": (
            "Report your proposed technical change and which shards' content "
            "influenced your reasoning. Call this tool exactly once per step."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "change": {
                    "type": "string",
                    "description":
                        "One concrete technical change (<=100 words). Be specific: "
                        "describe what you would edit in the codebase to move the "
                        "task forward."
                },
                "shards_used": {
                    "type": "array",
                    "items": {"type": "string", "enum": vocab_names},
                    "description":
                        "List EVERY shard whose content influenced your reasoning, "
                        "including shards you relied on from remembered state (not "
                        "just fresh this step). Do NOT list shards you never "
                        "encountered. Each entry must be one of the allowed names."
                }
            },
            "required": ["change", "shards_used"],
        },
    }

class VendorClient:
    name: str

    async def call_with_tool(
        self, *, system: str, messages: list, vocab: list,
        extra_headers: dict, force_tool: bool = True,
    ):
        raise NotImplementedError


class AnthropicClient(VendorClient):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, use_proxy: bool):
        base_url = PROXY_URL if use_proxy else None
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    async def call_with_tool(
        self, *, system, messages, vocab, extra_headers, force_tool=True,
    ):
        tool_def = report_change_tool(vocab)
        kwargs = dict(
            model       = self.model,
            max_tokens  = 500,
            temperature = TEMP,
            system      = system,
            messages    = messages,
            tools       = [tool_def],
            extra_headers = extra_headers,
        )
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": "report_change"}

        t0 = time.time()
        try:
            resp = await self.client.messages.create(**kwargs)
        except Exception as e:
            # Retry once without forced tool-use — sometimes the model
            # refuses to emit a tool_use block under specific conditions.
            try:
                kwargs.pop("tool_choice", None)
                resp = await self.client.messages.create(**kwargs)
            except Exception as e2:
                return set(), f"LLM error: {e2}", int((time.time() - t0) * 1000)

        llm_ms = int((time.time() - t0) * 1000)

        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" \
                    and getattr(block, "name", None) == "report_change":
                inp = getattr(block, "input", None) or {}
                used = inp.get("shards_used", [])
                change = inp.get("change", "")
                if isinstance(used, list):
                    shards = {s for s in used if isinstance(s, str) and s in vocab}
                else:
                    shards = set()
                return shards, (change if isinstance(change, str) else ""), llm_ms

        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", "") or "")
        s, c = _extract_text_fallback("\n".join(text_parts), set(vocab))
        return s, c, llm_ms


class GeminiClient(VendorClient):
    name = "gemini"

    def __init__(self, api_key: str, model: str, use_proxy: bool):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            print("ERROR: pip install google-genai"); sys.exit(1)
        self._genai = genai
        self._types = genai_types
        if use_proxy:
            self.client = genai.Client(
                api_key=api_key,
                http_options=genai_types.HttpOptions(base_url=PROXY_URL),
            )
        else:
            self.client = genai.Client(api_key=api_key)
        self.model = model

    def _build_tool(self, vocab):
        types = self._types
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="report_change",
                    description=(
                        "Report your proposed technical change and which shards' "
                        "content influenced your reasoning. Call this function "
                        "exactly once per step."
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "change": types.Schema(
                                type="STRING",
                                description=(
                                    "One concrete technical change (<=100 words). "
                                    "Be specific."
                                ),
                            ),
                            "shards_used": types.Schema(
                                type="ARRAY",
                                items=types.Schema(
                                    type="STRING",
                                    enum=vocab,
                                ),
                                description=(
                                    "List EVERY shard whose content influenced "
                                    "your reasoning, including shards from "
                                    "remembered state. Each entry must be one of "
                                    "the allowed names."
                                ),
                            ),
                        },
                        required=["change", "shards_used"],
                    ),
                )
            ]
        )

    def _translate_messages(self, system: str, messages: list):
        types = self._types
        contents = []
        for m in messages:
            role = m.get("role", "user")
            if role == "assistant":
                role = "model"
            content = m.get("content", "")
            if isinstance(content, str):
                parts = [types.Part.from_text(text=content)]
            elif isinstance(content, list):
                texts = []
                for blk in content:
                    if isinstance(blk, dict):
                        if blk.get("type") == "text":
                            texts.append(blk.get("text", ""))
                    elif isinstance(blk, str):
                        texts.append(blk)
                parts = [types.Part.from_text(text="\n".join(texts))]
            else:
                parts = [types.Part.from_text(text=str(content))]
            contents.append(types.Content(role=role, parts=parts))
        return contents, system

    async def call_with_tool(
        self, *, system, messages, vocab, extra_headers, force_tool=True,
    ):
        types = self._types

        contents, system_instruction = self._translate_messages(system, messages)
        tool = self._build_tool(vocab)

        tool_config = None
        if force_tool:
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["report_change"],
                )
            )

        thinking_config = types.ThinkingConfig(thinking_budget=0)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=TEMP,
            max_output_tokens=2000,  # generous budget; was 500
            tools=[tool],
            tool_config=tool_config,
            thinking_config=thinking_config,
            http_options=types.HttpOptions(
                headers=extra_headers,
            ),
        )

        t0 = time.time()
        try:
            resp = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            if force_tool:
                try:
                    config2 = types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=TEMP,
                        max_output_tokens=2000,
                        tools=[tool],
                        thinking_config=thinking_config,
                        http_options=types.HttpOptions(headers=extra_headers),
                    )
                    resp = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model,
                        contents=contents,
                        config=config2,
                    )
                except Exception as e2:
                    return set(), f"LLM error: {e2}", int((time.time() - t0) * 1000)
            else:
                return set(), f"LLM error: {e}", int((time.time() - t0) * 1000)

        llm_ms = int((time.time() - t0) * 1000)

        try:
            for cand in (resp.candidates or []):
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in (content.parts or []):
                    fc = getattr(part, "function_call", None)
                    if fc and getattr(fc, "name", "") == "report_change":
                        args = getattr(fc, "args", None) or {}
                        if hasattr(args, "items"):
                            args = dict(args.items())
                        elif not isinstance(args, dict):
                            args = {}
                        used = args.get("shards_used", [])
                        change = args.get("change", "")
                        if isinstance(used, list):
                            shards = {s for s in used if isinstance(s, str) and s in vocab}
                        else:
                            shards = set()
                        return shards, (change if isinstance(change, str) else ""), llm_ms

            text_parts = []
            for cand in (resp.candidates or []):
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in (content.parts or []):
                    txt = getattr(part, "text", "") or ""
                    if txt:
                        text_parts.append(txt)
            s, c = _extract_text_fallback("\n".join(text_parts), set(vocab))
            return s, c, llm_ms
        except Exception as e:
            return set(), f"parse error: {e}", llm_ms


def make_vendor_client(vendor: str, model: str, use_proxy: bool) -> VendorClient:
    if vendor == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(1)
        return AnthropicClient(api_key=api_key, model=model, use_proxy=use_proxy)
    elif vendor == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY (or GOOGLE_API_KEY) not set"); sys.exit(1)
        return GeminiClient(api_key=api_key, model=model, use_proxy=use_proxy)
    else:
        print(f"ERROR: unknown vendor '{vendor}' (expected: anthropic|gemini)")
        sys.exit(1)

def _http(method, url, body=None, params=None, timeout=10):
    if params:
        url = f"{url}?{urlencode(params)}"
    data = None
    headers = {"accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    req = Request(url, data=data, method=method, headers=headers)
    try:
        with _opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return resp.status, {"raw": raw}
    except HTTPError as e:
        try:
            b = e.read().decode("utf-8", errors="replace")
            return e.code, json.loads(b) if b else {}
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}

def http_get(url, params=None): return _http("GET",  url, params=params)
def http_post(url, body):       return _http("POST", url, body=body)

def health_check():
    st, _ = http_get(f"{SBUS_URL}/admin/delivery-log")
    return st == 200

def proxy_health_check():
    try:
        req = Request(f"{PROXY_URL}/health", method="GET")
        with _opener.open(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

def admin_delivery_log_for_agent(agent_id):
    st, data = http_get(f"{SBUS_URL}/admin/delivery-log")
    if st != 200 or not isinstance(data, dict):
        return set()
    agents = data.get("agents", {})
    if not isinstance(agents, dict):
        return set()
    entries = agents.get(agent_id, {})
    if isinstance(entries, dict):
        return set(entries.keys())
    if isinstance(entries, list):
        return {e.get("key") or e.get("shard_key")
                for e in entries
                if e.get("key") or e.get("shard_key")}
    return set()

def get_shard_version(full_key):
    st, data = http_get(f"{SBUS_URL}/shard/{full_key}")
    if st == 200 and isinstance(data, dict):
        return int(data.get("version", 0))
    return -1

@dataclass
class StepRow:
    condition:          str
    domain:             str
    task_id:            str
    run_id:             str
    agent_idx:          int
    step:               int
    backbone:           str
    r_obs_http:         str
    r_delivery_log:     str
    r_obs_proxy:        str
    r_self:             str
    f_obs_http:         float
    f_obs_total:        float
    commit_status:      str
    commit_http_code:   int
    commit_latency_ms:  int
    type_i_corr:        int
    llm_ms:             int

def shards_to_get(agent_idx, step, all_base, max_k=2):
    h = int(hashlib.sha1(f"{agent_idx}:{step}".encode()).hexdigest(), 16)
    k_hash = 1 + (h % max(1, max_k))
    k = min(max_k, k_hash)
    start = h % len(all_base)
    return [all_base[(start + i) % len(all_base)] for i in range(k)]

def target_shard(agent_idx, step, all_base):
    return all_base[(agent_idx + step) % len(all_base)]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def _extract_tool_use(resp, vocab):
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" \
                and getattr(block, "name", None) == "report_change":
            inp = getattr(block, "input", None) or {}
            used = inp.get("shards_used", [])
            change = inp.get("change", "")
            if isinstance(used, list):
                shards = {s for s in used if isinstance(s, str) and s in vocab}
            else:
                shards = set()
            return shards, (change if isinstance(change, str) else "")
    return set(), ""

def _extract_text_fallback(text, vocab):
    if not text:
        return set(), ""
    candidates = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i+1])
                start = None
    for c in reversed(candidates):
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        used = obj.get("shards_used", [])
        change = obj.get("change", "")
        if isinstance(used, list):
            shards = {s for s in used if isinstance(s, str) and s in vocab}
            return shards, (change if isinstance(change, str) else "")
    return set(), ""

def parse_response(resp, vocab):
    shards, change = _extract_tool_use(resp, vocab)
    if shards or change:
        return shards, change
    text_parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", "") or "")
    return _extract_text_fallback("\n".join(text_parts), vocab)

async def run_agent(
    client:       VendorClient,
    use_proxy:    bool,
    agent_id:     str,
    session_id:   str,
    shard_suffix: str,
    agent_idx:    int,
    domain:       dict,
    task:         dict,
    run_id:       str,
    n_steps:      int,
    condition:    str,
):
    rows = []
    history = []
    base = list(domain["shards"])
    full_keys = {b: f"{b}{shard_suffix}" for b in base}
    desc = f"{domain['desc']} Task: {task['goal']}"

    last_seen_version = {}
    last_seen_content = {}
    last_seen_step    = {}

    for step in range(n_steps):
        gets = shards_to_get(agent_idx, step, base, max_k=2)
        r_obs_http = set()
        fresh_block_lines = []
        for b in gets:
            fk = full_keys[b]
            st, data = http_get(f"{SBUS_URL}/shard/{fk}", {"agent_id": agent_id})
            if st == 200:
                v = int(data.get("version", 0))
                c = (data.get("content") or "")[:160]
                r_obs_http.add(b)
                last_seen_version[b] = v
                last_seen_content[b] = c
                last_seen_step[b]    = step
                fresh_block_lines.append(f"  - {b}: v{v} — {c[:100]}")

        remembered_lines = []
        for b in base:
            if b in r_obs_http or b not in last_seen_version:
                continue
            v_old = last_seen_version[b]
            c_old = last_seen_content.get(b, "")
            s_old = last_seen_step.get(b, -1)
            remembered_lines.append(
                f"  - {b}: v{v_old} (from step {s_old+1}) — {c_old[:100]} "
                f"[NOT refetched; may be stale if others wrote]"
            )

        fresh_block      = "\n".join(fresh_block_lines)   or "  (none fetched this step)"
        remembered_block = "\n".join(remembered_lines)    or "  (nothing remembered yet)"
        shard_list_str   = ", ".join(base)

        system_msg = (
            "You are a software engineering agent collaborating with 3 others "
            "on a shared codebase represented as shards. Each step, you see "
            "fresh state for a subset of shards and remembered state for the "
            "others. Your change must maintain consistency across the FULL "
            "shard state you rely on — both freshly fetched and remembered.\n"
            "\n"
            "You MUST respond by calling the `report_change` tool exactly "
            "once. Do not emit prose. In `shards_used`, list EVERY shard "
            "whose content influenced your reasoning, including shards you "
            "relied on from remembered state (not just fresh this step). Do "
            "NOT list shards you never encountered.\n"
            f"\n"
            f"Allowed shard names (exact match required): {shard_list_str}."
        )
        user_msg = (
            f"Task: {desc} (Step {step+1}/{n_steps}, Agent α{agent_idx})\n"
            f"\n"
            f"Shards fetched THIS STEP (values are current):\n"
            f"{fresh_block}\n"
            f"\n"
            f"Shards REMEMBERED from earlier (values may be stale — other "
            f"agents may have committed updates since):\n"
            f"{remembered_block}\n"
            f"\n"
            f"Call `report_change` with your proposed change and the "
            f"shards your reasoning depends on."
        )

        messages = []
        for m in history[-6:]:
            messages.append(m)
        messages.append({"role": "user", "content": user_msg})

        extra_headers = {
            "X-SBus-Agent-Id":     agent_id,
            "X-SBus-Session-Id":   session_id,
            "X-SBus-Shard-Suffix": shard_suffix,
        }

        r_self, ans_text, llm_ms = await client.call_with_tool(
            system        = system_msg,
            messages      = messages,
            vocab         = base,
            extra_headers = extra_headers,
            force_tool    = True,
        )

        dl_after = admin_delivery_log_for_agent(agent_id)
        dl_base_after = {
            k[: -len(shard_suffix)] if shard_suffix and k.endswith(shard_suffix) else k
            for k in dl_after
        } & set(base)
        r_obs_proxy = dl_base_after - r_obs_http

        tgt_base = target_shard(agent_idx, step, base)
        tgt_full = full_keys[tgt_base]
        exp_ver  = last_seen_version.get(tgt_base, get_shard_version(tgt_full))
        delta = f"α{agent_idx} step {step} [{client.name}]: {(ans_text or 'change')[:120]}"
        t_cmt_0 = time.time()
        st_cmt, data_cmt = http_post(f"{SBUS_URL}/commit/v2", {
            "agent_id":         agent_id,
            "key":              tgt_full,
            "expected_version": exp_ver,
            "delta":            delta,
        })
        commit_ms = int((time.time() - t_cmt_0) * 1000)

        if st_cmt == 200:
            commit_status = "ok"
            if isinstance(data_cmt, dict) and "version" in data_cmt:
                last_seen_version[tgt_base] = int(data_cmt["version"])
        elif st_cmt == 409:
            err = (data_cmt or {}).get("error", "")
            if "cross_shard" in err.lower() or "crossshard" in err.lower():
                commit_status = "cross_shard_stale"
            else:
                commit_status = "version_mismatch"
        else:
            commit_status = f"http_{st_cmt}"

        type_i = 0
        if commit_status == "ok":
            for b in r_self:
                if b == tgt_base:
                    continue
                vseen = last_seen_version.get(b, -1)
                if vseen < 0:
                    continue
                st_v, d_v = http_get(f"{SBUS_URL}/shard/{full_keys[b]}")
                if st_v == 200 and isinstance(d_v, dict):
                    v_now = int(d_v.get("version", -1))
                    if v_now > vseen:
                        type_i = 1
                        break

        def frac(n, d): return round(n / d, 4) if d else 0.0
        f_obs_http  = frac(len(r_obs_http    & r_self), len(r_self))
        f_obs_total = frac(len(dl_base_after & r_self), len(r_self))

        rows.append(StepRow(
            condition=condition,
            domain=domain.get("name", domain.get("desc", "")[:20]),
            task_id=task["id"],
            run_id=run_id,
            agent_idx=agent_idx,
            step=step,
            backbone=client.model,
            r_obs_http=",".join(sorted(r_obs_http)),
            r_delivery_log=",".join(sorted(dl_base_after)),
            r_obs_proxy=",".join(sorted(r_obs_proxy)),
            r_self=",".join(sorted(r_self)),
            f_obs_http=f_obs_http,
            f_obs_total=f_obs_total,
            commit_status=commit_status,
            commit_http_code=st_cmt,
            commit_latency_ms=commit_ms,
            type_i_corr=type_i,
            llm_ms=llm_ms,
        ))

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ans_text or "change reported"})

    return rows

async def run_trial(
    client, use_proxy, domain_key, domain, task,
    n_steps, condition, run_id,
):
    shard_suffix = f"_{run_id}"
    vendor_tag   = client.name
    session_id   = f"sess_{run_id}_{condition}_{vendor_tag}"
    for b in domain["shards"]:
        http_post(f"{SBUS_URL}/shard", {
            "key":      f"{b}{shard_suffix}",
            "content":  f"Initial state for {b} — {task['goal'][:60]}",
            "goal_tag": f"proxy_ph2_{vendor_tag}_{task['id']}",
        })

    agents = await asyncio.gather(*[
        run_agent(
            client=client,
            use_proxy=use_proxy,
            agent_id    = f"agent_{i}_{run_id}_{condition}_{vendor_tag}",
            session_id  = session_id,
            shard_suffix= shard_suffix,
            agent_idx   = i,
            domain      = {**domain, "name": domain_key},
            task        = task,
            run_id      = run_id,
            n_steps     = n_steps,
            condition   = condition,
        )
        for i in range(N_AGENTS)
    ], return_exceptions=True)

    out = []
    for r in agents:
        if isinstance(r, list):
            out.extend(r)
        elif isinstance(r, Exception):
            print(f"    ! agent error: {r}")
    return out

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * (p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def rule_of_three_upper(n, z=1.96):
    return 0.0 if n == 0 else 3.0 / n

def paired_bootstrap_ci(pairs, iters=5000, alpha=0.05):
    if not pairs:
        return 0.0, 0.0, 0.0
    diffs = [b - a for a, b in pairs]
    mean = sum(diffs) / len(diffs)
    rng  = random.Random(42)
    boot = []
    n = len(diffs)
    for _ in range(iters):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)
    boot.sort()
    lo = boot[int(iters *  (alpha/2))]
    hi = boot[int(iters * (1 - alpha/2))]
    return mean, lo, hi

def load_gpt_summary(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  (could not read GPT_SUMMARY at {path}: {e})")
        return None

def comparison_block(target_summary, gpt_summary, vendor_label="haiku_4_5",
                     vendor_display="Haiku-4.5"):
    def extract(s):
        cd = s.get("coverage_decomposition", {}) or {}
        pu = s.get("paired_uplift_proxy_on_minus_off", {}) or {}
        conds = s.get("conditions", {}) or {}
        off = conds.get("proxy_off", {}) or {}
        on  = conds.get("proxy_on",  {}) or {}
        return {
            "f_obs_http":              cd.get("http_this_step", off.get("mean_f_obs_http", 0)),
            "dl_accumulation":         cd.get("dl_accumulation_under_off",
                                              off.get("mean_within_row_uplift", 0)),
            "total_coverage_off":      cd.get("total_coverage_off", off.get("mean_f_obs_total", 0)),
            "proxy_marginal":          pu.get("mean_diff", 0),
            "proxy_marginal_95ci_lo":  pu.get("bs_95_lo", 0),
            "proxy_marginal_95ci_hi":  pu.get("bs_95_hi", 0),
            "commit_ok_off":           off.get("commit_ok_rate", 0),
            "commit_ok_on":            on.get("commit_ok_rate", 0),
            "type_i_off":              off.get("type_i_corruptions", 0),
            "type_i_on":               on.get("type_i_corruptions", 0),
            "n_rows_off":              off.get("n_rows", 0),
            "n_rows_on":               on.get("n_rows", 0),
        }

    h = extract(target_summary); g = extract(gpt_summary)
    delta = {k: round(h[k] - g[k], 4) for k in h
             if isinstance(h[k], (int, float)) and isinstance(g[k], (int, float))}

    block = {
        "gpt_4o_mini":                       g,
        vendor_label:                        h,
        f"absolute_delta_{vendor_label}_minus_gpt": delta,
        "dl_accumulation_replicates_within_0.05":
            abs(delta.get("dl_accumulation", 99)) <= 0.05,
        "type_i_parity_preserved":
            (h["type_i_off"] == 0 and h["type_i_on"] == 0),
    }

    if h["n_rows_on"] > 0:
        block["commit_throughput_collapse_replicated"] = h["commit_ok_on"] < h["commit_ok_off"]
    else:
        block["commit_throughput_collapse_replicated"] = None
        block["note_proxy_on_skipped"] = (
            f"proxy_on condition not run on {vendor_display}."
        )

    lines = [
        "", "─" * 74,
        "Cross-backbone comparison (Exp. PROXY-PH2)",
        "─" * 74,
        f"{'Metric':<32s} {'GPT-4o-mini':>14s} {vendor_display:>14s} {'Δ':>10s}",
        "-" * 74,
    ]
    def row(label, key, fmt="{:>14.4f}"):
        gv = g.get(key, 0); hv = h.get(key, 0); dv = delta.get(key, 0)
        return (f"{label:<32s} "
                f"{fmt.format(gv):>14s} "
                f"{fmt.format(hv):>14s} "
                f"{dv:>+10.4f}")
    lines.append(row("f_obs_http (this-step)",  "f_obs_http"))
    lines.append(row("DL-accumulation",         "dl_accumulation"))
    lines.append(row("total_coverage (off)",    "total_coverage_off"))
    lines.append(row("commit_ok proxy_off",     "commit_ok_off"))
    if h["n_rows_on"] > 0:
        lines.append(row("proxy marginal (paired)", "proxy_marginal"))
        lines.append(row("commit_ok proxy_on",      "commit_ok_on"))
    lines.append("-" * 74)
    ti_g = f"{g['type_i_off']}/{g['n_rows_off']} off, {g['type_i_on']}/{g['n_rows_on']} on"
    if h["n_rows_on"] > 0:
        ti_h = f"{h['type_i_off']}/{h['n_rows_off']} off, {h['type_i_on']}/{h['n_rows_on']} on"
    else:
        ti_h = f"{h['type_i_off']}/{h['n_rows_off']} off (on not run)"
    lines.append(f"{'Type-I corruptions':<32s} {ti_g:>14s} {ti_h:>14s}")
    lines.append("")
    lines.append("Replication verdict:")
    if block["dl_accumulation_replicates_within_0.05"]:
        lines.append(f"  ✓ DL-accumulation replicates within ±0.05 "
                     f"(|Δ|={abs(delta.get('dl_accumulation', 0)):.4f})")
    else:
        lines.append(f"  ✗ DL-accumulation drifts beyond ±0.05 "
                     f"(|Δ|={abs(delta.get('dl_accumulation', 0)):.4f}) — REPORT HONESTLY")
    if block["type_i_parity_preserved"]:
        lines.append(f"  ✓ Type-I = 0 preserved on {vendor_display}")
    else:
        lines.append(f"  ✗ Type-I non-zero on {vendor_display} "
                     f"(off={h['type_i_off']}, on={h['type_i_on']})")
    if block["commit_throughput_collapse_replicated"] is None:
        lines.append(f"  — commit-throughput-collapse: proxy_on not run on {vendor_display}")
    elif block["commit_throughput_collapse_replicated"]:
        lines.append(f"  ✓ Commit throughput collapse replicated "
                     f"({h['commit_ok_off']:.1%} → {h['commit_ok_on']:.1%} under proxy_on)")
    else:
        lines.append(f"  ✗ Commit throughput collapse NOT replicated "
                     f"({h['commit_ok_off']:.1%} → {h['commit_ok_on']:.1%})")
    lines.append("─" * 74)

    return block, "\n".join(lines)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor",            default="anthropic", choices=["anthropic", "gemini"],
                    help="LLM vendor: anthropic (Claude) or gemini (Google). "
                         "Default: anthropic.")
    ap.add_argument("--backbone",          default=None,
                    help="Model ID. Default depends on --vendor: "
                         "claude-haiku-4-5-20251001 for anthropic, "
                         "gemini-2.5-flash for gemini. "
                         "BACKBONE env var also accepted (deprecated).")
    ap.add_argument("--domains",           default="django_queryset,django_admin,django_migration,astropy_fits,astropy_wcs,astropy_units,sympy_solver,sympy_matrix,requests_session,scikit_estimator",
                    help="Comma-separated domain list. Default: all 10.")
    ap.add_argument("--tasks-per-domain",  type=int, default=2)
    ap.add_argument("--runs-per-task",     type=int, default=2)
    ap.add_argument("--steps",             type=int, default=6)
    ap.add_argument("--conditions",        default="proxy_off",
                    help="Comma-separated: proxy_off, proxy_on. Default: "
                         "proxy_off only.")
    ap.add_argument("--output",            default=None,
                    help="Default: results/proxy_ph2_<vendor>.csv")
    ap.add_argument("--summary",           default=None,
                    help="Default: results/proxy_ph2_<vendor>_summary.json")
    ap.add_argument("--gpt-summary",       default=os.environ.get("GPT_SUMMARY", ""),
                    help="Path to GPT-4o-mini full-sweep summary JSON for "
                         "cross-backbone comparison.")
    args = ap.parse_args()

    vendor = args.vendor
    if args.backbone:
        backbone = args.backbone
    elif os.environ.get("BACKBONE"):
        backbone = os.environ["BACKBONE"]
    elif vendor == "anthropic":
        backbone = "claude-haiku-4-5-20251001"
    elif vendor == "gemini":
        backbone = "gemini-2.5-flash"
    else:
        backbone = BACKBONE

    if args.output is None:
        args.output = f"results/proxy_ph2_{vendor}.csv"
    if args.summary is None:
        args.summary = f"results/proxy_ph2_{vendor}_summary.json"

    if not health_check():
        print(f"ERROR: S-Bus unreachable or admin off at {SBUS_URL}"); sys.exit(1)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]


    if "proxy_on" in conditions:
        if not proxy_health_check():
            print(f"ERROR: proxy_on requested but sbus-proxy unreachable at {PROXY_URL}")
            sys.exit(1)
        print("")
        print(f"NOTE: proxy_on with --vendor={vendor} requires proxy_patch_v3+")
        print("      (multi-upstream routing). If your proxy is older, requests")
        print("      will be misrouted and fail with 401.")
        print("")

    dkeys = [d.strip() for d in args.domains.split(",")]
    domains = {k: TASK_DOMAINS[k] for k in dkeys if k in TASK_DOMAINS}
    if not domains:
        print(f"ERROR: no valid domains in {args.domains!r}"); sys.exit(1)

    client_direct = make_vendor_client(vendor, backbone, use_proxy=False)
    client_proxy  = make_vendor_client(vendor, backbone, use_proxy=True)
    def client_for(cond):
        return (client_proxy, True) if cond == "proxy_on" else (client_direct, False)

    total_tasks = sum(min(args.tasks_per_domain, len(d["tasks"])) for d in domains.values())
    total_rows  = total_tasks * args.runs_per_task * len(conditions) * N_AGENTS * args.steps
    global_row_counter = [0]
    print("=" * 70)
    print(f"Exp. PROXY-PH2 multi-backbone replication (vendor={vendor})")
    print("=" * 70)
    print(f"Backbone: {backbone}   Output format: tool_use/function_call (forced)")
    print(f"Domains ({len(domains)}): {list(domains.keys())}")
    print(f"Tasks/dom: {args.tasks_per_domain}, runs/task: {args.runs_per_task}")
    print(f"conditions: {conditions}, N={N_AGENTS}, steps={args.steps}")
    print(f"Target rows: {total_rows}")
    print(f"SBUS_URL={SBUS_URL}")
    if "proxy_on" in conditions:
        print(f"PROXY_URL={PROXY_URL}")
    if args.gpt_summary:
        print(f"GPT comparison: {args.gpt_summary}")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fout = open(args.output, "w", newline="")
    w = csv.DictWriter(fout, fieldnames=[f.name for f in StepRow.__dataclass_fields__.values()])
    w.writeheader()

    t0 = time.time()
    all_rows = []
    r_self_empty_count = 0
    print(f"[{time.strftime('%H:%M:%S')}] Starting sweep. First LLM call "
          f"will fire within ~60s; progress heartbeats every 10 rows.",
          flush=True)
    for dk, d in domains.items():
        tasks = d["tasks"][:args.tasks_per_domain]
        for task in tasks:
            for r in range(args.runs_per_task):
                run_id = uuid.uuid4().hex[:8]
                for cond in conditions:
                    print(f"[{time.strftime('%H:%M:%S')}] "
                          f"[{dk}/{task['id']}/r{r}/{cond}] run_id={run_id}",
                          flush=True)
                    client, use_proxy = client_for(cond)
                    for i in range(N_AGENTS):
                        http_post(f"{SBUS_URL}/session", {
                            "agent_id": f"agent_{i}_{run_id}_{cond}_{vendor}"
                        })
                    rows = await run_trial(
                        client=client, use_proxy=use_proxy,
                        domain_key=dk, domain=d, task=task,
                        n_steps=args.steps, condition=cond, run_id=run_id,
                    )
                    for row in rows:
                        w.writerow(asdict(row))
                        if not row.r_self:
                            r_self_empty_count += 1
                        global_row_counter[0] += 1
                        if global_row_counter[0] % 10 == 0:
                            pct = 100.0 * global_row_counter[0] / total_rows
                            elapsed_so_far = time.time() - t0
                            rate = global_row_counter[0] / elapsed_so_far if elapsed_so_far > 0 else 0
                            remaining = (total_rows - global_row_counter[0]) / rate if rate > 0 else 0
                            print(f"  progress: {global_row_counter[0]}/{total_rows} "
                                  f"({pct:.1f}%)  "
                                  f"elapsed={elapsed_so_far/60:.1f}min  "
                                  f"rate={rate*60:.1f}row/min  "
                                  f"ETA={remaining/60:.1f}min",
                                  flush=True)
                    fout.flush()
                    sys.stdout.flush()
                    all_rows.extend(rows)
                    if rows:
                        empty_rs = sum(1 for x in rows if not x.r_self)
                        print(f"   rows: {len(rows)}  empty_r_self: {empty_rs}  "
                              f"f_obs_http={statistics.mean(x.f_obs_http for x in rows):.3f}  "
                              f"f_obs_total={statistics.mean(x.f_obs_total for x in rows):.3f}  "
                              f"type_i={sum(x.type_i_corr for x in rows)}")

    fout.close()
    elapsed = time.time() - t0
    print(f"\nCollected {len(all_rows)} rows in {elapsed:.1f}s")
    print(f"Rows with empty r_self: {r_self_empty_count}/{len(all_rows)} "
          f"({100*r_self_empty_count/max(1,len(all_rows)):.1f}%)")

    if r_self_empty_count > len(all_rows) * 0.10:
        print("")
        print("⚠ WARNING: >10% of rows have empty r_self.")
        print(f"  Check the {vendor} SDK version, the model ID ({backbone}),")
        print("  and that tool_use / function_call blocks are arriving.")
        print("  Inspect a single response manually to verify response shape.")
        print("")

    by_cond = {c: [] for c in conditions}
    for row in all_rows:
        by_cond.setdefault(row.condition, []).append(row)

    summary = {
        "conditions": {},
        "meta": {
            "sbus_url":             SBUS_URL,
            "proxy_url":            PROXY_URL,
            "vendor":               vendor,
            "backbone":             backbone,
            "output_mechanism":     "tool_use_forced",
            "n_agents":             N_AGENTS,
            "steps":                args.steps,
            "tasks_per_domain":     args.tasks_per_domain,
            "runs_per_task":        args.runs_per_task,
            "shards_per_domain":    4,
            "max_shards_per_step":  2,
            "n_rows_total":         len(all_rows),
            "n_rows_empty_r_self":  r_self_empty_count,
            "elapsed_s":            round(elapsed, 1),
            "conditions_requested": conditions,
        },
    }

    for cond, rows in by_cond.items():
        if not rows: continue
        n = len(rows)
        type_i = sum(r.type_i_corr for r in rows)
        ok     = sum(1 for r in rows if r.commit_status == "ok")
        css    = sum(1 for r in rows if r.commit_status == "cross_shard_stale")
        vm     = sum(1 for r in rows if r.commit_status == "version_mismatch")
        f_http  = [r.f_obs_http  for r in rows]
        f_total = [r.f_obs_total for r in rows]
        uplift  = [t - h for h, t in zip(f_http, f_total)]
        summary["conditions"][cond] = {
            "n_rows":                 n,
            "mean_f_obs_http":        round(statistics.mean(f_http),  4),
            "mean_f_obs_total":       round(statistics.mean(f_total), 4),
            "mean_within_row_uplift": round(statistics.mean(uplift),  4),
            "type_i_corruptions":     type_i,
            "type_i_rate":            round(type_i / n, 6),
            "type_i_upper_95":        round(rule_of_three_upper(n), 6) if type_i == 0 else None,
            "commit_ok":              ok,
            "commit_ok_rate":         round(ok / n, 4),
            "cross_shard_stale":      css,
            "version_mismatch":       vm,
        }

    if "proxy_off" in by_cond and by_cond["proxy_off"]:
        off = by_cond["proxy_off"]
        summary["coverage_decomposition"] = {
            "http_this_step":            round(statistics.mean(r.f_obs_http  for r in off), 4),
            "dl_accumulation_under_off": round(
                statistics.mean(r.f_obs_total - r.f_obs_http for r in off), 4),
            "total_coverage_off":        round(statistics.mean(r.f_obs_total for r in off), 4),
        }
        # Add proxy_marginal only if we have both conditions.
        if "proxy_on" in by_cond and by_cond["proxy_on"]:
            on = by_cond["proxy_on"]
            key = lambda r: (r.domain, r.task_id, r.run_id, r.agent_idx, r.step)
            off_map = {key(r): r.f_obs_total for r in off}
            pairs = [
                (off_map[key(r)], r.f_obs_total)
                for r in on
                if key(r) in off_map
            ]
            mean_d, lo, hi = paired_bootstrap_ci(pairs)
            summary["paired_uplift_proxy_on_minus_off"] = {
                "n_pairs":   len(pairs),
                "mean_diff": round(mean_d, 4),
                "bs_95_lo":  round(lo, 4),
                "bs_95_hi":  round(hi, 4),
            }
            summary["coverage_decomposition"]["proxy_marginal_paired"] = round(mean_d, 4)
            summary["coverage_decomposition"]["total_coverage_on"]     = round(
                statistics.mean(r.f_obs_total for r in on), 4)

    gpt = load_gpt_summary(args.gpt_summary)
    comparison_table = ""
    if gpt is not None:
        vendor_label_map = {
            "anthropic": ("haiku_4_5",        "Haiku-4.5"),
            "gemini":    ("gemini_2_5_flash", "Gemini-2.5-Flash"),
        }
        vlabel, vdisplay = vendor_label_map.get(vendor, (vendor, vendor.title()))
        block, comparison_table = comparison_block(
            summary, gpt, vendor_label=vlabel, vendor_display=vdisplay,
        )
        summary["cross_backbone_comparison"] = block

    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {vendor.upper()} SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if comparison_table:
        print(comparison_table)
    print(f"\nCSV     : {args.output}")
    print(f"Summary : {args.summary}")


if __name__ == "__main__":
    asyncio.run(main())