#!/usr/bin/env python3
"""
Exp. PH-3: p_hidden Measurement with Self-Reported Ground Truth
================================================================
Extends PH-2 (keyword-proxy p_hidden, no ground truth) by requesting the
agent to declare which shards it actually used via structured-output JSON.
This converts F3 from a keyword-proxy claim into a measurable coverage-
accuracy tradeoff: precision and recall of the keyword-scan mechanism
(used by phidden_wrapper) against agent self-report.

PH-3 OPTION B (keyword windowing, assistant-only)
--------------------------------------------------
The keyword-scan extraction used to walk the entire conversation history
and include user prompts. That caused two pathologies:

  1. Full-history saturation: once a shard name appears in any prior step's
     completion it stays in history forever, so after step ~3 the scan hits
     every shard every step regardless of what the agent is actually using.
     Recall climbs to 0.9+, precision collapses, both numbers stop reflecting
     per-step extraction quality.

  2. User-prompt pollution: the user prompt at each step contains a
     "fresh content block" listing all shards by name. Scanning user
     messages therefore always finds every shard, drowning the assistant's
     actual usage signal.

Option B fixes both:

  - Scan only the LAST HIDDEN_SCAN_WINDOW messages (default 2).
  - Within the window, consider only ASSISTANT turns (the agent's own
    recent outputs). User prompts are always skipped.

This captures the realistic operating target of phidden_wrapper, which
inspects the agent's output stream for shard references, not the full
transcript.

PH-3 produces, per step:
  - r_obs_set          : shards HTTP-GET'd (current workload: all of them)
  - r_hidden_set       : shards mentioned in WINDOWED conversation history
  - r_gt_set           : shards the agent SELF-REPORTS using (new, ground truth)
  - r_gt_subseteq_obs  : trivially yes for this workload
  - keyword_precision  : |r_gt_set ∩ r_hidden_set| / |r_hidden_set|
  - keyword_recall     : |r_gt_set ∩ r_hidden_set| / |r_gt_set|

These are the numbers the paper needs to state precision/recall of the
phidden_wrapper's keyword-scan extraction mechanism — closing F3 from
"74% structural ceiling" to "quantified tradeoff curve with error bounds".

N_AGENTS is 1 (not 4) because we measure per-agent self-report fidelity;
cross-agent state effects would confound the per-step causal-read interp.

METHODOLOGY
-----------
Per step, the agent is prompted to respond in strict JSON:
  {
    "change":       "<one concrete technical change, specific>",
    "shards_used":  ["models_state", "orm_query"]   # list of shard base names
  }

`shards_used` is the GROUND TRUTH for which shards influenced the agent's
output. It is compared against:
  - r_obs       : HTTP GET /shard/:key records (from the DeliveryLog)
  - r_hidden    : keyword-scan matches in conversation history (the proxy
                  mechanism that phidden_wrapper mimics at completion level)

Known limitations (documented, measured via Build 3b human annotation):
  - Self-report may suffer from observer effect (agent reasons differently
    when asked to self-report).
  - LLM may confabulate (claim to use shards it did not).
  - Format compliance isn't perfect; `parse_ok=False` rows are excluded
    from precision/recall stats but retained for rate counting.

Backward compat: the r_obs_count / r_hidden_count / p_hidden fields from
PH-2 are all preserved, so existing analysis scripts keep working.

TASK DOMAINS (10)
-----------------
  1. django_queryset    — queryset ordering, ORM fields
  2. django_admin       — admin action permissions
  3. django_migration   — migration squasher, circular deps
  4. astropy_fits       — FITS I/O, header manipulation
  5. astropy_wcs        — WCS coordinate transforms
  6. astropy_units      — unit conversion, quantity arithmetic
  7. sympy_solver       — algebraic solver, assumptions
  8. sympy_matrix       — matrix operations, eigenvalues
  9. requests_session   — HTTP session management, auth
  10. scikit_estimator  — sklearn estimator API, fit/transform

USAGE
-----
  export OPENAI_API_KEY=sk-...
  python3 measure_phidden_v2.py \\
      --domains all \\
      --runs-per-domain 5 \\
      --steps 20 \\
      --output results/phidden_v2.csv

  # Quick test (3 domains, 3 runs each, 10 steps → 360 step-logs)
  python3 measure_phidden_v2.py --domains django --runs-per-domain 3 --steps 10
"""

import asyncio
import argparse
import csv
import json
import os
import re
import socket
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, ProxyHandler, build_opener

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: pip install openai"); sys.exit(1)

# PH-3 Build 3c cross-family ablation: optional Anthropic client.
# We lazily import so users without the anthropic package keep working
# as long as they stick to OpenAI models.
_anthropic_available = False
try:
    from anthropic import AsyncAnthropic
    _anthropic_available = True
except ImportError:
    pass

SBUS_URL  = os.getenv("SBUS_URL", "http://localhost:7000")
BACKBONE  = "gpt-4o-mini"
# PH-3 Build 3c: semantic extraction analyst model. Deliberately stronger
# than BACKBONE — mini-model self-reports are the ground truth, but the
# extraction analyst has to reason about causal shard-usage from just the
# completion text. gpt-4o is the cheapest model with enough semantic
# reasoning to do this reliably. ~630 calls at ~$0.004 each ≈ $2.50.
# Override with SEMANTIC_MODEL env var for ablation:
#  - "gpt-4o"           OpenAI (default)
#  - "gpt-4o-mini"      OpenAI, cheap
#  - "claude-sonnet-4-6"      Anthropic (current stable, released Feb 2026)
#  - "claude-sonnet-4-5"      Anthropic (prior stable, Sept 2025)
#  - Any OpenAI chat model, or any Anthropic claude-* model.
# Cross-family testing (OpenAI worker + Anthropic analyst) is especially
# useful for disentangling capability from same-model-family alignment.
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL", "gpt-4o")
# If set to 0, skip semantic extraction entirely (recovers old PH-3 behaviour).
SEMANTIC_ENABLED = int(os.getenv("SEMANTIC_ENABLED", "1"))
N_AGENTS  = 1   # PH-3: isolate per-agent self-report (was 4 in PH-2)

# ── WORKLOAD selector (Option B, v49) ─────────────────────────────────────────
# Controls which workload regime the agent runs under. This is THE knob that
# determines whether the experiment measures coverage on the PH-3 original
# "all-GET" regime (p_hidden ~0.08) or the PH-2 v46 "GET-target-only" regime
# (p_hidden ~0.74). The semantic-extraction mechanism's recall in the two
# regimes is what Option B needs to measure.
#
# Values:
#   "ph3_all_get"        (default, original PH-3 v3 behaviour):
#                        Agent HTTP-GETs ALL shards every step before the LLM
#                        call. HTTP recall is 1.0 by construction. p_hidden
#                        stays low (~0.08) because r_obs is maximal.
#   "ph2_target_only"    (Option B, high-p_hidden transfer test):
#                        Agent HTTP-GETs ONLY the single shard it is about to
#                        commit this step. Other shards are referenced from
#                        context/memory (R_hidden). This reproduces PH-2's
#                        v46 workload, where p_hidden was 0.739. Used to test
#                        whether semantic extraction recall holds under a
#                        higher-p_hidden regime.
#
# Both regimes share the same task domains, same backbone, same self-report
# format, same semantic analyst — only the pre-LLM GET strategy differs.
WORKLOAD = os.getenv("WORKLOAD", "ph3_all_get")
if WORKLOAD not in ("ph3_all_get", "ph2_target_only"):
    raise ValueError(
        f"WORKLOAD={WORKLOAD!r} not recognised; expected 'ph3_all_get' "
        f"or 'ph2_target_only'"
    )

_opener = build_opener(ProxyHandler({}))


# ── Task domain definitions ───────────────────────────────────────────────────

TASK_DOMAINS = {
    "django_queryset": {
        "name":  "django_queryset",
        "desc":  "Fix Django queryset ordering with related model fields. "
                 "The bug causes incorrect ordering when using select_related() "
                 "with field traversal across foreign key relationships.",
        "shards": ["models_state", "orm_query", "test_fixtures", "migration_plan"],
        "tasks": [
            {"id": "django-11019", "goal": "Fix queryset ordering with select_related fields"},
            {"id": "django-11820", "goal": "Fix ordering on annotated querysets"},
            {"id": "django-14155", "goal": "Fix reverse FK ordering in admin views"},
        ],
    },
    "django_admin": {
        "name":  "django_admin",
        "desc":  "Fix Django admin action permission checking. "
                 "The check does not correctly validate per-object permissions "
                 "for custom admin actions.",
        "shards": ["admin_config", "permission_check", "action_registry", "view_logic"],
        "tasks": [
            {"id": "django-12286", "goal": "Fix admin action permission check"},
            {"id": "django-12308", "goal": "Fix has_change_permission for inline admin"},
        ],
    },
    "django_migration": {
        "name":  "django_migration",
        "desc":  "Fix Django migration squasher for circular dependencies. "
                 "The squasher incorrectly handles migrations that form cycles "
                 "through RunSQL operations.",
        "shards": ["migration_graph", "squash_plan", "dependency_resolver", "rollback_state"],
        "tasks": [
            {"id": "django-13230", "goal": "Fix migration squasher circular dep handling"},
            {"id": "django-14238", "goal": "Fix squash_migrations with replaces"},
        ],
    },
    "astropy_fits": {
        "name":  "astropy_fits",
        "desc":  "Fix Astropy FITS I/O for non-standard header values. "
                 "The reader fails on HIERARCH keywords exceeding 8 characters "
                 "in some FITS dialects.",
        "shards": ["header_parser", "io_handler", "keyword_registry", "card_formatter"],
        "tasks": [
            {"id": "astropy-7671", "goal": "Fix HIERARCH keyword parsing in FITS header"},
            {"id": "astropy-8765", "goal": "Fix FITS header continuation card reading"},
        ],
    },
    "astropy_wcs": {
        "name":  "astropy_wcs",
        "desc":  "Fix Astropy WCS coordinate transform for edge-case projections. "
                 "The zenithal equal-area projection (ZEA) fails near the poles "
                 "due to missing boundary checks.",
        "shards": ["wcs_transform", "projection_math", "coordinate_frame", "bounds_check"],
        "tasks": [
            {"id": "astropy-9999", "goal": "Fix ZEA projection near poles"},
            {"id": "astropy-8888", "goal": "Fix SIN projection wrapping for full-sky"},
        ],
    },
    "astropy_units": {
        "name":  "astropy_units",
        "desc":  "Fix Astropy unit conversion for composite quantities. "
                 "Converting between equivalent units fails when intermediate "
                 "representation uses non-SI base units.",
        "shards": ["unit_registry", "conversion_graph", "quantity_repr", "equivalency_map"],
        "tasks": [
            {"id": "astropy-7777", "goal": "Fix composite unit conversion via non-SI intermediates"},
            {"id": "astropy-6543", "goal": "Fix unit equivalency for photon flux"},
        ],
    },
    "sympy_solver": {
        "name":  "sympy_solver",
        "desc":  "Fix SymPy algebraic solver for systems with assumptions. "
                 "The solver drops solutions when assumptions like 'positive' "
                 "are combined with complex-valued intermediate steps.",
        "shards": ["solve_engine", "assumption_system", "simplification", "solution_filter"],
        "tasks": [
            {"id": "sympy-21345", "goal": "Fix solve() dropping solutions with positive assumption"},
            {"id": "sympy-19876", "goal": "Fix solveset for transcendental equations with assumptions"},
        ],
    },
    "sympy_matrix": {
        "name":  "sympy_matrix",
        "desc":  "Fix SymPy matrix eigenvalue computation for sparse symbolic matrices. "
                 "The eigenvals() method raises an exception on sparse matrices "
                 "with symbolic entries involving Rational coefficients.",
        "shards": ["matrix_engine", "eigenvalue_solver", "sparse_repr", "rational_arithmetic"],
        "tasks": [
            {"id": "sympy-22000", "goal": "Fix eigenvals() for sparse symbolic matrices"},
            {"id": "sympy-21100", "goal": "Fix det() for large symbolic matrices with rationals"},
        ],
    },
    "requests_session": {
        "name":  "requests_session",
        "desc":  "Fix requests library session auth handling for redirects. "
                 "Authorization headers are incorrectly stripped on cross-domain "
                 "redirects even when allow_redirects=True.",
        "shards": ["session_state", "auth_handler", "redirect_logic", "header_policy"],
        "tasks": [
            {"id": "requests-5430", "goal": "Fix auth header stripping on cross-domain redirect"},
            {"id": "requests-5200", "goal": "Fix session cookie handling with HTTPS redirects"},
        ],
    },
    "scikit_estimator": {
        "name":  "scikit_estimator",
        "desc":  "Fix scikit-learn estimator clone() for custom parameters. "
                 "The clone() function fails for estimators with non-standard "
                 "__init__ signatures that use *args or **kwargs.",
        "shards": ["estimator_api", "clone_logic", "param_inspector", "validation_utils"],
        "tasks": [
            {"id": "sklearn-25001", "goal": "Fix clone() for estimators with **kwargs in __init__"},
            {"id": "sklearn-24000", "goal": "Fix set_params() for nested pipeline estimators"},
        ],
    },
}

# Mapping from --domains CLI arg to domain keys
DOMAIN_GROUPS = {
    "all":     list(TASK_DOMAINS.keys()),
    "django":  ["django_queryset", "django_admin", "django_migration"],
    "astropy": ["astropy_fits", "astropy_wcs", "astropy_units"],
    "sympy":   ["sympy_solver", "sympy_matrix"],
    "other":   ["requests_session", "scikit_estimator"],
}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def http_get(url: str, params: dict = None) -> tuple[int, dict]:
    if params:
        url += "?" + urlencode(params)
    try:
        with _opener.open(url, timeout=15) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def http_post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def health_check(url: str = SBUS_URL) -> bool:
    try:
        s = socket.create_connection(("localhost", 7000), timeout=3)
        s.close()
    except Exception:
        return False
    status, _ = http_get(f"{url}/stats")
    return status == 200


def reset_bus() -> None:
    http_post(f"{SBUS_URL}/admin/reset", {})
    time.sleep(0.2)


# ── p_hidden computation ──────────────────────────────────────────────────────

def build_keyword_pattern(shard_keys: list[str]) -> re.Pattern:
    """Build regex to detect shard key mentions in conversation history."""
    escaped = [re.escape(k) for k in shard_keys]
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)


# PH-3 Option B (corrected): the keyword scan now (a) uses a rolling window
# and (b) scans ONLY ASSISTANT messages within that window. User messages
# always contain the fresh-content block (which lists every shard by name),
# so scanning them pollutes the hidden-read signal. Scanning only the agent's
# own recent outputs captures the realistic operating target of
# phidden_wrapper: "what shards did the agent itself reference in its
# recent completions?"
#
# History has interleaved [user, assistant, user, assistant, ...]. Default
# window of 2 captures the last one user + one assistant pair; the assistant
# filter means we effectively examine the last assistant message.
#
# The `current_fresh_content` exclusion parameter is kept for backward
# compatibility but is now redundant: user messages are already skipped by
# role. Passing None or "" disables the substring check.
HIDDEN_SCAN_WINDOW = 2


def count_hidden_reads(
    history: list[dict],
    shard_keys: list[str],
    current_fresh_content: str,
    window: int = HIDDEN_SCAN_WINDOW,
) -> int:
    """
    Count shard-key references in the last `window` ASSISTANT messages.
    User messages are always skipped (they contain the fresh-content
    block by construction). See HIDDEN_SCAN_WINDOW comment above.
    """
    pattern = build_keyword_pattern(shard_keys)
    windowed = history[-window:] if window > 0 else history
    count = 0
    for msg in windowed:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            count += len(pattern.findall(content))
    return count


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class StepLog:
    run_id:           str
    domain:           str
    task_id:          str
    agent_id:         str
    step:             int
    n_agents:         int

    # Read counts for this step
    r_obs_count:      int    # HTTP GET calls (observed by DeliveryLog)
    r_hidden_count:   int    # Keyword matches in conversation history
    r_total:          int    # r_obs + r_hidden

    # p_hidden for this step
    p_hidden:         float  # r_hidden / (r_obs + r_hidden)

    # Commit outcome
    commit_status:    str    # "ok" | "conflict_409" | "error_N"

    # Context growth (tracks accumulation effect)
    history_length:   int    # number of messages in history at this step
    history_tokens_est: int  # rough token estimate (chars / 4)

    # PH-3: ground-truth self-report from the agent (JSON string in CSV).
    # These are base shard names, NOT run-id-suffixed keys. Parsing layer
    # drops any names not in the domain's shard vocabulary.
    shards_used_gt_json:  str     # JSON list of shard base names, e.g. '["orm_query"]'
    n_shards_used_gt:     int     # len of that list, 0 if parse failed
    parse_ok:             bool    # did the LLM response parse into valid JSON
    r_obs_base_json:      str     # JSON list of shard base names that were HTTP-GET'd
    r_hidden_base_json:   str     # JSON list of shard base names found via keyword scan

    # PH-3 Build 3c: SEMANTIC extraction via a second LLM call (gpt-4o)
    # analysing the completion text. The analyst sees completion + vocabulary +
    # fresh content block (NOT the self-report) — same information a real
    # production extraction mechanism would have.
    r_semantic_base_json: str     # JSON list of shard base names inferred semantically
    semantic_parse_ok:    bool    # did the analyst's JSON parse cleanly

    # Build 3b / Phase B: raw text needed for human annotation. These fields
    # were added for the annotation harness — they carry the agent's actual
    # delta text and what the agent was shown at this step, so downstream
    # annotators can judge causal shard-use without re-running the experiment.
    change_text:          str     # the "change" field the agent produced
    fresh_content_block:  str     # what the agent was shown this step

    # Option B (v49): which workload regime this step was run under.
    # Lets downstream analysis separate ph3_all_get runs (low p_hidden) from
    # ph2_target_only runs (high p_hidden) when mixing CSVs.
    workload:             str     # "ph3_all_get" | "ph2_target_only"


# ── Agent runner ──────────────────────────────────────────────────────────────

def _keyword_matches_in_history(
    history: list[dict],
    shard_keys: list[str],
    current_fresh_content: str,
    window: int = HIDDEN_SCAN_WINDOW,
) -> list[str]:
    """
    Like count_hidden_reads but returns the SET of shard base names matched.
    Scans only assistant messages within the last `window` entries. User
    messages are skipped (they always contain the fresh-content block).
    """
    pattern = build_keyword_pattern(shard_keys)
    windowed = history[-window:] if window > 0 else history
    hit = set()
    for msg in windowed:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            for m in pattern.findall(content):
                hit.add(m.lower())
    keyed = {k.lower(): k for k in shard_keys}
    return sorted({keyed[h] for h in hit if h in keyed})


def _parse_agent_response(
    content: str,
    valid_base_names: list[str],
) -> tuple[str, list[str], bool]:
    """
    Parse the agent's JSON-formatted response.

    Returns (delta_text, shards_used_validated, parse_ok).

    valid_base_names is the domain's allowed shard vocabulary; any
    self-reported name not in this set is silently dropped (the LLM
    may confabulate shard names outside the vocabulary).

    On any failure, returns ("", [], False) — the caller treats this
    as a malformed step and excludes it from precision/recall stats,
    but still records it in the CSV for parse-rate accounting.
    """
    if not content:
        return "", [], False
    try:
        data = json.loads(content)
    except Exception:
        return "", [], False
    if not isinstance(data, dict):
        return "", [], False

    change = data.get("change", "")
    if not isinstance(change, str):
        return "", [], False

    raw = data.get("shards_used", [])
    if not isinstance(raw, list):
        return change.strip(), [], False  # delta recovered, shards not
    valid = set(valid_base_names)
    validated = [s for s in raw if isinstance(s, str) and s in valid]
    # De-duplicate while preserving order
    seen = set()
    dedup = []
    for s in validated:
        if s not in seen:
            seen.add(s); dedup.append(s)
    return change.strip(), dedup, True


# ── PH-3 Build 3c: Semantic extraction via analyst LLM ────────────────────────
#
# This is the Option 1 mechanism we committed to. Given ONLY what a
# production phidden_wrapper would see (completion text + shard vocabulary
# + fresh content block), ask a stronger LLM to reason about which shards
# from the vocabulary were actually used. The analyst NEVER sees the
# ground-truth self-report, so the comparison is fair.
#
# If this achieves significantly higher recall than keyword scan (~7%
# overall in our last measurement), F3 gets a genuine mitigation story.
# If it doesn't, F3 is confirmed as an open research problem requiring
# architectural rather than extraction-based solutions.

async def extract_shards_semantic(
    oai: AsyncOpenAI,
    completion_text: str,
    shard_vocab: list[str],
    fresh_content_block: str,
    domain_desc: str,
    anthropic_client=None,
) -> tuple[list[str], bool]:
    """
    Ask an analyst LLM to identify which shards from `shard_vocab` were
    causally used to produce `completion_text`, given what the agent saw
    in `fresh_content_block`.

    Returns (extracted_shard_list, parse_ok).

    CRITICAL METHODOLOGICAL NOTE: the analyst receives the same information
    that a deployed semantic-extraction system would have — the agent's
    output, the shard vocabulary, and the contextual state the agent was
    shown. It does NOT receive the self-report or any signal of ground
    truth. This is an honest capability measurement.

    Provider routing: if SEMANTIC_MODEL starts with "claude" the call is
    routed to Anthropic; otherwise OpenAI. This lets us run cross-family
    ablations (OpenAI worker + Anthropic analyst) to disentangle capability
    from same-family alignment.
    """
    if not SEMANTIC_ENABLED:
        return [], False
    if not completion_text or not shard_vocab:
        return [], False

    vocab_str = ", ".join(f'"{s}"' for s in shard_vocab)
    # The prompt is deliberately neutral — we do not bias the analyst toward
    # over- or under-reporting. "Causally used" is the key criterion.
    system_msg = (
        "You are a code-analysis assistant. Your job is to identify which "
        "shard keys from a known vocabulary were CAUSALLY USED by a software "
        "engineering agent when producing its output.\n\n"
        "A shard is 'causally used' if its content informed or directly "
        "shaped the agent's output — not if the agent merely mentioned it, "
        "saw it listed, or could have used it.\n\n"
        "Respond in strict JSON: {\"shards_used\": [\"<name>\", ...]}.\n"
        "List only shards from the provided vocabulary. If no shards from "
        "the vocabulary were causally used (e.g. the output is generic), "
        "return {\"shards_used\": []}."
    )
    user_msg = (
        f"Task domain: {domain_desc}\n\n"
        f"Shard vocabulary: [{vocab_str}]\n\n"
        f"What the agent was shown at this step:\n{fresh_content_block}\n\n"
        f"What the agent produced:\n{completion_text}\n\n"
        f"Which shards from the vocabulary were causally used? Respond in JSON."
    )

    # ── Provider routing ────────────────────────────────────────────────────
    raw = ""
    if SEMANTIC_MODEL.startswith("claude"):
        # Anthropic path. Messages API needs `system` as a separate arg and
        # does NOT support response_format=json_object; we hard-prompt for
        # JSON-only output. Claude follows this reliably.
        if anthropic_client is None:
            # Caller forgot to pass the Anthropic client. Fail closed so
            # these rows are excluded from precision/recall stats (they
            # count against parse_ok_rate, which is the correct signal).
            return [], False
        anthropic_user_msg = (
            user_msg
            + "\n\nIMPORTANT: Reply with ONLY the JSON object and nothing "
              "else. No preamble, no explanation, no markdown fences."
        )
        try:
            resp = await anthropic_client.messages.create(
                model=SEMANTIC_MODEL,
                system=system_msg,
                messages=[{"role": "user", "content": anthropic_user_msg}],
                max_tokens=200,
                temperature=0.0,
            )
            # Anthropic: response.content is a list of content blocks.
            blocks = getattr(resp, "content", None) or []
            for b in blocks:
                text = getattr(b, "text", None)
                if text is None and isinstance(b, dict):
                    text = b.get("text")
                if text:
                    raw = text.strip()
                    break
        except Exception:
            return [], False
    else:
        # OpenAI path (default).
        try:
            resp = await oai.chat.completions.create(
                model=SEMANTIC_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=120,
                temperature=0.0,   # deterministic analyst
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            return [], False

    # Strip a stray markdown fence if the analyst emitted one despite
    # instructions (occasionally seen from Claude models).
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[: -3].rstrip()

    try:
        data = json.loads(raw)
    except Exception:
        return [], False
    if not isinstance(data, dict):
        return [], False
    raw_list = data.get("shards_used", [])
    if not isinstance(raw_list, list):
        return [], False

    # Same vocabulary-validation as self-report: drop any confabulated names.
    valid = set(shard_vocab)
    out = []
    seen = set()
    for s in raw_list:
        if isinstance(s, str) and s in valid and s not in seen:
            seen.add(s); out.append(s)
    return out, True


async def run_agent_ph3(
    oai: AsyncOpenAI,
    agent_id: str,
    domain: dict,
    task: dict,
    run_id: str,
    n_steps: int,
    shard_pattern: re.Pattern,
    anthropic_client=None,
) -> list[StepLog]:
    """
    Run one agent for n_steps, requesting structured-output JSON with
    ground-truth self-report of shards actually used per step.
    """
    logs = []
    history = []
    shard_base_names = list(domain["shards"])  # e.g. ["models_state", "orm_query", ...]
    shared_shards = [f"{sk}_{run_id}" for sk in shard_base_names]
    # base → full key map, used below to tie self-reported base names back
    # to S-Bus shard IDs (not needed in this file, but documents the linkage)
    desc = f"{domain['desc']} Task: {task['goal']}"

    for step in range(n_steps):
        # ── R_obs: HTTP reads ─────────────────────────────────────────────────
        # Which shards the agent fetches depends on WORKLOAD:
        #   ph3_all_get     : fetch every shard every step (recall = 1.0)
        #   ph2_target_only : fetch only the shard about to be committed
        shard_data = {}
        read_set = []
        r_obs_base_set: set[str] = set()
        fresh_content_block = ""

        # Determine target shard for this step (same logic as commit below).
        target_idx = step % len(shared_shards)
        target_base = shard_base_names[target_idx]
        target_full = shared_shards[target_idx]

        if WORKLOAD == "ph3_all_get":
            # Original PH-3 v3 behaviour: HTTP-GET every shard, every step.
            shards_to_get = list(zip(shard_base_names, shared_shards))
        else:  # "ph2_target_only"
            # PH-2 v46 behaviour: HTTP-GET only the target shard. The other
            # three shards are not fetched; their content is known to the
            # LLM only via prior-step references in the conversation history.
            # This is the high-p_hidden regime.
            shards_to_get = [(target_base, target_full)]

        for base, full_key in shards_to_get:
            status, data = http_get(f"{SBUS_URL}/shard/{full_key}", {"agent_id": agent_id})
            if status == 200:
                shard_data[full_key] = data
                read_set.append({"key": full_key, "version_at_read": data.get("version", 0)})
                r_obs_base_set.add(base)

        # Build the fresh_content_block. The content varies by workload:
        #   ph3_all_get     : shows every shard's current content
        #   ph2_target_only : shows ONLY the target shard's content
        # In both cases the block is exactly what the agent sees in the prompt
        # this step. Shards not in shard_data are absent from the block —
        # the agent can reference them only via conversation history (R_hidden).
        fresh_content_block = "\n".join(
            f"  {base}: v{shard_data[full].get('version', 0)} "
            f"— {shard_data[full].get('content', '')[:80]}"
            for base, full in shards_to_get
            if full in shard_data
        )

        # ── R_hidden: count + set of keyword mentions in history ──────────────
        r_hidden_count = count_hidden_reads(history, shard_base_names, fresh_content_block)
        r_hidden_base_set = set(
            _keyword_matches_in_history(history, shard_base_names, fresh_content_block)
        )

        # ── LLM call with structured JSON output ──────────────────────────────
        shard_list_str = ", ".join(shard_base_names)
        system_msg = (
            "You are a software engineering agent. Reason from the fresh shard "
            "state provided. You MUST respond in strict JSON matching this schema:\n"
            "  {\n"
            '    "change":       "<one concrete technical change, specific, '
            '<=100 words>",\n'
            '    "shards_used":  ["<shard_base_name>", ...]\n'
            "  }\n"
            f"The shard vocabulary is exactly: [{shard_list_str}].\n"
            "In 'shards_used' list ONLY shards whose CONTENT you actually "
            "relied on to produce 'change'. Do NOT list shards you merely "
            "saw in history or were told about — only those whose values "
            "influenced your output. Be honest; this annotation is used to "
            "measure coverage of observation mechanisms."
        )
        user_msg = (
            f"Task: {desc} (Step {step+1}/{n_steps})\n"
            f"Current shard state (freshly fetched):\n{fresh_content_block}\n"
            f"Respond in JSON as specified."
        )
        messages = (
            [{"role": "system", "content": system_msg}]
            + history[-6:]
            + [{"role": "user", "content": user_msg}]
        )

        delta = ""
        shards_used: list[str] = []
        parse_ok = False
        try:
            resp = await oai.chat.completions.create(
                model=BACKBONE,
                messages=messages,
                max_tokens=250,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            raw_content = resp.choices[0].message.content or ""
            delta, shards_used, parse_ok = _parse_agent_response(
                raw_content, shard_base_names
            )
        except Exception as e:
            delta = f"[error: {e}]"
            shards_used = []
            parse_ok = False

        # ── PH-3 Build 3c: semantic extraction (analyst LLM on the delta) ─────
        # Called on the DELTA TEXT only (never the self-report or raw_content
        # which could leak the self-report). Same info a deployed
        # extraction mechanism would have.
        r_semantic_base_set: list[str] = []
        semantic_parse_ok = False
        if SEMANTIC_ENABLED and delta and not delta.startswith("[error:"):
            r_semantic_base_set, semantic_parse_ok = await extract_shards_semantic(
                oai,
                completion_text=delta,
                shard_vocab=shard_base_names,
                fresh_content_block=fresh_content_block,
                domain_desc=domain["desc"],
                anthropic_client=anthropic_client,
            )

        # Update conversation history (delta only, not the JSON wrapper —
        # keeps the "what the agent actually produced" view clean).
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant", "content": delta})

        # ── Commit ────────────────────────────────────────────────────────────
        target = shared_shards[step % len(shared_shards)]
        ev     = shard_data.get(target, {}).get("version", 0)
        status, _ = http_post(f"{SBUS_URL}/commit/v2", {
            "key":              target,
            "expected_version": ev,
            "delta":            delta if delta else f"[empty step {step}]",
            "agent_id":         agent_id,
            "read_set":         read_set,
        })

        # ── p_hidden for this step ────────────────────────────────────────────
        r_obs_count = len(r_obs_base_set)  # count of UNIQUE shards got
        r_total  = r_obs_count + r_hidden_count
        p_hidden = r_hidden_count / r_total if r_total > 0 else 0.0
        hist_chars = sum(len(m.get("content", "")) for m in history)

        logs.append(StepLog(
            run_id=run_id,
            domain=domain["name"],
            task_id=task["id"],
            agent_id=agent_id,
            step=step,
            n_agents=N_AGENTS,
            r_obs_count=r_obs_count,
            r_hidden_count=r_hidden_count,
            r_total=r_total,
            p_hidden=round(p_hidden, 4),
            commit_status="ok" if status == 200 else f"conflict_{status}",
            history_length=len(history),
            history_tokens_est=hist_chars // 4,
            shards_used_gt_json=json.dumps(shards_used),
            n_shards_used_gt=len(shards_used),
            parse_ok=parse_ok,
            r_obs_base_json=json.dumps(sorted(r_obs_base_set)),
            r_hidden_base_json=json.dumps(sorted(r_hidden_base_set)),
            r_semantic_base_json=json.dumps(sorted(r_semantic_base_set)),
            semantic_parse_ok=semantic_parse_ok,
            change_text=delta,
            fresh_content_block=fresh_content_block,
            workload=WORKLOAD,
        ))

    return logs


async def run_one_task(
    oai: AsyncOpenAI,
    domain: dict,
    task: dict,
    n_steps: int,
    anthropic_client=None,
) -> list[StepLog]:
    """Run N_AGENTS in parallel on one task and collect step logs."""
    run_id = uuid.uuid4().hex[:8]

    # Create shards for this run
    for sk in domain["shards"]:
        http_post(f"{SBUS_URL}/shard", {
            "key":      f"{sk}_{run_id}",
            "content":  f"Initial: {domain['desc'][:80]}",
            "goal_tag": f"ph2_{task['id']}",
        })

    shard_pattern = build_keyword_pattern(domain["shards"])

    # Run agents in parallel
    results = await asyncio.gather(*[
        run_agent_ph3(
            oai,
            agent_id=f"agent_{i}_{run_id}",
            domain=domain,
            task=task,
            run_id=run_id,
            n_steps=n_steps,
            shard_pattern=shard_pattern,
            anthropic_client=anthropic_client,
        )
        for i in range(N_AGENTS)
    ], return_exceptions=True)

    all_logs = []
    for r in results:
        if isinstance(r, list):
            all_logs.extend(r)
        elif isinstance(r, Exception):
            print(f"  Agent error: {r}")
    return all_logs


# ── CI computation ────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * (p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Exp. PH-2: Expanded p_hidden measurement")
    parser.add_argument("--domains",          default="all",
                        help="Domain group: all | django | astropy | sympy | other | "
                             "comma-separated domain names")
    parser.add_argument("--runs-per-domain",  type=int, default=5,
                        help="Runs per task per domain (3 tasks/domain × runs × 4 agents × steps = step-logs)")
    parser.add_argument("--steps",            type=int, default=20)
    parser.add_argument("--output",           default="results/phidden_v2.csv")
    parser.add_argument("--summary",          default="results/phidden_v2_summary.json")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)

    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}")
        print("  Start: SBUS_ADMIN_ENABLED=1 cargo run --release")
        sys.exit(1)

    oai = AsyncOpenAI(api_key=api_key)

    # ── PH-3 Build 3c: optional Anthropic analyst ────────────────────────────
    # If SEMANTIC_MODEL begins with "claude", we route the analyst call to
    # Anthropic's Messages API. Requires ANTHROPIC_API_KEY env var and the
    # `anthropic` package installed. We fail loudly if a Claude model is
    # requested but the setup is incomplete — better to stop now than to
    # silently produce 630 parse_ok=False rows.
    anthropic_client = None
    if SEMANTIC_MODEL.startswith("claude"):
        if not _anthropic_available:
            print("ERROR: SEMANTIC_MODEL is a Claude model but the `anthropic` "
                  "Python package is not installed.")
            print("  Fix: pip install anthropic")
            sys.exit(1)
        anth_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not anth_key:
            print("ERROR: SEMANTIC_MODEL is a Claude model but ANTHROPIC_API_KEY "
                  "is not set.")
            print("  Fix: export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)
        anthropic_client = AsyncAnthropic(api_key=anth_key)
        print(f"Anthropic analyst client ready (model: {SEMANTIC_MODEL})")

    # Resolve domain list
    if args.domains in DOMAIN_GROUPS:
        domain_keys = DOMAIN_GROUPS[args.domains]
    else:
        domain_keys = [d.strip() for d in args.domains.split(",")]

    domains = {k: TASK_DOMAINS[k] for k in domain_keys if k in TASK_DOMAINS}
    if not domains:
        print(f"ERROR: No valid domains in '{args.domains}'")
        print(f"Available: {list(TASK_DOMAINS.keys())}")
        sys.exit(1)

    # Estimate total step-logs
    total_tasks  = sum(len(d["tasks"]) for d in domains.values())
    total_steps  = total_tasks * args.runs_per_domain * N_AGENTS * args.steps
    print("=" * 70)
    print("Exp. PH-3: p_hidden Measurement & Extraction Mechanism Comparison")
    print("=" * 70)
    print(f"Domains ({len(domains)}): {list(domains.keys())}")
    print(f"Tasks total: {total_tasks}")
    print(f"Runs/task: {args.runs_per_domain}")
    print(f"Agents: {N_AGENTS}, Steps: {args.steps}")
    print(f"Target step-logs: {total_steps}")
    print(f"Backbone: {BACKBONE}")
    print(f"Semantic analyst: {SEMANTIC_MODEL} (enabled={SEMANTIC_ENABLED})")
    print(f"WORKLOAD: {WORKLOAD}")
    if WORKLOAD == "ph2_target_only":
        print("           → Agent HTTP-GETs ONLY the target shard each step.")
        print("           → Other shards referenced via context memory (R_hidden).")
        print("           → Expected p_hidden: ~0.6-0.8 (PH-2 regime).")
    else:
        print("           → Agent HTTP-GETs ALL shards each step.")
        print("           → HTTP recall = 1.0 by construction.")
        print("           → Expected p_hidden: ~0.05-0.15 (PH-3 regime).")
    print()

    # ── Run all domains ───────────────────────────────────────────────────────
    all_logs: list[StepLog] = []
    domain_summaries = {}

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    out_f  = open(args.output, "w", newline="")
    writer = None

    for domain_key, domain in domains.items():
        domain_logs = []
        print(f"\nDomain: {domain_key}")
        print(f"  Tasks: {[t['id'] for t in domain['tasks']]}")

        for task in domain["tasks"]:
            for run_idx in range(args.runs_per_domain):
                print(f"    [{task['id']}] run {run_idx+1}/{args.runs_per_domain} ...", end=" ", flush=True)
                reset_bus()
                t0 = time.time()
                logs = await run_one_task(oai, domain, task, args.steps,
                                          anthropic_client=anthropic_client)
                wall = time.time() - t0

                domain_logs.extend(logs)
                all_logs.extend(logs)

                # Stream to CSV
                for log in logs:
                    row = asdict(log)
                    if writer is None:
                        writer = csv.DictWriter(out_f, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                out_f.flush()

                mean_ph = statistics.mean(l.p_hidden for l in logs) if logs else 0
                print(f"n={len(logs)} mean_p_hidden={mean_ph:.4f} wall={wall:.0f}s")

        # Domain summary
        if domain_logs:
            ph_vals = [l.p_hidden for l in domain_logs]
            mean_ph = statistics.mean(ph_vals)
            # Count r_obs and r_hidden totals for Wilson CI
            total_r_obs    = sum(l.r_obs_count    for l in domain_logs)
            total_r_hidden = sum(l.r_hidden_count for l in domain_logs)
            total_r        = total_r_obs + total_r_hidden
            ci_lo, ci_hi   = wilson_ci(total_r_hidden, total_r)

            # ── PH-3 precision/recall against ground truth ───────────
            # Only usable rows: those with parse_ok=True
            usable = [l for l in domain_logs if l.parse_ok]
            n_parsed = len(usable)
            parse_rate = n_parsed / len(domain_logs) if domain_logs else 0.0

            kw_prec_vals, kw_rec_vals = [], []
            obs_rec_vals = []            # HTTP observation recall vs GT
            sem_prec_vals, sem_rec_vals = [], []   # PH-3 Build 3c: semantic
            combined_kw_sem_rec_vals = []           # HTTP + kw + semantic combined
            n_semantic_parsed = 0
            mean_gt_size = 0.0

            if usable:
                gt_sizes = [l.n_shards_used_gt for l in usable]
                mean_gt_size = sum(gt_sizes) / len(gt_sizes)
                for l in usable:
                    gt_set     = set(json.loads(l.shards_used_gt_json))
                    obs_set    = set(json.loads(l.r_obs_base_json))
                    hidden_set = set(json.loads(l.r_hidden_base_json))
                    sem_set    = set(json.loads(l.r_semantic_base_json))
                    if l.semantic_parse_ok:
                        n_semantic_parsed += 1
                    if not gt_set:
                        continue
                    inter_obs    = gt_set & obs_set
                    inter_hidden = gt_set & hidden_set
                    inter_sem    = gt_set & sem_set
                    obs_rec_vals.append(len(inter_obs) / len(gt_set))
                    kw_rec_vals.append(len(inter_hidden) / len(gt_set))
                    if hidden_set:
                        kw_prec_vals.append(len(inter_hidden) / len(hidden_set))
                    # Semantic extraction metrics (only on rows where the
                    # analyst returned valid JSON; malformed analyst outputs
                    # excluded from P/R to avoid biasing one way or the other)
                    if l.semantic_parse_ok:
                        sem_rec_vals.append(len(inter_sem) / len(gt_set))
                        if sem_set:
                            sem_prec_vals.append(len(inter_sem) / len(sem_set))
                    # Combined (HTTP observation + keyword + semantic) recall —
                    # this is what a deployed system using ALL THREE mechanisms
                    # would catch. If it's significantly higher than kw alone,
                    # semantic extraction is genuinely additive.
                    combined = obs_set | hidden_set | sem_set
                    combined_kw_sem_rec_vals.append(len(gt_set & combined) / len(gt_set))

            def _mean(xs): return round(sum(xs)/len(xs), 4) if xs else None

            domain_summaries[domain_key] = {
                "n_step_logs": len(domain_logs),
                "mean_p_hidden": round(mean_ph, 4),
                "wilson_ci_lo": round(ci_lo, 4),
                "wilson_ci_hi": round(ci_hi, 4),
                "total_r_obs":    total_r_obs,
                "total_r_hidden": total_r_hidden,
                "total_r":        total_r,
                # PH-3 ground-truth metrics
                "n_parsed":           n_parsed,
                "parse_ok_rate":      round(parse_rate, 4),
                "mean_shards_used_gt": round(mean_gt_size, 4),
                "http_obs_recall":    _mean(obs_rec_vals),
                "keyword_recall":     _mean(kw_rec_vals),
                "keyword_precision":  _mean(kw_prec_vals),
                "n_precision_samples": len(kw_prec_vals),
                "n_recall_samples":    len(kw_rec_vals),
                # PH-3 Build 3c: semantic extraction metrics
                "semantic_parse_ok_rate": round(n_semantic_parsed / max(1, n_parsed), 4),
                "semantic_recall":     _mean(sem_rec_vals),
                "semantic_precision":  _mean(sem_prec_vals),
                "n_semantic_precision_samples": len(sem_prec_vals),
                "n_semantic_recall_samples":    len(sem_rec_vals),
                "combined_http_kw_sem_recall":  _mean(combined_kw_sem_rec_vals),
            }
            print(f"  Domain summary: n={len(domain_logs)} "
                  f"p_hidden={mean_ph:.4f} | "
                  f"parse_ok={parse_rate:.2f} "
                  f"kw_rec={_mean(kw_rec_vals)} "
                  f"kw_prec={_mean(kw_prec_vals)} "
                  f"sem_rec={_mean(sem_rec_vals)} "
                  f"sem_prec={_mean(sem_prec_vals)} "
                  f"mean_gt={mean_gt_size:.2f}")

    out_f.close()

    # ── Overall summary ───────────────────────────────────────────────────────
    if not all_logs:
        print("No data collected."); sys.exit(1)

    total_r_obs    = sum(l.r_obs_count    for l in all_logs)
    total_r_hidden = sum(l.r_hidden_count for l in all_logs)
    total_r        = total_r_obs + total_r_hidden
    overall_ph     = total_r_hidden / total_r if total_r > 0 else 0.0
    ci_lo, ci_hi   = wilson_ci(total_r_hidden, total_r)

    # Step-by-step growth (shows context accumulation effect)
    step_means = {}
    for step in range(args.steps):
        vals = [l.p_hidden for l in all_logs if l.step == step]
        if vals:
            step_means[step] = statistics.mean(vals)

    # ── PH-3 overall precision/recall (aggregated across all domains) ─────────
    usable_all = [l for l in all_logs if l.parse_ok]
    n_parsed_all = len(usable_all)
    parse_rate_all = n_parsed_all / len(all_logs) if all_logs else 0.0

    kw_prec_all, kw_rec_all, obs_rec_all = [], [], []
    sem_prec_all, sem_rec_all = [], []           # PH-3 Build 3c
    n_semantic_parsed_all = 0
    mean_gt_all = 0.0
    if usable_all:
        gt_sizes = [l.n_shards_used_gt for l in usable_all]
        mean_gt_all = sum(gt_sizes) / len(gt_sizes)
        for l in usable_all:
            gt_set     = set(json.loads(l.shards_used_gt_json))
            obs_set    = set(json.loads(l.r_obs_base_json))
            hidden_set = set(json.loads(l.r_hidden_base_json))
            sem_set    = set(json.loads(l.r_semantic_base_json))
            if l.semantic_parse_ok:
                n_semantic_parsed_all += 1
            if not gt_set:
                continue
            inter_obs    = gt_set & obs_set
            inter_hidden = gt_set & hidden_set
            inter_sem    = gt_set & sem_set
            obs_rec_all.append(len(inter_obs) / len(gt_set))
            kw_rec_all.append(len(inter_hidden) / len(gt_set))
            if hidden_set:
                kw_prec_all.append(len(inter_hidden) / len(hidden_set))
            if l.semantic_parse_ok:
                sem_rec_all.append(len(inter_sem) / len(gt_set))
                if sem_set:
                    sem_prec_all.append(len(inter_sem) / len(sem_set))

    def _mean_all(xs): return round(sum(xs)/len(xs), 4) if xs else None

    # Combined-mechanism recall: HTTP + keyword-scan union against GT
    combined_rec_all = []
    combined_rec_hks_all = []    # HTTP + keyword + semantic
    if usable_all:
        for l in usable_all:
            gt_set     = set(json.loads(l.shards_used_gt_json))
            obs_set    = set(json.loads(l.r_obs_base_json))
            hidden_set = set(json.loads(l.r_hidden_base_json))
            sem_set    = set(json.loads(l.r_semantic_base_json))
            if not gt_set: continue
            combo = obs_set | hidden_set
            combined_rec_all.append(len(gt_set & combo) / len(gt_set))
            combo3 = obs_set | hidden_set | sem_set
            combined_rec_hks_all.append(len(gt_set & combo3) / len(gt_set))

    summary = {
        "experiment":         "PH-3",
        "workload":           WORKLOAD,
        "n_step_logs":        len(all_logs),
        "n_domains":          len(domains),
        "n_tasks":            total_tasks,
        "steps_per_agent":    args.steps,
        "n_agents":           N_AGENTS,
        "backbone":           BACKBONE,
        "semantic_model":     SEMANTIC_MODEL,
        "semantic_enabled":   bool(SEMANTIC_ENABLED),
        "overall_p_hidden":   round(overall_ph, 4),
        "wilson_ci_95":       [round(ci_lo, 4), round(ci_hi, 4)],
        "total_r_obs":        total_r_obs,
        "total_r_hidden":     total_r_hidden,
        "total_r":            total_r,
        "domain_summaries":   domain_summaries,
        "step_growth":        {str(k): round(v, 4) for k, v in step_means.items()},
        # PH-3 aggregates
        "ph3_n_parsed":            n_parsed_all,
        "ph3_parse_ok_rate":       round(parse_rate_all, 4),
        "ph3_mean_shards_used_gt": round(mean_gt_all, 4),
        "ph3_http_obs_recall":     _mean_all(obs_rec_all),
        "ph3_keyword_recall":      _mean_all(kw_rec_all),
        "ph3_keyword_precision":   _mean_all(kw_prec_all),
        "ph3_combined_recall":     _mean_all(combined_rec_all),
        # PH-3 Build 3c: semantic extraction aggregates
        "ph3_semantic_parse_ok_rate":  round(n_semantic_parsed_all / max(1, n_parsed_all), 4),
        "ph3_semantic_recall":         _mean_all(sem_rec_all),
        "ph3_semantic_precision":      _mean_all(sem_prec_all),
        "ph3_n_semantic_recall_samples":    len(sem_rec_all),
        "ph3_n_semantic_precision_samples": len(sem_prec_all),
        "ph3_combined_http_kw_sem_recall":  _mean_all(combined_rec_hks_all),
    }

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("EXP. PH-3 OVERALL RESULTS")
    print("=" * 70)
    print(f"  Total step-logs:          {len(all_logs)}")
    print(f"  Parse-OK rate (agent):    {parse_rate_all:.2%} ({n_parsed_all}/{len(all_logs)})")
    print(f"  Mean shards_used (GT):    {mean_gt_all:.2f}")
    print(f"  HTTP observation recall:  {_mean_all(obs_rec_all)}")
    print()
    print("  Extraction mechanism comparison")
    print(f"  Keyword-scan recall:      {_mean_all(kw_rec_all)}")
    print(f"  Keyword-scan precision:   {_mean_all(kw_prec_all)}")
    print(f"  Semantic (gpt-4o) recall: {_mean_all(sem_rec_all)}   "
          f"  [{len(sem_rec_all)} samples]")
    print(f"  Semantic precision:       {_mean_all(sem_prec_all)}   "
          f"  [{len(sem_prec_all)} samples]")
    print(f"  Semantic parse-ok:        {n_semantic_parsed_all/max(1,n_parsed_all):.2%} "
          f" ({n_semantic_parsed_all}/{n_parsed_all})")
    print()
    print(f"  Combined (HTTP+kw)        recall: {_mean_all(combined_rec_all)}")
    print(f"  Combined (HTTP+kw+sem)    recall: {_mean_all(combined_rec_hks_all)}")
    print()
    print("  Legacy p_hidden metrics (backward compat with PH-2):")
    print(f"    Total r_obs:            {total_r_obs}")
    print(f"    Total r_hidden:         {total_r_hidden}")
    print(f"    Overall p_hidden:       {overall_ph:.4f}")
    print(f"    95% Wilson CI:          [{ci_lo:.4f}, {ci_hi:.4f}]")
    print()
    print("  Per-domain PH-3 metrics:")
    for domain_key, ds in domain_summaries.items():
        print(f"    {domain_key:<20}: n={ds['n_step_logs']:>4} "
              f"parse={ds.get('parse_ok_rate',0):.2f} "
              f"kw_rec={ds.get('keyword_recall')} kw_prec={ds.get('keyword_precision')}  "
              f"sem_rec={ds.get('semantic_recall')} sem_prec={ds.get('semantic_precision')}  "
              f"mean_gt={ds.get('mean_shards_used_gt')}")
    print()

    print(f"\n  Summary: {args.summary}")
    print(f"  Full data: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
