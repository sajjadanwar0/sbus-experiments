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
    print("ERROR: uv add openai"); sys.exit(1)

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
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL", "gpt-4o")
SEMANTIC_ENABLED = int(os.getenv("SEMANTIC_ENABLED", "1"))
N_AGENTS  = 1

WORKLOAD = os.getenv("WORKLOAD", "ph3_all_get")
if WORKLOAD not in ("ph3_all_get", "ph2_target_only"):
    raise ValueError(
        f"WORKLOAD={WORKLOAD!r} not recognised; expected 'ph3_all_get' "
        f"or 'ph2_target_only'"
    )

_opener = build_opener(ProxyHandler({}))

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

def build_keyword_pattern(shard_keys: list[str]) -> re.Pattern:
    """Build regex to detect shard key mentions in conversation history."""
    escaped = [re.escape(k) for k in shard_keys]
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)

HIDDEN_SCAN_WINDOW = 2

def count_hidden_reads(
    history: list[dict],
    shard_keys: list[str],
    current_fresh_content: str,
    window: int = HIDDEN_SCAN_WINDOW,
) -> int:
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

@dataclass
class StepLog:
    run_id:           str
    domain:           str
    task_id:          str
    agent_id:         str
    step:             int
    n_agents:         int
    r_obs_count:      int
    r_hidden_count:   int
    r_total:          int
    p_hidden:         float
    commit_status:    str
    history_length:   int
    history_tokens_est: int
    shards_used_gt_json:  str
    n_shards_used_gt:     int
    parse_ok:             bool
    r_obs_base_json:      str
    r_hidden_base_json:   str
    r_semantic_base_json: str
    semantic_parse_ok:    bool
    change_text:          str
    fresh_content_block:  str
    workload:             str

def _keyword_matches_in_history(
    history: list[dict],
    shard_keys: list[str],
    current_fresh_content: str,
    window: int = HIDDEN_SCAN_WINDOW,
) -> list[str]:
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
    seen = set()
    dedup = []
    for s in validated:
        if s not in seen:
            seen.add(s); dedup.append(s)
    return change.strip(), dedup, True

async def extract_shards_semantic(
    oai: AsyncOpenAI,
    completion_text: str,
    shard_vocab: list[str],
    fresh_content_block: str,
    domain_desc: str,
    anthropic_client=None,
) -> tuple[list[str], bool]:
    if not SEMANTIC_ENABLED:
        return [], False
    if not completion_text or not shard_vocab:
        return [], False

    vocab_str = ", ".join(f'"{s}"' for s in shard_vocab)
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

    raw = ""
    if SEMANTIC_MODEL.startswith("claude"):
        if anthropic_client is None:
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
    logs = []
    history = []
    shard_base_names = list(domain["shards"])  # e.g. ["models_state", "orm_query", ...]
    shared_shards = [f"{sk}_{run_id}" for sk in shard_base_names]
    desc = f"{domain['desc']} Task: {task['goal']}"

    for step in range(n_steps):
        shard_data = {}
        read_set = []
        r_obs_base_set: set[str] = set()
        target_idx = step % len(shared_shards)
        target_base = shard_base_names[target_idx]
        target_full = shared_shards[target_idx]

        if WORKLOAD == "ph3_all_get":
            shards_to_get = list(zip(shard_base_names, shared_shards))
        else:
            shards_to_get = [(target_base, target_full)]

        for base, full_key in shards_to_get:
            status, data = http_get(f"{SBUS_URL}/shard/{full_key}", {"agent_id": agent_id})
            if status == 200:
                shard_data[full_key] = data
                read_set.append({"key": full_key, "version_at_read": data.get("version", 0)})
                r_obs_base_set.add(base)

        fresh_content_block = "\n".join(
            f"  {base}: v{shard_data[full].get('version', 0)} "
            f"— {shard_data[full].get('content', '')[:80]}"
            for base, full in shards_to_get
            if full in shard_data
        )

        r_hidden_count = count_hidden_reads(history, shard_base_names, fresh_content_block)
        r_hidden_base_set = set(
            _keyword_matches_in_history(history, shard_base_names, fresh_content_block)
        )

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

        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant", "content": delta})

        target = shared_shards[step % len(shared_shards)]
        ev     = shard_data.get(target, {}).get("version", 0)
        status, _ = http_post(f"{SBUS_URL}/commit/v2", {
            "key":              target,
            "expected_version": ev,
            "delta":            delta if delta else f"[empty step {step}]",
            "agent_id":         agent_id,
            "read_set":         read_set,
        })

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
    run_id = uuid.uuid4().hex[:8]

    for sk in domain["shards"]:
        http_post(f"{SBUS_URL}/shard", {
            "key":      f"{sk}_{run_id}",
            "content":  f"Initial: {domain['desc'][:80]}",
            "goal_tag": f"ph2_{task['id']}",
        })

    shard_pattern = build_keyword_pattern(domain["shards"])

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


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * (p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


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

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)

    if not health_check():
        print(f"ERROR: S-Bus not running at {SBUS_URL}")
        print("  Start: SBUS_ADMIN_ENABLED=1 cargo run --release")
        sys.exit(1)

    oai = AsyncOpenAI(api_key=api_key)

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

    if args.domains in DOMAIN_GROUPS:
        domain_keys = DOMAIN_GROUPS[args.domains]
    else:
        domain_keys = [d.strip() for d in args.domains.split(",")]

    domains = {k: TASK_DOMAINS[k] for k in domain_keys if k in TASK_DOMAINS}
    if not domains:
        print(f"ERROR: No valid domains in '{args.domains}'")
        print(f"Available: {list(TASK_DOMAINS.keys())}")
        sys.exit(1)

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
        print("           -> Agent HTTP-GETs ONLY the target shard each step.")
        print("           -> Other shards referenced via context memory (R_hidden).")
        print("           -> Expected p_hidden: ~0.6-0.8 (PH-2 regime).")
    else:
        print("           -> Agent HTTP-GETs ALL shards each step.")
        print("           -> HTTP recall = 1.0 by construction.")
        print("           -> Expected p_hidden: ~0.05-0.15 (PH-3 regime).")
    print()

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

                for log in logs:
                    row = asdict(log)
                    if writer is None:
                        writer = csv.DictWriter(out_f, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                out_f.flush()

                mean_ph = statistics.mean(l.p_hidden for l in logs) if logs else 0
                print(f"n={len(logs)} mean_p_hidden={mean_ph:.4f} wall={wall:.0f}s")

        if domain_logs:
            ph_vals = [l.p_hidden for l in domain_logs]
            mean_ph = statistics.mean(ph_vals)
            total_r_obs    = sum(l.r_obs_count    for l in domain_logs)
            total_r_hidden = sum(l.r_hidden_count for l in domain_logs)
            total_r        = total_r_obs + total_r_hidden
            ci_lo, ci_hi   = wilson_ci(total_r_hidden, total_r)
            usable = [l for l in domain_logs if l.parse_ok]
            n_parsed = len(usable)
            parse_rate = n_parsed / len(domain_logs) if domain_logs else 0.0
            kw_prec_vals, kw_rec_vals = [], []
            obs_rec_vals = []
            sem_prec_vals, sem_rec_vals = [], []
            combined_kw_sem_rec_vals = []
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
                    if l.semantic_parse_ok:
                        sem_rec_vals.append(len(inter_sem) / len(gt_set))
                        if sem_set:
                            sem_prec_vals.append(len(inter_sem) / len(sem_set))
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

    if not all_logs:
        print("No data collected."); sys.exit(1)

    total_r_obs    = sum(l.r_obs_count    for l in all_logs)
    total_r_hidden = sum(l.r_hidden_count for l in all_logs)
    total_r        = total_r_obs + total_r_hidden
    overall_ph     = total_r_hidden / total_r if total_r > 0 else 0.0
    ci_lo, ci_hi   = wilson_ci(total_r_hidden, total_r)

    step_means = {}
    for step in range(args.steps):
        vals = [l.p_hidden for l in all_logs if l.step == step]
        if vals:
            step_means[step] = statistics.mean(vals)

    usable_all = [l for l in all_logs if l.parse_ok]
    n_parsed_all = len(usable_all)
    parse_rate_all = n_parsed_all / len(all_logs) if all_logs else 0.0

    kw_prec_all, kw_rec_all, obs_rec_all = [], [], []
    sem_prec_all, sem_rec_all = [], []
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

    combined_rec_all = []
    combined_rec_hks_all = []
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
        "ph3_n_parsed":            n_parsed_all,
        "ph3_parse_ok_rate":       round(parse_rate_all, 4),
        "ph3_mean_shards_used_gt": round(mean_gt_all, 4),
        "ph3_http_obs_recall":     _mean_all(obs_rec_all),
        "ph3_keyword_recall":      _mean_all(kw_rec_all),
        "ph3_keyword_precision":   _mean_all(kw_prec_all),
        "ph3_combined_recall":     _mean_all(combined_rec_all),
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
