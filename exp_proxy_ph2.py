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
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: uv add openai"); sys.exit(1)


SBUS_URL  = os.getenv("SBUS_URL",  "http://localhost:7000")
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:9000")
BACKBONE  = os.getenv("BACKBONE",  "gpt-4o-mini")
TEMP      = float(os.getenv("TEMP", "0.3"))

_opener = build_opener(ProxyHandler({}))

EXTENDED_TASK_DOMAINS = {
    "django_queryset": {
        "desc":   "Fix Django queryset ordering with select_related() on FK traversal.",
        "shards": [
            "models_state", "orm_query", "test_fixtures", "migration_plan",
            "queryset_manager", "related_manager", "filter_engine", "join_strategy",
            "query_cache", "db_backend", "select_compiler", "result_iterator",
        ],
        "tasks": [
            {"id": "django-11019", "goal": "Fix queryset ordering with select_related fields"},
            {"id": "django-11820", "goal": "Fix ordering on annotated querysets"},
            {"id": "django-14155", "goal": "Fix reverse FK ordering in admin views"},
        ],
    },
    "django_admin": {
        "desc":   "Fix Django admin list_display rendering with computed fields.",
        "shards": [
            "admin_config", "model_fields", "template_render", "test_client",
            "widget_registry", "permission_check", "changelist_view", "form_validator",
            "media_static", "autocomplete_source", "batch_action", "url_patterns",
        ],
        "tasks": [
            {"id": "django-14034", "goal": "Fix list_display with @admin.display decorator"},
            {"id": "django-15127", "goal": "Fix admin inline formset validation"},
        ],
    },
    "django_migration": {
        "desc":   "Fix Django migration dependency resolution on multi-app state.",
        "shards": [
            "migration_graph", "app_state", "schema_editor", "test_db",
            "operation_queue", "state_projection", "dependency_order", "rollback_plan",
            "fake_migration", "migration_writer", "autodetector", "connection_wrapper",
        ],
        "tasks": [
            {"id": "django-15098", "goal": "Fix migration squash with RunPython"},
            {"id": "django-16105", "goal": "Fix migration dependency detection"},
        ],
    },
    "astropy_fits": {
        "desc":   "Fix FITS header parsing with CONTINUE cards and BLANK keyword.",
        "shards": [
            "header_state", "hdu_structure", "card_formatter", "io_path",
            "checksum_verify", "extension_list", "bintable_columns", "comment_block",
            "data_block", "wcs_projection", "bscale_bzero", "heap_area",
        ],
        "tasks": [
            {"id": "astropy-13969", "goal": "Fix FITS CONTINUE card round-tripping"},
            {"id": "astropy-14328", "goal": "Fix FITS BLANK/BZERO interaction"},
        ],
    },
    "astropy_wcs": {
        "desc":   "Fix WCS celestial-coordinate transformations with SIP distortion.",
        "shards": [
            "wcs_transform", "sip_coeffs", "coord_frame", "pixel_map",
            "projection_params", "celestial_sphere", "distortion_table", "cd_matrix",
            "radesys", "astropy_units_ctx", "reference_pixel", "transform_chain",
        ],
        "tasks": [
            {"id": "astropy-14539", "goal": "Fix WCS with SIP distortion on ref pixel"},
            {"id": "astropy-12907", "goal": "Fix celestial frame conversion"},
        ],
    },
    "astropy_units": {
        "desc":   "Fix astropy.units composite-unit equivalencies.",
        "shards": [
            "unit_registry", "equivalency_map", "quantity_state", "unit_parser",
            "prefix_table", "dimension_vector", "compose_cache", "conversion_graph",
            "format_handler", "si_base_units", "quantity_ops", "unit_decompose",
        ],
        "tasks": [
            {"id": "astropy-13745", "goal": "Fix composite-unit equivalency"},
            {"id": "astropy-14096", "goal": "Fix quantity dimensional analysis"},
        ],
    },
    "sympy_solver": {
        "desc":   "Fix SymPy solveset for transcendental equations with piecewise branches.",
        "shards": [
            "expr_tree", "assumption_ctx", "solver_strategy", "test_cases",
            "domain_lookup", "simplification_rules", "piecewise_split", "image_set",
            "condition_set", "finite_set", "complement_set", "interval_arith",
        ],
        "tasks": [
            {"id": "sympy-22714", "goal": "Fix solveset for piecewise transcendentals"},
            {"id": "sympy-23117", "goal": "Fix Intersection(FiniteSet, ImageSet)"},
        ],
    },
    "sympy_matrix": {
        "desc":   "Fix SymPy matrix-decomposition handling of symbolic entries.",
        "shards": [
            "matrix_repr", "decomp_method", "symbolic_engine", "numerical_fallback",
            "row_reducer", "pivot_strategy", "sparse_handler", "block_structure",
            "eigenvalue_cache", "determinant_path", "nullspace_basis", "lu_pivot_seq",
        ],
        "tasks": [
            {"id": "sympy-22456", "goal": "Fix matrix LU with symbolic pivots"},
            {"id": "sympy-23824", "goal": "Fix eigen for block-diagonal symbolic"},
        ],
    },
    "requests_session": {
        "desc":   "Fix Requests session cookie-jar merging on redirect chain.",
        "shards": [
            "cookie_jar", "session_state", "redirect_chain", "adapter_pool",
            "auth_handler", "proxy_config", "tls_context", "retry_policy",
            "header_merge", "hook_registry", "cert_bundle", "connection_pool",
        ],
        "tasks": [
            {"id": "requests-6028", "goal": "Fix cookie-jar merge on 3xx redirect"},
            {"id": "requests-6358", "goal": "Fix HTTPAdapter pool with proxy"},
        ],
    },
    "scikit_estimator": {
        "desc":   "Fix scikit-learn estimator __sklearn_is_fitted__ with partial fit.",
        "shards": [
            "estimator_state", "fit_params", "validation_hook", "test_grid",
            "feature_names", "scorer_callable", "pipeline_stage", "splitter",
            "metadata_tags", "partial_fit_ctx", "sample_weights", "fit_context",
        ],
        "tasks": [
            {"id": "sklearn-25443", "goal": "Fix is_fitted on partial_fit path"},
            {"id": "sklearn-26242", "goal": "Fix GridSearchCV with refit=False"},
        ],
    },
}

DOMAIN_GROUPS = {
    "all":     list(EXTENDED_TASK_DOMAINS.keys()),
    "django":  ["django_queryset", "django_admin", "django_migration"],
    "astropy": ["astropy_fits", "astropy_wcs", "astropy_units"],
    "sympy":   ["sympy_solver", "sympy_matrix"],
    "other":   ["requests_session", "scikit_estimator"],
}

DOMAIN_VOCAB = sorted({
    s for d in EXTENDED_TASK_DOMAINS.values() for s in d["shards"]
})

N_AGENTS = 4

def _http(method: str, url: str, body=None, params=None, timeout=10):
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
            body = e.read().decode("utf-8", errors="replace")
            return e.code, json.loads(body) if body else {}
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


def http_get(url, params=None):  return _http("GET",  url, params=params)
def http_post(url, body):        return _http("POST", url, body=body)

def reset_bus():
    http_post(f"{SBUS_URL}/admin/reset", {})

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

def admin_delivery_log_for_agent(agent_id: str) -> set[str]:
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

def get_shard_version(full_key: str) -> int:
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
    shards_per_domain:  int
    max_shards_per_step:int

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

def shards_to_get(agent_idx: int, step: int, all_base: list[str],
                  max_k: int = 2) -> list[str]:
    h = int(hashlib.sha1(f"{agent_idx}:{step}".encode()).hexdigest(), 16)
    k_hash = 1 + (h % max(1, max_k))
    k = min(max_k, k_hash)
    start = h % len(all_base)
    return [all_base[(start + i) % len(all_base)] for i in range(k)]

def target_shard(agent_idx: int, step: int, all_base: list[str]) -> str:
    return all_base[(agent_idx + step) % len(all_base)]

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

def parse_shards_used(text: str, vocab: list[str]) -> set[str]:
    if not text:
        return set()
    m = _JSON_BLOCK.search(text)
    raw = m.group(0) if m else text
    try:
        obj = json.loads(raw)
    except Exception:
        return set()
    if not isinstance(obj, dict):
        return set()
    used = obj.get("shards_used", [])
    if isinstance(used, list):
        return {s for s in used if isinstance(s, str) and s in vocab}
    return set()

async def run_agent(
    oai:                AsyncOpenAI,
    agent_id:           str,
    session_id:         str,
    shard_suffix:       str,
    agent_idx:          int,
    domain:             dict,
    task:               dict,
    run_id:             str,
    n_steps:            int,
    condition:          str,
    shards_per_domain:  int,
    max_shards_per_step:int,
) -> list[StepRow]:
    rows: list[StepRow] = []
    history: list[dict] = []
    base = list(domain["shards"])[:shards_per_domain]
    full_keys = {b: f"{b}{shard_suffix}" for b in base}
    desc = f"{domain['desc']} Task: {task['goal']}"

    last_seen_version: dict[str, int] = {}
    last_seen_content: dict[str, str] = {}
    last_seen_step:    dict[str, int] = {}

    for step in range(n_steps):
        gets = shards_to_get(agent_idx, step, base, max_k=max_shards_per_step)
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

        sys_msg = (
            "You are a software engineering agent collaborating with 3 others "
            "on a shared codebase represented as shards. Each step, you see "
            "fresh state for a subset of shards and remembered state for the "
            "others. Your change must maintain consistency across the FULL "
            "shard state you rely on — both freshly fetched and remembered.\n"
            "\n"
            "Respond in strict JSON matching this schema:\n"
            "  {\n"
            '    "change":       "<one concrete technical change, specific, <=100 words>",\n'
            '    "shards_used":  ["<shard_base_name>", ...]\n'
            "  }\n"
            f"The shard vocabulary is exactly: [{shard_list_str}].\n"
            "\n"
            "In 'shards_used' list EVERY shard whose content influenced your "
            "reasoning, INCLUDING shards you relied on from remembered state. "
            "If your change needs to be consistent with a shard's earlier value "
            "that you remember (even though you didn't refetch it this step), "
            "list that shard. Do NOT list shards you never encountered."
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
            f"Your change should be consistent with every shard your reasoning "
            f"depends on. Respond in JSON."
        )
        messages = [{"role": "system", "content": sys_msg}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_msg})

        t_llm_0 = time.time()
        try:
            resp = await oai.chat.completions.create(
                model=BACKBONE,
                messages=messages,
                temperature=TEMP,
                response_format={"type": "json_object"},
                max_tokens=350,
                extra_headers={
                    "X-SBus-Agent-Id":     agent_id,
                    "X-SBus-Session-Id":   session_id,
                    "X-SBus-Shard-Suffix": shard_suffix,
                },
            )
            ans = resp.choices[0].message.content or ""
        except Exception as e:
            ans = json.dumps({"change": f"LLM error: {e}", "shards_used": []})
        llm_ms = int((time.time() - t_llm_0) * 1000)

        r_self = parse_shards_used(ans, base)

        dl_after = admin_delivery_log_for_agent(agent_id)
        dl_base_after = {
            k[: -len(shard_suffix)] if shard_suffix and k.endswith(shard_suffix) else k
            for k in dl_after
        } & set(base)
        r_obs_proxy = dl_base_after - r_obs_http

        tgt_base = target_shard(agent_idx, step, base)
        tgt_full = full_keys[tgt_base]
        exp_ver  = last_seen_version.get(tgt_base, get_shard_version(tgt_full))
        delta = f"α{agent_idx} step {step}: {ans[:120]}"
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

        def frac(num, den):  return round(num / den, 4) if den else 0.0
        f_obs_http  = frac(len(r_obs_http    & r_self), len(r_self))
        f_obs_total = frac(len(dl_base_after & r_self), len(r_self))

        rows.append(StepRow(
            condition=condition,
            domain=domain.get("name", domain.get("desc", "")[:20]),
            task_id=task["id"],
            run_id=run_id,
            agent_idx=agent_idx,
            step=step,
            shards_per_domain=shards_per_domain,
            max_shards_per_step=max_shards_per_step,
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
        history.append({"role": "assistant", "content": ans})

    return rows

async def run_trial(
    oai: AsyncOpenAI,
    domain_key: str,
    domain: dict,
    task: dict,
    n_steps: int,
    condition: str,
    run_id: str,
    shards_per_domain: int,
    max_shards_per_step: int,
) -> list[StepRow]:
    shard_suffix = f"_{run_id}"
    session_id   = f"sess_{run_id}_{condition}"

    sliced_shards = list(domain["shards"])[:shards_per_domain]
    for b in sliced_shards:
        http_post(f"{SBUS_URL}/shard", {
            "key":      f"{b}{shard_suffix}",
            "content":  f"Initial state for {b} — {task['goal'][:60]}",
            "goal_tag": f"proxy_ph2_{task['id']}",
        })

    agents = await asyncio.gather(*[
        run_agent(
            oai=oai,
            agent_id    = f"agent_{i}_{run_id}_{condition}",
            session_id  = session_id,
            shard_suffix= shard_suffix,
            agent_idx   = i,
            domain      = {**domain, "name": domain_key},
            task        = task,
            run_id      = run_id,
            n_steps     = n_steps,
            condition   = condition,
            shards_per_domain   = shards_per_domain,
            max_shards_per_step = max_shards_per_step,
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

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return 0.0, 1.0
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * (p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def rule_of_three_upper(n: int, z: float = 1.96) -> float:
    return 0.0 if n == 0 else 3.0 / n

def paired_bootstrap_ci(pairs: list[tuple[float, float]], iters: int = 5000,
                        alpha: float = 0.05) -> tuple[float, float, float]:
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

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains",              default="all")
    ap.add_argument("--tasks-per-domain",     type=int, default=2)
    ap.add_argument("--runs-per-task",        type=int, default=3)
    ap.add_argument("--steps",                type=int, default=10)
    ap.add_argument("--shards-per-domain",    type=int, default=4,
                    help="How many base shards to expose per domain (1..12). "
                         "Main paper experiment uses 4; vocab-scaling appendix "
                         "sweeps 4, 8, 12.")
    ap.add_argument("--max-shards-per-step",  type=int, default=2,
                    help="Upper bound on shards HTTP-GET'd per step per agent. "
                         "Main experiment: 2. Vocab-scaling appendix: 1 "
                         "(forces sparser access, prevents fetch-history "
                         "saturation at higher vocabularies).")
    ap.add_argument("--conditions",           default="proxy_off,proxy_on")
    ap.add_argument("--output",               default="results/proxy_ph2.csv")
    ap.add_argument("--summary",              default="results/proxy_ph2_summary.json")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)
    if not health_check():
        print(f"ERROR: S-Bus unreachable or admin mode off at {SBUS_URL}"); sys.exit(1)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if "proxy_on" in conditions and not proxy_health_check():
        print(f"ERROR: proxy_on requested but sbus-proxy unreachable at {PROXY_URL}")
        sys.exit(1)

    if not (1 <= args.shards_per_domain <= 12):
        print(f"ERROR: --shards-per-domain must be in [1, 12], got {args.shards_per_domain}")
        sys.exit(1)
    if args.max_shards_per_step < 1:
        print(f"ERROR: --max-shards-per-step must be >= 1"); sys.exit(1)

    if args.domains in DOMAIN_GROUPS:
        dkeys = DOMAIN_GROUPS[args.domains]
    else:
        dkeys = [d.strip() for d in args.domains.split(",")]
    domains = {k: EXTENDED_TASK_DOMAINS[k] for k in dkeys if k in EXTENDED_TASK_DOMAINS}
    if not domains:
        print(f"ERROR: no valid domains in {args.domains!r}"); sys.exit(1)

    oai_direct = AsyncOpenAI(api_key=api_key)
    oai_proxy  = AsyncOpenAI(api_key=api_key, base_url=f"{PROXY_URL}/v1")
    def client_for(cond): return oai_proxy if cond == "proxy_on" else oai_direct

    total_tasks = sum(min(args.tasks_per_domain, len(d["tasks"])) for d in domains.values())
    total_rows  = total_tasks * args.runs_per_task * len(conditions) * N_AGENTS * args.steps
    print("=" * 70)
    print("Exp. PROXY-PH2 v50.1 — proxy coverage uplift on multi-agent PH-2")
    print("=" * 70)
    print(f"Domains ({len(domains)}): {list(domains.keys())}")
    print(f"Tasks: {total_tasks}, runs/task: {args.runs_per_task}, "
          f"conditions: {conditions}, N={N_AGENTS}, steps={args.steps}")
    print(f"Shards/domain: {args.shards_per_domain}  "
          f"Max shards/step: {args.max_shards_per_step}")
    print(f"Target rows: {total_rows}")
    print(f"Backbone: {BACKBONE} @ T={TEMP}")
    print(f"SBUS_URL={SBUS_URL}  PROXY_URL={PROXY_URL}")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fout = open(args.output, "w", newline="")
    w = csv.DictWriter(fout, fieldnames=[f.name for f in StepRow.__dataclass_fields__.values()])
    w.writeheader()

    t0 = time.time()
    all_rows: list[StepRow] = []
    for dk, d in domains.items():
        tasks = d["tasks"][:args.tasks_per_domain]
        for task in tasks:
            for r in range(args.runs_per_task):
                run_id = uuid.uuid4().hex[:8]
                for cond in conditions:
                    print(f"[{dk}/{task['id']}/r{r}/{cond}] run_id={run_id}")
                    for i in range(N_AGENTS):
                        http_post(f"{SBUS_URL}/session", {
                            "agent_id": f"agent_{i}_{run_id}_{cond}"
                        })
                    rows = await run_trial(
                        oai=client_for(cond),
                        domain_key=dk,
                        domain=d,
                        task=task,
                        n_steps=args.steps,
                        condition=cond,
                        run_id=run_id,
                        shards_per_domain=args.shards_per_domain,
                        max_shards_per_step=args.max_shards_per_step,
                    )
                    for row in rows:
                        w.writerow(asdict(row))
                    fout.flush()
                    all_rows.extend(rows)
                    if rows:
                        print(f"   rows: {len(rows)}  "
                              f"f_obs_http={statistics.mean(x.f_obs_http for x in rows):.3f}  "
                              f"f_obs_total={statistics.mean(x.f_obs_total for x in rows):.3f}")

    fout.close()
    elapsed = time.time() - t0
    print(f"\nCollected {len(all_rows)} rows in {elapsed:.1f}s")

    by_cond: dict[str, list[StepRow]] = {c: [] for c in conditions}
    for row in all_rows:
        by_cond.setdefault(row.condition, []).append(row)

    summary: dict = {
        "conditions": {},
        "meta": {
            "sbus_url":             SBUS_URL,
            "proxy_url":            PROXY_URL,
            "backbone":             BACKBONE,
            "n_agents":             N_AGENTS,
            "steps":                args.steps,
            "tasks_per_domain":     args.tasks_per_domain,
            "runs_per_task":        args.runs_per_task,
            "shards_per_domain":    args.shards_per_domain,
            "max_shards_per_step":  args.max_shards_per_step,
            "n_rows_total":         len(all_rows),
            "elapsed_s":            round(elapsed, 1),
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

    if "proxy_off" in by_cond and "proxy_on" in by_cond:
        key = lambda r: (r.domain, r.task_id, r.run_id, r.agent_idx, r.step)
        off_map = {key(r): r.f_obs_total for r in by_cond["proxy_off"]}
        pairs = [
            (off_map[key(r)], r.f_obs_total)
            for r in by_cond["proxy_on"]
            if key(r) in off_map
        ]
        mean_d, lo, hi = paired_bootstrap_ci(pairs)
        summary["paired_uplift_proxy_on_minus_off"] = {
            "n_pairs":   len(pairs),
            "mean_diff": round(mean_d, 4),
            "bs_95_lo":  round(lo, 4),
            "bs_95_hi":  round(hi, 4),
        }

        off = by_cond["proxy_off"]
        on  = by_cond["proxy_on"]
        summary["coverage_decomposition"] = {
            "http_this_step":          round(statistics.mean(r.f_obs_http  for r in off), 4),
            "dl_accumulation_under_off": round(
                statistics.mean(r.f_obs_total - r.f_obs_http for r in off), 4),
            "proxy_marginal_paired":   round(mean_d, 4),
            "total_coverage_off":      round(statistics.mean(r.f_obs_total for r in off), 4),
            "total_coverage_on":       round(statistics.mean(r.f_obs_total for r in on ), 4),
        }

    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV     : {args.output}")
    print(f"Summary : {args.summary}")


if __name__ == "__main__":
    asyncio.run(main())
