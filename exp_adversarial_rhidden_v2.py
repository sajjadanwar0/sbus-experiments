import asyncio
import csv
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, asdict

import httpx
from openai import AsyncOpenAI

SBUS_URL = os.getenv("SBUS_URL", "http://localhost:7000")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
N_TRIALS_PER_COND = int(os.getenv("N_TRIALS", "30"))
OUT_CSV = os.getenv("OUT_CSV", "results/adversarial_rhidden_v2.csv")
OUT_SUMMARY = os.getenv("OUT_SUMMARY", "results/adversarial_rhidden_v2_summary.json")

INITIAL_TABLE = "users"
RENAMED_TABLE = "customers"

CONDITIONS = ["ori_off", "ori_on", "ori_on_wrapper_naive", "ori_on_wrapper_regen"]


def http_get(path: str, params: dict | None = None) -> tuple[int, dict]:
    r = httpx.get(f"{SBUS_URL}{path}", params=params, timeout=30.0)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}

def http_post(path: str, body: dict) -> tuple[int, dict]:
    r = httpx.post(f"{SBUS_URL}{path}", json=body, timeout=60.0)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}

def reset_bus() -> None:
    http_post("/admin/reset", {})

def healthcheck() -> bool:
    try:
        s, _ = http_get("/stats")
        return s == 200
    except Exception:
        return False

# ─── Oracle ─────────────────────────────────────────────────────────────────

def scan_for_shard_refs(text: str, shard_base_names: list[str]) -> list[str]:
    hits = []
    if not text:
        return hits
    low = text.lower()
    for sk in shard_base_names:
        if sk.lower() in low:
            hits.append(sk)
    return hits

def classify_table_ref(text: str) -> str:
    """Returns 'stale' | 'fresh' | 'both' | 'none'."""
    low = (text or "").lower()
    has_stale = INITIAL_TABLE in low
    has_fresh = RENAMED_TABLE in low
    if has_stale and not has_fresh:
        return "stale"
    if has_fresh and not has_stale:
        return "fresh"
    if has_stale and has_fresh:
        return "both"
    return "none"

WORKER_SYSTEM = """You are a Django developer collaborating with three other
agents on a schema migration task. You write short, specific deltas.
Always reference table and column names accurately from the shared state
you have been shown. Respond in strict JSON:
{"change": "<one specific technical change>"}"""

WORKER_ROLE_DESC = {
    "alpha_2": "You own orm_query. Your job is to write a sample QuerySet that fetches all records from the primary table. Use the exact table name shown in db_schema.",
    "alpha_3": "You own test_fixture. Your job is to write a pytest fixture that inserts one row into the primary table. Use the exact table name shown in db_schema.",
    "alpha_4": "You own review_notes. Your job is to write a short code-review note listing the table(s) this PR touches. Use the exact table name shown in db_schema.",
}


@dataclass
class TrialResult:
    trial_id: str
    condition: str
    final_stale: int
    final_fresh: int
    final_ambiguous: int
    llm_stale: int
    llm_fresh: int
    llm_ambiguous: int
    n_409_triggers: int
    n_retries: int
    n_commits_landed: int
    wall_time_s: float
    raw_final_contents: str
    raw_llm_outputs: str

async def generate_change(oai: AsyncOpenAI, ctx: list[dict]) -> str:
    resp = await oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=ctx,
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw).get("change", "")
    except Exception:
        return raw[:200]

async def run_one_trial(oai: AsyncOpenAI, trial_id: str, condition: str) -> TrialResult:
    t0 = time.time()
    reset_bus()

    def sk(base: str) -> str:
        return f"{base}_{trial_id}"

    shard_base_names = ["db_schema", "orm_query", "test_fixture", "review_notes"]
    shard_keys = {b: sk(b) for b in shard_base_names}

    http_post("/shard", {"key": shard_keys["db_schema"],
                          "content": f"Primary table: {INITIAL_TABLE}. Fields: id, name, email.",
                          "goal_tag": f"adv_{trial_id}"})
    for b in ["orm_query", "test_fixture", "review_notes"]:
        http_post("/shard", {"key": shard_keys[b],
                              "content": "[empty]",
                              "goal_tag": f"adv_{trial_id}"})

    agent_ctx: dict[str, list[dict]] = {}
    for aid in ["alpha_2", "alpha_3", "alpha_4"]:
        agent_ctx[aid] = [{"role": "system", "content": WORKER_SYSTEM},
                          {"role": "user", "content": WORKER_ROLE_DESC[aid]}]
        s, data = http_get(f"/shard/{shard_keys['db_schema']}", {"agent_id": aid})
        if s != 200:
            raise RuntimeError(f"initial GET failed: {s} {data}")
        agent_ctx[aid].append({
            "role": "user",
            "content": f"Current db_schema (v={data['version']}):\n{data['content']}"
        })

    s, db = http_get(f"/shard/{shard_keys['db_schema']}", {"agent_id": "alpha_1"})
    s, _ = http_post("/commit/v2", {
        "key": shard_keys["db_schema"],
        "expected_version": db["version"],
        "delta": f"Primary table: {RENAMED_TABLE}. Fields: id, name, email. (renamed from {INITIAL_TABLE})",
        "agent_id": "alpha_1",
        "read_set": [{"key": shard_keys["db_schema"], "version_at_read": db["version"]}],
    })
    if s != 200:
        raise RuntimeError(f"alpha_1 rename failed: {s}")

    n_409 = 0
    n_retries = 0
    n_commits = 0

    llm_outputs: dict[str, str] = {}   # first LLM output per agent (pre-any-refresh)
    role_to_shard = {"alpha_2": "orm_query", "alpha_3": "test_fixture", "alpha_4": "review_notes"}

    for aid, shard_base in role_to_shard.items():
        target_key = shard_keys[shard_base]
        change = await generate_change(oai, agent_ctx[aid])
        llm_outputs[aid] = change

        commit_agent_id = aid
        read_set: list[dict] = []

        if condition == "ori_off":
            commit_agent_id = f"{aid}_noori_{uuid.uuid4().hex[:6]}"
            read_set = []

        elif condition == "ori_on":
            commit_agent_id = aid
            read_set = []

        elif condition == "ori_on_wrapper_naive":
            commit_agent_id = aid
            if (INITIAL_TABLE in change.lower()) or ("db_schema" in change.lower()):
                s, _fresh = http_get(f"/shard/{shard_keys['db_schema']}",
                                      {"agent_id": aid})
            read_set = []

        elif condition == "ori_on_wrapper_regen":
            commit_agent_id = aid
            if (INITIAL_TABLE in change.lower()) or ("db_schema" in change.lower()):
                s, fresh_db = http_get(f"/shard/{shard_keys['db_schema']}",
                                        {"agent_id": aid})
                if s == 200:
                    agent_ctx[aid].append({
                        "role": "user",
                        "content": (f"WRAPPER REFRESH: db_schema is now "
                                    f"v={fresh_db['version']}.\nFresh content:\n"
                                    f"{fresh_db['content']}\nRegenerate using "
                                    f"the fresh table name.")
                    })
                    change = await generate_change(oai, agent_ctx[aid])
                    n_retries += 1
            read_set = []

        s, cur = http_get(f"/shard/{target_key}", {"agent_id": commit_agent_id})
        if s != 200:
            continue
        expected_version = cur["version"]
        max_attempts = 1 if condition == "ori_off" else 3
        attempts = 0
        committed = False
        final_change = change
        while attempts < max_attempts and not committed:
            attempts += 1
            s, resp = http_post("/commit/v2", {
                "key": target_key,
                "expected_version": expected_version,
                "delta": final_change if final_change else "[empty]",
                "agent_id": commit_agent_id,
                "read_set": read_set,
            })
            if s == 200:
                n_commits += 1
                break
            if s == 409:
                n_409 += 1
                err = (resp.get("error") or "").lower() if isinstance(resp, dict) else ""

                if "cross" in err or "stale" in err:
                    s2, fresh = http_get(f"/shard/{shard_keys['db_schema']}",
                                          {"agent_id": aid})
                    if s2 == 200:
                        agent_ctx[aid].append({
                            "role": "user",
                            "content": (f"CROSS-SHARD STALE: db_schema is now "
                                        f"v={fresh['version']}.\nFresh content:\n"
                                        f"{fresh['content']}\nRegenerate using "
                                        f"the fresh table name.")
                        })
                        final_change = await generate_change(oai, agent_ctx[aid])
                        read_set = [{"key": shard_keys["db_schema"],
                                      "version_at_read": fresh["version"]}]
                        n_retries += 1
                        s3, cur2 = http_get(f"/shard/{target_key}",
                                             {"agent_id": commit_agent_id})
                        if s3 == 200:
                            expected_version = cur2["version"]
                        continue
                break
            break

    final_contents: dict[str, str] = {}
    for b in ["orm_query", "test_fixture", "review_notes"]:
        s, d = http_get(f"/shard/{shard_keys[b]}")
        final_contents[b] = d.get("content", "") if s == 200 else ""

    fstale = ffresh = famb = 0
    for _, c in final_contents.items():
        cls = classify_table_ref(c)
        if cls == "stale": fstale += 1
        elif cls == "fresh": ffresh += 1
        else: famb += 1

    lstale = lfresh = lamb = 0
    for _, c in llm_outputs.items():
        cls = classify_table_ref(c)
        if cls == "stale": lstale += 1
        elif cls == "fresh": lfresh += 1
        else: lamb += 1

    return TrialResult(
        trial_id=trial_id,
        condition=condition,
        final_stale=fstale, final_fresh=ffresh, final_ambiguous=famb,
        llm_stale=lstale,   llm_fresh=lfresh,   llm_ambiguous=lamb,
        n_409_triggers=n_409,
        n_retries=n_retries,
        n_commits_landed=n_commits,
        wall_time_s=round(time.time() - t0, 2),
        raw_final_contents=json.dumps(final_contents),
        raw_llm_outputs=json.dumps(llm_outputs),
    )

def bootstrap_mean_ci(xs: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    xs = list(xs)
    if not xs:
        return 0.0, 0.0, 0.0
    mean = sum(xs) / len(xs)
    boots = []
    for _ in range(n_boot):
        sample = [random.choice(xs) for _ in xs]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return mean, lo, hi

async def main():
    if not healthcheck():
        print(f"ERROR: S-Bus not running at {SBUS_URL}", file=sys.stderr)
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    oai = AsyncOpenAI()

    all_rows: list[TrialResult] = []
    for cond in CONDITIONS:
        print(f"\n=== Running condition: {cond} ({N_TRIALS_PER_COND} trials) ===",
              flush=True)
        for i in range(N_TRIALS_PER_COND):
            trial_id = f"{cond}_{i:03d}_{uuid.uuid4().hex[:6]}"
            try:
                r = await run_one_trial(oai, trial_id, cond)
                all_rows.append(r)
                print(f"  {i+1:2d}/{N_TRIALS_PER_COND}: "
                      f"final[stale={r.final_stale} fresh={r.final_fresh}] "
                      f"llm[stale={r.llm_stale} fresh={r.llm_fresh}] "
                      f"409s={r.n_409_triggers} commits={r.n_commits_landed} "
                      f"wall={r.wall_time_s:.1f}s", flush=True)
            except Exception as e:
                print(f"  trial {i+1} FAILED: {e}", flush=True)

    if all_rows:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(asdict(r))
        print(f"\nWrote {len(all_rows)} rows to {OUT_CSV}")

    summary: dict = {
        "experiment": "adversarial_rhidden_v2",
        "n_trials_per_condition": N_TRIALS_PER_COND,
        "backbone": OPENAI_MODEL,
        "conditions": {},
    }
    for cond in CONDITIONS:
        rows = [r for r in all_rows if r.condition == cond]
        if not rows:
            continue
        final_stale_rates = [r.final_stale / 3.0 for r in rows]
        llm_stale_rates = [r.llm_stale / 3.0 for r in rows]
        fs_m, fs_lo, fs_hi = bootstrap_mean_ci(final_stale_rates)
        ll_m, ll_lo, ll_hi = bootstrap_mean_ci(llm_stale_rates)
        summary["conditions"][cond] = {
            "n": len(rows),
            "mean_final_stale_per_trial": round(sum(r.final_stale for r in rows) / len(rows), 3),
            "mean_final_fresh_per_trial": round(sum(r.final_fresh for r in rows) / len(rows), 3),
            "mean_llm_stale_per_trial":   round(sum(r.llm_stale   for r in rows) / len(rows), 3),
            "mean_llm_fresh_per_trial":   round(sum(r.llm_fresh   for r in rows) / len(rows), 3),
            "final_stale_rate":     round(fs_m, 4),
            "final_stale_rate_95ci":[round(fs_lo, 4), round(fs_hi, 4)],
            "llm_stale_rate":       round(ll_m, 4),
            "llm_stale_rate_95ci":  [round(ll_lo, 4), round(ll_hi, 4)],
            "total_409_triggers":   sum(r.n_409_triggers for r in rows),
            "total_retries":        sum(r.n_retries for r in rows),
            "total_commits_landed": sum(r.n_commits_landed for r in rows),
            "mean_wall_time_s":     round(sum(r.wall_time_s for r in rows) / len(rows), 2),
        }

    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {OUT_SUMMARY}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
