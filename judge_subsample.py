import argparse
import asyncio
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

from openai import AsyncOpenAI


JUDGE_RUBRIC = """You are evaluating whether a 4-component data-pipeline \
design is internally coherent. Four agents (one per component) each wrote \
their component independently. You will read all four. Your job: do the \
components reference each other consistently?

A coherent pipeline:
- The transformation component's input source matches the ingestion component's output (e.g., if ingestion writes to a Kafka topic 'orders', transformation reads from 'orders').
- The storage component's input source matches the transformation component's output.
- The monitoring component instruments the technologies actually chosen by ingestion/transformation/storage (e.g., monitoring shouldn't reference Apache Spark if no other component mentions Spark).
- Cross-component entity references (technology names, topic/table names, schemas) appear in the components that should define them.

An incoherent pipeline:
- Storage references entities that don't exist in transformation (e.g., "consume from output topic" when transformation defined no such topic).
- Monitoring references technologies no other component uses.
- Ingestion writes to one technology, transformation reads from a different technology with no bridging mechanism.
- Components contradict each other on shared decisions (e.g., one says "exactly-once", another says "at-most-once" for the same event flow).

Output STRICT JSON with these fields:
- coherent: true if the pipeline is internally consistent, false otherwise
- confidence: float in [0.0, 1.0] indicating how confident you are
- problems: list of specific incoherences you found (empty if coherent=true)
- rationale: 1-2 sentences explaining your verdict"""


JUDGE_USER_TEMPLATE = """Pipeline ID: {trial_id}

INGESTION:
{ingestion}

TRANSFORMATION:
{transformation}

STORAGE:
{storage}

MONITORING:
{monitoring}

Output STRICT JSON: {{"coherent": bool, "confidence": float, "problems": [str], "rationale": str}}.
"""


def stable_shuffle_key(trial_id: str) -> str:
    return hashlib.sha256(trial_id.encode()).hexdigest()


async def judge_one_trial(client: AsyncOpenAI, model: str,
                            trial: Dict[str, Any]) -> Dict[str, Any]:
    final_state = trial.get("final_state", {})
    user_msg = JUDGE_USER_TEMPLATE.format(
        trial_id=trial["trial_id"],
        ingestion=(final_state.get("ingestion") or {}).get("content", "(missing)"),
        transformation=(final_state.get("transformation") or {}).get("content", "(missing)"),
        storage=(final_state.get("storage") or {}).get("content", "(missing)"),
        monitoring=(final_state.get("monitoring") or {}).get("content", "(missing)"),
    )
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_RUBRIC},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        verdict = json.loads(resp.choices[0].message.content)
        return {"trial_id": trial["trial_id"], "verdict": verdict, "error": None}
    except Exception as e:
        return {"trial_id": trial["trial_id"], "verdict": None,
                "error": f"{type(e).__name__}: {e}"}


def load_trials(jsonl_path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(jsonl_path)]


def select_subsample(trials: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    selected = []
    seen = {}
    for t in trials:
        key = (t["domain"], t["condition"])
        if seen.get(key, 0) < k:
            selected.append(t)
            seen[key] = seen.get(key, 0) + 1
    selected.sort(key=lambda t: stable_shuffle_key(t["trial_id"]))
    return selected


async def main_async(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: input file not found: {args.input}")

    all_trials = load_trials(args.input)
    print(f"Loaded {len(all_trials)} trials from {args.input}")
    subsample = select_subsample(all_trials, args.k_per_cell)
    print(f"Subsample: {len(subsample)} trials "
           f"({args.k_per_cell} per (domain, condition) cell)")

    client = AsyncOpenAI()
    out_path = args.output
    out_file = open(out_path, "w")
    cost_estimate = len(subsample) * 0.0075  # gpt-4o ~$0.0075/call
    print(f"Estimated cost: ~${cost_estimate:.2f}\n")

    verdicts = []
    for i, trial in enumerate(subsample, 1):
        print(f"  [{i:>2}/{len(subsample)}] {trial['trial_id']:<35} ", end="", flush=True)
        result = await judge_one_trial(client, args.judge_model, trial)
        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            v = result["verdict"]
            print(f"coherent={v.get('coherent')!s:<5} "
                   f"conf={v.get('confidence', 0):.2f}")
        out_file.write(json.dumps({
            **result,
            "domain": trial["domain"],
            "condition": trial["condition"],
        }) + "\n")
        out_file.flush()
        verdicts.append((trial, result))

    out_file.close()
    print()
    print("=" * 80)
    print("JUDGE VERDICTS BY CONDITION")
    print("=" * 80)

    by_cond = {"ori_on": {"coherent": 0, "incoherent": 0, "errors": 0},
                "ori_off": {"coherent": 0, "incoherent": 0, "errors": 0}}
    for trial, result in verdicts:
        c = trial["condition"]
        if c not in by_cond:
            continue
        if result["error"] or not result.get("verdict"):
            by_cond[c]["errors"] += 1
        elif result["verdict"].get("coherent"):
            by_cond[c]["coherent"] += 1
        else:
            by_cond[c]["incoherent"] += 1

    for cond in ["ori_on", "ori_off"]:
        c = by_cond[cond]
        n = c["coherent"] + c["incoherent"]
        rate = c["coherent"] / n if n else 0.0
        print(f"  {cond:<10} coherent: {c['coherent']:>2}/{n:<2}  "
               f"incoherent: {c['incoherent']:>2}/{n:<2}  rate: {rate:.0%}  "
               f"(errors: {c['errors']})")

    on_n = by_cond["ori_on"]["coherent"] + by_cond["ori_on"]["incoherent"]
    off_n = by_cond["ori_off"]["coherent"] + by_cond["ori_off"]["incoherent"]
    if on_n and off_n:
        on_rate = by_cond["ori_on"]["coherent"] / on_n
        off_rate = by_cond["ori_off"]["coherent"] / off_n
        print()
        print(f"  (coherent rate) = ori_on - ori_off = {on_rate - off_rate:+.2%}")
        print()
        if on_rate - off_rate >= 0.20:
            print(" JUDGE CROSS-VALIDATION: SUPPORTS ori_on outcome-quality advantage")
        elif on_rate - off_rate >= 0.05:
            print(" JUDGE CROSS-VALIDATION: WEAK SIGNAL favoring ori_on")
        else:
            print(" JUDGE CROSS-VALIDATION: NO CLEAR DIFFERENCE in outcome quality")
            print("    Note: this does NOT invalidate the server-side view-divergence")
            print("    finding. It means the LLM-judge proxy did not detect semantic")
            print("    consequences of the structural staleness ORI prevented.")
    print()
    print(f"Verdict file: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/workload_b_sweep.jsonl")
    p.add_argument("--output", default="results/judge_subsample.jsonl")
    p.add_argument("--k-per-cell", type=int, default=2,
                   help="trials per (domain, condition) cell to judge "
                        "(default 2 → 32 trials over 8 domains × 2 conds)")
    p.add_argument("--judge-model", default="gpt-4o",
                   help="OpenAI model for the judge (default gpt-4o)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()