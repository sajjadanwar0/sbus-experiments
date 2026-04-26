import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from typing import List, Dict, Any

from openai import AsyncOpenAI

from harness import run_trial
from evaluate import evaluate_trial
from domains import DOMAINS

DEFAULT_BASE_URL = "http://127.0.0.1:7000"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_N_STEPS = 6
RESULTS_DIR = "results"
JSONL_PATH = os.path.join(RESULTS_DIR, "workload_b_sweep.jsonl")
CSV_PATH = os.path.join(RESULTS_DIR, "workload_b_sweep.csv")


CSV_FIELDS = [
    "trial_id", "domain", "condition", "trial_idx", "n_steps", "elapsed_s",
    "n_commit_200", "n_commit_409", "n_commit_410", "n_commit_other",
    "n_rejections_total", "n_llm_errors", "total_commit_attempts",
    "view_divergent_commits", "view_checked_commits", "view_divergence_rate",
    "server_ori_enabled",
    "coherence_rate", "total_claimed_references", "coherent_references",
    "broken_references", "passed",
]


def _flatten_for_csv(trial_result: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trial_id": trial_result["trial_id"],
        "domain": trial_result["domain"],
        "condition": trial_result["condition"],
        "trial_idx": trial_result["trial_idx"],
        "n_steps": trial_result["n_steps"],
        "elapsed_s": trial_result["elapsed_s"],
        **trial_result["metrics"],
        "coherence_rate": round(evaluation["coherence_rate"], 4),
        "total_claimed_references": evaluation["total_claimed_references"],
        "coherent_references": evaluation["coherent_references"],
        "broken_references": evaluation["broken_references"],
        "passed": int(evaluation["pass"]),
    }


async def run_sweep(args) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set")
    print(f"OPENAI_API_KEY length = {len(api_key)}")

    client = AsyncOpenAI(api_key=api_key)

    domain_names = args.domains if args.domains else [d["name"] for d in DOMAINS]
    conditions = args.conditions

    cells = [(dn, c) for dn in domain_names for c in conditions]
    total_trials = len(cells) * args.n
    estimated_cost = 0.022 * total_trials
    estimated_minutes = total_trials * 0.25  # rough — depends on parallelism within trial
    print(f"Planned: {len(cells)} cells × {args.n} trials = {total_trials} trials, "
          f"~{estimated_minutes:.0f} min, ~${estimated_cost:.2f}")

    jsonl_file = open(JSONL_PATH, "a")
    csv_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    csv_file = open(CSV_PATH, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not csv_exists:
        csv_writer.writeheader()

    cell_idx = 0
    cell_results: Dict[str, List[Dict[str, Any]]] = {}
    for domain_name, condition in cells:
        cell_idx += 1
        cell_key = f"{domain_name}|{condition}"
        cell_results[cell_key] = []
        print(f"\n----- cell {cell_idx}/{len(cells)}: {cell_key} -----")
        for trial_idx in range(args.n):
            try:
                trial_result = await run_trial(
                    base_url=args.url,
                    domain_name=domain_name,
                    condition=condition,
                    trial_idx=trial_idx,
                    n_steps=args.n_steps,
                    openai_client=client,
                    model=args.model,
                )
            except Exception as e:
                print(f"  [trial {trial_idx}] CRASHED: {type(e).__name__}: {e}")
                continue

            evaluation = evaluate_trial(trial_result)
            trial_result["evaluation"] = evaluation

            m = trial_result["metrics"]
            print(
                f"  [trial {trial_idx:>2}] "
                f"200={m['n_commit_200']:<2} 409={m['n_commit_409']:<2} "
                f"410={m['n_commit_410']:<2} llm_err={m['n_llm_errors']} "
                f"coh={evaluation['coherence_rate']:.2f} "
                f"({evaluation['coherent_references']}/{evaluation['total_claimed_references']}) "
                f"pass={int(evaluation['pass'])} {trial_result['elapsed_s']}s"
            )

            # Persist
            jsonl_file.write(json.dumps(trial_result) + "\n")
            jsonl_file.flush()
            csv_writer.writerow(_flatten_for_csv(trial_result, evaluation))
            csv_file.flush()
            cell_results[cell_key].append(trial_result)

    jsonl_file.close()
    csv_file.close()

    print("\n" + "=" * 80)
    print(f"{'CELL':<48} {'PASS':>10} {'COHERENCE':>12} {'409s':>8}")
    print("=" * 80)
    for cell_key, trials in cell_results.items():
        if not trials:
            continue
        n_pass = sum(1 for t in trials if t["evaluation"]["pass"])
        avg_coh = sum(t["evaluation"]["coherence_rate"] for t in trials) / len(trials)
        avg_409 = sum(t["metrics"]["n_commit_409"] for t in trials) / len(trials)
        print(f"{cell_key:<48} {n_pass}/{len(trials):<8} {avg_coh:>11.3f} {avg_409:>7.1f}")
    print("=" * 80)


def main() -> None:
    p = argparse.ArgumentParser(description="Workload B: data-pipeline planning")
    p.add_argument("--n", type=int, default=10,
                   help="trials per cell (default 10)")
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS,
                   help="coordination steps per trial (default 6)")
    p.add_argument("--domains", nargs="+", default=None,
                   help="subset of domain names to run (default: all 8)")
    p.add_argument("--conditions", nargs="+", default=["ori_on", "ori_off"],
                   choices=["ori_on", "ori_off"],
                   help="conditions to test (default both)")
    p.add_argument("--url", default=DEFAULT_BASE_URL,
                   help=f"S-Bus server base URL (default {DEFAULT_BASE_URL})")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"OpenAI model (default {DEFAULT_MODEL})")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(run_sweep(args))


if __name__ == "__main__":
    main()
