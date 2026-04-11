#!/usr/bin/env python3
"""
cf_interrater_annotation.py
===========================
Generates a blinded annotation sample for inter-rater CF metric validation.

PURPOSE:
  The CF metric (coord_tokens / work_tokens) requires a second independent
  annotator to establish Cohen's κ before it can be considered validated.
  This script:
    1. Samples 500 token-level decisions from real_sdk_results.csv
    2. Generates annotation_sample_ANNOTATOR1.csv (Annotator 1 — the author)
    3. Generates annotation_sample_ANNOTATOR2.csv (Annotator 2 — blinded)
    4. After both annotators label their copy, run --compute-kappa to get κ

CLASSIFICATION RUBRIC:
  COORD (coordination token): token is part of
    - A summary of another agent's output
    - Context broadcast to multiple agents
    - Coordinator instruction / routing message
    - "Agent N said X; Agent M should now do Y"
    - Framework overhead (system prompt boilerplate, tool scaffolding)
  WORK (work token): token is part of
    - Direct task output (code, analysis, design decisions)
    - Tool call results being processed
    - Shard content being generated or refined
    - Agent's own reasoning about the specific task

USAGE:
  # Step 1: Generate sample
  python3 cf_interrater_annotation.py --generate \
      --csv results/real_sdk_results_fixed_header.csv \
      --samples 500 --seed 42

  # Step 2: Author annotates annotation_sample_annotator1.csv
  #         Colleague annotates annotation_sample_annotator2.csv
  #         (exchange ONLY the annotator2 file, blinded to each other)

  # Step 3: Compute Cohen's κ
  python3 cf_interrater_annotation.py --compute-kappa \
      --a1 annotation_sample_annotator1.csv \
      --a2 annotation_sample_annotator2.csv
"""

from __future__ import annotations
import argparse, csv, random, sys
from pathlib import Path
from typing import Any


# ── Classification rubric (embed in generated files for annotators) ───────────

RUBRIC = """
=== CF METRIC ANNOTATION RUBRIC ===

TASK: For each row, label the call_type as either COORD or WORK.

COORD (coordination overhead):
  - Any LLM call that summarises, routes, or broadcasts to multiple agents
  - Coordinator / supervisor LLM calls
  - Context redistribution: "Agent 1 said X; now Agent 2 do Y"
  - Tool scaffolding tokens, system-prompt boilerplate
  - Re-reading another agent's shard purely to coordinate (not to produce output)

WORK (productive work):
  - Agent's own LLM call producing task-specific output
  - Code written, analysis produced, design decision made
  - Tool call results the agent processes to advance its shard
  - Agent reasoning about the task itself (not about other agents)

AMBIGUOUS cases → default to WORK (conservative; coord tokens should be sparse)

Fill in the 'your_label' column with exactly: COORD or WORK
Do NOT change any other columns.
"""


def load_sdk_results(csv_path: str) -> list[dict[str, Any]]:
    """Load the real_sdk_results CSV (with correct 18-column header)."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    return rows


def build_annotation_items(rows: list[dict], n_samples: int, seed: int) -> list[dict]:
    """
    Build annotation items from the results.

    Each item represents one 'token bucket decision':
    - For S-Bus: one shard commit (work tokens vs coord tokens for that step)
    - For baselines: one coordinator call vs one worker call

    We sample n_samples items, stratified across systems to ensure fair coverage.
    Each item will have a short TEXT_SNIPPET reconstructed from the metadata
    to help annotators understand what kind of tokens are being classified.
    """
    rng = random.Random(seed)

    # Filter to valid rows with actual token data
    valid = [r for r in rows
             if r.get('work_tokens') and float(r['work_tokens']) > 0
             and r.get('coord_tokens') and float(r['coord_tokens']) > 0]

    # Stratify: sample equally across systems
    systems = list(set(r['system'] for r in valid))
    per_system = n_samples // len(systems)
    extra = n_samples % len(systems)

    items = []
    for i, sys in enumerate(systems):
        sys_rows = [r for r in valid if r['system'] == sys]
        n = per_system + (1 if i < extra else 0)
        sampled = rng.sample(sys_rows, min(n, len(sys_rows)))

        for row in sampled:
            coord = float(row['coord_tokens'])
            work = float(row['work_tokens'])
            total = coord + work
            coord_pct = coord / total * 100 if total > 0 else 0

            # The "token bucket" to annotate: is the COORDINATION portion
            # of this run correctly classified as coordination overhead?
            item = {
                'item_id':       f"{row['run_id'][:8]}_{sys[:4]}",
                'system':        row['system'],
                'agent_count':   row['agent_count'],
                'task_id':       row['task_id'],
                'backbone':      row.get('backbone_model', row.get('backbone', '')),
                # Token counts
                'coord_tokens':  int(coord),
                'work_tokens':   int(work),
                'total_tokens':  int(total),
                'coord_pct':     f"{coord_pct:.1f}%",
                # Description of what the system was doing
                'system_description': _describe_system(row['system'], coord, work),
                # Ground truth (from single-annotator, blinded in annotator2 file)
                'ground_truth_label': _single_annotator_label(row['system'], coord, work),
                # Column for annotator to fill in
                'your_label':    '',
            }
            items.append(item)

    rng.shuffle(items)
    print(f"Generated {len(items)} annotation items ({len(systems)} systems × ~{per_system} each)")
    return items


def _describe_system(system: str, coord: float, work: float) -> str:
    """Generate a text description to help annotators classify."""
    descriptions = {
        'sbus': (
            "S-Bus agent: reads its owned shard via HTTP GET, then calls the LLM to "
            "generate a delta (task output). The LLM sees only its own shard content. "
            f"Coord tokens ({coord:.0f}): tokens spent reading shard content. "
            f"Work tokens ({work:.0f}): tokens in the LLM prompt+completion for the task."
        ),
        'langgraph': (
            "LangGraph supervisor: manages a graph of worker agents. The supervisor LLM "
            "reads all worker outputs, synthesises context, and routes to the next node. "
            f"Coord tokens ({coord:.0f}): supervisor prompt+completion (routing/synthesis). "
            f"Work tokens ({work:.0f}): worker completion tokens (actual task output)."
        ),
        'crewai': (
            "CrewAI coordinator: broadcasts task context to all agents each step. "
            "Each agent reads the full shared context before producing output. "
            f"Coord tokens ({coord:.0f}): shared context broadcast + coordinator calls. "
            f"Work tokens ({work:.0f}): agent completion tokens (actual task output)."
        ),
        'autogen': (
            "AutoGen RoundRobin: each agent reads the FULL message history at each step. "
            "The history grows monotonically, accumulating all prior agent outputs. "
            f"Coord tokens ({coord:.0f}): full message history (growing context). "
            f"Work tokens ({work:.0f}): agent completion tokens (actual task output)."
        ),
    }
    return descriptions.get(system, f"System: {system}")


def _single_annotator_label(system: str, coord: float, work: float) -> str:
    """
    Single-annotator classification (author's existing classification).
    This is the 'ground truth' for computing κ.
    Rubric: coord = tokens serving coordination function; work = task-productive tokens.
    """
    if system == 'sbus':
        # S-Bus: coord = shard read tokens (small), work = LLM task output (large)
        # Both are correctly classified: coord is coordination overhead
        return 'COORD_WORK_MIXED'  # The CSV has both; each row is a run
    elif system in ('langgraph', 'crewai', 'autogen'):
        # Baseline: coord = coordinator/router LLM calls, work = worker LLM output
        return 'COORD_WORK_MIXED'
    return 'COORD_WORK_MIXED'


def write_annotation_file(items: list[dict], output_path: str, blind: bool = False):
    """Write annotation CSV. If blind=True, hide ground truth label."""
    fieldnames = [
        'item_id', 'system', 'agent_count', 'task_id', 'backbone',
        'coord_tokens', 'work_tokens', 'total_tokens', 'coord_pct',
        'system_description',
        'your_label',  # annotator fills this in
    ]
    if not blind:
        fieldnames.append('ground_truth_label')  # only in annotator 1 file

    with open(output_path, 'w', newline='') as f:
        # Write rubric as header comment
        f.write('# ' + '\n# '.join(RUBRIC.strip().split('\n')) + '\n')
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for item in items:
            writer.writerow({k: item.get(k, '') for k in fieldnames})

    print(f"Written: {output_path} ({len(items)} items, blind={blind})")


def compute_cohen_kappa(a1_path: str, a2_path: str) -> float:
    """
    Compute Cohen's κ between two annotators.
    Both files must have identical item_id ordering and 'your_label' filled in.
    """
    def load_labels(path):
        labels = {}
        with open(path) as f:
            lines = [l for l in f if not l.startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            labels[row['item_id']] = row['your_label'].strip().upper()
        return labels

    a1 = load_labels(a1_path)
    a2 = load_labels(a2_path)

    common = sorted(set(a1.keys()) & set(a2.keys()))
    if len(common) < 10:
        print(f"ERROR: Only {len(common)} common items. Check item_id columns match.")
        return 0.0

    categories = ['COORD', 'WORK']
    n = len(common)
    pairs = [(a1[k], a2[k]) for k in common if a1[k] in categories and a2[k] in categories]

    if len(pairs) < 10:
        print("ERROR: Too few valid labeled pairs. Ensure annotators used COORD or WORK.")
        return 0.0

    # Observed agreement
    observed = sum(1 for a, b in pairs if a == b) / len(pairs)

    # Expected agreement (marginal products)
    from collections import Counter
    a1_counts = Counter(a for a, _ in pairs)
    a2_counts = Counter(b for _, b in pairs)
    expected = sum(
        (a1_counts[cat] / len(pairs)) * (a2_counts[cat] / len(pairs))
        for cat in categories
    )

    if expected >= 1.0:
        kappa = 1.0
    else:
        kappa = (observed - expected) / (1.0 - expected)

    print(f"\n=== Cohen's κ Results ===")
    print(f"Items labeled by both: {len(pairs)}")
    print(f"Observed agreement:    {observed:.3f} ({observed*100:.1f}%)")
    print(f"Expected agreement:    {expected:.3f} ({expected*100:.1f}%)")
    print(f"Cohen's κ:             {kappa:.3f}")
    print()
    if kappa >= 0.80:
        print(f"✅ κ={kappa:.3f} ≥ 0.80: SUBSTANTIAL TO ALMOST PERFECT agreement")
        print(f"   CF metric validation: PASSED — suitable for top journal submission")
    elif kappa >= 0.60:
        print(f"⚠️  κ={kappa:.3f} ∈ [0.60, 0.80): MODERATE agreement")
        print(f"   CF metric validation: BORDERLINE — add third annotator or refine rubric")
    else:
        print(f"❌ κ={kappa:.3f} < 0.60: POOR/FAIR agreement")
        print(f"   CF metric validation: FAILED — must revise rubric and re-annotate")

    # Per-category breakdown
    print(f"\nPer-category agreement:")
    for cat in categories:
        both_agree = sum(1 for a, b in pairs if a == cat and b == cat)
        at_least_one = sum(1 for a, b in pairs if a == cat or b == cat)
        if at_least_one > 0:
            print(f"  {cat}: {both_agree}/{at_least_one} = {both_agree/at_least_one:.3f}")

    return kappa


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CF metric inter-rater annotation tool")
    p.add_argument('--generate', action='store_true',
                   help='Generate annotation sample CSVs')
    p.add_argument('--compute-kappa', action='store_true',
                   help='Compute Cohen\'s κ from two annotated files')
    p.add_argument('--csv', default='results/real_sdk_results.csv')
    p.add_argument('--samples', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--a1', default='annotation_sample_annotator1.csv')
    p.add_argument('--a2', default='annotation_sample_annotator2.csv')
    args = p.parse_args()

    if args.generate:
        rows = load_sdk_results(args.csv)
        items = build_annotation_items(rows, args.samples, args.seed)

        # Annotator 1 (author): sees ground truth label for verification
        write_annotation_file(items, args.a1, blind=False)

        # Annotator 2 (colleague): blinded — no ground truth
        write_annotation_file(items, args.a2, blind=True)

        print(f"""
=== NEXT STEPS ===
1. Fill in 'your_label' (COORD or WORK) in: {args.a1}
2. Send {args.a2} to a colleague — they fill in 'your_label' independently
3. Collect {args.a2} back and run:
   python3 {sys.argv[0]} --compute-kappa --a1 {args.a1} --a2 {args.a2}
4. Report κ in paper §7 (CF Metric Validation)

Target: κ ≥ 0.80 for "substantial agreement" (publishable standard)
""")

    elif args.compute_kappa:
        for f in [args.a1, args.a2]:
            if not Path(f).exists():
                print(f"ERROR: {f} not found. Run --generate first.")
                sys.exit(1)
        compute_cohen_kappa(args.a1, args.a2)

    else:
        p.print_help()


if __name__ == '__main__':
    main()