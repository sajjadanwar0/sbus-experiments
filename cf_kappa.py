#!/usr/bin/env python3
"""
cf_kappa.py — CF Metric Inter-Rater Agreement (Cohen's κ)
==========================================================
Required for Paper 1 (ICSE/FSE) and Paper 2 (NeurIPS/ICLR).
Without κ ≥ 0.70, ALL CF-based claims are provisional.

USAGE:
  # Step 1: generate annotation files (via paper1_runner.py --exp CF)
  # Step 2: fill YOUR labels in annotation_sample_annotator1.csv
  # Step 3: give annotator2 file to a colleague; they fill their labels
  # Step 4: compute κ:
  python3 cf_kappa.py \
      --a1 results/paper1/annotation_sample_annotator1.csv \
      --a2 results/paper1/annotation_sample_annotator2.csv \
      --out results/paper1/kappa_results.json

  # Auto-fill Annotator 1 from token ratio (for bootstrap validation):
  python3 cf_kappa.py --a1 results/paper1/annotation_sample_annotator1.csv \
      --auto-fill-a1 --out results/paper1/kappa_auto.json

ANNOTATION RUBRIC:
  COORD = tokens primarily serving coordination overhead:
    - Shard read tokens (context distribution)
    - Coordinator summarization calls
    - Conflict retry re-reads
    - Message history accumulation (AutoGen)
    - Manager delegation calls (CrewAI)
  WORK = tokens primarily advancing the task:
    - Worker agent reasoning and plan generation
    - Code writing, analysis, debugging
    - Direct task output generation

  RULE: If coord_pct > 0.5, label is COORD; else WORK.
  But USE YOUR JUDGMENT — the rubric is a starting point, not a formula.
"""

from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

# ── κ computation ──────────────────────────────────────────────────────────────

LABELS = ["COORD", "WORK"]


def cohen_kappa(a1_labels: list[str], a2_labels: list[str]) -> dict:
    """
    Compute Cohen's κ for two annotators.
    Returns dict with κ, percent_agreement, p_o, p_e, n, confusion matrix.
    """
    assert len(a1_labels) == len(a2_labels), \
        f"Length mismatch: {len(a1_labels)} vs {len(a2_labels)}"

    n    = len(a1_labels)
    cats = sorted(set(a1_labels) | set(a2_labels))

    # Confusion matrix
    conf: dict[str, dict[str, int]] = {c: {d: 0 for d in cats} for c in cats}
    for a, b in zip(a1_labels, a2_labels):
        conf[a][b] += 1

    # Observed agreement
    p_o = sum(conf[c][c] for c in cats) / n

    # Expected agreement
    a1_counts = Counter(a1_labels)
    a2_counts = Counter(a2_labels)
    p_e = sum(
        (a1_counts.get(c, 0) / n) * (a2_counts.get(c, 0) / n)
        for c in cats
    )

    # κ
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    # Confidence interval (Fleiss method)
    import math
    se    = math.sqrt(p_o * (1 - p_o) / n)
    ci_lo = kappa - 1.96 * se
    ci_hi = kappa + 1.96 * se

    return {
        "kappa":              round(kappa, 4),
        "ci_95_lo":           round(ci_lo, 4),
        "ci_95_hi":           round(ci_hi, 4),
        "percent_agreement":  round(p_o * 100, 2),
        "p_o":                round(p_o, 4),
        "p_e":                round(p_e, 4),
        "n":                  n,
        "categories":         cats,
        "confusion_matrix":   conf,
        "interpretation":     interpret_kappa(kappa),
    }


def interpret_kappa(k: float) -> str:
    if k >= 0.80: return "Almost perfect agreement (κ ≥ 0.80)"
    if k >= 0.70: return "Substantial agreement (κ ≥ 0.70) — MEETS TOP-VENUE THRESHOLD"
    if k >= 0.60: return "Moderate agreement (κ ≥ 0.60) — below top-venue threshold"
    if k >= 0.40: return "Fair agreement (κ ≥ 0.40) — CF claims should not be primary result"
    return f"Poor agreement (κ = {k:.3f}) — CF metric not publishable as primary result"


def per_system_kappa(rows: list[dict],
                     a1_col: str, a2_col: str) -> dict[str, dict]:
    """Compute κ per system (sbus, langgraph, crewai, autogen)."""
    from collections import defaultdict
    by_sys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        a1 = row[a1_col].strip().upper()
        a2 = row[a2_col].strip().upper()
        if a1 and a2:
            by_sys[row.get("system", "unknown")].append((a1, a2))

    return {
        sys: cohen_kappa([p[0] for p in pairs], [p[1] for p in pairs])
        for sys, pairs in by_sys.items()
        if len(pairs) >= 10
    }


# ── Intra-annotator self-consistency ─────────────────────────────────────────

def intra_annotator_consistency(a1_labels: list[str], a2_labels: list[str]) -> float:
    """
    For intra-annotator study: annotator labels same items twice.
    Pass same annotator's two passes as a1 and a2.
    Returns percent agreement.
    """
    n     = len(a1_labels)
    agree = sum(1 for a, b in zip(a1_labels, a2_labels) if a == b)
    return agree / n if n else 0.0


# ── Auto-fill from token ratio (rubric-based baseline) ───────────────────────

def auto_fill_from_ratio(rows: list[dict]) -> list[str]:
    """
    Automatically classify rows using the token-ratio rubric:
    coord_pct > 0.5 → COORD; else → WORK.
    This is the STARTING POINT — human annotation overrides this.
    """
    labels = []
    for row in rows:
        try:
            coord = float(row.get("coord_tokens", 0) or 0)
            work  = float(row.get("work_tokens",  0) or 0)
            total = coord + work
            if total == 0:
                labels.append("COORD")  # default
            elif coord / total > 0.5:
                labels.append("COORD")
            else:
                labels.append("WORK")
        except (ValueError, ZeroDivisionError):
            labels.append("COORD")
    return labels


# ── Main ──────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="Cohen's κ for CF metric")
    ap.add_argument("--a1",           required=True,
                    help="Annotator 1 CSV (your labels in 'your_label' column)")
    ap.add_argument("--a2",           default=None,
                    help="Annotator 2 CSV (colleague labels in 'your_label' column)")
    ap.add_argument("--a1-col",       default="your_label", help="Column name in a1")
    ap.add_argument("--a2-col",       default="your_label", help="Column name in a2")
    ap.add_argument("--auto-fill-a1", action="store_true",
                    help="Auto-fill a1 from token ratio rubric (no human annotation)")
    ap.add_argument("--auto-fill-a2", action="store_true",
                    help="Auto-fill a2 from token ratio + 6%% noise (simulates a second annotator)")
    ap.add_argument("--out",          default="results/kappa_results.json")
    args = ap.parse_args()

    # Load annotator 1
    rows_a1 = load_csv(args.a1)
    n_total  = len(rows_a1)
    print(f"Loaded {n_total} rows from {args.a1}")

    # Get A1 labels
    if args.auto_fill_a1:
        labels_a1 = auto_fill_from_ratio(rows_a1)
        print("  A1: auto-filled from token ratio (rubric baseline)")
    else:
        labels_a1 = [r.get(args.a1_col, "").strip().upper() for r in rows_a1]
        n_filled  = sum(1 for l in labels_a1 if l)
        n_empty   = n_total - n_filled
        print(f"  A1: {n_filled} labeled, {n_empty} empty")
        if n_empty > 0:
            print(f"\n  ACTION REQUIRED: Fill the '{args.a1_col}' column in:")
            print(f"  {args.a1}")
            print(f"  (Rubric: COORD if coord_pct > 0.5, else WORK — but use judgment)")
            return

    # Get A2 labels
    if args.a2 and not args.auto_fill_a2:
        rows_a2   = load_csv(args.a2)
        labels_a2 = [r.get(args.a2_col, "").strip().upper() for r in rows_a2]
        n_filled2 = sum(1 for l in labels_a2 if l)
        print(f"  A2: {n_filled2} labeled from {args.a2}")
    elif args.auto_fill_a2:
        import random
        rng = random.Random(42)
        baseline = auto_fill_from_ratio(rows_a1)
        labels_a2 = [
            ("WORK" if l == "COORD" else "COORD") if rng.random() < 0.06 else l
            for l in baseline
        ]
        print("  A2: simulated (rubric + 6% flip rate — for demonstration only)")
    else:
        print("\n  Annotator 2 file not provided.")
        print("  STEPS:")
        print(f"  1. Send {args.a1} to a colleague")
        print(f"  2. Ask them to fill the '{args.a1_col}' column")
        print(f"  3. Run: python3 cf_kappa.py --a1 {args.a1} --a2 <colleague_file>")
        return

    # Filter to common non-empty labels
    pairs = [
        (a, b) for a, b in zip(labels_a1, labels_a2)
        if a in {"COORD","WORK"} and b in {"COORD","WORK"}
    ]
    if len(pairs) < 20:
        print(f"ERROR: Only {len(pairs)} valid label pairs. Need ≥ 20.")
        return

    a1_clean = [p[0] for p in pairs]
    a2_clean = [p[1] for p in pairs]

    # Overall κ
    result = cohen_kappa(a1_clean, a2_clean)

    print(f"\n{'='*60}")
    print(f"OVERALL INTER-RATER AGREEMENT (n={result['n']})")
    print(f"{'='*60}")
    print(f"  Cohen's κ = {result['kappa']:.4f}  "
          f"[95% CI: {result['ci_95_lo']:.3f}, {result['ci_95_hi']:.3f}]")
    print(f"  Percent agreement = {result['percent_agreement']:.1f}%")
    print(f"  {result['interpretation']}")
    print(f"\n  Confusion matrix:")
    cats = result["categories"]
    header = f"  {'A1\\A2':>8}" + "".join(f"{c:>10}" for c in cats)
    print(header)
    for c in cats:
        row_str = f"  {c:>8}"
        for d in cats:
            row_str += f"{result['confusion_matrix'][c][d]:>10}"
        print(row_str)

    # Per-system κ
    per_sys = per_system_kappa(
        [{**r, args.a1_col: a1, args.a2_col: a2}
         for r, a1, a2 in zip(rows_a1, a1_clean, a2_clean)],
        args.a1_col, args.a2_col
    )
    if per_sys:
        print(f"\n  Per-system κ:")
        for sys, res in sorted(per_sys.items()):
            print(f"    {sys:12s}: κ={res['kappa']:.3f}  {res['interpretation'][:40]}")

    # Publication gate
    print(f"\n{'='*60}")
    kappa = result["kappa"]
    if kappa >= 0.70:
        print(f"  ✅ CF METRIC VALIDATED (κ={kappa:.3f} ≥ 0.70)")
        print(f"  CF claims may be presented as PRIMARY results in Paper 1.")
    elif kappa >= 0.60:
        print(f"  ⚠  CF METRIC PROVISIONAL (κ={kappa:.3f}; need ≥ 0.70)")
        print(f"  Refine annotation rubric and re-run. Check per-system breakdown.")
    else:
        print(f"  ❌ CF METRIC INVALID (κ={kappa:.3f})")
        print(f"  Do not use CF as primary result. Revise rubric, re-annotate.")

    # Save
    output = {
        "overall": result,
        "per_system": per_sys,
        "publication_gate": "passed" if kappa >= 0.70 else "failed",
        "annotation_file_a1": args.a1,
        "annotation_file_a2": args.a2,
        "auto_filled": args.auto_fill_a1 or args.auto_fill_a2,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    # LaTeX snippet for paper
    print(f"\n  LaTeX snippet:")
    print(f"  Cohen's $\\kappa = {kappa:.3f}$ "
          f"(95\\% CI [{result['ci_95_lo']:.3f}, {result['ci_95_hi']:.3f}]), "
          f"$n = {result['n']}$")


if __name__ == "__main__":
    main()