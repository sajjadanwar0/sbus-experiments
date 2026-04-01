"""
s50_analysis.py
─────────────────────────────────────────────────────────────────────────────
Standalone S@50 statistical analysis with Wilson score 95% CIs and
Fisher's exact tests.

Fills the gap: the original paper reported S@50 as bare percentages with
no uncertainty quantification.  With n=5 tasks, CIs are wide and must be
shown honestly.

Run:
  python3 s50_analysis.py --csv results/real_sdk_results.csv
  python3 s50_analysis.py --csv results/real_sdk_results.csv --latex
"""

import argparse
import csv
import math
from collections import defaultdict
from scipy.stats import fisher_exact


# ─── Wilson score CI ──────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p     = k / n
    denom = 1 + z**2 / n
    ctr   = (p + z**2 / (2 * n)) / denom
    marg  = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, ctr - marg), min(1.0, ctr + marg)


# ─── load CSV ─────────────────────────────────────────────────────────────────

def load(csv_path: str) -> dict:
    """Returns {(system, n_agents): [0/1, ...]} mapping."""
    data = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("excluded", "False").lower() == "true":
                continue
            system   = row["system"]
            n_agents = int(row["n_agents"])
            success  = row["success"].lower() == "true"
            data[(system, n_agents)].append(int(success))
    return dict(data)


# ─── main analysis ────────────────────────────────────────────────────────────

def analyse(data: dict, latex: bool = False):
    all_n = sorted(set(k[1] for k in data.keys()))
    all_s = sorted(set(k[0] for k in data.keys()))

    # Table: S@50 with Wilson CIs
    print("\n" + "="*75)
    print("S@50 with Wilson 95% CIs")
    print("="*75)
    print(f"{'System':<15} {'N':>4} {'Succ':>6} {'n':>5} {'S@50':>7} "
          f"{'95% CI':>16}  {'Note'}")
    print("-"*75)

    for n in all_n:
        for system in all_s:
            vals = data.get((system, n), [])
            if not vals:
                continue
            k_s  = sum(vals)
            n_   = len(vals)
            p    = k_s / n_
            lo, hi = wilson_ci(k_s, n_)
            ci_str = f"[{lo:.0%} – {hi:.0%}]"

            # Flag wide CIs (the honest note reviewers need to see)
            width = hi - lo
            note  = ""
            if width > 0.5:
                note = "← wide CI: n too small for strong inference"
            elif width > 0.3:
                note = "← moderate CI"

            print(f"{system:<15} {n:>4} {k_s:>6} {n_:>5} {p:>7.0%} {ci_str:>16}  {note}")

    # Fisher's exact tests between S-Bus and each baseline
    print("\n" + "="*75)
    print("Fisher's exact test: S-Bus vs baselines (S@50, one-sided)")
    print("="*75)
    print(f"{'Comparison':<40} {'p-value':>10}  {'Interpretation'}")
    print("-"*75)

    for n in all_n:
        sbus_vals = data.get(("sbus", n), [])
        if not sbus_vals:
            continue
        sbus_k = sum(sbus_vals); sbus_n = len(sbus_vals)

        for baseline in ["autogen", "langgraph", "crewai"]:
            base_vals = data.get((baseline, n), [])
            if not base_vals:
                continue
            base_k = sum(base_vals); base_n = len(base_vals)

            table   = [[sbus_k, sbus_n - sbus_k], [base_k, base_n - base_k]]
            _, p    = fisher_exact(table, alternative="greater")
            label   = f"sbus({sbus_k}/{sbus_n}) > {baseline}({base_k}/{base_n}) N={n}"
            if p < 0.05:
                interp = f"p={p:.4f} *  significant"
            else:
                interp = (
                    f"p={p:.4f} ns  [IMPORTANT: not significant at n={sbus_n} tasks. "
                    f"Report honestly — larger n needed for strong S@50 claim.]"
                )
            print(f"{label:<40} {p:>10.4f}  {interp}")

    # Interpretive note for the paper
    print("\n" + "="*75)
    print("Guidance for paper §7 (S@50 reporting):")
    print("="*75)
    print("""
The Wilson CIs above must be included in Table 8 of the revised paper.
Do NOT report S@50 as bare percentages without uncertainty bounds.

If Fisher's exact tests are non-significant (p > 0.05), the paper should
acknowledge this explicitly:
  "At n=5 tasks, the S@50 difference between S-Bus and [baseline] is
   directionally consistent with the CWR findings but does not reach
   statistical significance (p=X.XX, Fisher's exact). A larger task
   sample is required for a strong S@50 claim; we report the 50-step
   results as preliminary evidence and identify extended evaluation
   as the primary open item."

This is the honest framing. Reviewers at IEEE TPDS will respect it.
Claiming significance where n=5 does not support it will be caught.
""")

    # LaTeX table
    if latex:
        print("\n% ─── LaTeX table for revised Table 8 ───────────────────────")
        print("\\begin{table}[t]")
        print("\\caption{S@50 results with 95\\% Wilson confidence intervals.}")
        print("\\begin{tabular}{llrrrr}")
        print("\\toprule")
        print("System & N & Successes & n & S@50 & 95\\% CI \\\\ \\midrule")
        for n in all_n:
            for system in all_s:
                vals = data.get((system, n), [])
                if not vals:
                    continue
                k_s  = sum(vals); n_ = len(vals)
                p    = k_s / n_
                lo, hi = wilson_ci(k_s, n_)
                print(
                    f"{system} & {n} & {k_s} & {n_} & "
                    f"{p:.0%} & [{lo:.0%}, {hi:.0%}] \\\\"
                )
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",   required=True, help="Path to real_sdk_results.csv")
    ap.add_argument("--latex", action="store_true", help="Also print LaTeX table")
    args = ap.parse_args()

    data = load(args.csv)
    if not data:
        print("No valid rows found in CSV. Run sdk_compare.py first.")
        return
    analyse(data, latex=args.latex)


if __name__ == "__main__":
    main()