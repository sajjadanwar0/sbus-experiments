#!/usr/bin/env python3
"""
deduplicate_sjv4.py
===================
Cleans the merged SJ-v4 CSV by removing duplicate rows,
then runs the full statistical analysis.

WHY DUPLICATES EXIST:
  The first 20-task experiment attempt crashed (NameError/TypeError)
  but tasks 0-4 were already running as background processes.
  When the second attempt started, both old and new processes wrote
  to the same result_0 through result_4 CSV files simultaneously,
  creating duplicate run_id entries.

USAGE:
  python3 deduplicate_sjv4.py results/sj_v4_results.csv
"""

import csv
import sys
import os
from collections import defaultdict

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: pip install scipy for p-values")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 deduplicate_sjv4.py results/sj_v4_results.csv")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = input_path.replace(".csv", "_clean.csv")

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    # ── Load all rows ─────────────────────────────────────────────────────────
    with open(input_path) as f:
        reader  = csv.DictReader(f)
        fields  = reader.fieldnames
        all_rows = list(reader)

    print(f"Loaded {len(all_rows)} rows from {input_path}")

    # ── Deduplicate by run_id ─────────────────────────────────────────────────
    # Each run has a unique run_id (uuid hex). Duplicate rows = same run_id.
    seen_run_ids = set()
    clean_rows   = []
    dupes        = 0

    for row in all_rows:
        rid = row.get("run_id", "")
        if rid and rid in seen_run_ids:
            dupes += 1
        else:
            seen_run_ids.add(rid)
            clean_rows.append(row)

    print(f"Duplicates removed: {dupes}")
    print(f"Clean rows: {len(clean_rows)}")

    # ── Also remove rows with empty verdict ───────────────────────────────────
    valid_rows = [r for r in clean_rows if r.get("verdict") in
                  ("CORRECT", "INCOMPLETE", "CORRUPTED")]
    invalid    = len(clean_rows) - len(valid_rows)
    if invalid:
        print(f"Rows with invalid/empty verdict removed: {invalid}")
    print(f"Valid rows for analysis: {len(valid_rows)}")
    print()

    # ── Save clean CSV ────────────────────────────────────────────────────────
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(valid_rows)
    print(f"Clean CSV saved: {output_path}")
    print()

    # ── Statistical analysis ──────────────────────────────────────────────────
    counts = defaultdict(lambda: {"corrupted":0,"correct":0,"incomplete":0,"total":0})
    for row in valid_rows:
        cond = row.get("condition","")
        counts[cond]["total"] += 1
        v = row.get("verdict","")
        if v == "CORRUPTED":   counts[cond]["corrupted"]  += 1
        elif v == "CORRECT":   counts[cond]["correct"]    += 1
        else:                  counts[cond]["incomplete"] += 1

    print("=" * 65)
    print("SJ-v4 RESULTS (deduplicated)")
    print("=" * 65)
    print()

    for cond in ["structural_fresh", "structural_stale"]:
        c = counts[cond]
        t = c["total"]
        if t == 0:
            print(f"  {cond}: NO DATA")
            continue
        cr = c["corrupted"] / t
        co = c["correct"]   / t
        print(f"  {cond:<25} n={t:4d} | "
              f"CORRECT={c['correct']:4d}({co*100:5.1f}%) | "
              f"INCOMPLETE={c['incomplete']:4d} | "
              f"CORRUPTED={c['corrupted']:4d}({cr*100:5.1f}%)")

    a = counts["structural_fresh"]
    b = counts["structural_stale"]
    p_a  = a["corrupted"] / max(1, a["total"])
    p_b  = b["corrupted"] / max(1, b["total"])
    diff = p_b - p_a

    print()
    print(f"  FRESH corruption:  {p_a*100:.2f}%  ({a['corrupted']}/{a['total']})")
    print(f"  STALE corruption:  {p_b*100:.2f}%  ({b['corrupted']}/{b['total']})")
    print(f"  R_hidden lift:     {diff*100:+.2f}pp")
    print()

    if HAS_SCIPY:
        k1, n1 = a["corrupted"], a["total"]
        k2, n2 = b["corrupted"], b["total"]
        if n1 > 0 and n2 > 0:
            _, p_fisher = scipy_stats.fisher_exact(
                [[k1, n1-k1], [k2, n2-k2]], alternative="less"
            )
            # Two-proportion z-test
            p_pool = (k1+k2) / (n1+n2)
            import math
            se = math.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
            z  = (p_a - p_b) / se if se > 0 else 0
            p_z = scipy_stats.norm.sf(abs(z)) * 2  # two-sided

            # 95% CI for difference
            se2    = math.sqrt(p_a*(1-p_a)/n1 + p_b*(1-p_b)/n2)
            ci_lo  = diff - 1.96 * se2
            ci_hi  = diff + 1.96 * se2

            print(f"  Fisher's exact (one-sided, STALE>FRESH): p = {p_fisher:.4f}")
            print(f"  Two-proportion z-test (two-sided):        p = {p_z:.4f}")
            print(f"  95% CI for lift: [{ci_lo*100:.1f}pp, {ci_hi*100:.1f}pp]")
            print()

            if p_fisher < 0.05:
                print("  ✅ SIGNIFICANT at α=0.05")
                print("  -> R_hidden stale context causes measurably more")
                print("     semantic corruption than fresh context.")
                print("  -> POS model VALIDATED empirically.")
            elif p_fisher < 0.10:
                print("  ⚠️  MARGINAL (0.05 < p < 0.10)")
                print("  -> Weak evidence of R_hidden corruption lift.")
                print("     Report as marginal; do not overclaim.")
            else:
                print("  ❌ NOT SIGNIFICANT (p > 0.10)")
                print("  -> No detectable semantic lift from stale context.")
                print("     Paper should frame as structural-only contribution.")
            print()

            # Power analysis
            from scipy.stats import norm
            effect = abs(diff)
            if effect > 0 and n1 > 0:
                z_alpha = 1.645  # one-sided α=0.05
                z_beta  = norm.ppf(0.80)   # 80% power
                p_bar   = (p_a + p_b) / 2
                n_needed = ((z_alpha + z_beta) / effect)**2 * 2 * p_bar*(1-p_bar)
                print(f"  Power analysis (to detect {effect*100:.1f}pp lift):")
                print(f"    n per condition needed (80% power): {n_needed:.0f}")
                print(f"    n you have per condition: {n1}")
                if n1 >= n_needed:
                    print(f"    -> ADEQUATELY POWERED ✓")
                else:
                    print(f"    -> UNDERPOWERED (need {n_needed-n1:.0f} more per condition)")
            print()

    # ── Per-task breakdown ────────────────────────────────────────────────────
    print("Per-task breakdown:")
    print(f"  {'Task':<35} {'Fresh':>8} {'Stale':>8} {'Lift':>8} {'n_f':>5} {'n_s':>5}")
    print("  " + "-"*70)

    task_counts = defaultdict(lambda: defaultdict(
        lambda: {"corrupted":0,"total":0}))
    for row in valid_rows:
        task_counts[row["task_id"]][row["condition"]]["total"] += 1
        if row["verdict"] == "CORRUPTED":
            task_counts[row["task_id"]][row["condition"]]["corrupted"] += 1

    for task, conds in sorted(task_counts.items()):
        f = conds["structural_fresh"]
        s = conds["structural_stale"]
        pf = f["corrupted"] / max(1, f["total"])
        ps = s["corrupted"] / max(1, s["total"])
        print(f"  {task[-33:]:<35} {pf*100:>7.1f}% {ps*100:>7.1f}% "
              f"{(ps-pf)*100:>+7.1f}pp {f['total']:>5} {s['total']:>5}")

    print()

    # ── Stale steps validation ────────────────────────────────────────────────
    print("Stale injection validation:")
    stale_rows  = [r for r in valid_rows if r["condition"] == "structural_stale"]
    zero_stale  = [r for r in stale_rows if r.get("n_stale_steps","0") == "0"]
    nonzero     = [r for r in stale_rows if r.get("n_stale_steps","0") != "0"]
    print(f"  Stale runs with n_stale_steps=0:   {len(zero_stale)}/{len(stale_rows)}")
    print(f"  Stale runs with n_stale_steps>0:   {len(nonzero)}/{len(stale_rows)}")
    if nonzero:
        vals = [int(r.get("n_stale_steps",0)) for r in nonzero]
        import statistics
        print(f"  Mean stale steps when injected: {statistics.mean(vals):.1f}")
    print()
    if len(zero_stale) == len(stale_rows):
        print("  WARNING: ALL stale runs have n_stale_steps=0")
        print("  This means the stale injection did NOT engage (same bug as SJ-v3)")
        print("  Check: injection_step > n_steps? Or stale agent still calling GET?")

    # ── Paper text ────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("PAPER TEXT (§9.19 Exp. SJ-v4)")
    print("=" * 65)
    print()
    n_tasks_run = len(task_counts)
    n_total     = len(valid_rows)
    n_fresh     = a["total"]
    n_stale     = b["total"]

    if HAS_SCIPY and n1 > 0 and n2 > 0:
        sig_str = (f"statistically significant ($p={p_fisher:.4f}$, Fisher's exact, one-sided)"
                   if p_fisher < 0.05 else
                   f"not statistically significant ($p={p_fisher:.4f}$)")
        print(f"Exp.~SJ-v4 ({n_tasks_run}~tasks, {n_fresh}~Fresh runs, {n_stale}~Stale runs;")
        print(f"GPT-4o-mini backbone, {counts['structural_fresh']['total']//n_tasks_run}~runs/condition/task):")
        print(f"Fresh ($R_{{\\text{{obs}}}}$, agent fetches current state): "
              f"${p_a*100:.1f}\\%$ semantic corruption;")
        print(f"Stale ($R_{{\\text{{hidden}}}}$ simulated, agent uses frozen snapshot "
              f"from step~0): ${p_b*100:.1f}\\%$ corruption.")
        print(f"Lift $= {diff*100:+.1f}$\\,pp (95\\% CI: "
              f"[${ci_lo*100:.1f}$\\,pp, ${ci_hi*100:.1f}$\\,pp]); {sig_str}.")
    print()
    print(f"Clean results: {output_path}")


if __name__ == "__main__":
    main()