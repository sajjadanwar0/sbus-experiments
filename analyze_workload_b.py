import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def wilson_ci(s: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p_hat = s / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z / denom * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def chi2_2x2(a_pos: int, a_neg: int, b_pos: int, b_neg: int) -> Tuple[float, float]:
    n = a_pos + a_neg + b_pos + b_neg
    if n == 0:
        return 0.0, 1.0
    rA, rB = a_pos + a_neg, b_pos + b_neg
    cP, cN = a_pos + b_pos, a_neg + b_neg
    if rA == 0 or rB == 0 or cP == 0 or cN == 0:
        return 0.0, 1.0
    e = [rA*cP/n, rA*cN/n, rB*cP/n, rB*cN/n]
    o = [a_pos, a_neg, b_pos, b_neg]
    chi2 = sum((oi-ei)**2/ei if ei > 0 else 0 for oi, ei in zip(o, e))
    z = math.sqrt(chi2)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return chi2, p


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def cell_summary(rows: List[Dict[str, str]]) -> None:
    cells: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        cells[(r["domain"], r["condition"])].append(r)

    print()
    print(f"{'CELL':<32} {'N':<4} {'PASS':<8} {'COMPLETE':<10} "
           f"{'REJ.RATE':<10} {'REJECTIONS':<12} {'200s':<8}")
    print("=" * 100)
    domains = sorted(set(r["domain"] for r in rows))
    for dn in domains:
        for cond in ["ori_on", "ori_off"]:
            key = (dn, cond)
            if key not in cells:
                continue
            cell = cells[key]
            n = len(cell)
            n_pass = sum(int(r["passed"]) for r in cell)
            avg_rej_rate = sum(float(r["coherence_rate"]) for r in cell) / n
            avg_rejections = sum(int(r["broken_references"]) for r in cell) / n
            avg_200 = sum(int(r["n_commit_200"]) for r in cell) / n
            n_steps = int(cell[0]["n_steps"])
            expected = 4 * n_steps
            completion = avg_200 / expected if expected else 0.0
            print(f"{dn+'|'+cond:<32} {n:<4} {n_pass}/{n:<6} "
                   f"{completion:<10.2f} {avg_rej_rate:<10.3f} "
                   f"{avg_rejections:<12.1f} {avg_200:<8.1f}")
        print()


def pairwise_rejection_tests(rows: List[Dict[str, str]]) -> List[Tuple]:
    print("=" * 100)
    print("PAIRWISE TESTS — server rejection rate (ori_on vs ori_off)")
    print("Substantive ORI claim: ori_on should have substantially HIGHER rejection rate")
    print("(every rejection = server detected and prevented a stale cross-shard read)")
    print("=" * 100)
    cells: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        cells[(r["domain"], r["condition"])].append(r)

    domains = sorted(set(r["domain"] for r in rows))
    results = []
    for dn in domains:
        on = cells.get((dn, "ori_on"), [])
        off = cells.get((dn, "ori_off"), [])
        if not on or not off:
            continue

        on_attempts = sum(int(r["total_claimed_references"]) for r in on)
        on_rejections = sum(int(r["broken_references"]) for r in on)
        off_attempts = sum(int(r["total_claimed_references"]) for r in off)
        off_rejections = sum(int(r["broken_references"]) for r in off)

        on_rate = on_rejections / on_attempts if on_attempts else 0
        off_rate = off_rejections / off_attempts if off_attempts else 0
        delta = on_rate - off_rate

        chi2, p = chi2_2x2(on_rejections, on_attempts - on_rejections,
                            off_rejections, off_attempts - off_rejections)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {dn:<22} on={on_rate:>5.0%} ({on_rejections}/{on_attempts}) "
               f"off={off_rate:>5.0%} ({off_rejections}/{off_attempts}) "
               f"Δ={delta:+5.0%}  chi2={chi2:>7.2f}  p={p:.6f}  {sig}")
        results.append((dn, delta, p, on_rate, off_rate))

    on_t = sum(int(r["total_claimed_references"]) for r in rows if r["condition"] == "ori_on")
    on_rj = sum(int(r["broken_references"]) for r in rows if r["condition"] == "ori_on")
    off_t = sum(int(r["total_claimed_references"]) for r in rows if r["condition"] == "ori_off")
    off_rj = sum(int(r["broken_references"]) for r in rows if r["condition"] == "ori_off")
    if on_t > 0 and off_t > 0:
        on_r = on_rj / on_t
        off_r = off_rj / off_t
        chi2, p = chi2_2x2(on_rj, on_t-on_rj, off_rj, off_t-off_rj)
        print()
        print(f"  AGGREGATE      on={on_r:>5.0%} ({on_rj}/{on_t}) "
               f"off={off_r:>5.0%} ({off_rj}/{off_t}) "
               f"Δ={on_r-off_r:+5.0%}  chi2={chi2:>7.2f}  p={p:.6f}")

    return results


def decision(crit_results: List[Tuple]) -> None:
    print()
    print("=" * 100)
    print("DECISION CRITERIA — does workload-B close R1.3 / ET1?")
    print("=" * 100)
    print()
    print("[1] On at least 6/8 domains, ori_on rejection_rate >= ori_off by >= 25pp, p < 0.05")
    pass_count = 0
    for dn, delta, p, on_r, off_r in crit_results:
        ok = (delta >= 0.25) and (p < 0.05)
        pass_count += int(ok)
        marker = "PASS" if ok else "FAIL"
        print(f"    {marker} {dn:<22} Δ={delta:+.2f} p={p:.4f} (ori_on={on_r:.0%}, ori_off={off_r:.0%})")
    n_total = len(crit_results)
    print(f"    {pass_count}/{n_total} domains pass")
    print()
    if not crit_results:
        print(">>> RESULT: NO DATA")
    elif n_total >= 8:
        crit_pass = pass_count >= 6
    else:
        crit_pass = pass_count == n_total
    if crit_results:
        if crit_pass:
            print(">>> RESULT: GO (workload-B closes R1.3)")
            print("    S-Bus's structural-conflict-prevention demonstrably fires on")
            print("    a non-code workload across multiple domains. The C2 safety")
            print("    claim generalises beyond SWE-bench-derived Python tasks.")
        else:
            print(">>> RESULT: PARTIAL or NO-GO")
            print("    Inspect domains where the gap is small. Could indicate the")
            print("    workload generates less concurrency contention than expected,")
            print("    not necessarily that ORI fails to fire.")
    print("=" * 100)


def view_divergence_summary(rows: List[Dict[str, str]]) -> None:
    print("=" * 100)
    print("VIEW-DIVERGENCE — server-side stale-commit detection (R1.3 PRIMARY METRIC)")
    print("=" * 100)
    by_cond_div = defaultdict(lambda: [0, 0])  # [divergent, checked]
    has_data = False
    for r in rows:
        c = r["condition"]
        vd = int(r.get("view_divergent_commits", 0) or 0)
        vc = int(r.get("view_checked_commits", 0) or 0)
        if vc > 0:
            has_data = True
        by_cond_div[c][0] += vd
        by_cond_div[c][1] += vc

    if not has_data:
        print("  No view-divergence data in CSV. Server may not have the v4 ")
        print("  instrumentation patch applied. Skipping primary metric.")
        return

    for cond in ["ori_on", "ori_off"]:
        if cond not in by_cond_div:
            continue
        d, c = by_cond_div[cond]
        rate = d / c if c else 0.0
        print(f"  {cond:<10}  divergent/checked = {d}/{c} = {rate:.4f}")

    on = by_cond_div.get("ori_on", [0, 0])
    off = by_cond_div.get("ori_off", [0, 0])
    if on[1] > 0 and off[1] > 0:
        on_rate = on[0] / on[1]
        off_rate = off[0] / off[1]
        chi2, p = chi2_2x2(on[0], on[1]-on[0], off[0], off[1]-off[0])
        print(f"\n  Δ(divergence rate) = ori_off - ori_on = {off_rate - on_rate:+.4f}")
        print(f"  chi2 = {chi2:.2f}, p = {p:.6f}")
        print()
        if off[0] > 0 and on[0] == 0:
            print(f"  CLEAN STRUCTURAL CLAIM: ORI prevented {off[0]} stale cross-shard")
            print(f"  reads from being committed (out of {off[1]} commits where the")
            print(f"  cross-shard view was checked). Zero divergent commits got through")
            print(f"  under ori_on; {off[0]} divergent commits got through under ori_off.")
        elif off[0] > 0:
            print(f"  PARTIAL STRUCTURAL CLAIM: ori_off allowed {off[0]} divergent")
            print(f"  commits; ori_on allowed {on[0]} (should be zero — investigate).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/workload_b_sweep.csv")
    args = p.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")
    rows = load_csv(args.csv)
    if not rows:
        raise SystemExit(f"CSV is empty: {args.csv}")
    print(f"Loaded {len(rows)} trial rows from {args.csv}")
    cell_summary(rows)
    view_divergence_summary(rows)
    print()
    crit = pairwise_rejection_tests(rows)
    decision(crit or [])


if __name__ == "__main__":
    main()
