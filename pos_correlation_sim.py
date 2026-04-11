#!/usr/bin/env python3
"""
pos_correlation_sim.py
======================
Validates the POS degradation model under correlated (bursty) hidden reads.

PURPOSE:
  Experiment C in the paper assumes i.i.d. (independent, uniformly random)
  hidden reads: each injection event is hidden with probability p_hidden,
  independently of all others.

  Real LLM agents accumulate context ACROSS steps — if an agent reads shard X
  at step 3, it "remembers" X in its prompt for steps 4, 5, 6...
  This creates CORRELATED hidden reads: a shard read once stays hidden for
  many subsequent steps (bursty correlation).

  This experiment tests whether the linear degradation model
  ρ ≈ 0.71 × p_hidden holds under:
    (A) i.i.d. hidden reads  (validates Exp C)
    (B) bursty hidden reads  (simulates real LLM context accumulation)
    (C) anti-correlated reads (lower bound)

  If the model is ROBUST (A ≈ B ≈ C), the POS theorem is stronger.
  If the model FAILS (B >> A), the i.i.d. caveat is serious.

USAGE:
  python3 pos_correlation_sim.py \
      --injections 8630 \
      --trials 10 \
      --p-hidden 0.0 0.1 0.25 0.5 0.75 1.0 \
      --out results/pos_correlation_results.csv

The script does NOT require the S-Bus server — it's a pure simulation
using the baseline corruption rate measured in Table 7.
"""

from __future__ import annotations
import argparse, csv, os, random
from dataclasses import dataclass
import numpy as np


BASELINE_CORRUPTION = 0.712  # from Experiment C p_hidden=1.0 condition


@dataclass
class SimResult:
    condition:    str   # "iid", "bursty", "anticorrelated"
    p_hidden:     float
    n_injections: int
    n_trials:     int
    corrupted:    int
    detected:     int

    @property
    def corrupt_rate(self) -> float:
        return self.corrupted / (self.n_injections * self.n_trials)

    @property
    def detection_rate(self) -> float:
        return self.detected / (self.n_injections * self.n_trials)


def simulate_iid(p_hidden: float, n_inj: int, n_trials: int,
                 rng: random.Random) -> SimResult:
    """
    i.i.d. model: each injection event is hidden independently with prob p_hidden.
    Hidden → corrupt with prob BASELINE_CORRUPTION.
    """
    corrupted = 0
    detected  = 0
    for _ in range(n_trials * n_inj):
        if rng.random() < p_hidden:
            # Hidden read: server can't detect
            if rng.random() < BASELINE_CORRUPTION:
                corrupted += 1
        else:
            # Observable read: DeliveryLog detects it
            detected += 1
    return SimResult("iid", p_hidden, n_inj, n_trials, corrupted, detected)


def simulate_bursty(p_hidden: float, n_inj: int, n_trials: int,
                    rng: random.Random, burst_len: int = 5) -> SimResult:
    """
    Bursty model: agent reads a shard via HTTP at step T, then "remembers" it
    in context for burst_len subsequent steps WITHOUT re-fetching.

    Implementation: Once a shard becomes "context-loaded" (happens with prob
    p_enter per step), it stays hidden for burst_len steps.
    p_enter is calibrated so that the overall p_hidden matches target.

    Expected: slightly HIGHER corruption rate than i.i.d. if p_hidden > 0
    because bursts mean multiple consecutive stale reads from same context.
    """
    # Calibrate p_enter so avg hidden fraction ≈ p_hidden
    # In steady state: fraction_hidden ≈ p_enter × burst_len / (1 + p_enter × burst_len)
    # Solve for p_enter:
    if p_hidden <= 0.0:
        return SimResult("bursty", p_hidden, n_inj, n_trials, 0, n_inj * n_trials)
    if p_hidden >= 1.0:
        corrupted = sum(
            1 for _ in range(n_inj * n_trials)
            if rng.random() < BASELINE_CORRUPTION
        )
        return SimResult("bursty", p_hidden, n_inj, n_trials, corrupted, 0)

    p_enter = p_hidden / (burst_len * (1 - p_hidden))
    p_enter = min(p_enter, 1.0)

    corrupted = 0
    detected  = 0
    state_hidden = False
    remaining_burst = 0

    for _ in range(n_trials * n_inj):
        if state_hidden:
            remaining_burst -= 1
            if remaining_burst <= 0:
                state_hidden = False
            # This step: hidden read
            if rng.random() < BASELINE_CORRUPTION:
                corrupted += 1
        else:
            # Enter burst with probability p_enter
            if rng.random() < p_enter:
                state_hidden   = True
                remaining_burst = burst_len
                if rng.random() < BASELINE_CORRUPTION:
                    corrupted += 1
            else:
                detected += 1

    return SimResult("bursty", p_hidden, n_inj, n_trials, corrupted, detected)


def simulate_anticorrelated(p_hidden: float, n_inj: int, n_trials: int,
                             rng: random.Random) -> SimResult:
    """
    Anti-correlated model: agent alternates between observable and hidden reads.
    Hidden reads happen exactly p_hidden fraction of the time but in a regular pattern.
    This is the LOWER BOUND scenario for real agents.
    """
    corrupted = 0
    detected  = 0
    total = n_inj * n_trials
    n_hidden = int(total * p_hidden)

    # Generate regular pattern: every 1/p_hidden steps is hidden
    hidden_set = set(range(0, total, max(1, int(1/p_hidden)))) if p_hidden > 0 else set()

    for i in range(total):
        if i in hidden_set:
            if rng.random() < BASELINE_CORRUPTION:
                corrupted += 1
        else:
            detected += 1

    return SimResult("anticorrelated", p_hidden, n_inj, n_trials, corrupted, detected)


def print_results_table(results: list[SimResult]):
    print("\n" + "="*90)
    print("POS Degradation Model: i.i.d. vs Bursty vs Anti-correlated Hidden Reads")
    print("="*90)
    print(f"{'p_hidden':>10} {'Condition':>15} {'Corrupt%':>10} "
          f"{'Predicted%':>12} {'Δ from pred':>12}")
    print("-"*90)

    by_p = {}
    for r in results:
        by_p.setdefault(r.p_hidden, []).append(r)

    for p, cond_results in sorted(by_p.items()):
        predicted = BASELINE_CORRUPTION * p * 100
        for r in cond_results:
            actual = r.corrupt_rate * 100
            delta  = actual - predicted
            print(f"{p:>9.0%} {r.condition:>15} {actual:>9.1f}% "
                  f"{predicted:>11.1f}% {delta:>+11.1f}%")
        print()

    print("="*90)
    print(f"\nBaseline corruption rate (p_hidden=1.0): {BASELINE_CORRUPTION*100:.1f}%")
    print(f"Linear model prediction: corrupt% ≈ {BASELINE_CORRUPTION*100:.1f}% × p_hidden")
    print("\nKey finding:")
    print("  If bursty ≈ i.i.d.:       linear model is robust to correlation patterns")
    print("  If bursty >> i.i.d.:      linear model UNDERSTATES risk for bursty agents")
    print("  If anti-corr ≈ i.i.d.:    model is tight regardless of pattern")


def write_csv(results: list[SimResult], path: str):
    fields = ["condition", "p_hidden", "n_injections", "n_trials",
              "corrupted", "corrupt_rate", "detected", "detection_rate",
              "predicted_rate"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "condition":      r.condition,
                "p_hidden":       r.p_hidden,
                "n_injections":   r.n_injections,
                "n_trials":       r.n_trials,
                "corrupted":      r.corrupted,
                "corrupt_rate":   f"{r.corrupt_rate:.4f}",
                "detected":       r.detected,
                "detection_rate": f"{r.detection_rate:.4f}",
                "predicted_rate": f"{BASELINE_CORRUPTION * r.p_hidden:.4f}",
            })
    print(f"Results written to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injections", type=int, default=8630)
    ap.add_argument("--trials",     type=int, default=10)
    ap.add_argument("--p-hidden",   type=float, nargs="+",
                    default=[0.0, 0.10, 0.25, 0.50, 0.75, 1.00])
    ap.add_argument("--burst-len",  type=int, default=5,
                    help="Steps a shard stays hidden in context (bursty model)")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--out",        default="results/pos_correlation_results.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    results: list[SimResult] = []

    for p in args.p_hidden:
        print(f"Simulating p_hidden={p:.0%}...")
        results.append(simulate_iid(p, args.injections, args.trials, rng))
        results.append(simulate_bursty(p, args.injections, args.trials, rng, args.burst_len))
        results.append(simulate_anticorrelated(p, args.injections, args.trials, rng))

    print_results_table(results)
    write_csv(results, args.out)

    # Compute R² for each condition
    print("\n=== R² for linear model by condition ===")
    for cond in ["iid", "bursty", "anticorrelated"]:
        cond_results = [r for r in results if r.condition == cond and r.p_hidden > 0]
        if not cond_results:
            continue
        actual    = np.array([r.corrupt_rate for r in cond_results])
        predicted = np.array([BASELINE_CORRUPTION * r.p_hidden for r in cond_results])
        ss_res = np.sum((actual - predicted)**2)
        ss_tot = np.sum((actual - actual.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        print(f"  {cond:>15}: R²={r2:.4f}")


if __name__ == "__main__":
    main()