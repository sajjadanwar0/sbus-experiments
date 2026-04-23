#!/usr/bin/env python3
"""
diagnose_disagreements.py

Error analysis for inter-LLM disagreements. Surfaces patterns to help
decide whether:
  (a) the task is genuinely ambiguous (report kappa honestly), or
  (b) one judge has a systematic failure mode (fix the JUDGE, not the
      prompt — e.g. raise max_tokens, handle empty shards differently), or
  (c) the dataset has a quality issue (fix the dataset).

It does NOT suggest prompt changes. If you're tempted to change the
frozen prompt because of what this script shows you, stop.

Usage:
    python3 diagnose_disagreements.py gpt4o_labels.csv claude_sonnet_labels.csv [N_SAMPLES]

N_SAMPLES = how many disagreements to print per direction (default 8).
"""
import csv
import sys
from collections import Counter, defaultdict


def load(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["row_idx"], row["candidate_shard"])
            out[key] = row
    return out


def pct(x, n):
    return f"{x/n:.1%}" if n else "n/a"


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} judge_a.csv judge_b.csv [N_SAMPLES]")
        sys.exit(1)
    n_samples = int(sys.argv[3]) if len(sys.argv) >= 4 else 8

    path_a, path_b = sys.argv[1], sys.argv[2]
    a, b = load(path_a), load(path_b)
    keys = sorted(set(a) & set(b), key=lambda k: (int(k[0]) if k[0].isdigit() else k[0], k[1]))
    n = len(keys)

    yy = [k for k in keys if a[k]["llm_label"] == "yes" and b[k]["llm_label"] == "yes"]
    nn = [k for k in keys if a[k]["llm_label"] == "no"  and b[k]["llm_label"] == "no"]
    ay_bn = [k for k in keys if a[k]["llm_label"] == "yes" and b[k]["llm_label"] == "no"]
    an_by = [k for k in keys if a[k]["llm_label"] == "no"  and b[k]["llm_label"] == "yes"]

    print("=" * 72)
    print("Disagreement diagnosis")
    print("=" * 72)
    print(f"Total pairs: {n}")
    print(f"  both yes: {len(yy):4d}   both no: {len(nn):4d}")
    print(f"  A=yes B=no: {len(ay_bn):3d}  (A more generous)")
    print(f"  A=no B=yes: {len(an_by):3d}  (B more generous)")
    print(f"  agreement:  {(len(yy)+len(nn))}/{n} = {pct(len(yy)+len(nn), n)}")
    print()

    # Extraction failures (suggests format-parsing bugs, not prompt issues)
    def fails(d):
        return sum(1 for v in d.values()
                   if "EXTRACTION FAILED" in (v.get("reasoning") or ""))
    print("Extraction failures (should be ~0 for a well-behaved judge):")
    print(f"  Judge A: {fails(a)}    Judge B: {fails(b)}")
    print()

    # Empty-shard handling (Rule R1). If a judge says Yes when content is empty
    # or evidence is NONE, R1 is being ignored.
    def r1_violations(d):
        v = 0
        for row in d.values():
            if row["llm_label"] == "yes" and (row.get("evidence") or "").strip().upper() == "NONE":
                v += 1
        return v
    print("Rule R1/R2 violations (judge said YES but evidence is NONE):")
    print(f"  Judge A: {r1_violations(a)}")
    print(f"  Judge B: {r1_violations(b)}")
    print()

    # Step-reached distribution, split by agree/disagree
    def step_counts(subset, d):
        return Counter(d[k].get("step_reached", "") or "?" for k in subset)
    agreeing = yy + nn
    disagreeing = ay_bn + an_by
    print("Step reached, Judge A, on agreements:      ", dict(step_counts(agreeing, a)))
    print("Step reached, Judge A, on disagreements:   ", dict(step_counts(disagreeing, a)))
    print("Step reached, Judge B, on agreements:      ", dict(step_counts(agreeing, b)))
    print("Step reached, Judge B, on disagreements:   ", dict(step_counts(disagreeing, b)))
    print("  (If disagreements cluster at step=2, the Step 2 rule is the")
    print("   ambiguous part of the task. That's a finding, not a bug.)")
    print()

    # Per-domain agreement — is the κ dragged down by one bad domain?
    per_domain = defaultdict(lambda: [0, 0])  # [agree, total]
    for k in keys:
        dom = a[k].get("domain", "?")
        per_domain[dom][1] += 1
        if a[k]["llm_label"] == b[k]["llm_label"]:
            per_domain[dom][0] += 1
    print("Per-domain agreement (sorted worst → best):")
    rows = sorted(per_domain.items(), key=lambda x: x[1][0] / max(x[1][1], 1))
    for dom, (ag, tot) in rows:
        print(f"  {dom:32s}  {ag:3d}/{tot:<3d} = {pct(ag, tot)}")
    print()

    # Self-report as a prior on disagreement
    def agent_yes(k):
        return a[k].get("agent_said_used_it", "").lower() == "yes"
    a_yes_on_agent_yes = sum(1 for k in disagreeing
                             if agent_yes(k) and a[k]["llm_label"] == "yes")
    a_yes_on_agent_no  = sum(1 for k in disagreeing
                             if not agent_yes(k) and a[k]["llm_label"] == "yes")
    print("Self-report alignment in the DISAGREEMENT set:")
    print(f"  When agent self-reports 'used': {sum(1 for k in disagreeing if agent_yes(k))} disagreements")
    print(f"    Judge A said yes: {a_yes_on_agent_yes}")
    print(f"  When agent self-reports 'not used': {sum(1 for k in disagreeing if not agent_yes(k))} disagreements")
    print(f"    Judge A said yes: {a_yes_on_agent_no}")
    print("  (If a judge's Yes tracks the agent's self-report > chance, the")
    print("   judge may be anchoring on the self-report hint rather than")
    print("   the shard content. Your prompt doesn't mention self-report,")
    print("   but the change text sometimes echoes it.)")
    print()

    # Evidence length distribution on Yes labels
    def ev_lens(d, label):
        return [len((v.get("evidence") or "").strip())
                for v in d.values() if v["llm_label"] == label
                and (v.get("evidence") or "").strip().upper() != "NONE"]
    for name, d in [("A", a), ("B", b)]:
        lens = ev_lens(d, "yes")
        if lens:
            avg = sum(lens) / len(lens)
            print(f"Judge {name}: evidence length on YES (chars), "
                  f"n={len(lens)}, mean={avg:.0f}, max={max(lens)}")
    print()

    # Sample disagreements for qualitative inspection
    print("=" * 72)
    print(f"Sample disagreements (up to {n_samples} per direction)")
    print("=" * 72)
    for title, ks in [(f"Judge A said YES, Judge B said NO ({len(ay_bn)} total)", ay_bn[:n_samples]),
                      (f"Judge A said NO,  Judge B said YES ({len(an_by)} total)", an_by[:n_samples])]:
        print(f"\n--- {title} ---")
        for k in ks:
            ra, rb = a[k], b[k]
            print(f"\n  row_idx={k[0]}  shard={k[1]}  domain={ra.get('domain', '')}")
            print(f"    A  step={ra.get('step_reached','?')}  ev=\"{(ra.get('evidence','') or '')[:140]}\"")
            print(f"    B  step={rb.get('step_reached','?')}  ev=\"{(rb.get('evidence','') or '')[:140]}\"")
            print(f"    agent_self_reported: {ra.get('agent_said_used_it','')}")
    print()
    print("=" * 72)
    print("What to do with these results")
    print("=" * 72)
    print("1. If extraction failures > 0:  fix the parser, rerun those rows.")
    print("2. If R1 violations > 0:        one judge ignores empty-shard rule;")
    print("                                consider logging and filtering those")
    print("                                rows, not rewriting the prompt.")
    print("3. If one domain drives most:   report per-domain κ in the paper.")
    print("4. If step=2 dominates disag:   the task is genuinely ambiguous on")
    print("                                'required state'. Write that in §VII-Q.")
    print("5. If none of the above:        run the full 400 rows, accept κ, and")
    print("                                report honestly. Agent over-claims")
    print("                                usage either way — that's the finding.")
    print()
    print("Do not change the prompt based on what you see above.")


if __name__ == "__main__":
    main()
