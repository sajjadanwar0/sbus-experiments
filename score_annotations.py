import csv
import sys


def load_labels(path):
    """Return dict: (row_idx, candidate_shard) -> 'yes' / 'no' / 'unclear' / ''."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("llm_label") or row.get("human_label") or "").strip().lower()
            key = (row["row_idx"], row["candidate_shard"])
            out[key] = label
    return out


def agent_claim(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["row_idx"], row["candidate_shard"])
            out[key] = row["agent_said_used_it"].strip().lower() == "yes"
    return out


def cohen_kappa(labels_a, labels_b, classes):
    """Cohen's kappa for categorical labels."""
    keys = [k for k in labels_a
            if k in labels_b
            and labels_a[k] in classes
            and labels_b[k] in classes]
    n = len(keys)
    if n == 0:
        return None, 0

    agree = sum(1 for k in keys if labels_a[k] == labels_b[k])
    p_o = agree / n

    p_e = 0.0
    for c in classes:
        p_a = sum(1 for k in keys if labels_a[k] == c) / n
        p_b = sum(1 for k in keys if labels_b[k] == c) / n
        p_e += p_a * p_b

    if p_e == 1.0:
        return (1.0 if p_o == 1.0 else 0.0), n
    return (p_o - p_e) / (1 - p_e), n


def confusion_matrix(labels_a, labels_b, classes, label_a="A", label_b="B"):
    keys = [k for k in labels_a
            if k in labels_b
            and labels_a[k] in classes
            and labels_b[k] in classes]
    matrix = {(ca, cb): 0 for ca in classes for cb in classes}
    for k in keys:
        matrix[(labels_a[k], labels_b[k])] += 1

    col = 12
    print(f"{'':>{col}}", end="")
    for cb in classes:
        print(f"{(label_b + ':' + cb):>{col}}", end="")
    print()
    for ca in classes:
        print(f"{(label_a + ':' + ca):>{col}}", end="")
        for cb in classes:
            print(f"{matrix[(ca, cb)]:>{col}}", end="")
        print()


def self_report_vs_labels(path_csv, judge_label):
    claim = agent_claim(path_csv)
    labels = load_labels(path_csv)
    keys = [k for k in claim if k in labels and labels[k] in ("yes", "no")]
    tp = sum(1 for k in keys if claim[k] and labels[k] == "yes")
    fp = sum(1 for k in keys if claim[k] and labels[k] == "no")
    fn = sum(1 for k in keys if not claim[k] and labels[k] == "yes")
    tn = sum(1 for k in keys if not claim[k] and labels[k] == "no")
    total = tp + fp + fn + tn
    if total == 0:
        return None
    return {
        "judge": judge_label,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall":    tp / (tp + fn) if (tp + fn) else 0.0,
        "accuracy":  (tp + tn) / total,
        "n": total,
    }


def judge_name_from_path(path):
    base = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return base.replace("_labels", "").replace("_", "-")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} judge_a_labels.csv judge_b_labels.csv")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    name_a = judge_name_from_path(path_a)
    name_b = judge_name_from_path(path_b)

    labels_a = load_labels(path_a)
    labels_b = load_labels(path_b)

    classes_strict = ["yes", "no"]
    classes_lenient = ["yes", "no", "unclear"]
    kappa_strict, n_strict = cohen_kappa(labels_a, labels_b, classes_strict)
    kappa_lenient, n_lenient = cohen_kappa(labels_a, labels_b, classes_lenient)

    keys_strict = [k for k in labels_a
                   if k in labels_b
                   and labels_a[k] in classes_strict
                   and labels_b[k] in classes_strict]
    agree_strict = sum(1 for k in keys_strict if labels_a[k] == labels_b[k])

    print("=" * 64)
    print("Inter-LLM agreement report")
    print("=" * 64)
    print(f"Judge A: {name_a}   ({path_a})")
    print(f"Judge B: {name_b}   ({path_b})")
    print()
    print("Pairs labelled by both judges:")
    print(f"  yes/no only (strict):  {n_strict}")
    print(f"  including unclear:     {n_lenient}")
    print()
    if kappa_strict is not None:
        print(f"Strict  kappa (yes/no):  {kappa_strict:+.4f}")
    else:
        print("Strict  kappa: no comparable data")
    if kappa_lenient is not None:
        print(f"Lenient kappa (3-class): {kappa_lenient:+.4f}")
    else:
        print("Lenient kappa: no comparable data")
    if n_strict:
        print(f"Raw agreement (strict):  "
              f"{agree_strict}/{n_strict} = {agree_strict/n_strict:.1%}")
    print()
    print("Landis & Koch (1977) interpretation:")
    print("  <0.00 poor | 0.00-0.20 slight | 0.21-0.40 fair")
    print("  0.41-0.60 moderate | 0.61-0.80 substantial | 0.81-1.00 almost perfect")
    print()
    print("NOTE: This is inter-LLM agreement, not human IAA. The kappa")
    print("measures consistency between two LLM judges, not correctness.")
    print()

    print("Confusion matrix (strict, yes/no only):")
    confusion_matrix(labels_a, labels_b, classes_strict,
                     label_a=name_a[:8], label_b=name_b[:8])
    print()

    print("=" * 64)
    print("Self-report vs LLM-judge (weak validation)")
    print("=" * 64)
    for path in (path_a, path_b):
        r = self_report_vs_labels(path, judge_name_from_path(path))
        if r is None:
            print(f"{path}: insufficient data")
            continue
        print(f"{r['judge']} (n={r['n']}):")
        print(f"  self-report precision: {r['precision']:.3f}  "
              f"(when agent says 'used', how often does judge agree?)")
        print(f"  self-report recall:    {r['recall']:.3f}  "
              f"(of judge-marked 'used', how many did agent self-report?)")
        print(f"  self-report accuracy:  {r['accuracy']:.3f}")
        print(f"    TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  TN={r['tn']}")
        print()


if __name__ == "__main__":
    main()
