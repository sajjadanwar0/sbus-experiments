from typing import Dict, Any


COMPLETION_THRESHOLD = 1.0
ORI_ON_REJECTION_THRESHOLD = 0.30


def _compute_completion(trial_result: Dict[str, Any]) -> Dict[str, Any]:
    final_state = trial_result["final_state"]
    n_shards = len(final_state)
    n_committed = sum(1 for v in final_state.values()
                       if (v.get("version") or 0) >= 1)
    return {
        "completion_rate": n_committed / n_shards if n_shards else 0.0,
        "n_committed_shards": n_committed,
        "n_total_shards": n_shards,
        "incomplete_shards": [k for k, v in final_state.items()
                                if (v.get("version") or 0) == 0],
    }


def _compute_rejection_rate(trial_result: Dict[str, Any]) -> Dict[str, Any]:
    m = trial_result.get("metrics", {})
    rejections = m.get("n_rejections_total")
    attempts = m.get("total_commit_attempts", 0)

    if rejections is None:
        rejections = max(0, attempts - m.get("n_commit_200", 0))

    rate = rejections / attempts if attempts > 0 else 0.0
    return {
        "rejection_rate": rate,
        "n_rejections": rejections,
        "n_attempts": attempts,
    }


def evaluate_trial(trial_result: Dict[str, Any]) -> Dict[str, Any]:
    completion = _compute_completion(trial_result)
    rej = _compute_rejection_rate(trial_result)
    condition = trial_result.get("condition", "")

    completed = completion["completion_rate"] >= COMPLETION_THRESHOLD
    if condition == "ori_on":
        rej_ok = rej["rejection_rate"] >= ORI_ON_REJECTION_THRESHOLD
    else:
        rej_ok = True
    passed = completed and rej_ok

    return {
        "rejection_rate": rej["rejection_rate"],
        "n_rejections": rej["n_rejections"],
        "n_attempts": rej["n_attempts"],
        "completion_rate": completion["completion_rate"],
        "n_committed_shards": completion["n_committed_shards"],
        "n_total_shards": completion["n_total_shards"],
        "incomplete_shards": completion["incomplete_shards"],
        "pass": passed,
        "coherence_rate": rej["rejection_rate"],
        "total_claimed_references": rej["n_attempts"],
        "coherent_references": rej["n_attempts"] - rej["n_rejections"],
        "broken_references": rej["n_rejections"],
        "per_agent": {},
    }
