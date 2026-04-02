# convert_swebench.py
import json
from datasets import load_dataset

# Load SWE-bench (use "lite" for faster runs)
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

tasks = []
for item in ds:
    tasks.append({
        "task_id": item["instance_id"],
        "description": (
            f"Fix the following GitHub issue in {item['repo']}:\n\n"
            f"{item['problem_statement']}"
        ),
        "category": "software_engineering",
        "shared_state_keys": [
            "bug_analysis",
            "reproduction_steps",
            "patch_implementation",
            "test_verification"
        ],
        "ground_truth_outputs": [
            "A concrete patch or code fix is described",
            "The root cause of the bug is identified",
            "Steps to verify the fix are outlined"
        ]
    })

import os
os.makedirs("datasets", exist_ok=True)
with open("datasets/long_horizon_tasks.json", "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Converted {len(tasks)} SWE-bench tasks")