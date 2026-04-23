#!/usr/bin/env python3
"""
run_llm_judges.py

Runs two independent LLM judges (GPT-4o and Claude Sonnet 4.6) over the
PH-3 annotation tasks and produces two CSV files of labels. The resulting
CSVs are consumed by score_annotations.py to compute inter-model
agreement (reported in the paper as inter-LLM kappa, NOT human IAA).

Column naming reflects this: the label column is `llm_label`.

Usage:
    python3 run_llm_judges.py                  # full run (2 * 400 calls)
    python3 run_llm_judges.py --pilot 50       # 50-row pilot first
    python3 run_llm_judges.py --resume         # skip rows already in both CSVs

Environment:
    OPENAI_API_KEY      required
    ANTHROPIC_API_KEY   required

Dependencies:
    pip install openai anthropic
"""

import argparse
import concurrent.futures
import csv
import json
import os
import re
import time

import anthropic
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL_A = "gpt-4o"
MODEL_A_SHORT = "gpt4o"
MODEL_B = "claude-sonnet-4-6"
MODEL_B_SHORT = "claude_sonnet"

INPUT_JSON = "tasks.json"
CSV_A = f"{MODEL_A_SHORT}_labels.csv"
CSV_B = f"{MODEL_B_SHORT}_labels.csv"

MAX_CONCURRENT_THREADS = 10
MAX_RETRIES = 3
RETRY_BASE_WAIT = 2.0  # seconds; exponential

CSV_HEADERS = [
    "row_idx", "run_id", "domain", "task_id", "step",
    "candidate_shard", "llm_label", "evidence", "step_reached",
    "reasoning", "agent_said_used_it",
]

# ---------------------------------------------------------------------------
# Prompt (frozen — do NOT tune to hit a kappa target)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a strict code auditor. Decide whether the Candidate Shard provided content that the agent demonstrably needed to produce the Code Change.

DECISION PROCEDURE (apply in order; stop at the first that fires):

Step 1 — Direct entity definition.
Does the shard's content *define* (not merely mention) the specific function, class, variable, field, or constant that is being modified or read by the Change?
    If yes -> <label>Yes</label>.

Step 2 — Required state or schema.
Does the Change transform a value, structure, or invariant whose concrete shape is only recoverable from this shard's content (e.g. a schema, a prior version, a signature, a type)?
    If yes -> <label>Yes</label>.

Step 3 — Default.
Topical overlap ("both relate to the database"), shared vocabulary, or mere availability in the context is NOT sufficient.
    -> <label>No</label>.

HARD RULES (override everything above):
R1. If the shard content shown is empty, truncated, or does not actually contain the entity referenced in Step 1/2 -> <label>No</label>.
R2. If you cannot quote specific words or tokens from the shard that the Change depends on -> <label>No</label>.
R3. Mere name collisions (two shards both contain the word "user") do not count as evidence. The evidence must be semantic.

CALIBRATION:

Ex 1 (Step 1 Yes): Change modifies fetch_data()'s requests.get call. Shard = source of fetch_data(). -> <label>Yes</label>
Ex 2 (Step 3 No):  Change updates CSS colour of Submit button. Shard = submit_handler.js (JS logic). -> <label>No</label>
Ex 3 (Step 3 No):  Change fixes typo in "user_not_found" string. Shard = database_schema.sql. -> <label>No</label>
Ex 4 (Step 2 Yes): Change adds migration renaming column `created`->`created_at`. Shard = models_state showing the original column. -> <label>Yes</label>
Ex 5 (R1 No):      Change references orm_query. Shard = orm_query whose content block is empty/placeholder. -> <label>No</label>
Ex 6 (R3 No):      Change mentions "user". Shard = test_fixtures that mentions "user" in an unrelated fixture. -> <label>No</label>

---
TASK:
Change: "{change}"
Candidate Shard: "{candidate_shard}"
Shard Content:
{fresh_content_block}

OUTPUT FORMAT (all three lines required, exactly in this order, nothing else):
Evidence: "<verbatim <=25-word quote from the shard content that the Change depends on, or NONE>"
Step: <1|2|3>
<label>Yes</label> or <label>No</label>
"""

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
LABEL_RE = re.compile(r"<label>\s*(Yes|No)\s*</label>", re.IGNORECASE)
EVIDENCE_RE_QUOTED = re.compile(r'Evidence:\s*"([^"]*)"', re.IGNORECASE)
EVIDENCE_RE_PLAIN = re.compile(r"Evidence:\s*(.+?)(?:\n|$)", re.IGNORECASE)
STEP_RE = re.compile(r"Step:\s*([123])", re.IGNORECASE)


def extract_fields(text: str):
    """Return (label, evidence, step_reached, full_reasoning)."""
    label_m = LABEL_RE.search(text)
    if label_m:
        label = label_m.group(1).strip().lower()
    else:
        label = "no"

    ev_m = EVIDENCE_RE_QUOTED.search(text) or EVIDENCE_RE_PLAIN.search(text)
    evidence = ev_m.group(1).strip() if ev_m else "NONE"

    step_m = STEP_RE.search(text)
    step_reached = step_m.group(1) if step_m else ""

    reasoning = text.strip()
    if not label_m:
        reasoning += " [EXTRACTION FAILED]"
    return label, evidence, step_reached, reasoning


# ---------------------------------------------------------------------------
# API wrappers with retry
# ---------------------------------------------------------------------------
def _with_retry(fn, *args, **kwargs):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — want to retry everything
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
    raise last_err  # type: ignore[misc]


def call_openai(prompt: str) -> str:
    def _do():
        resp = openai_client.chat.completions.create(
            model=MODEL_A,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return resp.choices[0].message.content
    return _with_retry(_do)


def call_claude(prompt: str) -> str:
    def _do():
        resp = anthropic_client.messages.create(
            model=MODEL_B,
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    return _with_retry(_do)


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------
def process_single_row(task: dict):
    prompt = PROMPT_TEMPLATE.format(
        change=task["change"],
        candidate_shard=task["candidate_shard"],
        fresh_content_block=task["fresh_content_block"],
    )

    try:
        a_text = call_openai(prompt)
        a_label, a_evidence, a_step, a_reasoning = extract_fields(a_text)
    except Exception as e:  # noqa: BLE001
        a_label, a_evidence, a_step, a_reasoning = "no", "NONE", "", f"API Error: {e}"

    try:
        b_text = call_claude(prompt)
        b_label, b_evidence, b_step, b_reasoning = extract_fields(b_text)
    except Exception as e:  # noqa: BLE001
        b_label, b_evidence, b_step, b_reasoning = "no", "NONE", "", f"API Error: {e}"

    agent_claim = "yes" if task.get("agent_said_used_it", False) else "no"
    base = {
        "row_idx": task["row_idx"],
        "run_id": task.get("run_id", ""),
        "domain": task.get("domain", ""),
        "task_id": task.get("task_id", ""),
        "step": task.get("step", ""),
        "candidate_shard": task["candidate_shard"],
        "agent_said_used_it": agent_claim,
    }
    row_a = {**base, "llm_label": a_label, "evidence": a_evidence,
             "step_reached": a_step, "reasoning": a_reasoning}
    row_b = {**base, "llm_label": b_label, "evidence": b_evidence,
             "step_reached": b_step, "reasoning": b_reasoning}
    return task["row_idx"], row_a, row_b


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def load_processed_keys(csv_path: str):
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((str(row["row_idx"]), row["candidate_shard"]))
    return keys


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def process_tasks(pilot_n: int | None = None, resume: bool = False):
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if pilot_n is not None:
        tasks = tasks[:pilot_n]
        print(f"PILOT MODE: running first {len(tasks)} tasks only")

    if resume and (os.path.exists(CSV_A) or os.path.exists(CSV_B)):
        done_a = load_processed_keys(CSV_A)
        done_b = load_processed_keys(CSV_B)
        done_both = done_a & done_b
        before = len(tasks)
        tasks = [t for t in tasks
                 if (str(t["row_idx"]), t["candidate_shard"]) not in done_both]
        print(f"RESUME: {before - len(tasks)} already processed in both CSVs; "
              f"{len(tasks)} remaining")
        write_mode = "a"
    else:
        write_mode = "w"

    if not tasks:
        print("Nothing to do.")
        return

    print(f"Judge A: {MODEL_A}")
    print(f"Judge B: {MODEL_B}")
    print(f"Tasks to run: {len(tasks)}  (= {2 * len(tasks)} API calls total)")
    print(f"Concurrency: {MAX_CONCURRENT_THREADS}")
    print("-" * 60)

    start = time.time()

    with open(CSV_A, write_mode, newline="", encoding="utf-8") as fa, \
         open(CSV_B, write_mode, newline="", encoding="utf-8") as fb:
        writer_a = csv.DictWriter(fa, fieldnames=CSV_HEADERS)
        writer_b = csv.DictWriter(fb, fieldnames=CSV_HEADERS)
        if write_mode == "w":
            writer_a.writeheader()
            writer_b.writeheader()

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_CONCURRENT_THREADS) as pool:
            futures = {pool.submit(process_single_row, t): t for t in tasks}
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    _, row_a, row_b = fut.result()
                    writer_a.writerow(row_a)
                    writer_b.writerow(row_b)
                    fa.flush()
                    fb.flush()
                except Exception as e:  # noqa: BLE001
                    print(f"  [{i}] row failed entirely: {e}")
                if i % 20 == 0 or i == len(tasks):
                    elapsed = time.time() - start
                    rate = i / elapsed if elapsed > 0 else 0.0
                    eta = (len(tasks) - i) / rate if rate > 0 else 0.0
                    print(f"  [{i}/{len(tasks)}]  {rate:.2f} rows/s  "
                          f"ETA {eta/60:.1f} min")

    mins = (time.time() - start) / 60
    print("-" * 60)
    print(f"Done in {mins:.2f} min")
    print(f"  {CSV_A}")
    print(f"  {CSV_B}")
    print("\nNext step:")
    print(f"  python3 score_annotations.py {CSV_A} {CSV_B}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--pilot", type=int, default=None,
                        help="Only run first N tasks (e.g. 50 for calibration)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (row_idx, candidate_shard) pairs already "
                             "in BOTH CSVs; append to existing files")
    args = parser.parse_args()
    process_tasks(pilot_n=args.pilot, resume=args.resume)
