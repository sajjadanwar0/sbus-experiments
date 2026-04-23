#!/usr/bin/env python3
"""
run_sjv3_parallel.py
====================
Runs exp_semantic_judge_v3.py in parallel — one process per task,
each against its own S-Bus instance on a separate port.

USAGE:
    python3 run_sjv3_parallel.py \
        --tasks datasets/tasks_30_multidomain.json \
        --n-tasks 5 \
        --n-runs 25 \
        --n-steps 20 \
        --base-port 7000 \
        --sbus-bin ../../sbus/target/release/sbus \
        --output results/sj_v3_results.csv

WHAT IT DOES:
    1. Splits tasks JSON into n-tasks separate single-task JSON files
    2. Starts one S-Bus instance per task on ports base-port to base-port+n
    3. Starts one exp_semantic_judge_v3.py process per task
    4. Waits for all to finish
    5. Merges all per-task CSVs into one final CSV
    6. Runs statistical tests on merged results
    7. Kills all S-Bus instances

REQUIREMENTS:
    - S-Bus binary compiled: cargo build --release
    - exp_semantic_judge_v3.py in same directory
    - OPENAI_API_KEY set in environment
"""

import argparse
import csv
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def split_tasks(tasks_file: str, n_tasks: int, out_dir: str) -> list[str]:
    """Split tasks JSON into one file per task. Returns list of file paths."""
    with open(tasks_file) as f:
        all_tasks = json.load(f)

    # Prefer django tasks
    django = [t for t in all_tasks if "django" in t.get("task_id", "").lower()]
    other  = [t for t in all_tasks if "django" not in t.get("task_id", "").lower()]
    tasks  = (django + other)[:n_tasks]

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, task in enumerate(tasks):
        path = os.path.join(out_dir, f"task_{i}_{task['task_id'][:20]}.json")
        with open(path, "w") as f:
            json.dump([task], f)
        paths.append(path)
        print(f"  Task {i}: {task['task_id']} → {path}")

    return paths


def wait_for_sbus(port: int, timeout: int = 30) -> bool:
    """Wait until S-Bus is accepting connections on port."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection(("localhost", port), timeout=1)
            s.close()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def start_sbus(port: int, sbus_bin: str, log_dir: str) -> subprocess.Popen:
    """Start one S-Bus instance on the given port."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"sbus_{port}.log")
    env = os.environ.copy()
    env["SBUS_PORT"]          = str(port)
    env["SBUS_ADMIN_ENABLED"] = "1"
    env["RUST_LOG"]           = "warn"  # suppress debug noise

    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [sbus_bin],
        env=env,
        stdout=log_f,
        stderr=log_f,
    )
    return proc


def start_experiment(
    task_file:      str,
    port:           int,
    n_runs:         int,
    n_steps:        int,
    n_agents:       int,
    injection_step: int,
    conditions:     list[str],
    output:         str,
    log_dir:        str,
    use_claude_judge: bool,
) -> subprocess.Popen:
    """Start one exp_semantic_judge_v3.py process."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"exp_port{port}.log")

    env = os.environ.copy()
    env["SBUS_URL"] = f"http://localhost:{port}"

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "exp_semantic_judge_v3.py")

    cmd = [
        sys.executable, script,
        "--tasks",          task_file,
        "--n-tasks",        "1",           # each process handles exactly 1 task
        "--n-runs",         str(n_runs),
        "--n-steps",        str(n_steps),
        "--n-agents",       str(n_agents),
        "--injection-step", str(injection_step),
        "--conditions",     *conditions,
        "--output",         output,
    ]
    if use_claude_judge:
        cmd.append("--use-claude-judge")

    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=log_f,
    )
    return proc


def merge_csvs(pattern: str, output: str) -> int:
    """Merge all CSVs matching pattern into output. Returns total row count."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  ERROR: no files matching {pattern}")
        return 0

    rows   = []
    header = None
    for f in files:
        try:
            with open(f) as fp:
                reader = csv.DictReader(fp)
                if reader.fieldnames:
                    header = reader.fieldnames
                file_rows = list(reader)
                rows.extend(file_rows)
                print(f"  {f}: {len(file_rows)} rows")
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}")

    if not rows or header is None:
        print("  ERROR: no rows collected")
        return 0

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    with open(output, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Merged {len(rows)} total rows → {output}")
    return len(rows)


def print_summary(csv_path: str, conditions: list[str]) -> None:
    """Print per-condition corruption rates from merged CSV."""
    if not os.path.exists(csv_path):
        return

    from collections import defaultdict
    counts = defaultdict(lambda: {"corrupted": 0, "correct": 0, "incomplete": 0, "total": 0})

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cond = row.get("condition", "")
            if cond not in conditions:
                continue
            counts[cond]["total"] += 1
            v = row.get("verdict", "")
            if v == "CORRUPTED":
                counts[cond]["corrupted"] += 1
            elif v == "CORRECT":
                counts[cond]["correct"] += 1
            else:
                counts[cond]["incomplete"] += 1

    print("\n" + "=" * 60)
    print("MERGED RESULTS SUMMARY")
    print("=" * 60)
    for cond in conditions:
        c = counts[cond]
        t = c["total"]
        if t == 0:
            continue
        corr_r = c["corrupted"] / t
        comp_r = c["correct"] / t
        print(f"  {cond:<25}: n={t:3d} | "
              f"CORRECT={c['correct']:3d}({comp_r*100:5.1f}%) | "
              f"CORRUPTED={c['corrupted']:3d}({corr_r*100:5.1f}%)")

    # Key comparison: FRESH vs STALE
    a = counts.get("structural_fresh",  {"corrupted": 0, "total": 1})
    b = counts.get("structural_stale",  {"corrupted": 0, "total": 1})
    p_a = a["corrupted"] / max(1, a["total"])
    p_b = b["corrupted"] / max(1, b["total"])
    print(f"\n  R_hidden lift (STALE vs FRESH): {(p_b-p_a)*100:+.1f}pp")

    try:
        from scipy import stats
        k1 = a["corrupted"]; n1 = a["total"]
        k2 = b["corrupted"]; n2 = b["total"]
        _, p = stats.fisher_exact([[k1, n1-k1], [k2, n2-k2]], alternative="less")
        print(f"  Fisher's exact (one-sided):     p = {p:.4f}")
        if p < 0.05:
            print("  -> SIGNIFICANT at α=0.05 ✓")
        elif p < 0.10:
            print("  -> Marginal (0.05 < p < 0.10)")
        else:
            print("  -> Not significant (p > 0.10)")
    except ImportError:
        print("  (install scipy for p-value: pip install scipy)")


def main():
    parser = argparse.ArgumentParser(description="Run SJ-v3 in parallel across tasks")
    parser.add_argument("--tasks",            default="datasets/tasks_30_multidomain.json")
    parser.add_argument("--n-tasks",          type=int, default=5)
    parser.add_argument("--n-runs",           type=int, default=25)
    parser.add_argument("--n-steps",          type=int, default=20)
    parser.add_argument("--n-agents",         type=int, default=4)
    parser.add_argument("--injection-step",   type=int, default=8)
    parser.add_argument("--conditions",       nargs="+",
                        default=["structural_fresh", "structural_stale"])
    parser.add_argument("--base-port",        type=int, default=7000)
    parser.add_argument("--sbus-bin",         default="../../sbus/target/release/sbus",
                        help="Path to compiled S-Bus binary")
    parser.add_argument("--output",           default="results/sj_v3_results.csv")
    parser.add_argument("--work-dir",         default="sj_v3_parallel_work")
    parser.add_argument("--use-claude-judge", action="store_true")
    parser.add_argument("--skip-sbus",        action="store_true",
                        help="Skip starting S-Bus (if already running on base-port..base-port+n)")
    args = parser.parse_args()

    # ── Checks ────────────────────────────────────────────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)

    if not args.skip_sbus and not os.path.exists(args.sbus_bin):
        print(f"ERROR: S-Bus binary not found at {args.sbus_bin}")
        print("  Build with: cargo build --release")
        print("  Then set --sbus-bin path/to/target/release/sbus")
        sys.exit(1)

    print("=" * 60)
    print("SJ-v3 Parallel Runner")
    print("=" * 60)
    print(f"Tasks:      {args.n_tasks}")
    print(f"Runs/task:  {args.n_runs}")
    print(f"Steps:      {args.n_steps}")
    print(f"Agents:     {args.n_agents}")
    print(f"Conditions: {args.conditions}")
    print(f"Base port:  {args.base_port}")
    total = args.n_tasks * args.n_runs * len(args.conditions)
    print(f"Total runs: {total}")
    print()

    # ── Split tasks ───────────────────────────────────────────────────────────
    task_dir   = os.path.join(args.work_dir, "tasks")
    log_dir    = os.path.join(args.work_dir, "logs")
    result_dir = os.path.join(args.work_dir, "partial_results")
    os.makedirs(result_dir, exist_ok=True)

    print("Splitting tasks...")
    task_files = split_tasks(args.tasks, args.n_tasks, task_dir)
    print(f"Created {len(task_files)} task files\n")

    sbus_procs = []
    exp_procs  = []

    # ── Cleanup on Ctrl+C ─────────────────────────────────────────────────────
    def cleanup(sig=None, frame=None):
        print("\nCleaning up...")
        for p in exp_procs + sbus_procs:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # ── Start S-Bus instances ─────────────────────────────────────────────────
    ports = [args.base_port + i for i in range(len(task_files))]

    if not args.skip_sbus:
        print("Starting S-Bus instances...")
        for port in ports:
            proc = start_sbus(port, args.sbus_bin, log_dir)
            sbus_procs.append(proc)
            print(f"  S-Bus PID {proc.pid} on port {port}")

        print("Waiting for S-Bus instances to be ready...")
        for port in ports:
            if wait_for_sbus(port, timeout=30):
                print(f"  Port {port}: ready")
            else:
                print(f"  Port {port}: TIMEOUT — check {log_dir}/sbus_{port}.log")
                cleanup()
        print()
    else:
        print("Skipping S-Bus start (--skip-sbus set)\n")

    # ── Start experiment processes ────────────────────────────────────────────
    print("Starting experiment processes...")
    for i, (task_file, port) in enumerate(zip(task_files, ports)):
        task_name  = Path(task_file).stem
        out_path   = os.path.join(result_dir, f"result_{i}_{task_name}.csv")

        proc = start_experiment(
            task_file      = task_file,
            port           = port,
            n_runs         = args.n_runs,
            n_steps        = args.n_steps,
            n_agents       = args.n_agents,
            injection_step = args.injection_step,
            conditions     = args.conditions,
            output         = out_path,
            log_dir        = log_dir,
            use_claude_judge = args.use_claude_judge,
        )
        exp_procs.append(proc)
        print(f"  Task {i} ({Path(task_file).stem}): PID {proc.pid} "
              f"port {port} → {out_path}")

    print(f"\nAll {len(exp_procs)} processes running.")
    print(f"Logs: {log_dir}/")
    print(f"Partial results: {result_dir}/")
    print()

    # ── Monitor progress ──────────────────────────────────────────────────────
    print("Monitoring (Ctrl+C to stop cleanly)...")
    start_time = time.time()
    done = [False] * len(exp_procs)

    while not all(done):
        time.sleep(30)  # check every 30 seconds
        elapsed = int(time.time() - start_time)
        h, m = divmod(elapsed // 60, 60)
        still_running = 0

        for i, proc in enumerate(exp_procs):
            if done[i]:
                continue
            ret = proc.poll()
            if ret is not None:
                done[i] = True
                status = "done" if ret == 0 else f"FAILED (exit {ret})"
                print(f"  [{elapsed//60:3d}m] Task {i}: {status}")
            else:
                still_running += 1

        # Count rows written so far
        partial_files = glob.glob(os.path.join(result_dir, "*.csv"))
        total_rows = 0
        for f in partial_files:
            try:
                with open(f) as fp:
                    total_rows += sum(1 for _ in fp) - 1  # subtract header
            except Exception:
                pass

        print(f"  [{elapsed//60:3d}m] Running: {still_running}/{len(exp_procs)} | "
              f"Rows written: {total_rows}/{total}")

    elapsed = int(time.time() - start_time)
    print(f"\nAll processes finished in {elapsed//60}m {elapsed%60}s")

    # ── Kill S-Bus instances ──────────────────────────────────────────────────
    if not args.skip_sbus:
        print("\nStopping S-Bus instances...")
        for proc in sbus_procs:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("Done.")

    # ── Merge results ─────────────────────────────────────────────────────────
    print("\nMerging results...")
    pattern = os.path.join(result_dir, "*.csv")
    n_rows = merge_csvs(pattern, args.output)

    if n_rows == 0:
        print("ERROR: no results collected. Check logs in:", log_dir)
        sys.exit(1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(args.output, args.conditions)
    print(f"\nFinal results: {args.output}")
    print(f"Logs:          {log_dir}/")


if __name__ == "__main__":
    main()
