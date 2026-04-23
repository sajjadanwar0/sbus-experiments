#!/usr/bin/env python3
"""
run_sjv4_parallel.py — Parallel runner for exp_semantic_judge_v4.py
"""

import argparse
import csv
import glob
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def wait_for_sbus(port: int, timeout: int = 30) -> bool:
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
    os.makedirs(log_dir, exist_ok=True)
    env = os.environ.copy()
    env["SBUS_PORT"]          = str(port)
    env["SBUS_ADMIN_ENABLED"] = "1"
    env["SBUS_SESSION_TTL"]   = "3600"
    env["RUST_LOG"]           = "warn"
    log_f = open(os.path.join(log_dir, f"sbus_{port}.log"), "w")
    return subprocess.Popen([sbus_bin], env=env, stdout=log_f, stderr=log_f)


def start_experiment(
    task_index:     int,
    port:           int,
    n_runs:         int,
    n_steps:        int,
    n_agents:       int,
    injection_step: int,
    output:         str,
    log_dir:        str,
    tasks_file:     str = None,   # ← parameter, not args.tasks_file
) -> subprocess.Popen:
    os.makedirs(log_dir, exist_ok=True)
    env = os.environ.copy()
    env["SBUS_URL"]          = f"http://localhost:{port}"
    env["SJV4_TASK_OFFSET"]  = str(task_index)

    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "exp_semantic_judge_v4.py"
    )

    cmd = [
        sys.executable, script,
        "--n-tasks",        "1",
        "--n-runs",         str(n_runs),
        "--n-steps",        str(n_steps),
        "--n-agents",       str(n_agents),
        "--injection-step", str(injection_step),
        "--output",         output,
    ]
    if tasks_file:
        cmd += ["--tasks-file", tasks_file]

    log_f = open(os.path.join(log_dir, f"exp_port{port}.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)


def merge_csvs(pattern: str, output: str) -> int:
    files  = sorted(glob.glob(pattern))
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
                print(f"  {Path(f).name}: {len(file_rows)} rows")
        except Exception as e:
            print(f"  WARNING {f}: {e}")

    if not rows or not header:
        return 0

    os.makedirs(
        os.path.dirname(output) if os.path.dirname(output) else ".",
        exist_ok=True
    )
    with open(output, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"  Merged {len(rows)} rows → {output}")
    return len(rows)


def print_summary(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        return

    from collections import defaultdict
    counts = defaultdict(
        lambda: {"corrupted": 0, "correct": 0, "incomplete": 0, "total": 0}
    )

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cond = row.get("condition", "")
            counts[cond]["total"] += 1
            v = row.get("verdict", "")
            if v == "CORRUPTED":
                counts[cond]["corrupted"] += 1
            elif v == "CORRECT":
                counts[cond]["correct"] += 1
            else:
                counts[cond]["incomplete"] += 1

    print("\n" + "=" * 60)
    print("SJ-v4 MERGED RESULTS")
    print("=" * 60)

    for cond in ["structural_fresh", "structural_stale"]:
        c = counts[cond]
        t = c["total"]
        if t == 0:
            continue
        print(f"  {cond:<25} n={t:3d} | "
              f"CORRECT={c['correct']:3d}({c['correct']/t*100:5.1f}%) | "
              f"CORRUPTED={c['corrupted']:3d}({c['corrupted']/t*100:5.1f}%)")

    a = counts["structural_fresh"]
    b = counts["structural_stale"]
    p_a = a["corrupted"] / max(1, a["total"])
    p_b = b["corrupted"] / max(1, b["total"])
    print(f"\n  R_hidden lift: {(p_b - p_a)*100:+.1f}pp  "
          f"(FRESH={p_a*100:.1f}%  STALE={p_b*100:.1f}%)")

    try:
        from scipy import stats
        k1, n1 = a["corrupted"], a["total"]
        k2, n2 = b["corrupted"], b["total"]
        if n1 > 0 and n2 > 0:
            _, p = stats.fisher_exact(
                [[k1, n1 - k1], [k2, n2 - k2]], alternative="less"
            )
            print(f"  Fisher's exact (one-sided):  p = {p:.4f}")
            if p < 0.05:
                print("  ✅ SIGNIFICANT — R_hidden causes semantic corruption")
            elif p < 0.10:
                print("  ⚠️  MARGINAL (0.05 < p < 0.10)")
            else:
                print("  ❌ NOT SIGNIFICANT")
    except ImportError:
        print("  (pip install scipy for p-value)")


def main():
    parser = argparse.ArgumentParser(description="SJ-v4 parallel runner")
    parser.add_argument("--tasks-file",
                        default=None,
                        help="JSON task list from generate_sjv4_tasks.py")
    parser.add_argument("--n-tasks",        type=int, default=5)
    parser.add_argument("--n-runs",         type=int, default=25)
    parser.add_argument("--n-steps",        type=int, default=20)
    parser.add_argument("--n-agents",       type=int, default=4)
    parser.add_argument("--injection-step", type=int, default=5)
    parser.add_argument("--base-port",      type=int, default=7010)
    parser.add_argument("--sbus-bin",
                        default="/home/neo/RustroverProjects/sbus/target/release/sbus-server")
    parser.add_argument("--output",
                        default="results/sj_v4_results.csv")
    parser.add_argument("--work-dir",       default="sj_v4_parallel_work")
    parser.add_argument("--skip-sbus",      action="store_true",
                        help="Skip starting S-Bus (already running)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY")
        sys.exit(1)

    if not args.skip_sbus and not os.path.exists(args.sbus_bin):
        print(f"ERROR: S-Bus binary not found: {args.sbus_bin}")
        sys.exit(1)

    log_dir    = os.path.join(args.work_dir, "logs")
    result_dir = os.path.join(args.work_dir, "partial_results")
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("SJ-v4 Parallel Runner")
    print("=" * 60)
    print(f"Tasks:          {args.n_tasks}")
    print(f"Runs/condition: {args.n_runs}")
    print(f"Steps:          {args.n_steps}")
    print(f"Agents:         {args.n_agents}")
    print(f"Injection step: {args.injection_step}")
    print(f"Tasks file:     {args.tasks_file or 'built-in (5 tasks)'}")
    total = args.n_tasks * args.n_runs * 2
    print(f"Total runs:     {total}")
    print()

    ports      = [args.base_port + i for i in range(args.n_tasks)]
    sbus_procs = []
    exp_procs  = []

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
    if not args.skip_sbus:
        print("Starting S-Bus instances...")
        for port in ports:
            proc = start_sbus(port, args.sbus_bin, log_dir)
            sbus_procs.append(proc)
            print(f"  PID {proc.pid} on port {port}")

        print("Waiting for S-Bus instances to be ready...")
        for port in ports:
            ok = wait_for_sbus(port, timeout=30)
            print(f"  Port {port}: {'ready ✓' if ok else 'TIMEOUT ✗'}")
        print()
    else:
        print("Skipping S-Bus start (--skip-sbus)\n")

    # ── Start experiment processes ────────────────────────────────────────────
    print("Starting experiment processes...")
    for i, port in enumerate(ports):
        out_path = os.path.join(result_dir, f"result_{i}_task_{i}.csv")
        proc = start_experiment(
            task_index=i,
            port=port,
            n_runs=args.n_runs,
            n_steps=args.n_steps,
            n_agents=args.n_agents,
            injection_step=args.injection_step,
            output=out_path,
            log_dir=log_dir,
            tasks_file=args.tasks_file,   # ← passed correctly
        )
        exp_procs.append(proc)
        print(f"  Task {i}: PID {proc.pid}  port {port}  → {out_path}")

    print(f"\nAll {len(exp_procs)} processes running.")
    print(f"Monitor: watch -n 30 'wc -l {result_dir}/*.csv'")
    print()

    # ── Monitor ───────────────────────────────────────────────────────────────
    start_time = time.time()
    done       = [False] * len(exp_procs)

    while not all(done):
        time.sleep(30)
        elapsed = int(time.time() - start_time)

        for i, proc in enumerate(exp_procs):
            if done[i]:
                continue
            if proc.poll() is not None:
                done[i] = True
                status  = "done ✓" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                print(f"  [{elapsed//60}m] Task {i}: {status}")

        partial    = glob.glob(os.path.join(result_dir, "*.csv"))
        total_rows = 0
        for f in partial:
            try:
                with open(f) as fp:
                    total_rows += max(0, sum(1 for _ in fp) - 1)
            except Exception:
                pass

        running = sum(1 for d in done if not d)
        print(f"  [{elapsed//60}m] Running: {running}/{len(exp_procs)}  "
              f"Rows: {total_rows}/{total}")

    elapsed = int(time.time() - start_time)
    print(f"\nAll done in {elapsed // 60}m {elapsed % 60}s")

    # ── Stop S-Bus ────────────────────────────────────────────────────────────
    if not args.skip_sbus:
        print("Stopping S-Bus instances...")
        for proc in sbus_procs:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("Done.")

    # ── Merge ─────────────────────────────────────────────────────────────────
    print("\nMerging results...")
    pattern = os.path.join(result_dir, "*.csv")
    n = merge_csvs(pattern, args.output)

    if n == 0:
        print(f"ERROR: no results collected. Check logs in: {log_dir}")
        sys.exit(1)

    print_summary(args.output)
    print(f"\nDone. Results: {args.output}")


if __name__ == "__main__":
    main()
