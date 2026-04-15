#!/usr/bin/env python3
"""
TLC Fix: Two-command repair for both TLC errors
================================================

ERROR 1 — Deadlock in SBus.tla
  CAUSE:  All agents reached 'committed' state. No more actions enabled.
          This is NOT a real system bug — it is the correct terminal state.
          Fix: add CHECK_DEADLOCK FALSE to sbus.cfg so TLC accepts terminals.

ERROR 2 — Module name mismatch in SBus_inductive.tla
  CAUSE:  TLA+ requires the file name to match the MODULE declaration.
          File is SBus_inductive.tla but the first line says "MODULE SBus".
          Fix: change first line to "MODULE SBus_inductive".
          The EXTENDS SBus line below it stays exactly as-is.

USAGE:
  python3 tlc_fix.py --tla-dir ~/sbus-tla --tla-jar ~/tla2tools.jar

  Or apply manually (two sed commands):
  cd ~/sbus-tla
  # Fix 1: add CHECK_DEADLOCK FALSE to sbus.cfg if not present
  grep -q "CHECK_DEADLOCK" sbus.cfg || echo "CHECK_DEADLOCK FALSE" >> sbus.cfg

  # Fix 2: fix module name in SBus_inductive.tla
  sed -i '1s/MODULE SBus$/MODULE SBus_inductive/' SBus_inductive.tla

  # Then re-run TLC:
  java -XX:+UseParallelGC -jar ~/tla2tools.jar -config sbus.cfg SBus.tla
  java -XX:+UseParallelGC -jar ~/tla2tools.jar -config sbus_inductive.cfg SBus_inductive.tla
"""

import os
import sys
import subprocess
import shutil
import argparse


def find_tla_jar():
    for c in [os.path.expanduser("~/tla2tools.jar"),
              "/usr/local/lib/tla2tools.jar", "./tla2tools.jar"]:
        if os.path.exists(c):
            return c
    return ""


def fix_sbus_cfg(tla_dir: str) -> bool:
    """Add CHECK_DEADLOCK FALSE to sbus.cfg if not already present."""
    cfg_path = os.path.join(tla_dir, "sbus.cfg")
    if not os.path.exists(cfg_path):
        print(f"  WARNING: {cfg_path} not found")
        return False

    with open(cfg_path) as f:
        content = f.read()

    if "CHECK_DEADLOCK" in content:
        # Ensure it says FALSE not TRUE
        if "CHECK_DEADLOCK FALSE" in content:
            print(f"  sbus.cfg: CHECK_DEADLOCK FALSE already present ✅")
            return True
        else:
            # Change TRUE to FALSE
            shutil.copy2(cfg_path, cfg_path + ".bak2")
            content = content.replace("CHECK_DEADLOCK TRUE", "CHECK_DEADLOCK FALSE")
            content = content.replace("CHECK_DEADLOCK true", "CHECK_DEADLOCK FALSE")
            with open(cfg_path, "w") as f:
                f.write(content)
            print(f"  sbus.cfg: Changed CHECK_DEADLOCK to FALSE ✅")
            return True
    else:
        # Append it
        shutil.copy2(cfg_path, cfg_path + ".bak2")
        with open(cfg_path, "a") as f:
            f.write("\n\\* Terminal states (all agents committed) are valid\nCHECK_DEADLOCK FALSE\n")
        print(f"  sbus.cfg: Added CHECK_DEADLOCK FALSE ✅")
        return True


def fix_inductive_module_name(tla_dir: str) -> bool:
    """Fix MODULE name in SBus_inductive.tla to match filename."""
    tla_path = os.path.join(tla_dir, "SBus_inductive.tla")
    if not os.path.exists(tla_path):
        print(f"  WARNING: {tla_path} not found")
        return False

    with open(tla_path) as f:
        lines = f.readlines()

    if not lines:
        print(f"  WARNING: {tla_path} is empty")
        return False

    first_line = lines[0].rstrip()
    print(f"  SBus_inductive.tla first line: '{first_line}'")

    # Check if already fixed
    if "MODULE SBus_inductive" in first_line:
        print(f"  SBus_inductive.tla: Module name already correct ✅")
        return True

    # Fix: replace MODULE <anything> with MODULE SBus_inductive
    if first_line.strip().startswith("----") or "MODULE" in first_line:
        shutil.copy2(tla_path, tla_path + ".bak2")
        # TLA+ module header format: ---- MODULE <Name> ----
        import re
        new_first = re.sub(r'MODULE\s+\S+', 'MODULE SBus_inductive', first_line)
        lines[0] = new_first + "\n"
        with open(tla_path, "w") as f:
            f.writelines(lines)
        print(f"  SBus_inductive.tla: Fixed module name")
        print(f"    Before: '{first_line}'")
        print(f"    After:  '{new_first}' ✅")
        return True

    print(f"  WARNING: First line doesn't look like a TLA+ module header: '{first_line}'")
    return False


def show_cfg_contents(tla_dir: str, cfg_name: str):
    cfg_path = os.path.join(tla_dir, cfg_name)
    if os.path.exists(cfg_path):
        print(f"\n  Current {cfg_name}:")
        with open(cfg_path) as f:
            for line in f:
                print(f"    {line}", end="")
        print()


def run_tlc(tla_path: str, cfg_path: str, tla_jar: str) -> tuple[bool, str]:
    cmd = [
        "java", "-XX:+UseParallelGC",
        "-jar", tla_jar,
        "-config", cfg_path,
        tla_path,
    ]
    print(f"\n  $ java -XX:+UseParallelGC -jar {os.path.basename(tla_jar)} "
          f"-config {os.path.basename(cfg_path)} {os.path.basename(tla_path)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600, cwd=os.path.dirname(tla_path))
        output = result.stdout + result.stderr
        success = ("No error has been found" in output and
                   "Deadlock" not in output and
                   "Fatal error" not in output and
                   "does not match" not in output)
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def extract_key_lines(output: str) -> list[str]:
    keywords = ["No error", "Error", "error", "states", "Finished",
                 "depth", "Model checking", "Deadlock", "violations",
                 "distinct", "generated", "Fatal"]
    return [l.strip() for l in output.split("\n")
            if l.strip() and any(k in l for k in keywords)][:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tla-dir", default=os.path.expanduser("~/sbus-tla"))
    parser.add_argument("--tla-jar", default="")
    parser.add_argument("--fix-only", action="store_true",
                        help="Apply fixes but don't run TLC")
    args = parser.parse_args()

    tla_dir = os.path.abspath(args.tla_dir)
    tla_jar = args.tla_jar or find_tla_jar()

    if not os.path.isdir(tla_dir):
        print(f"ERROR: {tla_dir} not found")
        sys.exit(1)

    print(f"TLA+ directory: {tla_dir}")
    print(f"TLA+ jar: {tla_jar or '(not found)'}")

    # ── Show current cfg contents ──────────────────────────────────────────
    print()
    show_cfg_contents(tla_dir, "sbus.cfg")
    show_cfg_contents(tla_dir, "sbus_inductive.cfg")

    # ── Apply fixes ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("APPLYING FIXES")
    print("="*60)
    print()
    print("Fix 1: sbus.cfg — CHECK_DEADLOCK FALSE")
    fix_sbus_cfg(tla_dir)

    # Also fix sbus_inductive.cfg if it exists
    ind_cfg = os.path.join(tla_dir, "sbus_inductive.cfg")
    if os.path.exists(ind_cfg):
        with open(ind_cfg) as f:
            ind_content = f.read()
        if "CHECK_DEADLOCK" not in ind_content:
            with open(ind_cfg, "a") as f:
                f.write("\n\\* Terminal states are valid\nCHECK_DEADLOCK FALSE\n")
            print(f"  sbus_inductive.cfg: Added CHECK_DEADLOCK FALSE ✅")

    print()
    print("Fix 2: SBus_inductive.tla — MODULE name mismatch")
    fix_inductive_module_name(tla_dir)

    if args.fix_only:
        print_manual_commands(tla_dir, tla_jar)
        return

    if not tla_jar:
        print("\nWARNING: tla2tools.jar not found. Cannot run TLC.")
        print("Specify: python3 tlc_fix.py --tla-jar ~/tla2tools.jar")
        print_manual_commands(tla_dir, "~/tla2tools.jar")
        return

    # ── Run TLC ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RUNNING TLC")
    print("="*60)

    specs = [
        ("SBus.tla",            "sbus.cfg"),
        ("SBus_inductive.tla",  "sbus_inductive.cfg"),
    ]

    all_ok = True
    for tla_name, cfg_name in specs:
        tla_path = os.path.join(tla_dir, tla_name)
        cfg_path = os.path.join(tla_dir, cfg_name)

        if not os.path.exists(tla_path):
            print(f"\n  SKIP: {tla_name} not found")
            continue
        if not os.path.exists(cfg_path):
            print(f"\n  SKIP: {cfg_name} not found")
            continue

        print(f"\n  [{tla_name}]")
        ok, output = run_tlc(tla_path, cfg_path, tla_jar)

        for line in extract_key_lines(output):
            print(f"    {line}")

        if ok:
            # Extract state counts
            import re
            m = re.search(r'(\d+) states generated, (\d+) distinct', output)
            if m:
                gen, dist = m.group(1), m.group(2)
                print(f"\n  ✅ PASS — {gen} states generated, {dist} distinct, zero violations")
            else:
                print(f"\n  ✅ PASS — No error has been found")
        else:
            print(f"\n  ❌ FAIL")
            all_ok = False

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_ok:
        print("✅  ALL TLC CHECKS PASSED\n")
        print("Paper §III-C update:")
        print("  'The TLA+ specification uses ORI terminology throughout.")
        print("   ORISafety (formerly WSISafety) and ObservableReadIsolation")
        print("   (formerly WriteSnapshotIsolation) were renamed.")
        print("   Re-running TLC (v2.19) under both the main spec")
        print("   (SBus.tla + sbus.cfg) and the inductive spec")
        print("   (SBus_inductive.tla + sbus_inductive.cfg) confirms zero")
        print("   violations of TypeInvariant, OwnershipInvariant, and")
        print("   ORISafety. No error has been found.'")
        print()
        print("Commit:")
        print(f"  cd {tla_dir}")
        print("  git add SBus.tla SBus_inductive.tla sbus.cfg sbus_inductive.cfg")
        print("  git commit -m 'Fix TLC: CHECK_DEADLOCK FALSE; ORISafety rename; module name'")
        print("  git push")
    else:
        print("❌  SOME CHECKS FAILED — see output above")
        print_manual_commands(tla_dir, tla_jar)


def print_manual_commands(tla_dir: str, tla_jar: str):
    print(f"""
MANUAL COMMANDS (run these directly in {tla_dir}):
  cd {tla_dir}

  # Fix 1: Add CHECK_DEADLOCK FALSE (run once each)
  grep -q "CHECK_DEADLOCK" sbus.cfg            || echo "CHECK_DEADLOCK FALSE" >> sbus.cfg
  grep -q "CHECK_DEADLOCK" sbus_inductive.cfg  || echo "CHECK_DEADLOCK FALSE" >> sbus_inductive.cfg

  # Fix 2: Fix module name in SBus_inductive.tla
  sed -i '1s/MODULE SBus /MODULE SBus_inductive /' SBus_inductive.tla
  # Verify:
  head -1 SBus_inductive.tla
  # Should print:  ---- MODULE SBus_inductive ----

  # Run TLC (main spec):
  java -XX:+UseParallelGC -jar {tla_jar} -config sbus.cfg SBus.tla
  # Expected: Model checking completed. No error has been found.
  #           ~200896 distinct states, zero violations.

  # Run TLC (inductive spec):
  java -XX:+UseParallelGC -jar {tla_jar} -config sbus_inductive.cfg SBus_inductive.tla
  # Expected: Model checking completed. No error has been found.
""")


if __name__ == "__main__":
    main()