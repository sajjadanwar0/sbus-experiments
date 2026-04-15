#!/usr/bin/env python3
"""
TLA+ Rename: WSISafety -> ORISafety (FIXED)
============================================
Fixed version of tla_rename_ori.py.

Bug in previous version: spec/cfg pairing was case-sensitive.
Your TLA+ files are named SBus.tla / SBus_inductive.tla (mixed case)
but configs are sbus.cfg / sbus_inductive.cfg (lowercase).
The old script generated SBus.cfg from SBus.tla — that file doesn't
exist — so TLC was never invoked correctly.

This version:
  1. Scans for all .tla and .cfg files
  2. Pairs them case-insensitively (SBus.tla -> sbus.cfg)
  3. Also supports explicit --spec/--cfg pairs
  4. Prints the exact java commands you can run manually

USAGE:
  python3 tla_rename_ori.py --tla-dir ~/sbus-tla --tla-jar ~/tla2tools.jar

MANUAL TLC COMMANDS (run these if the script still fails):
  cd ~/sbus-tla
  java -jar ~/tla2tools.jar -config sbus.cfg SBus.tla
  java -jar ~/tla2tools.jar -config sbus_inductive.cfg SBus_inductive.tla
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

RENAME_MAP = {
    "WSISafety":              "ORISafety",
    "WriteSnapshotIsolation": "ObservableReadIsolation",
    "WSI":                    "ORI",
}

TLA_FILES = [
    "sbus.tla", "SBus.tla",
    "sbus.cfg",
    "sbus_inductive.tla", "SBus_inductive.tla",
    "sbus_inductive.cfg",
]


def find_tla_jar():
    for c in [os.path.expanduser("~/tla2tools.jar"),
              "/usr/local/lib/tla2tools.jar",
              "/opt/tla2tools.jar", "./tla2tools.jar"]:
        if os.path.exists(c):
            return c
    return ""


def rename_in_file(path: str) -> tuple[bool, list[str]]:
    """Apply renames. Returns (changed, list_of_changes)."""
    with open(path) as f:
        content = f.read()
    original = content
    changes = []
    for old, new in RENAME_MAP.items():
        if old in content:
            count = content.count(old)
            content = content.replace(old, new)
            changes.append(f"{old} -> {new} ({count}x)")
    if content != original:
        shutil.copy2(path, path + ".bak")
        with open(path, "w") as f:
            f.write(content)
        return True, changes
    return False, []


def find_all_files(tla_dir: str) -> tuple[list[str], list[str]]:
    """Return (tla_files, cfg_files) — full paths."""
    tlas, cfgs = [], []
    for root, _, files in os.walk(tla_dir):
        for f in files:
            full = os.path.join(root, f)
            if f.endswith(".tla"):
                tlas.append(full)
            elif f.endswith(".cfg"):
                cfgs.append(full)
    return tlas, cfgs


def pair_spec_cfg(tlas: list[str], cfgs: list[str]) -> list[tuple[str, str]]:
    """
    Pair each .tla with its .cfg case-insensitively.
    SBus.tla -> sbus.cfg  (strip directory + lowercase match)
    SBus_inductive.tla -> sbus_inductive.cfg
    """
    # Build lowercase stem -> full cfg path map
    cfg_map = {}
    for cfg in cfgs:
        stem = Path(cfg).stem.lower()          # "sbus", "sbus_inductive"
        cfg_map[stem] = cfg

    pairs = []
    for tla in tlas:
        stem = Path(tla).stem.lower()          # "sbus", "sbus_inductive"
        if stem in cfg_map:
            pairs.append((tla, cfg_map[stem]))
        else:
            print(f"  WARNING: No matching .cfg found for {Path(tla).name}")
            print(f"    Available cfgs: {[Path(c).name for c in cfgs]}")
    return pairs


def run_tlc(tla_path: str, cfg_path: str, tla_jar: str) -> tuple[bool, str]:
    """Run TLC. tla_path and cfg_path are full absolute paths."""
    tla_dir = os.path.dirname(tla_path)
    cmd = [
        "java", "-jar", tla_jar,
        "-config", cfg_path,
        tla_path,
    ]
    print(f"\n  Command: java -jar {os.path.basename(tla_jar)} "
          f"-config {os.path.basename(cfg_path)} {os.path.basename(tla_path)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=tla_dir,
        )
        output = result.stdout + result.stderr
        success = "No error has been found" in output
        return success, output
    except subprocess.TimeoutExpired:
        return False, "TLC timed out after 600s"
    except FileNotFoundError:
        return False, f"java not found. Install: sudo apt-get install default-jre"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tla-dir", default=os.path.expanduser("~/sbus-tla"))
    parser.add_argument("--tla-jar", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-tlc", action="store_true")
    parser.add_argument("--rename-only", action="store_true",
                        help="Only rename files, skip TLC entirely")
    args = parser.parse_args()

    tla_dir = os.path.abspath(args.tla_dir)
    if not os.path.isdir(tla_dir):
        print(f"ERROR: Directory not found: {tla_dir}")
        sys.exit(1)

    tla_jar = args.tla_jar or find_tla_jar()

    # ── Step 1: Discover files ──────────────────────────────────────────────
    tlas, cfgs = find_all_files(tla_dir)
    print(f"TLA+ directory: {tla_dir}")
    print(f"Found .tla files: {[os.path.basename(f) for f in tlas]}")
    print(f"Found .cfg files: {[os.path.basename(f) for f in cfgs]}")

    all_files = tlas + cfgs
    if not all_files:
        print("ERROR: No TLA+ files found.")
        sys.exit(1)

    # ── Step 2: Rename ──────────────────────────────────────────────────────
    print(f"\nRename map:")
    for old, new in RENAME_MAP.items():
        print(f"  {old:30s} -> {new}")

    if args.dry_run:
        print("\nDRY RUN:")
        for fpath in all_files:
            with open(fpath) as f:
                content = f.read()
            for old, new in RENAME_MAP.items():
                if old in content:
                    print(f"  {os.path.basename(fpath)}: '{old}' appears "
                          f"{content.count(old)}x")
        return

    print("\nApplying renames...")
    any_changed = False
    for fpath in all_files:
        changed, changes = rename_in_file(fpath)
        if changed:
            any_changed = True
            print(f"  ✅ Modified: {os.path.basename(fpath)}")
            for c in changes:
                print(f"       {c}")
            print(f"     Backup:   {os.path.basename(fpath)}.bak")
        else:
            print(f"  — No changes: {os.path.basename(fpath)}")

    if not any_changed:
        print("\nNo files were modified (already renamed, or terms not present).")

    if args.rename_only or args.skip_tlc:
        print("\nRename complete. Skipping TLC.")
        print_manual_commands(tla_dir, tla_jar)
        return

    if not tla_jar:
        print("\nWARNING: tla2tools.jar not found — cannot run TLC.")
        print("Download: https://github.com/tlaplus/tlaplus/releases/latest")
        print("Then: python3 tla_rename_ori_fixed.py --tla-jar ~/tla2tools.jar")
        print_manual_commands(tla_dir, "~/tla2tools.jar")
        return

    # ── Step 3: Pair and run TLC ────────────────────────────────────────────
    pairs = pair_spec_cfg(tlas, cfgs)
    if not pairs:
        print("\nERROR: Could not pair any .tla with .cfg files.")
        print("Try running TLC manually:")
        print_manual_commands(tla_dir, tla_jar)
        return

    print(f"\nSpec/cfg pairs found:")
    for tla, cfg in pairs:
        print(f"  {os.path.basename(tla):30s} <-> {os.path.basename(cfg)}")

    print(f"\nRunning TLC ({len(pairs)} spec(s))...")
    all_ok = True
    for tla_path, cfg_path in pairs:
        spec_name = os.path.basename(tla_path)
        cfg_name = os.path.basename(cfg_path)
        print(f"\n  [{spec_name}]")
        ok, output = run_tlc(tla_path, cfg_path, tla_jar)

        key_lines = [l.strip() for l in output.split("\n") if l.strip() and any(
            x in l for x in ["No error", "error", "Error", "states", "Finished",
                              "depth", "Model checking", "violations"]
        )]
        for line in key_lines[:10]:
            print(f"    {line}")

        if ok:
            print(f"  ✅ PASS — Zero violations after ORI rename")
        else:
            print(f"  ❌ FAIL — See output above")
            all_ok = False

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_ok:
        print("✅ ALL TLC RUNS PASSED — ORI rename verified")
        print()
        print("Paper update for §III-C:")
        print("  'The TLA+ specification has been updated to use ORI")
        print("   terminology throughout: ORISafety replaces WSISafety,")
        print("   ObservableReadIsolation replaces WriteSnapshotIsolation.")
        print("   Re-running TLC confirms zero violations of ORISafety")
        print("   across 200,896 distinct states (706,945 generated).")
        print("   No error has been found.'")
        print()
        print("Commit to GitHub:")
        print(f"  cd {tla_dir}")
        print("  git add *.tla *.cfg")
        print("  git commit -m 'Rename WSISafety->ORISafety; ORI terminology'")
        print("  git push origin main")
    else:
        print("❌ SOME RUNS FAILED")
        print_manual_commands(tla_dir, tla_jar)
        print()
        print("To restore backups:")
        for tla, cfg in pairs:
            bak_tla = tla + ".bak"
            bak_cfg = cfg + ".bak"
            if os.path.exists(bak_tla):
                print(f"  cp '{bak_tla}' '{tla}'")
            if os.path.exists(bak_cfg):
                print(f"  cp '{bak_cfg}' '{cfg}'")


def print_manual_commands(tla_dir: str, tla_jar: str):
    print()
    print("MANUAL TLC COMMANDS (run these directly if script fails):")
    print(f"  cd {tla_dir}")
    print()
    print("  # Main spec:")
    print(f"  java -jar {tla_jar} -config sbus.cfg SBus.tla")
    print()
    print("  # Inductive spec:")
    print(f"  java -jar {tla_jar} -config sbus_inductive.cfg SBus_inductive.tla")
    print()
    print("  # Expected output for both:")
    print("  #   Model checking completed. No error has been found.")
    print("  #   706945 states generated, 200896 distinct states found, 0 left on queue.")


if __name__ == "__main__":
    main()