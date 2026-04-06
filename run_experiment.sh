#!/bin/bash
# =============================================================================
# S-Bus Dynamic Experiment Runner
# Auto-discovers and runs ALL Python files in the project directory.
#
# Usage:
#   ./run_experiments.sh                       # run everything
#   ./run_experiments.sh --skip "arsi_*"       # skip files matching pattern
#   ./run_experiments.sh --only sdk_compare.py # run one specific file
#   ./run_experiments.sh --dry-run             # print commands, don't execute
# =============================================================================

set -a
source /home/ubuntu/exp/.env 2>/dev/null || true
set +a

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON="/home/ubuntu/exp/.venv/bin/python3"
WORKDIR="/home/ubuntu/exp"
LOG_DIR="$WORKDIR/logs"
RESULTS_DIR="$WORKDIR/results"
SERVER="${SERVER:-http://localhost:7000}"

# Files to never auto-run (utility/helper scripts, not experiments)
EXCLUDE_PATTERNS="convert_swebench.py s50_analysis.py"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# ── CLI args ──────────────────────────────────────────────────────────────────
SKIP_PATTERN=""
ONLY_FILE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip)    SKIP_PATTERN="$2"; shift 2 ;;
    --only)    ONLY_FILE="$2";    shift 2 ;;
    --dry-run) DRY_RUN=true;      shift ;;
    *) shift ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/master.log"
}

is_excluded() {
  local file=$(basename "$1")
  for pattern in $EXCLUDE_PATTERNS $SKIP_PATTERN; do
    [[ "$file" == $pattern ]] && return 0
  done
  return 1
}

run_script() {
  local script="$1"
  local name=$(basename "$script" .py)
  local log_file="$LOG_DIR/${name}.log"

  log "=============================="
  log "START: $name"
  log "=============================="

  if $DRY_RUN; then
    log "DRY-RUN: $PYTHON $script"
    return
  fi

  # Try with --server and --out first; fall back to bare run if argparse rejects them
  {
    $PYTHON "$script" --server "$SERVER" --out "$RESULTS_DIR/${name}.csv" 2>&1 \
    || $PYTHON "$script" --server "$SERVER" 2>&1 \
    || $PYTHON "$script" 2>&1
  } | tee -a "$log_file" "$LOG_DIR/master.log"

  local exit_code=${PIPESTATUS[0]}
  if [[ $exit_code -eq 0 ]]; then
    log "DONE: $name ✓"
  else
    log "FAILED: $name (exit $exit_code) — continuing..."
  fi
  echo "" | tee -a "$LOG_DIR/master.log"
}

# ── Server check ──────────────────────────────────────────────────────────────
log "Checking S-Bus server at $SERVER..."
if ! curl -sf "$SERVER/stats" > /dev/null 2>&1; then
  log "ERROR: S-Bus server not reachable at $SERVER. Exiting."
  exit 1
fi
log "Server OK."

# ── Auto-discover and run ─────────────────────────────────────────────────────
cd "$WORKDIR"

log "Scanning $WORKDIR for Python scripts..."

SCRIPTS=()
for f in "$WORKDIR"/*.py; do
  [[ -f "$f" ]] || continue
  is_excluded "$f" && { log "EXCLUDED: $(basename $f)"; continue; }
  [[ -n "$ONLY_FILE" && "$(basename $f)" != "$ONLY_FILE" ]] && continue
  SCRIPTS+=("$f")
done

log "Found ${#SCRIPTS[@]} scripts to run:"
for s in "${SCRIPTS[@]}"; do log "  - $(basename $s)"; done
log ""

for script in "${SCRIPTS[@]}"; do
  run_script "$script"
done

log "=============================="
log "ALL EXPERIMENTS COMPLETE"
log "Logs:    $LOG_DIR"
log "Results: $RESULTS_DIR"
log "=============================="