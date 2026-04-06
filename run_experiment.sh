#!/bin/bash
set -a
source /home/ubuntu/.bashrc
set +a

PYTHON="/home/ubuntu/exp/.venv/bin/python3"
WORKDIR="/home/ubuntu/exp"
LOG_DIR="$WORKDIR/logs"
RESULTS_DIR="$WORKDIR/results"
CONFIG="$WORKDIR/experiments.conf"
SERVER="${SERVER:-http://localhost:7000}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/master.log"
}

# Server check
log "Checking S-Bus server at $SERVER..."
if ! curl -sf "$SERVER/stats" > /dev/null 2>&1; then
  log "ERROR: S-Bus server not reachable. Exiting."
  exit 1
fi
log "Server OK."

# Read config and run each line
while IFS= read -r line; do
  # Skip empty lines and comments
  [[ -z "$line" || "$line" == \#* ]] && continue

  script=$(echo "$line" | awk '{print $1}')
  args=$(echo "$line" | cut -d' ' -f2-)
  name=$(basename "$script" .py)

  log "=============================="
  log "START: $name"
  log "CMD: $script $args"
  log "=============================="

  cd "$WORKDIR"
  $PYTHON "$script" $args 2>&1 | tee -a "$LOG_DIR/${name}.log" "$LOG_DIR/master.log"

  if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    log "DONE: $name ✓"
  else
    log "FAILED: $name — continuing..."
  fi
  echo "" | tee -a "$LOG_DIR/master.log"

done < "$CONFIG"

log "ALL EXPERIMENTS COMPLETE"