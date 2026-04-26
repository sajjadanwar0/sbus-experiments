#!/usr/bin/env bash
set -euo pipefail

: "${SBUS_WORKSPACE:=$HOME/RustroverProjects/sbus-workspace}"
: "${PROXY_REPO:=$HOME/RustroverProjects/sbus-proxy}"
: "${PY_DIR:=$HOME/PycharmProjects/agenticPaper}"


: "${SBUS_BIN:=$SBUS_WORKSPACE/target/release/sbus-server}"
: "${PROXY_BIN:=$PROXY_REPO/target/release/sbus-proxy}"


if [[ ! -x "$SBUS_BIN" && -x "$SBUS_WORKSPACE/target/release/sbus" ]]; then
    SBUS_BIN="$SBUS_WORKSPACE/target/release/sbus"
fi

: "${RESULTS_DIR:=$PY_DIR/results}"
: "${LOG_DIR:=$PY_DIR/logs}"

: "${SBUS_URL:=http://localhost:7000}"
: "${PROXY_URL:=http://localhost:9000}"
: "${SBUS_PORT:=7000}"
: "${PROXY_PORT:=9000}"
: "${BACKBONE:=gpt-4o-mini}"
: "${TEMP:=0.3}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set"; exit 1
fi

MODE="${1:-pilot}"
case "$MODE" in
    pilot)
        DOMAINS="django_queryset,astropy_fits"
        TASKS_PER_DOMAIN=2
        RUNS_PER_TASK=2
        STEPS=6
        ;;
    half)
        DOMAINS="django_queryset,django_migration,astropy_fits,astropy_wcs,sympy_solver"
        TASKS_PER_DOMAIN=2
        RUNS_PER_TASK=3
        STEPS=10
        ;;
    full)
        DOMAINS="all"
        TASKS_PER_DOMAIN=3
        RUNS_PER_TASK=5
        STEPS=20
        ;;
    *)  echo "Usage: $0 [pilot|half|full]"; exit 2 ;;
esac

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
RUN_TAG="proxy_ph2_$(date +%Y%m%d_%H%M%S)_$MODE"
CSV="$RESULTS_DIR/${RUN_TAG}.csv"
SUMMARY="$RESULTS_DIR/${RUN_TAG}_summary.json"
SBUS_LOG="$LOG_DIR/${RUN_TAG}_sbus.log"
PROXY_LOG="$LOG_DIR/${RUN_TAG}_proxy.log"

echo "=========================================================="
echo "Exp. PROXY-PH2 (mode=$MODE)"
echo "=========================================================="
echo "Paths:"
echo "  SBUS_WORKSPACE = $SBUS_WORKSPACE"
echo "  SBUS_BIN       = $SBUS_BIN"
echo "  PROXY_REPO     = $PROXY_REPO"
echo "  PROXY_BIN      = $PROXY_BIN"
echo "  PY_DIR         = $PY_DIR"
echo "URLs:"
echo "  SBUS_URL       = $SBUS_URL"
echo "  PROXY_URL      = $PROXY_URL"
echo "Output:"
echo "  CSV            = $CSV"
echo "  Summary        = $SUMMARY"
echo "=========================================================="

VOCAB=$(cd "$PY_DIR" && python3 - <<'PY'
from exp_proxy_ph2 import DOMAIN_VOCAB
print(",".join(DOMAIN_VOCAB))
PY
)
echo "Vocab ($(echo "$VOCAB" | tr ',' '\n' | wc -l) shards)"

is_listening() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -lnt 2>/dev/null | awk -v p=":$port" '$4 ~ p {found=1} END {exit !found}'
    elif command -v netstat >/dev/null 2>&1; then
        netstat -lnt 2>/dev/null | awk -v p=":$port" '$4 ~ p {found=1} END {exit !found}'
    else
        (exec 3<>/dev/tcp/localhost/"$port") 2>/dev/null && { exec 3<&-; exec 3>&-; return 0; } || return 1
    fi
}

wait_for_http() {
    local url="$1" name="$2" max_wait="${3:-30}"
    for i in $(seq 1 "$max_wait"); do
        if curl -fs "$url" >/dev/null 2>&1; then
            echo "  $name up at $url (after ${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: $name did not come up at $url within ${max_wait}s"
    return 1
}

cleanup() {
    [[ -f "$LOG_DIR/${RUN_TAG}_sbus.pid"  ]] && {
        echo "Stopping sbus-server (we started it)"
        kill "$(cat "$LOG_DIR/${RUN_TAG}_sbus.pid")"  2>/dev/null || true
        rm -f "$LOG_DIR/${RUN_TAG}_sbus.pid"
    }
    [[ -f "$LOG_DIR/${RUN_TAG}_proxy.pid" ]] && {
        echo "Stopping sbus-proxy (we started it)"
        kill "$(cat "$LOG_DIR/${RUN_TAG}_proxy.pid")" 2>/dev/null || true
        rm -f "$LOG_DIR/${RUN_TAG}_proxy.pid"
    }
}
trap cleanup EXIT

if is_listening "$SBUS_PORT"; then
    echo "S-Bus already listening on :$SBUS_PORT — using existing instance"
    admin_code=$(curl -s -o /dev/null -w "%{http_code}" "$SBUS_URL/admin/delivery-log" || echo "000")
    if [[ "$admin_code" == "403" ]]; then
        echo "ERROR: S-Bus is running but admin mode is OFF (/admin/delivery-log → 403)"
        echo "       The experiment requires admin mode to read DeliveryLog state."
        echo
        echo "       If S-Bus runs as a systemd service, add to the unit file:"
        echo "         Environment=SBUS_ADMIN_ENABLED=1"
        echo "       then: sudo systemctl daemon-reload && sudo systemctl restart sbus-server"
        echo
        echo "       If you started it manually, stop it and restart with:"
        echo "         SBUS_ADMIN_ENABLED=1 $SBUS_BIN --http-port $SBUS_PORT"
        exit 1
    elif [[ "$admin_code" != "200" ]]; then
        echo "WARNING: unexpected HTTP $admin_code from /admin/delivery-log; continuing"
    fi
else
    if [[ ! -x "$SBUS_BIN" ]]; then
        echo "ERROR: SBUS_BIN not found or not executable: $SBUS_BIN"
        echo "       Build it first:  (cd $SBUS_WORKSPACE && cargo build --release)"
        echo "       Or override:     SBUS_BIN=/path/to/binary ./run_proxy_ph2.sh $MODE"
        exit 1
    fi
    echo "Starting sbus-server on :$SBUS_PORT (log: $SBUS_LOG)"
    SBUS_ADMIN_ENABLED=1 RUST_LOG="sbus_server=info" \
        "$SBUS_BIN" --http-port "$SBUS_PORT" > "$SBUS_LOG" 2>&1 &
    echo $! > "$LOG_DIR/${RUN_TAG}_sbus.pid"
    wait_for_http "$SBUS_URL/health" "sbus-server" 30
fi

if is_listening "$PROXY_PORT"; then
    echo "sbus-proxy already listening on :$PROXY_PORT — using existing instance"
    echo "NOTE: if this proxy pre-dates the v50 shard-suffix patch, the preflight"
    echo "      check at step 3 will fail. Restart with the patched binary if so."
else
    if [[ ! -x "$PROXY_BIN" ]]; then
        echo "ERROR: PROXY_BIN not found or not executable: $PROXY_BIN"
        echo "       Build it first:  (cd $PROXY_REPO && cargo build --release)"
        echo "       Or override:     PROXY_BIN=/path/to/binary ./run_proxy_ph2.sh $MODE"
        exit 1
    fi
    echo "Starting sbus-proxy on :$PROXY_PORT (log: $PROXY_LOG)"
    SBUS_PROXY_VOCAB="$VOCAB" \
    SBUS_PROXY_UPSTREAM_URL="https://api.openai.com" \
    SBUS_URL="$SBUS_URL" \
    RUST_LOG="sbus_proxy=info" \
        "$PROXY_BIN" --listen-port "$PROXY_PORT" > "$PROXY_LOG" 2>&1 &
    echo $! > "$LOG_DIR/${RUN_TAG}_proxy.pid"
    wait_for_http "$PROXY_URL/health" "sbus-proxy" 15
fi

echo "Running preflight…"
SBUS_URL="$SBUS_URL" PROXY_URL="$PROXY_URL" \
PREFLIGHT_VOCAB_HIT="models_state" \
python3 "$PY_DIR/preflight_proxy.py"

echo
echo "Running Exp. PROXY-PH2 ($MODE)…"
echo "  domains=$DOMAINS  tasks/domain=$TASKS_PER_DOMAIN"
echo "  runs/task=$RUNS_PER_TASK  steps=$STEPS"
echo

cd "$PY_DIR"
SBUS_URL="$SBUS_URL" PROXY_URL="$PROXY_URL" BACKBONE="$BACKBONE" TEMP="$TEMP" \
python3 exp_proxy_ph2.py \
    --domains          "$DOMAINS" \
    --tasks-per-domain "$TASKS_PER_DOMAIN" \
    --runs-per-task    "$RUNS_PER_TASK" \
    --steps            "$STEPS" \
    --output           "$CSV" \
    --summary          "$SUMMARY"

echo
echo "=========================================================="
echo "Done."
echo "  CSV     : $CSV"
echo "  Summary : $SUMMARY"
[[ -f "$SBUS_LOG"  ]] && echo "  S-Bus log  : $SBUS_LOG"
[[ -f "$PROXY_LOG" ]] && echo "  Proxy log  : $PROXY_LOG"
echo "=========================================================="