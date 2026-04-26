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

DOMAINS="django_queryset,astropy_fits,sympy_solver"
TASKS_PER_DOMAIN=2
RUNS_PER_TASK=2
STEPS=6

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
RUN_TAG="proxy_ph2_vocab_$(date +%Y%m%d_%H%M%S)"
SBUS_LOG="$LOG_DIR/${RUN_TAG}_sbus.log"
PROXY_LOG="$LOG_DIR/${RUN_TAG}_proxy.log"

echo "Exp. PROXY-PH2 VOCAB-SCALING APPENDIX (v50.1)"
echo "Domains:         $DOMAINS"
echo "Tasks/domain:    $TASKS_PER_DOMAIN"
echo "Runs/task:       $RUNS_PER_TASK"
echo "Steps:           $STEPS"
echo "Vocab sweep:     4, 8, 12 shards per domain"
echo "Max fetch/step:  1 (forces sparsity)"

VOCAB=$(cd "$PY_DIR" && python3 - <<'PY'
from exp_proxy_ph2 import DOMAIN_VOCAB
print(",".join(DOMAIN_VOCAB))
PY
)
echo "Proxy vocab: $(echo "$VOCAB" | tr ',' '\n' | wc -l) shards"

is_listening() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -lnt 2>/dev/null | awk -v p=":$port" '$4 ~ p {f=1} END {exit !f}'
    elif command -v netstat >/dev/null 2>&1; then
        netstat -lnt 2>/dev/null | awk -v p=":$port" '$4 ~ p {f=1} END {exit !f}'
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
        echo "ERROR: S-Bus admin mode is OFF. See run_proxy_ph2.sh for fix."
        exit 1
    fi
else
    if [[ ! -x "$SBUS_BIN" ]]; then
        echo "ERROR: SBUS_BIN not found at $SBUS_BIN. Build:  (cd $SBUS_WORKSPACE && cargo build --release)"
        exit 1
    fi
    SBUS_ADMIN_ENABLED=1 RUST_LOG="sbus_server=info" \
        "$SBUS_BIN" --http-port "$SBUS_PORT" > "$SBUS_LOG" 2>&1 &
    echo $! > "$LOG_DIR/${RUN_TAG}_sbus.pid"
    wait_for_http "$SBUS_URL/admin/delivery-log" "sbus-server" 30
fi

if is_listening "$PROXY_PORT"; then
    echo "sbus-proxy already listening on :$PROXY_PORT — using existing instance"
    echo "NOTE: restart with the full 120-shard SBUS_PROXY_VOCAB if you haven't yet."
else
    if [[ ! -x "$PROXY_BIN" ]]; then
        echo "ERROR: PROXY_BIN not found at $PROXY_BIN. Build:  (cd $PROXY_REPO && cargo build --release)"
        exit 1
    fi
    SBUS_PROXY_VOCAB="$VOCAB" \
    SBUS_PROXY_UPSTREAM_URL="https://api.openai.com" \
    SBUS_URL="$SBUS_URL" \
    RUST_LOG="sbus_proxy=info" \
        "$PROXY_BIN" --listen-port "$PROXY_PORT" > "$PROXY_LOG" 2>&1 &
    echo $! > "$LOG_DIR/${RUN_TAG}_proxy.pid"
    wait_for_http "$PROXY_URL/health" "sbus-proxy" 15
fi

for V in 4 8 12; do
    CSV="$RESULTS_DIR/${RUN_TAG}_vocab${V}.csv"
    SUMMARY="$RESULTS_DIR/${RUN_TAG}_vocab${V}_summary.json"

    echo
    echo "──────────────────────────────────────────────────────────"
    echo "  vocab = $V shards per domain"
    echo "──────────────────────────────────────────────────────────"

    cd "$PY_DIR"
    SBUS_URL="$SBUS_URL" PROXY_URL="$PROXY_URL" BACKBONE="$BACKBONE" TEMP="$TEMP" \
    python3 exp_proxy_ph2.py \
        --domains             "$DOMAINS" \
        --tasks-per-domain    "$TASKS_PER_DOMAIN" \
        --runs-per-task       "$RUNS_PER_TASK" \
        --steps               "$STEPS" \
        --shards-per-domain   "$V" \
        --max-shards-per-step 1 \
        --output              "$CSV" \
        --summary             "$SUMMARY"
done

echo
echo "Vocab-scaling sweep complete."
echo "Results: $RESULTS_DIR/${RUN_TAG}_vocab{4,8,12}_summary.json"
echo
echo "Quick comparison table:"
python3 - <<PY
import json, os
tag = "$RUN_TAG"
results_dir = "$RESULTS_DIR"
print(f"{'vocab':>6} {'f_http':>8} {'dl_acc':>8} {'proxy_Δ':>8} {'total_on':>10} {'type_i':>8}")
print("-" * 60)
for v in [4, 8, 12]:
    path = os.path.join(results_dir, f"{tag}_vocab{v}_summary.json")
    if not os.path.exists(path):
        print(f"{v:>6}  (missing)")
        continue
    with open(path) as f: s = json.load(f)
    cd = s.get("coverage_decomposition", {})
    ti = sum(c.get("type_i_corruptions", 0) for c in s.get("conditions", {}).values())
    print(f"{v:>6} {cd.get('http_this_step', 0):>8.3f} "
          f"{cd.get('dl_accumulation_under_off', 0):>8.3f} "
          f"{cd.get('proxy_marginal_paired', 0):>8.3f} "
          f"{cd.get('total_coverage_on', 0):>10.3f} "
          f"{ti:>8d}")
PY