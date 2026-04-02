# sbus-experiments/Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility targets for all paper tables.
# Run `make help` to see available targets.
#
# Prerequisites:
#   export OPENAI_API_KEY="sk-..."
#   cd path/to/sbus && cargo run     # in a separate terminal
#
# Quick smoke test (< $1, ~5 min):
#   make smoke

PYTHON     ?= python3
SERVER     ?= http://localhost:3000
AGENTS_LHP ?= 4 8
STEPS_LHP  ?= 50
TASKS      ?= 5
RUNS       ?= 1

.PHONY: help smoke check-server \
        table3 table4 table5 table6 table7 table11 \
        all-tables clean

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "S-Bus Experiment Reproducibility"
	@echo "=================================="
	@echo ""
	@echo "Targets:"
	@echo "  smoke        Quick 1-task test (~\$$1, ~5 min)"
	@echo "  table3       CWR comparison (real SDKs, Table 3)  ~2h"
	@echo "  table5       SCR contention benchmark (Table 5)   ~30min"
	@echo "  table6       Cross-shard validation (Table 6)     ~1h"
	@echo "  table7       S@50 results (generated from table3)"
	@echo "  table11      Ablation study (Table 11)            ~30min"
	@echo "  all-tables   Run tables 3,5,6,11 sequentially"
	@echo "  clean        Remove results/ directory"
	@echo ""
	@echo "Environment variables:"
	@echo "  OPENAI_API_KEY   Required"
	@echo "  SERVER           S-Bus server URL (default: $(SERVER))"
	@echo "  STEPS_LHP        Steps per run (default: $(STEPS_LHP))"
	@echo "  TASKS            Number of LHP tasks (default: $(TASKS))"
	@echo "  RUNS             Runs per condition (default: $(RUNS))"
	@echo "  SBUS_RETRY_BUDGET  Retry budget B (default: 1)"
	@echo ""
	@echo "Ablation study requires server started with flags:"
	@echo "  SBUS_TOKEN=0 cargo run       # –token condition"
	@echo "  SBUS_VERSION=0 cargo run     # –version condition"
	@echo "  SBUS_LOG=0 cargo run         # –log condition"
	@echo ""

# ── Prerequisites ────────────────────────────────────────────────────────────

check-server:
	@echo "Checking S-Bus server at $(SERVER)..."
	@curl -sf $(SERVER)/stats > /dev/null || \
		(echo "ERROR: S-Bus server not running. Run: cd ../sbus && cargo run" && exit 1)
	@echo "Server OK."
	@curl -s $(SERVER)/stats | python3 -c \
		"import sys,json; cfg=json.load(sys.stdin).get('acp_config',{}); \
		print('ACP config:', json.dumps(cfg, indent=2))"

results/:
	mkdir -p results/

# ── Smoke test ────────────────────────────────────────────────────────────────

smoke: check-server results/
	@echo "\nSmoke test: 1 task, N=4, 10 steps..."
	$(PYTHON) sdk_compare.py \
		--system sbus \
		--agents 4 \
		--steps 10 \
		--tasks-limit 1 \
		--server $(SERVER) \
		--out results/smoke.csv
	@echo "Smoke test complete. Results in results/smoke.csv"

# ── Table 3: CWR comparison (canonical) ──────────────────────────────────────

table3: check-server results/
	@echo "\nTable 3: CWR comparison (real SDKs)"
	@echo "Canonical paper run: $(TASKS) tasks, N=$(AGENTS_LHP), $(STEPS_LHP) steps"
	@echo "Expected: S-Bus CWR=0.186 (N=4), LangGraph CWR=4.592 (N=4)"
	$(PYTHON) sdk_compare.py \
		--system all \
		--agents $(AGENTS_LHP) \
		--steps $(STEPS_LHP) \
		--tasks-limit $(TASKS) \
		--server $(SERVER) \
		--out results/table3_cwr.csv
	@echo "\nResults: results/table3_cwr.csv"

# ── Table 5: SCR contention benchmark ────────────────────────────────────────

table5: check-server results/
	@echo "\nTable 5: SCR contention benchmark"
	@echo "N=4 (2 runs), N=8 (3 runs), N=16 (1 run — point estimate only)"
	$(PYTHON) contention.py \
		--mode scr \
		--agents 4 8 16 \
		--steps 5 \
		--runs $(RUNS) \
		--server $(SERVER)

# ── Table 6: Cross-shard validation — all three conditions ───────────────────

table6: check-server results/
	@echo "\nTable 6: Cross-shard validation (3 conditions)"
	@echo "  Condition 1: no_read_set baseline"
	@echo "  Condition 2: v2_naive (NEW control — unordered lock acquisition)"
	@echo "  Condition 3: v2_sorted (Havender sorted order — paper result)"
	@echo "Expected: v2_sorted=0 corruptions, v2_naive>0, no_read_set=57.5%"
	$(PYTHON) contention.py \
		--mode cross-shard \
		--agents 4 8 16 \
		--trials 10 \
		--server $(SERVER)

# ── Table 7: S@50 (derived from table3 results) ──────────────────────────────

table7: results/table3_cwr.csv
	@echo "\nTable 7: S@50 with Clopper-Pearson exact CIs"
	$(PYTHON) sdk_compare.py \
		--analyse-only \
		--out results/table3_cwr.csv

# ── Table 11: Ablation study ─────────────────────────────────────────────────
# IMPORTANT: Restart server with correct ablation flag for each condition.
# This target runs the full S-Bus (no flags). Run partial ablations manually.

table11: check-server results/
	@echo "\nTable 11: Ablation study"
	@echo "Running FULL S-Bus condition (all flags enabled)..."
	$(PYTHON) benchmark.py \
		--tasks-limit 3 \
		--runs 3 \
		--steps 20 \
		--agents 4 \
		--server $(SERVER) \
		--out results/table11_full.csv
	@echo ""
	@echo "For other ablation conditions, restart the server with:"
	@echo "  SBUS_TOKEN=0 cargo run    then: make table11-no-token"
	@echo "  SBUS_VERSION=0 cargo run  then: make table11-no-version"
	@echo "  SBUS_LOG=0 cargo run      then: make table11-no-log"
	@echo "  SBUS_TOKEN=0 SBUS_VERSION=0 cargo run  then: make table11-no-both"

table11-no-token: check-server results/
	$(PYTHON) benchmark.py \
		--tasks-limit 3 --runs 3 --steps 20 --agents 4 \
		--server $(SERVER) --out results/table11_no_token.csv
	@echo "Results: results/table11_no_token.csv"

table11-no-version: check-server results/
	$(PYTHON) benchmark.py \
		--tasks-limit 3 --runs 3 --steps 20 --agents 4 \
		--server $(SERVER) --out results/table11_no_version.csv
	@echo "Results: results/table11_no_version.csv"

table11-no-log: check-server results/
	$(PYTHON) benchmark.py \
		--tasks-limit 3 --runs 3 --steps 20 --agents 4 \
		--server $(SERVER) --out results/table11_no_log.csv
	@echo "Results: results/table11_no_log.csv"

table11-no-both: check-server results/
	$(PYTHON) benchmark.py \
		--tasks-limit 3 --runs 3 --steps 20 --agents 4 \
		--server $(SERVER) --out results/table11_no_both.csv
	@echo "Results: results/table11_no_both.csv"

# ── Run all tables ────────────────────────────────────────────────────────────

all-tables: table3 table5 table6 table11
	@echo "\nAll tables complete. Results in results/"
	$(PYTHON) sdk_compare.py --analyse-only --out results/table3_cwr.csv

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf results/
	@echo "Cleaned results/"
