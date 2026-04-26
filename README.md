# sbus-experiments

**Python experimental harness for the S-Bus paper.**

This repository contains the experiment scripts, datasets, and analysis
code that produce every empirical measurement reported in:

> *S-Bus: Automatic Read-Set Reconstruction for Multi-Agent LLM State
> Coordination.* Sajjad Khan, 2026. [arXiv link — TBA]

**Companion repositories:**

- [`sbus`](https://github.com/sajjadanwar0/sbus) — the Rust workspace
  containing `sbus-server` (the measured system), `sbus-baselines`
  (PG and Redis adapters), and `sbus-proxy` (transparent LLM-API proxy)
- [`sbus-formals`](https://github.com/sajjadanwar0/sbus-formals) —
  TLA+, TLAPS, and Dafny mechanised proofs

---

## Paper experiment → script mapping

The paper organises evidence into five argument arcs (A1–A5) plus
distributed-systems support. The full mapping below covers every
experiment in the paper.

### A1: Structural conflict prevention

| Paper experiment | Script(s) |
|---|---|
| Exp. B (SDK comparison: CF, S@50) | `sdk_compare_v2.py` |
| Exp. SR (200/200 stale-read injection) | `cross_shard_validation.py` |
| Exp. CSV (9,304 cross-shard injections) | `cross_shard_validation.py --csv` |
| Exp. Sequential (wall-time speedup) | `exp_sequential_wall_time_v2.py` |
| Exp. ORI-Isolation (959 paired trials) | `exp_ori_isolation_v2.py` |
| Exp. Workload-B (data-pipeline, 8 domains) | `run_workload_b.py`, `analyze_workload_b.py` |

### A2: Coverage-gap measurement

| Paper experiment | Script(s) |
|---|---|
| Exp. PH-2 (`p_hidden = 0.739`) | `measure_phidden_v2.py` (legacy mode) |
| Exp. PH-3 (semantic extraction, recall/precision) | `measure_phidden_v2.py` (PH-3 mode) |
| Exp. PH-3 validation (κ = 0.46, 2 LLM judges) | `run_llm_judges.py`, `score_annotations.py`, `diagnose_disagreements.py` |
| Exp. Adversarial-Rhidden | `exp_adversarial_rhidden_v2.py` |
| Exp. PROXY-PH2 (structural-coverage decomposition) | `exp_proxy_ph2.py` |
| Exp. PROXY-PH2 cross-backbone (Anthropic Haiku 4.5) | `exp_proxy_ph2_haiku.py` |
| Exp. PROXY-PH2 cross-backbone (Google Gemini 2.5 Flash) | `exp_proxy_ph2_multibackbone.py` |

### A3: Concurrency-control safety parity

| Paper experiment | Script(s) |
|---|---|
| Exp. E + Exp. SCALE (shared-shard, N ≤ 64) | `exp_contention_scale.py` |
| Exp. PG-Comparison (Python adapters) | `pg_bench_full.py` |
| Exp. PG-Comparison Rust-Native (against `sbus-baselines`) | `pg_comparison.py` |
| Exp. PG-Contention (three backends, contention) | `pg_bench_contention.py`, `exp_pg_contention.py` |

### A4: Backbone generalisation

| Paper experiment | Script(s) |
|---|---|
| Exp. T3-A / T3-B (Haiku-3, Llama-3.1-8b) | `backbone_replication.py` |

### A5: Topology-conditional operating envelope

| Paper experiment | Script(s) |
|---|---|
| Exp. SJ-V3 (semantic judge pilot) | `exp_semantic_judge_v3.py`, `run_sjv3_parallel.py` |
| Exp. SJ-V4 (context diversity, 1,000 runs) | `exp_semantic_judge_v4.py`, `run_sjv4_parallel.py` |
| Exp. SJ-V5 (SCR dose-response) | `exp_sjv5_parallel.py` |
| Exp. Dedicated-Shard (n=600) | `exp_dedicated_shard_semantic.py` |
| Exp. Shared-State (n=180) | `exp_shared_state.py` |
| Exp. Merge (OCC vs LLM-merge) | `merge_baseline.py` |

### Distributed / Raft (supporting)

| Paper experiment | Script(s) |
|---|---|
| Exp. DR-9 (P1 session replication, leader failover) | `exp_session_replication_dr9.py` |
| Exp. DR (8 sub-experiments) | `exp_distributed.py` |

---

## Repository layout

```
sbus-experiments/
├── pyproject.toml                  uv-managed dependency manifest
├── README.md                       this file
├── LICENSE                         MIT (code) + CC-BY-4.0 (datasets)
│
├── agent.py                        Agent abstraction over OpenAI API
├── harness.py                      Shared experiment-harness primitives
├── domains.py                      Task-domain definitions
├── evaluate.py                     Common evaluation utilities
├── phidden_wrapper.py              Keyword-scan R_hidden inference (reference)
├── swe_bench_lite.py               SWE-bench-Lite task pool
├── preflight_proxy.py              Pre-experiment proxy reachability check
├── judge_subsample.py              Subsampling for inter-LLM-judge IAA studies
│
├── exp_*.py, run_*.py, *_compare.py, *_baseline.py, *_bench_*.py, ...
│                                   (40+ experiment scripts; see mapping above)
│
├── tasks.json                      400-row evaluation task pool
├── sjv4_tasks.json                 Tasks for Exp. SJ-V4
├── shared_state_tasks.json         Tasks for Exp. Shared-State
├── datasets/
│   ├── tasks_30_multidomain.json   30 tasks across 4 domains (Exp. SJ-V3)
│   └── long_horizon_tasks.json     15-task long-horizon-planning bench (Exp. B)
│
├── results/                        captured experiment outputs (CSV + JSON)
├── run_proxy_ph2.sh                Convenience driver for PROXY-PH2 sweep
└── run_vocab_scaling.sh            Convenience driver for vocab-size sweep
```

Most scripts are standalone and support `--help`.

---

## Datasets

| File | Used by | Rows | License |
|---|---|---|---|
| `tasks.json` | `run_llm_judges.py` | 400 | CC-BY-4.0 |
| `sjv4_tasks.json` | `exp_semantic_judge_v4.py` | 20 | CC-BY-4.0 |
| `shared_state_tasks.json` | `exp_shared_state.py` | 30 | CC-BY-4.0 |
| `datasets/tasks_30_multidomain.json` | `run_sjv3_parallel.py` | 30 | CC-BY-4.0 |
| `datasets/long_horizon_tasks.json` | `sdk_compare_v2.py` | 15 | CC-BY-4.0 |

---

## Sample results

The `results/` directory ships actual experiment outputs from late
April 2026:

- `proxy_ph2_*_full_summary.json` — main PROXY-PH2 run on GPT-4o-mini
  (n=8,400 paired)
- `haiku_paired_*_summary.json` — cross-backbone replication on
  Anthropic Haiku 4.5 (n=2,400 paired)
- `gemini_paired_*_summary.json` — cross-backbone replication on
  Google Gemini 2.5 Flash (n=2,400 paired)
- `proxy_ph2_vocab_*` — proxy-vocabulary scaling sweep at V ∈ {4, 8, 12}
- `workload_b_sweep.csv` + `.jsonl` — Workload-B raw data and per-trial
  records

These are the source files for the paper's Tables and figures. Fresh
re-runs produce statistically equivalent but not identical outputs (see
"Reproducibility" below).

---

## Prerequisites

### Python environment

This repo uses [`uv`](https://github.com/astral-sh/uv) for dependency
management (Python 3.12+).

```bash
uv sync                 # creates .venv/ and installs all deps
uv run python <script>  # runs scripts inside the managed environment
```

The `pyproject.toml` pins all required packages: `openai`, `anthropic`,
`google-genai`, `groq`, `httpx`, `psycopg`, `redis`, `pandas`, `scipy`,
`tiktoken`, `langgraph`, `crewai`, `autogen-agentchat`, `litellm`,
`mitmproxy`, `swebench`, plus development tooling.

### Backends

Most experiments require a running S-Bus server:

```bash
cd ../sbus
SBUS_ADMIN_ENABLED=1 cargo run --release -p sbus-server
# → http://localhost:7000
```

PG-Comparison and PG-Contention also require the Rust adapters:

```bash
cd ../sbus
PG_DSN="host=localhost dbname=sbus_baseline user=sbus_user password=..." \
  cargo run --release -p sbus-baselines --bin pg-adapter &
# → http://localhost:7001

REDIS_URL=redis://127.0.0.1:6379 \
  cargo run --release -p sbus-baselines --bin redis-adapter &
# → http://localhost:7002
```

PROXY-PH2 experiments require the LLM-API proxy:

```bash
cd ../sbus
SBUS_PROXY_VOCAB="models_state,query_compiler,test_fixture,review_notes" \
  cargo run --release -p sbus-proxy
# → http://localhost:9000
```

### API keys

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...   # cross-backbone, PH-3 validation
export GEMINI_API_KEY=...             # Gemini replication
export GROQ_API_KEY=gsk-...           # Llama-3.1-8b replication
```

---

## Running experiments

Each script is standalone and supports `--help`.

### Quickest sanity check (≈2 min, no API keys needed)

```bash
python3 cross_shard_validation.py --n-trials 50
```

Verifies stale commits are rejected with HTTP 409 and fresh commits
accepted with HTTP 200 (paper §VII-C, Exp. SR).

### Standard reproduction (≈15 min, ≈$2 API spend)

```bash
# Structural-SCR dose-response — exact analytic match per Table XVIII
python3 exp_sjv5_parallel.py --tasks-limit 1
```

### Full paper reproductions

Most experiments take 30–120 minutes and $5–$50 in API spend. Specific
estimates are in each script's `--help` output. The largest individual
run is `exp_proxy_ph2.py --full` (≈2 hours, ≈$30 on GPT-4o-mini).

```bash
# PH-3 semantic extraction with cross-family analyst ablation
python3 measure_phidden_v2.py \
  --domains all --runs-per-domain 5 \
  --output  results/ph3.csv \
  --summary results/ph3_summary.json

# PH-3 inter-LLM-judge validation (κ = 0.46)
python3 run_llm_judges.py \
  --input results/ph3.csv \
  --output-gpt    gpt4o_labels.csv \
  --output-claude claude_sonnet_labels.csv
python3 score_annotations.py gpt4o_labels.csv claude_sonnet_labels.csv

# Workload-B (cross-shard view-divergence on 8 domains)
python3 run_workload_b.py --domains all --trials 5
python3 analyze_workload_b.py results/workload_b_sweep.jsonl

# PROXY-PH2 cross-backbone trio
python3 exp_proxy_ph2.py               --output results/proxy_ph2_gpt.csv
python3 exp_proxy_ph2_haiku.py         --output results/proxy_ph2_haiku.csv
python3 exp_proxy_ph2_multibackbone.py --output results/proxy_ph2_gemini.csv

# PG-Comparison full sweep (requires PG and Redis adapters running)
python3 pg_bench_full.py --backends sbus,pg,redis --agents 4 8 16 32 64
```

The convenience drivers `run_proxy_ph2.sh` and `run_vocab_scaling.sh`
wrap multi-cell parameter sweeps for the proxy experiments.

---

## Reproducibility

LLM-driven experiments are not bit-reproducible. Expect:

- **Identical structural results** — commit counts, conflict counts,
  HTTP-status distributions, and view-divergence counters. These are
  deterministic given the ACP retry logic, version checks, and the
  server-side counters; they do not depend on LLM stochasticity.
- **Approximate semantic results** — judge labels, content-quality
  scores, and IAA estimates vary by ≤5 pp across runs at temperature 0,
  more under higher temperatures. The shipped `results/*.csv` files
  are exact run records; fresh runs produce statistically equivalent
  but not bit-identical outputs.

For deterministic structural validation, set `OPENAI_TEMPERATURE=0` and
use the `--seed` flag where supported. Numeric results in the paper that
are zero-variance by protocol specification (Remark 7 in §VII-E) will
reproduce exactly.

The Workload-B experiment relies on the server-side
`view_divergent_commits` / `view_checked_commits` counters exposed via
`GET /stats` and the runtime ORI toggle at `POST /admin/config`
(both in `sbus-server`). The harness reads these counters at the end of
each trial; the structural results are independent of LLM determinism.

---

## Citation

```bibtex
@techreport{khan2026sbus,
  author      = {Khan, Sajjad},
  title       = {S-Bus: Automatic Read-Set Reconstruction for Multi-Agent
                 LLM State Coordination},
  institution = {Independent},
  year        = {2026},
  note        = {arXiv preprint},
  url         = {https://arxiv.org/abs/...}
}
```

---

## License

- Code: MIT (see `LICENSE`).
- Datasets in `datasets/` and the four top-level `*_tasks.json` files:
  CC-BY-4.0.