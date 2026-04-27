# sbus-experiments

**Python experimental harness for the S-Bus paper.**

This repository contains the experiment scripts, datasets, and analysis
code that produce every empirical measurement reported in:

> *S-Bus: Automatic Read-Set Reconstruction for Multi-Agent LLM State
> Coordination.* Sajjad Khan, 2026. _arXiv ID forthcoming._

**Companion repositories:**

- [`sbus`](https://github.com/sajjadanwar0/sbus) — the Rust workspace
  containing `sbus-server` (the measured system), `sbus-baselines`
  (PG and Redis adapters), and `sbus-proxy` (transparent LLM-API proxy)
- [`sbus-formals`](https://github.com/sajjadanwar0/sbus-formals) —
  TLA+, TLAPS, and Dafny mechanised proofs

---

## Quickstart (≈ 2 min, no API keys)

The cheapest sanity check verifies the cross-shard validation logic on
the running server: stale commits should be rejected (HTTP 409), fresh
commits accepted (HTTP 200).

```bash
# Terminal 1: start the server
cd ../sbus
SBUS_ADMIN_ENABLED=1 cargo run --release -p sbus-server

# Terminal 2: run the validation
cd ../sbus-experiments
uv sync
uv run python cross_shard_validation.py --n-trials 50
```

Expected output: `200/200 stale rejected, 50/50 fresh accepted, 0 corruptions`.
This corresponds to Exp. SR in the paper (§VII-C).

For LLM-driven experiments, see [Running experiments](#running-experiments).

---

## What's in `results/`

Actual outputs from late April 2026, used as the source data for the
paper's tables and figures:

| File | Experiment | Sample size |
|---|---|---|
| `proxy_ph2_*_full_summary.json` | PROXY-PH2 main run (GPT-4o-mini) | n = 8,400 paired |
| `haiku_paired_*_summary.json` | Cross-backbone (Anthropic Haiku 4.5) | n = 2,400 paired |
| `gemini_paired_*_summary.json` | Cross-backbone (Google Gemini 2.5 Flash) | n = 2,400 paired |
| `proxy_ph2_vocab_*` | Vocabulary scaling sweep | V ∈ {4, 8, 12} |
| `workload_b_sweep.csv` + `.jsonl` | Workload-B raw + per-trial records | 8 domains × 10 trials |

Fresh re-runs produce statistically equivalent but not bit-identical
outputs (see [Reproducibility](#reproducibility)).

---

## Running experiments

Each script is standalone and supports `--help`. The three most
informative reproductions, in increasing order of cost:

```bash
# 1. Structural-SCR dose-response — exact analytic match per Table XVIII
#    ≈ 5 min, ≈ $0.50
uv run python exp_sjv5_parallel.py --tasks-limit 1

# 2. Workload-B (cross-shard view-divergence on 8 domains)
#    ≈ 30 min, ≈ $5
uv run python run_workload_b.py --domains all --trials 5
uv run python analyze_workload_b.py results/workload_b_sweep.jsonl

# 3. PROXY-PH2 cross-backbone trio
#    ≈ 6 hours total, ≈ $50 across three vendors
uv run python exp_proxy_ph2.py               --output results/proxy_ph2_gpt.csv
uv run python exp_proxy_ph2_haiku.py         --output results/proxy_ph2_haiku.csv
uv run python exp_proxy_ph2_multibackbone.py --output results/proxy_ph2_gemini.csv
```

The convenience drivers `run_proxy_ph2.sh` and `run_vocab_scaling.sh`
wrap multi-cell parameter sweeps. See per-script `--help` for full
options. The largest individual run is `exp_proxy_ph2.py --full`
(≈ 2 hours, ≈ $30 on GPT-4o-mini).

For the PG-Comparison sweep, the Rust adapters from the `sbus`
repository must be running on ports 7001 (PG) and 7002 (Redis); see
[Backends](#backends) below.

---

## Setup

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
│                                   (40+ experiment scripts; see mapping below)
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

## Paper experiment → script mapping

_For paper reviewers: the table below maps every experiment in the
paper to the script(s) that produce it._

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

## Reproducibility

LLM-driven experiments are not bit-reproducible. Expect:

- **Identical structural results** — commit counts, conflict counts,
  HTTP-status distributions, and view-divergence counters. These are
  deterministic given the ACP retry logic, version checks, and the
  server-side counters; they do not depend on LLM stochasticity.
- **Approximate semantic results** — judge labels, content-quality
  scores, and IAA estimates vary by ≈ 5 pp across runs at temperature 0,
  more under higher temperatures. The shipped `results/*.csv` files
  are exact run records; fresh runs produce statistically equivalent
  but not bit-identical outputs.

For deterministic structural validation, set `OPENAI_TEMPERATURE=0` and
use the `--seed` flag where supported. Numeric results in the paper
that are zero-variance by protocol specification (Remark 7 in §VII-E)
will reproduce exactly.

The Workload-B experiment relies on the server-side
`view_divergent_commits` / `view_checked_commits` counters exposed via
`GET /stats` and the runtime ORI toggle at `POST /admin/config` (both
in `sbus-server`). The harness reads these counters at the end of
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
  note        = {arXiv preprint}
}
```

_Once the arXiv ID is assigned, add `url = {https://arxiv.org/abs/XXXX.XXXXX}`
and the eprint ID to the BibTeX above._

---

## License

- Code: MIT (see `LICENSE`).
- Datasets in `datasets/` and the four top-level `*_tasks.json` files:
  CC-BY-4.0.