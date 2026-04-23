# sbus-experiments

Experimental pipeline for the S-Bus paper
(*Observable-Read Consistency for Concurrent Multi-Agent LLM State*).

This repository contains the Python harnesses, analysis scripts, and
seed data needed to reproduce every measurement reported in the paper.

## Companion repositories

- [`sbus`](https://github.com/sajjadanwar0/sbus) — the Rust implementation
  (measured system)
- [`sbus-baselines`](https://github.com/sajjadanwar0/sbus-baselines) —
  Rust-native PostgreSQL + Redis adapters (required for
  Exp.~PG-Comparison and Exp.~PG-Contention)
- [`sbus-formals`](https://github.com/sajjadanwar0/sbus-formals) — TLA+,
  TLAPS, and Dafny proofs

---

## Paper experiment → script mapping (complete)

### Structural validation and contention

| Paper experiment                            | Script(s)                                                                     |
|---------------------------------------------|-------------------------------------------------------------------------------|
| Exp. B (SDK comparison, CF, S@50)           | `sdk_compare_v2.py`                                                           |
| Exp. SR (structural validation, 200/200)    | `cross_shard_validation.py`                                                   |
| Exp. CSV (cross-shard validation, 9,304 inj)| `cross_shard_validation.py` (same script, `--csv` mode)                       |
| Exp. E (shared-shard at N≤16)               | `exp_contention_scale.py`                                                     |
| Exp. Scale (N≤64, 74,400 attempts)          | `exp_contention_scale.py`                                                     |
| Exp. T3-B (Llama/Haiku backbones)           | `backbone_replication.py`                                                     |

### R_hidden and semantic extraction

| Paper experiment                            | Script(s)                                                                     |
|---------------------------------------------|-------------------------------------------------------------------------------|
| Exp. PH-2 (p_hidden = 0.739)                | `measure_phidden_v2.py` (legacy mode)                                         |
| Exp. PH-3 (semantic extraction)             | `measure_phidden_v2.py` (PH-3 mode)                                           |
| Exp. PH-3 validation (κ = 0.46)             | `run_llm_judges.py`, `score_annotations.py`, `diagnose_disagreements.py`      |
| Exp. SJ-v3                                  | `exp_semantic_judge_v3.py`, `run_sjv3_parallel.py`                            |
| Exp. SJ-v4                                  | `exp_semantic_judge_v4.py`, `run_sjv4_parallel.py`                            |
| Exp. SJ-v5 (dose-response)                  | `exp_sjv5_parallel.py`                                                        |
| Exp. Adversarial-Rhidden                    | `exp_adversarial_rhidden_v2.py`                                               |

### Coordination architectures

| Paper experiment                            | Script(s)                                                                     |
|---------------------------------------------|-------------------------------------------------------------------------------|
| Exp. ORI-Isolation (959 paired trials)      | `exp_ori_isolation_v2.py`                                                     |
| Exp. Dedicated-Shard (LWW-equivalent)       | `exp_dedicated_shard_semantic.py`                                             |
| Exp. Shared-State (multi-domain)            | `exp_shared_state.py`                                                         |
| Exp. Merge (OCC vs LLM-assisted)            | `merge_baseline.py`                                                           |
| Exp. Sequential (wall-time speedup)         | `exp_sequential_wall_time_v2.py`                                              |

### PostgreSQL + Redis comparison (requires sbus-baselines)

| Paper experiment                            | Script(s)                                                                     |
|---------------------------------------------|-------------------------------------------------------------------------------|
| Exp. PG-Comparison (full)                   | `pg_bench_full.py`, `pg_comparison.py`                                        |
| Exp. PG-Comparison Rust-Native              | `pg_comparison.py` (against Rust adapters)                                    |
| Exp. PG-Contention (three backends)         | `exp_pg_contention.py`, `pg_bench_contention.py`                              |

### Distributed / Raft

| Paper experiment                            | Script(s)                                                                     |
|---------------------------------------------|-------------------------------------------------------------------------------|
| Exp. DR-9 (leader failover)                 | `exp_session_replication_dr9.py`                                              |
| Exp. DR (8 sub-experiments)                 | `exp_distributed.py`                                                          |

---

## Shared infrastructure

| File                  | Purpose                                                     |
|-----------------------|-------------------------------------------------------------|
| `phidden_wrapper.py`  | Keyword-scan `R_hidden` inference (reference implementation)|
| `swe_bench_lite.py`   | SWE-bench-lite task definitions (used by `pg_bench_full`)   |

## Seed data files

| File                                   | Used by                                          |
|----------------------------------------|--------------------------------------------------|
| `tasks.json` (400 rows)                | `run_llm_judges.py`                              |
| `sjv4_tasks.json`                      | `exp_semantic_judge_v4.py`                       |
| `shared_state_tasks.json`              | `exp_shared_state.py`                            |
| `datasets/tasks_30_multidomain.json`   | `run_sjv3_parallel.py`                           |
| `datasets/long_horizon_tasks.json`     | `sdk_compare_v2.py` (Exp. B default task set)    |

---

## Prerequisites

### Python packages

```bash
uv sync
```

Core: `openai`, `httpx`, `requests`, `psycopg2-binary`. For SDK-comparison
experiments: `langgraph`, `crewai`, `autogen_agentchat`, `tiktoken`,
`pandas`. Optional: `anthropic` (cross-family analyst ablations), `scipy`
(statistical tests).

### Running backends

Most experiments require one or more HTTP backends:

| Backend            | Repo              | Default port | Required for                         |
|--------------------|-------------------|--------------|--------------------------------------|
| S-Bus              | `sbus`            | 7000         | All experiments                      |
| PG adapter (Rust)  | `sbus-baselines`  | 7001         | Exp.~PG-Comparison, Exp.~PG-Contention |
| Redis adapter (Rust)| `sbus-baselines` | 7002         | Exp.~PG-Comparison, Exp.~PG-Contention |

Start S-Bus:
```bash
cd ../sbus
SBUS_ADMIN_ENABLED=1 cargo run --release
```

Start baselines (only needed for PG/Redis experiments):
```bash
cd ../sbus-baselines
PG_DSN="host=localhost dbname=sbus_baseline user=sbus_user password=sbus_pass" \
  cargo run --release --bin pg-adapter &
cargo run --release --bin redis-adapter &
```

### API keys

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...   # optional
export GROQ_API_KEY=gsk-...           # optional, for Exp.~T3-B Llama backbone
```

---

## Running an experiment

Each script is standalone and prints usage with `--help`.

```bash
# Example: PH-3 semantic extraction (full paper run)
python3 measure_phidden_v2.py \
  --domains all --runs-per-domain 5 \
  --output  results/ph3.csv \
  --summary results/ph3_summary.json

# Example: PH-3 validation study (inter-LLM-judge)
python3 run_llm_judges.py \
  --input results/ph3.csv \
  --output-gpt    gpt4o_labels.csv \
  --output-claude claude_sonnet_labels.csv
python3 score_annotations.py gpt4o_labels.csv claude_sonnet_labels.csv
python3 diagnose_disagreements.py gpt4o_labels.csv claude_sonnet_labels.csv

# Example: ORI-Isolation (959 paired trials)
python3 exp_ori_isolation_v2.py --n-tasks 10 --n-runs 50

# Example: Adversarial-Rhidden (30/30 failure demonstration)
python3 exp_adversarial_rhidden_v2.py
```

---

## Reproducing the paper's numbers
```bash
# 1. Start backends (see above)

# 2. Run structural validation (fast)
python3 cross_shard_validation.py

# 3. Run contention scale
python3 exp_contention_scale.py --agents 4 8 16 32 64

# 4. Run PH-3
python3 measure_phidden_v2.py --domains all --runs-per-domain 5

# 5. Run validation study
python3 run_llm_judges.py ...

# 6. Run ORI-Isolation
python3 exp_ori_isolation_v2.py

# 7. Run Adversarial-Rhidden
python3 exp_adversarial_rhidden_v2.py

# 8. Run PG comparison (requires sbus-baselines; ~2-3 hours)
python3 pg_bench_full.py
python3 exp_pg_contention.py --n-agents 4 8 16 32

# 9. Run SDK comparison (requires langgraph/crewai/autogen; ~2 hours)
python3 sdk_compare_v2.py --agents 4 8 --tasks-limit 5

# 10. SJ-v3, v4, v5 dose-response
python3 exp_semantic_judge_v3.py
python3 exp_semantic_judge_v4.py
python3 exp_sjv5_parallel.py

# 11. Backbone replication (optional; Llama via Groq)
python3 backbone_replication.py --backbone llama-3.1-8b-instant

# 12. Distributed (Raft) tests
python3 exp_distributed.py
python3 exp_session_replication_dr9.py
```

---

## Linting

```bash
pip install ruff
ruff check --config ruff.toml *.py
```

The configuration tolerates research-code norms: relaxed line length,
multi-statement lines, bare `except`, and conventionally ambiguous
variable names.

---

## Data files

Large result CSVs and JSONs (10 KB–2 MB each) are not included in this
repository. They are available in the Zenodo deposit associated with
the paper.

---

## Citation

```bibtex
@techreport{khan2026sbus,
  author      = {Khan, Sajjad},
  title       = {Observable-Read Consistency for Concurrent Multi-Agent
                 LLM State},
  institution = {Independent},
  year        = {2026},
  note        = {arXiv preprint. Version 50.2.},
  url         = {https://arxiv.org/abs/...}
}
```

## License

MIT. See `LICENSE`.

## Contact

Sajjad Khan — sajjadanwar0@gmail.com
