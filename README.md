# sbus-experiments

Experiment harness for the S-Bus paper.

> **"Reliable Autonomous Orchestration: A Rust-Based Transactional Middleware
> for Mitigating Semantic Synchronization Overhead in Multi-Agent Systems"**
> Sajjad Khan

**GitHub (experiments):** https://github.com/sajjadanwar0/sbus-experiments  
**GitHub (S-Bus server):** https://github.com/sajjadanwar0/sbus

---

## What is in this repo

Three experiment scripts and the benchmark dataset used to produce all
results in the paper.

| File | Purpose |
|------|---------|
| `contention.py` | Measures Semantic Conflict Rate (SCR) under distinct-shard vs shared-shard topologies |
| `benchmark.py` | Measures Coordination-to-Work Ratio (CWR) for S-Bus vs coordinator-worker baselines |
| `sdk_compare.py` | Compares S-Bus against real CrewAI, AutoGen, and LangGraph SDK implementations |
| `datasets/long_horizon_tasks.json` | 15-task Long-Horizon Planning benchmark (CC-BY-4.0) |

---

## Key results

| System | CWR N=4 | CWR N=8 | Reduction | S@50 N=8 |
|--------|---------|---------|-----------|----------|
| **S-Bus** | **0.238** | **0.210** | — | **80%** |
| LangGraph | 4.384 | 4.213 | 94.8% | 40% |
| CrewAI | 7.099 | 8.168 | 97.1% | 20% |
| AutoGen | 11.970 | 12.070 | 98.1% | 44% |

Mann-Whitney U=0, p<0.0001, r=1.000 for all three comparisons.  
Zero state corruptions across 272 ACP commit attempts.

---

## Setup

```bash
git clone https://github.com/sajjadanwar0/sbus-experiments
cd sbus-experiments

# Install dependencies
uv add httpx tiktoken openai pandas scipy \
            crewai "autogen-ext[openai]" autogen-agentchat langgraph
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="API KEY"
```

Start the S-Bus server in a separate terminal:

```bash
cd path/to/sbus
cargo run
# Server starts on http://localhost:3000
```

---

## Running the experiments

### 1. `contention.py` — SCR experiment

Measures the Semantic Conflict Rate under two coordination topologies:

- **Experiment A:** Each agent owns a separate shard → SCR ≈ 0.000
- **Experiment B:** All agents compete for one shared shard → SCR > 0

```bash
# Default: 4 agents distinct shards + 8 agents shared shard, 5 steps
python3 contention.py

# Paper-quality data (more steps)
python3 contention.py --agents-a 4 --agents-b 8 --steps 10

# Only the shared shard stress test
python3 contention.py --skip-a --agents-b 16 --steps 5
```

Expected output:
```
EXPERIMENT A: 4 agents × 4 DISTINCT shards
  agent-0   commits=  5  conflicts=  0  SCR=0.000
  Global SCR : 0.0000   ← agents own separate shards, no conflicts

EXPERIMENT B: 8 agents × 1 SHARED shard
  agent-2   commits=  3  conflicts=  2  SCR=0.400
  Global SCR : 0.5210   ← conflicts detected and resolved, zero corruptions
```

### 2. `benchmark.py` — CWR benchmark

Measures Coordination-to-Work Ratio across S-Bus and coordinator-worker
baselines (models the pattern used by CrewAI, AutoGen, LangGraph).

```bash
# Quick test — S-Bus only, 1 task (~$0.20)
python3 benchmark.py --tasks-limit 1 --agents 4 --steps 10

# Full run — all systems, 5 tasks (~$5)
python3 benchmark.py --all --agents 4 8 --steps 20

# Analyse existing results
python3 benchmark.py --analyse-only --out results/cwr_results.csv
```

### 3. `sdk_compare.py` — Real SDK comparison

Benchmarks S-Bus against actual SDK implementations of CrewAI, AutoGen,
and LangGraph. This produces the main results table in the paper.

```bash
# Smoke test — 1 task, all systems (~$1, ~5 min)
python3 sdk_compare.py \
    --system all --agents 4 --steps 10 \
    --tasks-limit 1 --out results/smoke.csv

# Full paper run — 5 tasks, 4+8 agents (~$15-20, ~2 hours)
python3 sdk_compare.py \
    --agents 4 8 --steps 20 \
    --tasks-limit 5 \
    --out results/real_sdk_results.csv

# Run one system at a time (useful if a system crashes mid-run)
python3 sdk_compare.py --system sbus      --agents 4 8 --steps 20
python3 sdk_compare.py --system crewai    --agents 4 8 --steps 20
python3 sdk_compare.py --system autogen   --agents 4 8 --steps 20
python3 sdk_compare.py --system langgraph --agents 4 8 --steps 20

# Analyse results
python3 sdk_compare.py --analyse-only --out results/real_sdk_results.csv
```

Analysis output:
```
CWR by (system, agent_count)
System          N    Mean CWR     ±95%CI     n
sbus            4       0.238  ±   0.004    10
langgraph       4       4.384  ±   0.082     5
crewai          4       7.099  ±   0.431     8
autogen         4      11.970  ±   0.801    10

Mann-Whitney U: S-Bus CWR < each baseline
  sbus < langgraph : U=0  p=0.0000  r=1.000  ***
  sbus < crewai    : U=0  p=0.0000  r=1.000  ***
  sbus < autogen   : U=0  p=0.0000  r=1.000  ***
```

---

## Project structure

```
sbus-experiments/
├── datasets/
│   └── long_horizon_tasks.json   # 15-task LHP benchmark (CC-BY-4.0)
├── results/
│   ├── real_sdk_results.csv      # Real SDK experiment output
│   ├── cwr_results.csv           # CWR benchmark output
│   └── smoke*.csv                # Smoke test results
├──contention.py             # SCR contention experiment
├──benchmark.py              # CWR benchmark (S-Bus vs coord-worker)
├──sdk_compare.py            # Real SDK comparison experiment
├── pyproject.toml
└── README.md
```

---

## Token accounting

The CWR metric is only meaningful if all systems use identical token
counting rules. All token counts use the GPT-4o tokenizer via
[tiktoken](https://github.com/openai/tiktoken).

| Token type | Definition |
|-----------|-----------|
| `coord_tokens` | Tokens spent on inter-agent coordination (context reads, summarisation calls, shard reads) |
| `work_tokens` | Tokens spent on task-directed reasoning (agent LLM completions) |
| `CWR` | `coord_tokens / work_tokens` — lower is better |

See Table 3 in the paper for the full taxonomy.

---

## Troubleshooting

| Error | Fix                                                               |
|-------|-------------------------------------------------------------------|
| `Connection refused localhost:3000` | Start the S-Bus server: `cargo run` in the sbus repo              |
| `OPENAI_API_KEY not set` | `export OPENAI_API_KEY="API KEY"`                                 |
| `ModuleNotFoundError: crewai` | `uv add crewai`                                                   |
| `ModuleNotFoundError: autogen_ext` | `uv add "autogen-ext[openai]"`                                    |
| `InvalidUpdateError: messages` | Use `sdk_compare.py` — older versions had a LangGraph state bug   |
| Run interrupted mid-way | Results save after every run — rerun with `--tasks-limit` reduced |

---

## Dataset

`datasets/long_horizon_tasks.json` contains 15 long-horizon planning
tasks spanning:

- Software architecture (3 tasks, 31–35 steps)
- Security and compliance (2 tasks, 32–36 steps)  
- Data and ML pipelines (3 tasks, 22–38 steps)
- Codebase refactoring (2 tasks, 24–40 steps)
- System design (2 tasks, 25–33 steps)
- Research synthesis (1 task, 30 steps)
- Product and API design (2 tasks, 20–26 steps)

Released under **CC-BY-4.0**.

---

## Citation

```bibtex
@article{khan2026sbus,
  title   = {Reliable Autonomous Orchestration: A Rust-Based Transactional
             Middleware for Mitigating Semantic Synchronization Overhead
             in Multi-Agent Systems},
  author  = {Khan, Sajjad},
  journal = {Uploading...},
  year    = {2026},
  note    = {Under review}
}
```

---

## Author

**Sajjad Khan  
GitHub: [@sajjadanwar0](https://github.com/sajjadanwar0)