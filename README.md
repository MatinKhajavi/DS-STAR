# DS-STAR
Implementation of the paper "DS-STAR: A state-of-the-art versatile data science agent". The system follows the paper's multi-agent pipeline to analyze tabular/JSON/markdown data, plan an analysis, implement code, verify sufficiency, and auto-debug until a final answer is produced.

## Project structure
```
data/                   # Place your task data here (context files live here after downloads)
evaluations/            # DABStep benchmark runners and helper scripts
  run_dabstep.py        # Main benchmark driver (downloads context + runs agent)
  quick_test.sh         # 10-task smoke test on DABStep dev split
  run_full_benchmark.sh # Full default split runner
logs/                   # Prompt + iteration logs (auto-created)
runs/                   # Benchmark results/submission files (auto-created)
src/                    # DS-STAR implementation
  cli/main.py           # CLI entrypoint
  core/                 # Orchestrator, prompts, executor, data models
  agents/               # Analyzer, Planner, Coder, Verifier, Router, Debugger, Finalizer
  llm/                  # Gemini client (primary), OpenAI/Anthropic stubs
  utils/                # Filesystem + logging helpers
pyproject.toml          # Poetry project (Python 3.13+)
```

## Highlights
- Paper-faithful DS-STAR loop: plan -> code -> execute -> verify -> route (add/remove steps) -> debug -> finalize.
- Seven cooperating agents: Analyzer (file profiling), Planner, Coder, Verifier, Router, Debugger, Finalizer.
- Cached data descriptions so multiple queries reuse pre-analysis of `data/`.
- Auto-debug with retries and execution timeout to keep LLM code safe from hangs.


## Multi-agent framework (paper-aligned)
DS-STAR orchestrates specialist LLM agents: Analyzer profiles every file; Planner proposes incremental steps; Coder implements them; Verifier judges sufficiency; Router decides to add or truncate steps; Debugger repairs failing scripts; Finalizer formats the final answer under optional guidelines. The orchestrator runs Algorithm 1 from the paper until the verifier deems the plan sufficient or max rounds are reached.

## Setup
1) Install prerequisites  
   - Python 3.13+  
   - Poetry (preferred)  

2) Install dependencies  
   ```bash
   poetry install
   ```

3) Configure LLM credentials (Gemini is the only provider fully wired end-to-end right now)  
   - `GEMINI_API_KEY` or `GOOGLE_API_KEY`  
   - OpenAI/Anthropic clients exist, but the orchestrator currently raises `NotImplementedError` for them.  

4) (Optional) Hugging Face token if your environment requires authentication for `adyen/DABstep` pulls.

## Running the agent
### CLI (single question)
```bash
poetry run python -m src.cli.main \
  "Which country has the highest total fee revenue?" \
  --data-dir data \
  --max-rounds 10 \
  --log-dir logs/run1
```
- `--guidelines` lets you constrain the final formatter (Finalizer agent).
- Results and prompts are saved under `logs/`; per-run artifacts go into the chosen log dir.

### Python API
```python
from ds_star import DSStar
from src.config import DSStarConfig, LLMConfig

config = DSStarConfig(
    llm=LLMConfig(provider="gemini", model="gemini-2.5-pro"),
    max_rounds=15,
    data_dir="data",
    log_dir="logs/example",
)
agent = DSStar(config)
agent.prepare_data()  # cache file descriptions once
code, answer = agent.run("How many merchants are active in 2023?")
print(answer)
```

## DABStep benchmark runners
- Smoke test (first 10 dev tasks):  
  `bash evaluations/quick_test.sh`
- Full/default split:  
  `bash evaluations/run_full_benchmark.sh`
- Manual run with custom slice:  
  ```bash
  poetry run python evaluations/run_dabstep.py \
    --split dev \
    --max-tasks 50 \
    --provider gemini \
    --model gemini-2.5-pro \
    --max-rounds 20
  ```
  The script downloads required context files into `data/`, caches analyses, streams progress, and writes `runs/<run_id>.jsonl` plus `<run_id>_submission.jsonl`.
- Parallel/offset workflows: see `evaluations/run_parallel.sh` and `evaluations/run_missing_tasks.py`; merge outputs with `evaluations/combine_results.sh`.

## Configuration knobs (src/config.py)
- `llm`: `provider` (`gemini`), `model`, `api_key`
- `max_rounds`: Max refinement iterations in Algorithm 1
- `data_dir`: Source data directory (analyzer runs over all files)
- `execution`: `max_retries`, `capture_output`
- `log_dir`: Where prompts/iterations/final solutions are saved
- `debug_mode`: Enable verbose logging
- `retrieval_threshold`, `top_k_files`: placeholders for future retrieval (not yet implemented)

## Tips
- Keep all task files under `data/`; the Analyzer agent recursively profiles everything non-hidden.
- Reuse `prepare_data()` when running many questions against the same dataset to skip redundant analysis.
- Inspect `logs/<query_id>/prompts/` to audit prompts/responses if behavior drifts.
