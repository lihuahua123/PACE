# PACE replication package

This package contains the PACE controller implementation, run and analysis scripts, task-level figure data, and 50-task archived trajectories for the main PACE versus no-control comparisons. It was exported from the experimental workspace without its Git history, credentials, evaluator logs, or local environment.

## Contents

- `mini-swe-agent/`: the modified MiniSWEAgent source used for PACE. The pre-execution controller is in `src/minisweagent/agents/default.py`; local price normalization is in `src/minisweagent/pricing.py`. The upstream MiniSWEAgent license is retained.
- `scripts/`: the archived launch and analysis scripts. Launch scripts require a SWE-bench evaluator, model access, and environment-specific paths. API credentials are not included.
- `traces/`: each run has 50 task directories containing an agent trajectory, task summary, prediction where available, and verifier outcome where available. The MiniMax PACE directories contain the success-selected task reruns used in the paper main table.
- `traces_original/`: the same three MiniMax PACE runs before task replacement, each with 50 original task records.
- `data/figure_tables/`: task-level data and summaries used for the paper figures and tables.
- `data/trace_manifest.csv`: mapping from neutral run labels to archived run IDs and files.
- `reproduce_main.py`: recomputes success, API calls, tokens, and estimated CNY cost for the main no-control and PACE rows.

The bundled controller includes a later fix that refreshes the edit epoch after `git checkout`. The archived trajectories were generated before this fix; their recorded actions and paper statistics have not been changed.

## Recompute the main comparison

Use Python 3.10 or newer. Install the dependencies in `mini-swe-agent/pyproject.toml` plus `pandas`, then run:

```bash
python3 reproduce_main.py
```

The script checks that each included run has 50 task records. It uses the archived token pricing implementation and task summaries/trajectories. The expected main results are:

| Model and configuration | Successes by run | Mean success | Mean cost (CNY) |
| --- | --- | ---: | ---: |
| MiniMax-M2.7 no control | 38, 35, 39 | 74.7% | 21.560 |
| MiniMax-M2.7 PACE (selected reruns) | 37, 39, 37 | 75.3% | 13.363 |
| MiniMax-M2.7 PACE (original independent runs) | 36, 35, 35 | 70.7% | 13.408 |
| gpt-5.6-luna no control | 23, 27, 29 | 52.7% | 3.604 |
| gpt-5.6-luna PACE | 34, 31 | 65.0% | 2.881 |
| DeepSeek-V4-Flash no control | 40 | 80.0% | 59.199 |
| DeepSeek-V4-Flash PACE | 40 | 80.0% | 31.797 |

The selected MiniMax PACE row has sample standard deviations of 2.3 percentage points for success, 4.168 CNY for cost, 136 API calls, and 9.205 million tokens. Its 113 successes out of 150 task records are a descriptive selected outcome, not an estimate from three independent replications. Selected-trace cost excludes discarded attempts, so 13.363 CNY is not the total cost of obtaining the selected outcomes. The original independent PACE success rate is 70.7%, compared with 74.7% without control. The `luna_pace_1` directory is an archived case-study run and is excluded from the two-run luna main-table mean. The `minimax_case_pace` directory is also a case-study run, excluded from the three-run MiniMax main-table mean.

## MiniMax provenance and later reruns

The three MiniMax PACE runs use the same 50 task IDs and limits `(8, 35, 7, 4)`. Of the original executions, `minimax_pace_1` and `_2` used AutoDL's OpenAI-compatible endpoint and `minimax_pace_3` used MiniMax's Anthropic-compatible endpoint. The seven selected replacements used MiniMax's Anthropic-compatible endpoint, including within the two AutoDL runs. The original independent records are preserved in `traces_original/`.

The main-table PACE values use the selected records in `traces/`: one failed task was replaced in `minimax_pace_1`, four in `minimax_pace_2`, and two in `minimax_pace_3`, only when a rerun succeeded. This choice biases success upward and mixes endpoints within two runs. The replacement metadata is in `data/rep2_replacement_manifest.json`, `data/rep3_replacement_manifest.json`, and `data/minimax_anthropic_selected_rerun_summary.json`. `data/minimax_anthropic_original_summary.json` records the original Anthropic run. `data/selected_rerun_provenance.json` describes a separate, superseded August cohort; it is **not** used for the main-table 75.3% result. `data/figure_tables/minimax_pace_runs.csv` records the selected per-run metrics; `data/figure_tables/minimax_pace_runs_original.csv` records the original metrics.

## Scope and privacy

This export includes the main PACE and no-control trajectories and derived task-level data. It does not include the full evaluator containers, model weights, private API credentials, or every exploratory run. The two external comparison methods' launch scripts and published summary data are included, but their full trajectories are not in this compact package. Absolute paths from the experimental machine were replaced with `/workspace` placeholders in copied text files. To launch a new run, adapt the copied shell scripts' local paths and provide your own credentials through environment variables or private key files.
