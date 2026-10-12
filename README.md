# PACE replication package

This package contains the PACE controller implementation, run and analysis scripts, task-level figure data, and 50-task archived trajectories for the main PACE versus no-control comparisons. It was exported from the experimental workspace without its Git history, credentials, evaluator logs, or local environment.

## Contents

- `mini-swe-agent/`: the modified MiniSWEAgent source used for PACE. The pre-execution controller is in `src/minisweagent/agents/default.py`; local price normalization is in `src/minisweagent/pricing.py`. The upstream MiniSWEAgent license is retained.
- `scripts/`: the archived launch and analysis scripts. Launch scripts require a SWE-bench evaluator, model access, and environment-specific paths. API credentials are not included.
- `traces/`: each run has 50 task directories containing an agent trajectory, task summary, prediction where available, and verifier outcome where available.
- `data/figure_tables/`: task-level data and summaries used for the paper figures and tables.
- `data/trace_manifest.csv`: mapping from neutral run labels to archived run IDs and files.
- `reproduce_main.py`: recomputes success, API calls, tokens, and estimated CNY cost for the main no-control and PACE rows.

## Recompute the main comparison

Use Python 3.10 or newer. Install the dependencies in `mini-swe-agent/pyproject.toml` plus `pandas`, then run:

```bash
python3 reproduce_main.py
```

The script checks that each included run has 50 task records. It uses the archived token pricing implementation and task summaries/trajectories. The expected main results are:

| Model and configuration | Successes by run | Mean success | Mean cost (CNY) |
| --- | --- | ---: | ---: |
| MiniMax-M2.7 no control | 38, 35, 39 | 74.7% | 21.560 |
| MiniMax-M2.7 PACE | 36, 35, 35 | 70.7% | 13.408 |
| gpt-5.6-luna no control | 23, 27, 29 | 52.7% | 3.604 |
| gpt-5.6-luna PACE | 34, 31 | 65.0% | 2.881 |
| DeepSeek-V4-Flash no control | 40 | 80.0% | 59.199 |
| DeepSeek-V4-Flash PACE | 40 | 80.0% | 31.797 |

The `luna_pace_1` directory is an archived case-study run and is excluded from the two-run luna main-table mean. The `minimax_case_pace` directory is also a case-study run, excluded from the three-run MiniMax main-table mean.

## MiniMax provenance and later reruns

The main-table MiniMax PACE runs use the same 50 task IDs and limits `(8, 35, 7, 4)`. `minimax_pace_1` and `_2` used AutoDL's OpenAI-compatible endpoint; `minimax_pace_3` used MiniMax's Anthropic-compatible endpoint. The original, independent runs are reconstructed from the archived pre-replacement task backups. Their counts are 36, 35, and 35 successes.

Several source run directories were later modified by replacing selected failed tasks with successful reruns. Those selected outcomes are **not** independent 50-task replications. The source-workspace replacement metadata is preserved in `data/rep2_replacement_manifest.json`, `data/rep3_replacement_manifest.json`, `data/minimax_anthropic_selected_rerun_summary.json`, and `data/selected_rerun_provenance.json`. `data/minimax_anthropic_original_summary.json` records the original Anthropic run. The main-table values above use the original task records.

## Scope and privacy

This export includes the main PACE and no-control trajectories and derived task-level data. It does not include the full evaluator containers, model weights, private API credentials, or every exploratory run. The two external comparison methods' launch scripts and published summary data are included, but their full trajectories are not in this compact package. Absolute paths from the experimental machine were replaced with `/workspace` placeholders in copied text files. To launch a new run, adapt the copied shell scripts' local paths and provide your own credentials through environment variables or private key files.
