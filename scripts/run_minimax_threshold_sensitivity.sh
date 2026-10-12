#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y-%m-%d__%H-%M-%S")}"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/pace_source/swebench/log/minimax-thresholds-${RUN_STAMP}}"
MODEL="anthropic/MiniMax-M2.7"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
export MAX_PARALLEL

run_profile() {
  local name="$1"
  local consecutive="$2"
  local total="$3"
  local scratch="$4"
  local tests="$5"
  RUN_ID="minimax-threshold-${name}-${RUN_STAMP}" \
  OUTPUT_ROOT="${OUTPUT_BASE}/${name}" \
  MODEL="${MODEL}" \
  TRACE_STATUS_BAR_ENABLED=false \
  TRACE_STATUS_REPEAT_BLOCK_ENABLED=true \
  TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS="${consecutive}" \
  TRACE_STATUS_REPEAT_BLOCK_MAX_TOTAL_INSPECTIONS="${total}" \
  TRACE_STATUS_REPEAT_BLOCK_MAX_SCRATCH_EXPERIMENTS="${scratch}" \
  TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS="${tests}" \
  "${SCRIPT_DIR}/run_my_subset_eval.sh"
}

# Tight/default/loose are online operating points, not percentile labels.
# Replace these values with the exact per-metric empirical percentiles if desired.
run_profile tight 4 20 4 2
run_profile default 8 35 7 4
run_profile loose 12 50 10 8
