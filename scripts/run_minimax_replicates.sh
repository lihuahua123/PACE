#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y-%m-%d__%H-%M-%S")}"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/pace_source/swebench/log/minimax-replicates-${RUN_STAMP}}"
REPLICATES="${REPLICATES:-3}"
MODEL="anthropic/MiniMax-M2.7"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
export MAX_PARALLEL

if ! [[ "${REPLICATES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPLICATES must be a positive integer. Got: ${REPLICATES}" >&2
  exit 2
fi

run_config() {
  local replicate="$1"
  local name="$2"
  shift 2
  RUN_ID="minimax-replicate-${replicate}-${name}-${RUN_STAMP}" \
  OUTPUT_ROOT="${OUTPUT_BASE}/replicate-${replicate}/${name}" \
  MODEL="${MODEL}" \
  "$@" \
  "${SCRIPT_DIR}/run_my_subset_eval.sh"
}

for ((replicate = 1; replicate <= REPLICATES; replicate++)); do
  run_config "${replicate}" org \
    env TRACE_STATUS_BAR_ENABLED=false TRACE_STATUS_REPEAT_BLOCK_ENABLED=false
  run_config "${replicate}" full_guard \
    env TRACE_STATUS_BAR_ENABLED=false TRACE_STATUS_REPEAT_BLOCK_ENABLED=true \
    TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS=8 \
    TRACE_STATUS_REPEAT_BLOCK_MAX_TOTAL_INSPECTIONS=35 \
    TRACE_STATUS_REPEAT_BLOCK_MAX_SCRATCH_EXPERIMENTS=7 \
    TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS=4
done
