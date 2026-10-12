#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKILL_REPO_ROOT="/workspace/skill"

DATASET_PATH="${DATASET_PATH:-${SKILL_REPO_ROOT}/data/swebench_verified/test-00000-of-00001.parquet}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-Bench_Verified}"
SUBSET_SEED="${SUBSET_SEED:-251016786}"
MODEL="${MODEL:-autodl/DeepSeek-V4-Flash}"
MODEL_TAG="${MODEL##*/}"
RUNNER_PYTHON="${RUNNER_PYTHON:-/workspace/skillsbench/.venv/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y-%m-%d__%H-%M-%S")}"
RUN_ID="${RUN_ID:-turn-controller-${RUN_STAMP}-${MODEL_TAG}-last_50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SKILL_REPO_ROOT}/swebench/log/${RUN_ID}}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/subset_instance_ids.txt}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/subset_summary.json}"
TASK_LOG_DIR="${TASK_LOG_DIR:-${OUTPUT_ROOT}/task_logs}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_TASKS="${MAX_TASKS:-50}"
# The selector emits the reproducible 100-task manifest; run its latter half by default.
LAST_TASKS="${LAST_TASKS:-50}"
REDO_EXISTING="${REDO_EXISTING:-true}"
DOCKER_PRUNE_BETWEEN_BATCHES="${DOCKER_PRUNE_BETWEEN_BATCHES:-1}"
DOCKER_IMAGE_PRUNE_BETWEEN_BATCHES="${DOCKER_IMAGE_PRUNE_BETWEEN_BATCHES:-1}"
DOCKER_PRUNE_AFTER_FINAL_BATCH="${DOCKER_PRUNE_AFTER_FINAL_BATCH:-1}"

TURN_CONTROL_INITIAL_LIMIT="${TURN_CONTROL_INITIAL_LIMIT:-27}"
TURN_CONTROL_EXTENSION_LIMIT="${TURN_CONTROL_EXTENSION_LIMIT:-44}"
TURN_CONTROL_REMINDER="${TURN_CONTROL_REMINDER:-true}"
TURN_CONTROL_BASELINE_TRACE_ROOT="${TURN_CONTROL_BASELINE_TRACE_ROOT:-}"
TURN_CONTROL_INITIAL_PERCENTILE="${TURN_CONTROL_INITIAL_PERCENTILE:-0.5}"
TURN_CONTROL_EXTENSION_PERCENTILE="${TURN_CONTROL_EXTENSION_PERCENTILE:-0.75}"

if [[ ! -x "${RUNNER_PYTHON}" ]]; then
  echo "Runner python is not executable: ${RUNNER_PYTHON}" >&2
  exit 2
fi

if (( MAX_PARALLEL <= 0 )); then
  echo "MAX_PARALLEL must be positive. Got: ${MAX_PARALLEL}" >&2
  exit 2
fi
if (( BATCH_SIZE <= 0 )); then
  echo "BATCH_SIZE must be positive. Got: ${BATCH_SIZE}" >&2
  exit 2
fi
if ! [[ "${MAX_TASKS}" =~ ^[0-9]+$ ]]; then
  echo "MAX_TASKS must be a non-negative integer. Got: ${MAX_TASKS}" >&2
  exit 2
fi
if ! [[ "${LAST_TASKS}" =~ ^[0-9]+$ ]]; then
  echo "LAST_TASKS must be a non-negative integer. Got: ${LAST_TASKS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${TASK_LOG_DIR}"

HF_HOME=/tmp/hf_cache XDG_CACHE_HOME=/tmp/xdg_cache "${RUNNER_PYTHON}" \
  "${SKILL_REPO_ROOT}/scripts/select_swebench_verified_subset.py" \
  --dataset-path "${DATASET_PATH}" \
  --dataset-split "${DATASET_SPLIT}" \
  --seed "${SUBSET_SEED}" \
  --output "${MANIFEST_PATH}" \
  --summary-output "${SUMMARY_PATH}"

mapfile -t INSTANCE_IDS < <(sed '/^[[:space:]]*$/d' "${MANIFEST_PATH}")
if (( LAST_TASKS > 0 )) && (( ${#INSTANCE_IDS[@]} > LAST_TASKS )); then
  INSTANCE_IDS=("${INSTANCE_IDS[@]: -LAST_TASKS}")
fi
if (( MAX_TASKS > 0 )) && (( ${#INSTANCE_IDS[@]} > MAX_TASKS )); then
  INSTANCE_IDS=("${INSTANCE_IDS[@]:0:MAX_TASKS}")
fi
if [[ "${#INSTANCE_IDS[@]}" -eq 0 ]]; then
  echo "No instance ids selected. Manifest: ${MANIFEST_PATH}" >&2
  exit 3
fi

# Keep manifest aligned with the actual run list so downstream summary
# always reflects the true task set executed in this run.
printf '%s\n' "${INSTANCE_IDS[@]}" > "${MANIFEST_PATH}"
echo "[$(date +"%F %T")] EFFECTIVE_TASKS task_count=${#INSTANCE_IDS[@]} manifest=${MANIFEST_PATH}"

if [[ -n "${TURN_CONTROL_BASELINE_TRACE_ROOT}" ]]; then
  if [[ ! -d "${TURN_CONTROL_BASELINE_TRACE_ROOT}" ]]; then
    echo "TURN_CONTROL_BASELINE_TRACE_ROOT is not a directory: ${TURN_CONTROL_BASELINE_TRACE_ROOT}" >&2
    exit 2
  fi

  TURN_LIMITS="$("${RUNNER_PYTHON}" - \
    "${TURN_CONTROL_BASELINE_TRACE_ROOT}" \
    "${MANIFEST_PATH}" \
    "${TURN_CONTROL_INITIAL_PERCENTILE}" \
    "${TURN_CONTROL_EXTENSION_PERCENTILE}" \
    "${MODEL}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

trace_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
initial_p = float(sys.argv[3])
extension_p = float(sys.argv[4])
model = sys.argv[5]

if not (0.0 <= initial_p <= 1.0 and 0.0 <= extension_p <= 1.0):
    raise SystemExit("initial/extension percentile must be in [0,1]")
if extension_p <= initial_p:
    raise SystemExit("extension percentile must be greater than initial percentile")

instance_ids = [
    line.strip()
    for line in manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

turns: list[int] = []
missing: list[str] = []
mismatched_model: list[str] = []
for instance_id in instance_ids:
    traj_path = trace_root / instance_id / "attempt_1" / "traj.json"
    if not traj_path.exists():
        missing.append(instance_id)
        continue
    payload = json.loads(traj_path.read_text(encoding="utf-8"))
    used_model = (
        (payload.get("info") or {})
        .get("config", {})
        .get("model", {})
        .get("name")
    )
    if isinstance(used_model, str) and used_model and used_model != model:
        mismatched_model.append(f"{instance_id}:{used_model}")
        continue
    usage_per_round = payload.get("usage_per_round")
    if isinstance(usage_per_round, list) and usage_per_round:
        turns.append(len(usage_per_round))
        continue
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        turns.append(len(messages))
        continue
    missing.append(instance_id)

if not turns:
    raise SystemExit("no usable baseline traces found for selected subset")

turns.sort()

def quantile(values: list[int], p: float) -> float:
    if len(values) == 1:
        return float(values[0])
    idx = (len(values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(values[lo])
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)

initial_limit = int(math.ceil(quantile(turns, initial_p)))
extension_limit = int(math.ceil(quantile(turns, extension_p)))
if extension_limit <= initial_limit:
    extension_limit = initial_limit + 1

print(f"{initial_limit} {extension_limit} {len(turns)} {len(missing)} {len(mismatched_model)}")
PY
)"

  read -r AUTO_INITIAL_LIMIT AUTO_EXTENSION_LIMIT AUTO_SAMPLE_COUNT AUTO_MISSING_COUNT AUTO_MISMATCHED_MODEL_COUNT <<<"${TURN_LIMITS}"
  TURN_CONTROL_INITIAL_LIMIT="${AUTO_INITIAL_LIMIT}"
  TURN_CONTROL_EXTENSION_LIMIT="${AUTO_EXTENSION_LIMIT}"

  echo "[$(date +"%F %T")] AUTO_TURN_LIMITS baseline_root=${TURN_CONTROL_BASELINE_TRACE_ROOT}"
  echo "[$(date +"%F %T")] AUTO_TURN_LIMITS model=${MODEL} initial_percentile=${TURN_CONTROL_INITIAL_PERCENTILE} extension_percentile=${TURN_CONTROL_EXTENSION_PERCENTILE}"
  echo "[$(date +"%F %T")] AUTO_TURN_LIMITS initial_limit=${TURN_CONTROL_INITIAL_LIMIT} extension_limit=${TURN_CONTROL_EXTENSION_LIMIT} sample_count=${AUTO_SAMPLE_COUNT} missing_count=${AUTO_MISSING_COUNT} model_mismatch_count=${AUTO_MISMATCHED_MODEL_COUNT}"
fi

run_instance() {
  local instance_id="$1"
  local task_log="${TASK_LOG_DIR}/${instance_id}.log"
  local instance_output_root="${OUTPUT_ROOT}/${instance_id}"
  local task_summary="${instance_output_root}/task_summary.json"
  local instance_run_id="${RUN_ID}-${instance_id}"
  local eval_run_id="${instance_run_id}__verify"

  if [[ -f "${task_summary}" ]]; then
    {
      echo "[$(date +"%F %T")] SKIP ${instance_id}"
      echo "reason=existing_task_summary"
      echo "task_summary=${task_summary}"
    } >"${task_log}"
    return 0
  fi

  mkdir -p "${instance_output_root}"

  {
    echo "[$(date +"%F %T")] START ${instance_id}"
    INSTANCE_ID="${instance_id}" \
    DATASET_NAME="${DATASET_NAME}" \
    DATASET_SPLIT="${DATASET_SPLIT}" \
    MODEL="${MODEL}" \
    RUNNER_PYTHON="${RUNNER_PYTHON}" \
    RUN_ID="${instance_run_id}" \
    OUTPUT_ROOT="${instance_output_root}" \
    REDO_EXISTING="${REDO_EXISTING}" \
    TURN_CONTROL_INITIAL_LIMIT="${TURN_CONTROL_INITIAL_LIMIT}" \
    TURN_CONTROL_EXTENSION_LIMIT="${TURN_CONTROL_EXTENSION_LIMIT}" \
    TURN_CONTROL_REMINDER="${TURN_CONTROL_REMINDER}" \
    "${SCRIPT_DIR}/turn_controller_single.sh"

    REPORT_PATH="$(find "${instance_output_root}/logs/run_evaluation/${eval_run_id}" -type f -name report.json 2>/dev/null | head -n 1 || true)"
    if [[ -z "${REPORT_PATH}" || ! -f "${REPORT_PATH}" ]]; then
      if [[ -f "${task_summary}" ]]; then
        echo "report.json not found for ${instance_id}; using existing task_summary=${task_summary}"
        echo "[$(date +"%F %T")] DONE ${instance_id}"
        return 0
      fi
      echo "report.json not found for ${instance_id}" >&2
      exit 4
    fi

    "${RUNNER_PYTHON}" - <<PY
import json
from pathlib import Path

instance_id = ${instance_id@Q}
output_root = Path(${instance_output_root@Q})
report_path = Path(${REPORT_PATH@Q})
payload = json.loads(report_path.read_text(encoding="utf-8"))
resolved = bool(payload[instance_id]["resolved"])
task_summary = {
    "instance_id": instance_id,
    "success_within_budget": resolved,
    "resolved": resolved,
    "run_id": ${instance_run_id@Q},
    "output_root": str(output_root),
    "report_path": str(report_path),
}
(output_root / "task_summary.json").write_text(json.dumps(task_summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(f"task_summary={output_root / 'task_summary.json'}")
PY
    echo "[$(date +"%F %T")] DONE ${instance_id}"
  } >"${task_log}" 2>&1
}

run_batch() {
  local active_jobs=0
  local instance_id
  for instance_id in "$@"; do
    run_instance "${instance_id}" &
    ((active_jobs += 1))
    if (( active_jobs >= MAX_PARALLEL )); then
      wait -n || true
      active_jobs=$((active_jobs - 1))
    fi
  done

  while (( active_jobs > 0 )); do
    wait -n || true
    active_jobs=$((active_jobs - 1))
  done
}

cleanup_docker_artifacts() {
  local batch_number="$1"
  local total_batches="$2"

  if [[ "${DOCKER_PRUNE_BETWEEN_BATCHES}" != "1" ]]; then
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "[$(date +"%F %T")] WARN batch=${batch_number}/${total_batches} docker not found; skip prune" >&2
    return 0
  fi

  echo "[$(date +"%F %T")] CLEANUP batch=${batch_number}/${total_batches} docker builder prune -af"
  docker builder prune -af || echo "[$(date +"%F %T")] WARN docker builder prune failed" >&2

  if [[ "${DOCKER_IMAGE_PRUNE_BETWEEN_BATCHES}" == "1" ]]; then
    echo "[$(date +"%F %T")] CLEANUP batch=${batch_number}/${total_batches} docker image prune -af"
    docker image prune -af || echo "[$(date +"%F %T")] WARN docker image prune failed" >&2
  fi
}

total_instances="${#INSTANCE_IDS[@]}"
total_batches="$(((total_instances + BATCH_SIZE - 1) / BATCH_SIZE))"
batch_number=0

echo "[$(date +"%F %T")] SUBSET_START task_count=${total_instances} batch_size=${BATCH_SIZE} total_batches=${total_batches}"
for ((batch_start = 0; batch_start < total_instances; batch_start += BATCH_SIZE)); do
  batch_number=$((batch_number + 1))
  batch_ids=("${INSTANCE_IDS[@]:batch_start:BATCH_SIZE}")
  batch_end=$((batch_start + ${#batch_ids[@]}))
  echo "[$(date +"%F %T")] BATCH_START ${batch_number}/${total_batches} instances=${#batch_ids[@]} range=$((batch_start + 1))-${batch_end}"
  run_batch "${batch_ids[@]}"
  echo "[$(date +"%F %T")] BATCH_DONE ${batch_number}/${total_batches}"

  if (( batch_number < total_batches )) || [[ "${DOCKER_PRUNE_AFTER_FINAL_BATCH}" == "1" ]]; then
    cleanup_docker_artifacts "${batch_number}" "${total_batches}"
  fi
done
echo "[$(date +"%F %T")] SUBSET_DONE"

HF_HOME=/tmp/hf_cache XDG_CACHE_HOME=/tmp/xdg_cache "${RUNNER_PYTHON}" - <<'PY' \
  "${MANIFEST_PATH}" \
  "${OUTPUT_ROOT}" \
  "${SUMMARY_PATH}"
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

instance_ids = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
completed = 0
resolved = 0
failed = 0
missing = []
for instance_id in instance_ids:
    task_summary_path = output_root / instance_id / "task_summary.json"
    if not task_summary_path.exists():
        missing.append(instance_id)
        continue
    completed += 1
    payload = json.loads(task_summary_path.read_text(encoding="utf-8"))
    if bool(payload.get("success_within_budget")):
        resolved += 1
    else:
        failed += 1

summary = {}
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary.update(
    {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "task_count": len(instance_ids),
        "completed_count": completed,
        "resolved_count": resolved,
        "failed_count": failed,
        "missing_count": len(missing),
        "missing_instance_ids": missing,
    }
)
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"task_count={len(instance_ids)}")
print(f"completed_count={completed}")
print(f"resolved_count={resolved}")
print(f"failed_count={failed}")
print(f"missing_count={len(missing)}")
print(f"manifest_path={manifest_path}")
print(f"summary_path={summary_path}")
print(f"output_root={output_root}")
PY
