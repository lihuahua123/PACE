#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKILL_REPO_ROOT="/workspace/skill"

DATASET_PATH="${DATASET_PATH:-${SKILL_REPO_ROOT}/data/swebench_verified/test-00000-of-00001.parquet}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-Bench_Verified}"
SUBSET_SEED="${SUBSET_SEED:-251016786}"
MODEL="${MODEL:-autodl/DeepSeek-V4-Flash}" #MODEL="${MODEL:-autodl/gpt-5.4-mini}" anthropic/MiniMax-M2.7 
MODEL_TAG="${MODEL##*/}"
RUNNER_PYTHON="${RUNNER_PYTHON:-/workspace/skillsbench/.venv/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y-%m-%d__%H-%M-%S")}"
RUN_ID="${RUN_ID:-eet-${RUN_STAMP}-${MODEL_TAG}-last_50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SKILL_REPO_ROOT}/swebench/log/${RUN_ID}}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/subset_instance_ids.txt}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/subset_summary.json}"
TASK_LOG_DIR="${TASK_LOG_DIR:-${OUTPUT_ROOT}/task_logs}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_TASKS="${MAX_TASKS:-50}"
# The selector emits the reproducible 100-task manifest; run its latter half by default.
LAST_TASKS="${LAST_TASKS:-50}"
REDO_EXISTING="${REDO_EXISTING:-true}"
DOCKER_PRUNE_BETWEEN_BATCHES="${DOCKER_PRUNE_BETWEEN_BATCHES:-1}"
DOCKER_IMAGE_PRUNE_BETWEEN_BATCHES="${DOCKER_IMAGE_PRUNE_BETWEEN_BATCHES:-1}"
DOCKER_PRUNE_AFTER_FINAL_BATCH="${DOCKER_PRUNE_AFTER_FINAL_BATCH:-1}"

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
printf '%s\n' "${INSTANCE_IDS[@]}" >"${MANIFEST_PATH}"

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
    "${SCRIPT_DIR}/run_EET.sh"

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
import re
from pathlib import Path

instance_id = ${instance_id@Q}
output_root = Path(${instance_output_root@Q})
report_path = Path(${REPORT_PATH@Q})
payload = json.loads(report_path.read_text(encoding="utf-8"))
resolved = bool(payload[instance_id]["resolved"])

# Parse trajectory-level signals for early-stop / confidence / retrieval diagnostics.
traj_candidates = sorted(output_root.glob(f"{instance_id}/{instance_id}.traj.json"))
traj_payload = {}
if traj_candidates:
    try:
        traj_payload = json.loads(traj_candidates[0].read_text(encoding="utf-8"))
    except Exception:
        traj_payload = {}

info = traj_payload.get("info") if isinstance(traj_payload, dict) else {}
if not isinstance(info, dict):
    info = {}
messages = traj_payload.get("messages") if isinstance(traj_payload, dict) else []
if not isinstance(messages, list):
    messages = []

agent_exit_status = info.get("exit_status")
early_stop_reason = info.get("stop_reason")
early_stop_triggered = str(agent_exit_status or "").strip() == "StopCheckTerminated"
confidence_scores: list[int] = []
experience_retrieved = False

for msg in messages:
    if not isinstance(msg, dict):
        continue
    content = str(msg.get("content") or "")
    if "## Experience" in content and "<retrieved_experiences>" in content:
        experience_retrieved = True

    # Detect confidence_score from JSON snippets in message text.
    for m in re.finditer(r'\{[^{}]*"confidence_score"\s*:\s*(\d{1,3})[^{}]*\}', content):
        try:
            score = int(m.group(1))
        except Exception:
            continue
        if 0 <= score <= 100:
            confidence_scores.append(score)

task_summary = {
    "instance_id": instance_id,
    "success_within_budget": resolved,
    "resolved": resolved,
    "run_id": ${instance_run_id@Q},
    "output_root": str(output_root),
    "report_path": str(report_path),
    "agent_exit_status": agent_exit_status,
    "early_stop_triggered": early_stop_triggered,
    "early_stop_reason": early_stop_reason,
    "confidence_scores": confidence_scores,
    "last_confidence_score": confidence_scores[-1] if confidence_scores else None,
    "experience_retrieved": experience_retrieved,
    "traj_path": str(traj_candidates[0]) if traj_candidates else None,
}
(output_root / "task_summary.json").write_text(json.dumps(task_summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(f"task_summary={output_root / 'task_summary.json'}")
PY
    echo "[$(date +"%F %T")] DONE ${instance_id}"
  } >"${task_log}" 2>&1
}

run_batch() {
  local active_jobs=0
  local failed_jobs=0
  local instance_id
  for instance_id in "$@"; do
    run_instance "${instance_id}" &
    ((active_jobs += 1))
    if (( active_jobs >= MAX_PARALLEL )); then
      wait -n || ((failed_jobs += 1))
      active_jobs=$((active_jobs - 1))
    fi
  done

  while (( active_jobs > 0 )); do
    wait -n || ((failed_jobs += 1))
    active_jobs=$((active_jobs - 1))
  done

  if (( failed_jobs > 0 )); then
    echo "[$(date +"%F %T")] ERROR batch_failed_jobs=${failed_jobs}" >&2
    return 1
  fi
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
failed_batches=0

echo "[$(date +"%F %T")] SUBSET_START task_count=${total_instances} batch_size=${BATCH_SIZE} total_batches=${total_batches}"
for ((batch_start = 0; batch_start < total_instances; batch_start += BATCH_SIZE)); do
  batch_number=$((batch_number + 1))
  batch_ids=("${INSTANCE_IDS[@]:batch_start:BATCH_SIZE}")
  batch_end=$((batch_start + ${#batch_ids[@]}))
  echo "[$(date +"%F %T")] BATCH_START ${batch_number}/${total_batches} instances=${#batch_ids[@]} range=$((batch_start + 1))-${batch_end}"
  if ! run_batch "${batch_ids[@]}"; then
    failed_batches=$((failed_batches + 1))
    echo "[$(date +"%F %T")] WARN continuing_after_batch_failure batch=${batch_number}/${total_batches}" >&2
  fi
  echo "[$(date +"%F %T")] BATCH_DONE ${batch_number}/${total_batches}"

  if (( batch_number < total_batches )) || [[ "${DOCKER_PRUNE_AFTER_FINAL_BATCH}" == "1" ]]; then
    cleanup_docker_artifacts "${batch_number}" "${total_batches}"
  fi
done
echo "[$(date +"%F %T")] SUBSET_DONE"
if (( failed_batches > 0 )); then
  echo "[$(date +"%F %T")] WARN subset_failed_batches=${failed_batches}" >&2
fi

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

if (( failed_batches > 0 )); then
  exit 1
fi
