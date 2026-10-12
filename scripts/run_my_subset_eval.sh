#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKILL_REPO_ROOT="/workspace/pace_source"

DATASET_PATH="${DATASET_PATH:-${SKILL_REPO_ROOT}/data/swebench_verified/test-00000-of-00001.parquet}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-Bench_Verified}"
SUBSET_SEED="${SUBSET_SEED:-251016786}"
MODEL="${MODEL:-autodl/DeepSeek-V4-Flash}" #anthropic/MiniMax-M2.7 autodl/DeepSeek-V4-Flash
MODEL_TAG="${MODEL##*/}"
case "${MODEL}" in
  anthropic/MiniMax-*) DEFAULT_MINIMAX_QUOTA_GATE_ENABLED=true ;;
  *) DEFAULT_MINIMAX_QUOTA_GATE_ENABLED=false ;;
esac
MINIMAX_QUOTA_GATE_ENABLED="${MINIMAX_QUOTA_GATE_ENABLED:-${DEFAULT_MINIMAX_QUOTA_GATE_ENABLED}}"
MINIMAX_QUOTA_MIN_REMAINING="${MINIMAX_QUOTA_MIN_REMAINING:-150}"
MINIMAX_QUOTA_POLL_SECONDS="${MINIMAX_QUOTA_POLL_SECONDS:-600}"
MINIMAX_API_KEY_FILE="${MINIMAX_API_KEY_FILE:-/workspace/.minimaxapikey}"
MINIMAX_QUOTA_API_BASE="${MINIMAX_QUOTA_API_BASE:-https://api.minimaxi.com}"
RUNNER_PYTHON="${RUNNER_PYTHON:-/workspace/skillsbench/.venv/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y-%m-%d__%H-%M-%S")}"
RUN_ID="${RUN_ID:-my-${RUN_STAMP}-${MODEL_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SKILL_REPO_ROOT}/swebench/log/${RUN_ID}}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/subset_instance_ids.txt}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/subset_summary.json}"
TASK_LOG_DIR="${TASK_LOG_DIR:-${OUTPUT_ROOT}/task_logs}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_TASKS="${MAX_TASKS:-50}"
LAST_TASKS="${LAST_TASKS:-0}"
REDO_EXISTING="${REDO_EXISTING:-true}"
DOCKER_PRUNE_BETWEEN_BATCHES="${DOCKER_PRUNE_BETWEEN_BATCHES:-1}"
DOCKER_IMAGE_PRUNE_BETWEEN_BATCHES="${DOCKER_IMAGE_PRUNE_BETWEEN_BATCHES:-1}"
DOCKER_PRUNE_AFTER_FINAL_BATCH="${DOCKER_PRUNE_AFTER_FINAL_BATCH:-1}"
TRACE_REPETITION_NUDGE_ENABLED="${TRACE_REPETITION_NUDGE_ENABLED:-false}"
TRACE_REPETITION_NUDGE_ENABLED_FORCE="${TRACE_REPETITION_NUDGE_ENABLED_FORCE:-false}"
TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM="${TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM:-6}"
CONFIG_PATH="${CONFIG_PATH:-/workspace/pace_source/mini-swe-agent/src/minisweagent/config/extra/swebench_org.yaml}"

TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED="${TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED:-false}"
TRACE_STATUS_BAR_ENABLED="${TRACE_STATUS_BAR_ENABLED:-false}"
TRACE_STATUS_BAR_MAX_ITEMS="${TRACE_STATUS_BAR_MAX_ITEMS:-8}"
TRACE_STATUS_REPEAT_BLOCK_ENABLED="${TRACE_STATUS_REPEAT_BLOCK_ENABLED:-false}"
TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED="${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED:-false}"
TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS="${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS:-30}"
TRACE_STATUS_POST_VERIFY_GUARD_ENABLED="${TRACE_STATUS_POST_VERIFY_GUARD_ENABLED:-false}"

INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT="${INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT:-false}"
INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_MD="${INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_MD:-${SKILL_REPO_ROOT}/my_skills/read_phase_token_efficiency_skill.md}"

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
if [[ "${MINIMAX_QUOTA_GATE_ENABLED}" == "true" ]]; then
  if ! [[ "${MINIMAX_QUOTA_MIN_REMAINING}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MINIMAX_QUOTA_MIN_REMAINING must be a positive integer. Got: ${MINIMAX_QUOTA_MIN_REMAINING}" >&2
    exit 2
  fi
  if ! [[ "${MINIMAX_QUOTA_POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MINIMAX_QUOTA_POLL_SECONDS must be a positive integer. Got: ${MINIMAX_QUOTA_POLL_SECONDS}" >&2
    exit 2
  fi
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
if (( MAX_TASKS > 0 )) && (( ${#INSTANCE_IDS[@]} > MAX_TASKS )); then
  INSTANCE_IDS=("${INSTANCE_IDS[@]:0:MAX_TASKS}")
fi
if (( LAST_TASKS > 0 )) && (( ${#INSTANCE_IDS[@]} > LAST_TASKS )); then
  INSTANCE_IDS=("${INSTANCE_IDS[@]: -LAST_TASKS}")
fi
if [[ "${#INSTANCE_IDS[@]}" -eq 0 ]]; then
  echo "No instance ids selected. Manifest: ${MANIFEST_PATH}" >&2
  exit 3
fi

minimax_remaining_requests() {
  local api_key quota_json
  if [[ ! -f "${MINIMAX_API_KEY_FILE}" ]]; then
    echo "MiniMax API key file not found: ${MINIMAX_API_KEY_FILE}" >&2
    return 1
  fi
  api_key="$(tr -d '\r\n' < "${MINIMAX_API_KEY_FILE}")"
  quota_json="$(curl --fail --silent --show-error --max-time 30 \
    -H "Authorization: Bearer ${api_key}" \
    "${MINIMAX_QUOTA_API_BASE%/}/v1/token_plan/remains")" || return 1

  printf '%s' "${quota_json}" | "${RUNNER_PYTHON}" -c '
import json
import sys

payload = json.load(sys.stdin)
remaining = []
unmetered = False
for item in payload.get("model_remains", []):
    model_name = str(item.get("model_name", ""))
    if not (model_name.startswith("MiniMax-M") or model_name == "general"):
        continue
    total = item.get("current_interval_total_count")
    reported = item.get("current_interval_usage_count")
    if item.get("current_interval_status") == 2:
        remaining.append(0)
        continue
    percent = item.get("current_interval_remaining_percent")
    # The CN Token Plan may report general text quota as a percentage only
    # (total_count=0). It cannot support a request-count reserve, but a
    # positive percentage still proves the next task may start.
    if isinstance(total, (int, float)) and total == 0:
        if isinstance(percent, (int, float)) and percent > 0:
            unmetered = True
        elif isinstance(percent, (int, float)) and percent == 0:
            remaining.append(0)
        continue
    if not isinstance(total, (int, float)) or not isinstance(reported, (int, float)) or total < 0:
        continue
    value = reported
    if isinstance(percent, (int, float)):
        as_remaining = reported / total * 100
        as_used = (total - reported) / total * 100
        if abs(as_used - percent) < abs(as_remaining - percent):
            value = total - reported
    if 0 <= value <= total:
        remaining.append(int(value))
if remaining:
    print(min(remaining))
elif unmetered:
    print("available")
else:
    raise SystemExit("MiniMax text quota was absent or malformed")
'
}

wait_for_minimax_quota() {
  local instance_id="$1"
  local remaining
  if [[ "${MINIMAX_QUOTA_GATE_ENABLED}" != "true" ]]; then
    return 0
  fi
  while true; do
    if remaining="$(minimax_remaining_requests)"; then
      if [[ "${remaining}" == "available" ]] || (( remaining >= MINIMAX_QUOTA_MIN_REMAINING )); then
        echo "[$(date +"%F %T")] QUOTA_OK ${instance_id} remaining=${remaining} reserve=${MINIMAX_QUOTA_MIN_REMAINING}"
        return 0
      fi
      echo "[$(date +"%F %T")] QUOTA_WAIT ${instance_id} remaining=${remaining} reserve=${MINIMAX_QUOTA_MIN_REMAINING} retry_seconds=${MINIMAX_QUOTA_POLL_SECONDS}"
    else
      echo "[$(date +"%F %T")] QUOTA_QUERY_FAILED ${instance_id} retry_seconds=${MINIMAX_QUOTA_POLL_SECONDS}" >&2
    fi
    sleep "${MINIMAX_QUOTA_POLL_SECONDS}"
  done
}

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
    wait_for_minimax_quota "${instance_id}"
    echo "[$(date +"%F %T")] START ${instance_id}"
    INSTANCE_ID="${instance_id}" \
    DATASET_NAME="${DATASET_NAME}" \
    DATASET_SPLIT="${DATASET_SPLIT}" \
    MODEL="${MODEL}" \
    RUNNER_PYTHON="${RUNNER_PYTHON}" \
    RUN_ID="${instance_run_id}" \
    OUTPUT_ROOT="${instance_output_root}" \
    REDO_EXISTING="${REDO_EXISTING}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT="${INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT}" \
    TRACE_REPETITION_NUDGE_ENABLED="${TRACE_REPETITION_NUDGE_ENABLED}" \
    TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED="${TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED}" \
    TRACE_REPETITION_NUDGE_ENABLED_FORCE="${TRACE_REPETITION_NUDGE_ENABLED_FORCE}" \
    TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM="${TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM}" \
    TRACE_STATUS_BAR_ENABLED="${TRACE_STATUS_BAR_ENABLED}" \
    TRACE_STATUS_BAR_MAX_ITEMS="${TRACE_STATUS_BAR_MAX_ITEMS}" \
    TRACE_STATUS_REPEAT_BLOCK_ENABLED="${TRACE_STATUS_REPEAT_BLOCK_ENABLED}" \
    TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS="${TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS:-10}" \
    TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS="${TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS:-18}" \
    TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED="${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED}" \
    TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS="${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS}" \
    TRACE_STATUS_POST_VERIFY_GUARD_ENABLED="${TRACE_STATUS_POST_VERIFY_GUARD_ENABLED}" \
    INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_MD="${INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_MD}" \
    "${SCRIPT_DIR}/run_my.sh"

    REPORT_PATH="$(find "${instance_output_root}/logs/run_evaluation/${eval_run_id}" -type f -name report.json | head -n 1)"
    if [[ -z "${REPORT_PATH}" || ! -f "${REPORT_PATH}" ]]; then
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
echo "[$(date +"%F %T")] trace_repetition_nudge_enabled=${TRACE_REPETITION_NUDGE_ENABLED}"
echo "[$(date +"%F %T")] trace_repetition_read_loop_nudge_enabled=${TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED}"
echo "[$(date +"%F %T")] trace_repetition_nudge_enabled_force=${TRACE_REPETITION_NUDGE_ENABLED_FORCE}"
echo "[$(date +"%F %T")] trace_repetition_nudge_enabled_force_num=${TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM}"
echo "[$(date +"%F %T")] trace_status_bar_enabled=${TRACE_STATUS_BAR_ENABLED}"
echo "[$(date +"%F %T")] trace_status_bar_max_items=${TRACE_STATUS_BAR_MAX_ITEMS}"
echo "[$(date +"%F %T")] trace_status_repeat_block_enabled=${TRACE_STATUS_REPEAT_BLOCK_ENABLED}"
echo "[$(date +"%F %T")] trace_status_repeat_block_dynamic_inspection_enabled=${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED}"
echo "[$(date +"%F %T")] trace_status_repeat_block_dynamic_inspection_max_bonus=${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS}"
echo "[$(date +"%F %T")] trace_status_post_verify_guard_enabled=${TRACE_STATUS_POST_VERIFY_GUARD_ENABLED}"
echo "[$(date +"%F %T")] minimax_quota_gate_enabled=${MINIMAX_QUOTA_GATE_ENABLED}"
echo "[$(date +"%F %T")] minimax_quota_min_remaining=${MINIMAX_QUOTA_MIN_REMAINING}"
echo "[$(date +"%F %T")] minimax_quota_poll_seconds=${MINIMAX_QUOTA_POLL_SECONDS}"
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
