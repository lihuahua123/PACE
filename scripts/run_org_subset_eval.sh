#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKILL_REPO_ROOT="/workspace/skill"

DATASET_PATH="${DATASET_PATH:-${SKILL_REPO_ROOT}/data/swebench_verified/test-00000-of-00001.parquet}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-Bench_Verified}"
SUBSET_SEED="${SUBSET_SEED:-251016786}"
MODEL="${MODEL:-autodl/DeepSeek-V4-Flash}" #MODEL="${MODEL:-autodl/gpt-5.4-mini}" #MODEL="${MODEL:-anthropic/MiniMax-M2.5}"
MODEL_TAG="${MODEL##*/}"
RUNNER_PYTHON="${RUNNER_PYTHON:-/workspace/skillsbench/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/mini-swe-agent/src/minisweagent/config/extra/swebench_org.yaml}"
MINISWEAGENT_SRC="${REPO_ROOT}/mini-swe-agent/src"
MINIMAX_API_KEY_FILE="${MINIMAX_API_KEY_FILE:-/workspace/.minimaxapikey}"
MINIMAX_ANTHROPIC_BASE_URL="${MINIMAX_ANTHROPIC_BASE_URL:-https://api.minimaxi.com/anthropic}"
AUTODL_KEY_FILE="${AUTODL_KEY_FILE:-/workspace/.autodlkey}"
AUTODL_BASE_DEFAULT="${AUTODL_BASE_DEFAULT:-https://www.autodl.art/api/v1}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/tmp/minisweagent-config}"
HF_HOME="${HF_HOME:-/tmp/hf_cache}"
EET_EARLY_STOP_ENABLED="${EET_EARLY_STOP_ENABLED:-false}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y-%m-%d__%H-%M-%S")}"
RUN_ID="${RUN_ID:-org-${RUN_STAMP}-${MODEL_TAG}-last_50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SKILL_REPO_ROOT}/swebench/log/${RUN_ID}}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/subset_instance_ids.txt}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/subset_summary.json}"
TASK_LOG_DIR="${TASK_LOG_DIR:-${OUTPUT_ROOT}/task_logs}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_TASKS="${MAX_TASKS:-50}"
# The selector emits the reproducible 100-task manifest; run its latter half by default.
TASK_OFFSET="${TASK_OFFSET:-50}"
REDO_EXISTING="${REDO_EXISTING:-true}"
DOCKER_PRUNE_BETWEEN_BATCHES="${DOCKER_PRUNE_BETWEEN_BATCHES:-1}"
DOCKER_IMAGE_PRUNE_BETWEEN_BATCHES="${DOCKER_IMAGE_PRUNE_BETWEEN_BATCHES:-1}"
DOCKER_PRUNE_AFTER_FINAL_BATCH="${DOCKER_PRUNE_AFTER_FINAL_BATCH:-1}"

if [[ ! -x "${RUNNER_PYTHON}" ]]; then
  echo "Runner python is not executable: ${RUNNER_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
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
if ! [[ "${TASK_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "TASK_OFFSET must be a non-negative integer. Got: ${TASK_OFFSET}" >&2
  exit 2
fi

configure_clash_proxy_env() {
  local existing_proxy="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-${ALL_PROXY:-${all_proxy:-}}}}}}"
  if [[ -n "${existing_proxy}" ]]; then
    export HTTP_PROXY="${HTTP_PROXY:-${existing_proxy}}"
    export HTTPS_PROXY="${HTTPS_PROXY:-${existing_proxy}}"
    export ALL_PROXY="${ALL_PROXY:-${existing_proxy}}"
    export http_proxy="${http_proxy:-${HTTP_PROXY}}"
    export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
    export all_proxy="${all_proxy:-${ALL_PROXY}}"
    return 0
  fi

  local candidate_proxy=""
  if command -v ss >/dev/null 2>&1 && ss -lnt '( sport = :7897 )' 2>/dev/null | rg -q ':7897'; then
    candidate_proxy="http://127.0.0.1:7897"
  elif command -v ss >/dev/null 2>&1 && ss -lnt '( sport = :7890 )' 2>/dev/null | rg -q ':7890'; then
    candidate_proxy="http://127.0.0.1:7890"
  fi

  if [[ -n "${candidate_proxy}" ]]; then
    export HTTP_PROXY="${candidate_proxy}"
    export HTTPS_PROXY="${candidate_proxy}"
    export ALL_PROXY="${candidate_proxy}"
    export http_proxy="${candidate_proxy}"
    export https_proxy="${candidate_proxy}"
    export all_proxy="${candidate_proxy}"
  fi
}

configure_minimax_anthropic_env() {
  local model_id="${1:-}"
  case "${model_id}" in
    anthropic/MiniMax-*)
      ;;
    *)
      return 0
      ;;
  esac

  local anthropic_api_key="${ANTHROPIC_API_KEY:-}"
  if [[ -z "${anthropic_api_key}" || "${anthropic_api_key}" == dummy-* ]]; then
    if [[ ! -f "${MINIMAX_API_KEY_FILE}" ]]; then
      echo "Missing MiniMax API key file: ${MINIMAX_API_KEY_FILE}" >&2
      echo "Set ANTHROPIC_API_KEY or MINIMAX_API_KEY_FILE before running ${model_id}." >&2
      exit 2
    fi
    export ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY="$(tr -d '\r\n' < "${MINIMAX_API_KEY_FILE}")"
  fi

  export ANTHROPIC_BASE_URL="${MINIMAX_ANTHROPIC_BASE_URL}"
}

configure_autodl_openai_env() {
  local model_id="${1:-}"
  case "${model_id}" in
    autodl/*)
      ;;
    *)
      return 0
      ;;
  esac

  local autodl_key="${AUTODL_API_KEY:-${OPENAI_API_KEY:-}}"
  if [[ -z "${autodl_key}" && -f "${AUTODL_KEY_FILE}" ]]; then
    autodl_key="$(tr -d '\r\n' < "${AUTODL_KEY_FILE}")"
  fi
  if [[ -z "${autodl_key}" ]]; then
    echo "Missing AutoDL API key. Set AUTODL_API_KEY/OPENAI_API_KEY or create ${AUTODL_KEY_FILE}." >&2
    exit 2
  fi

  export OPENAI_API_KEY="${autodl_key}"
  export OPENAI_BASE_URL="${AUTODL_API_BASE:-${AUTODL_API_BASE_URL:-${OPENAI_API_BASE:-${OPENAI_BASE_URL:-${AUTODL_BASE_DEFAULT}}}}}"
}

normalize_model_id() {
  local model_id="${1:-}"
  case "${model_id}" in
    autodl/*)
      printf 'openai/%s\n' "${model_id#autodl/}"
      ;;
    *)
      printf '%s\n' "${model_id}"
      ;;
  esac
}

configure_model_provider_env() {
  local model_id="${1:-}"
  configure_minimax_anthropic_env "${model_id}"
  configure_autodl_openai_env "${model_id}"
}

configure_clash_proxy_env
configure_model_provider_env "${MODEL}"
MODEL="$(normalize_model_id "${MODEL}")"
MODEL_LOG_NAME="${MODEL//\//__}"

export XDG_CONFIG_HOME
export HF_HOME
export PYTHONPATH="${MINISWEAGENT_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}" "${TASK_LOG_DIR}"

HF_HOME=/tmp/hf_cache XDG_CACHE_HOME=/tmp/xdg_cache "${RUNNER_PYTHON}" \
  "${SKILL_REPO_ROOT}/scripts/select_swebench_verified_subset.py" \
  --dataset-path "${DATASET_PATH}" \
  --dataset-split "${DATASET_SPLIT}" \
  --seed "${SUBSET_SEED}" \
  --output "${MANIFEST_PATH}" \
  --summary-output "${SUMMARY_PATH}"

mapfile -t INSTANCE_IDS < <(sed '/^[[:space:]]*$/d' "${MANIFEST_PATH}")
if (( TASK_OFFSET > 0 )); then
  if (( TASK_OFFSET >= ${#INSTANCE_IDS[@]} )); then
    echo "TASK_OFFSET (${TASK_OFFSET}) is out of range for manifest size ${#INSTANCE_IDS[@]}" >&2
    exit 3
  fi
  INSTANCE_IDS=("${INSTANCE_IDS[@]:TASK_OFFSET}")
fi
if (( MAX_TASKS > 0 )) && (( ${#INSTANCE_IDS[@]} > MAX_TASKS )); then
  INSTANCE_IDS=("${INSTANCE_IDS[@]:0:MAX_TASKS}")
fi
if [[ "${#INSTANCE_IDS[@]}" -eq 0 ]]; then
  echo "No instance ids selected. Manifest: ${MANIFEST_PATH}" >&2
  exit 3
fi

printf '%s\n' "${INSTANCE_IDS[@]}" > "${MANIFEST_PATH}"

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
    echo "config_path=${CONFIG_PATH}"
    echo "eet_early_stop_enabled=${EET_EARLY_STOP_ENABLED}"

    EFFECTIVE_CONFIG_PATH="${instance_output_root}/config.eet.no_early_stop.yaml"
    "${RUNNER_PYTHON}" - <<PY
import json
from pathlib import Path
import yaml

config_in = Path(${CONFIG_PATH@Q})
config_out = Path(${EFFECTIVE_CONFIG_PATH@Q})
force_early_stop = str(${EET_EARLY_STOP_ENABLED@Q}).strip().lower() in {"1", "true", "yes", "on"}

config = yaml.safe_load(config_in.read_text(encoding="utf-8")) or {}
agent = config.setdefault("agent", {})

# EET early-stop switch: keep disabled unless explicitly enabled.
if not force_early_stop:
    agent["confidence_check_enabled"] = False

config_out.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
print(f"effective_config={config_out}")
print(f"confidence_check_enabled={agent.get('confidence_check_enabled')}")
PY

    FILTER_REGEX="^${instance_id}$"
    REDO_ARGS=()
    if [[ "${REDO_EXISTING}" == "true" ]]; then
      REDO_ARGS+=(--redo-existing)
    fi

    "${RUNNER_PYTHON}" -m minisweagent.run.extra.swebench \
      --subset verified \
      --split "${DATASET_SPLIT}" \
      --filter "${FILTER_REGEX}" \
      --workers 1 \
      --output "${instance_output_root}" \
      --config "${EFFECTIVE_CONFIG_PATH}" \
      --model "${MODEL}" \
      "${REDO_ARGS[@]}"

    PREDICTIONS_PATH="${instance_output_root}/preds.json"
    if [[ ! -f "${PREDICTIONS_PATH}" ]]; then
      echo "preds.json not found after agent run: ${PREDICTIONS_PATH}" >&2
      exit 3
    fi

    VERIFICATION_LOG_PATH="${instance_output_root}/verification.log"
    VERIFICATION_STATUS="passed"
    if ! (
      cd "${instance_output_root}"
      "${RUNNER_PYTHON}" -m swebench.harness.run_evaluation \
        --dataset_name "${DATASET_NAME}" \
        --predictions_path "${PREDICTIONS_PATH}" \
        --max_workers 1 \
        --run_id "${eval_run_id}"
    ) >"${VERIFICATION_LOG_PATH}" 2>&1; then
      VERIFICATION_STATUS="failed"
      echo "run_evaluation failed for ${instance_id}; see ${VERIFICATION_LOG_PATH}" >&2
    fi

    REPORT_PATH="${instance_output_root}/logs/run_evaluation/${eval_run_id}/${MODEL_LOG_NAME}/${instance_id}/report.json"
    if [[ ! -f "${REPORT_PATH}" ]]; then
      REPORT_PATH="$(find "${instance_output_root}/logs/run_evaluation/${eval_run_id}" -type f -name report.json | head -n 1 || true)"
    fi
    if [[ -z "${REPORT_PATH}" || ! -f "${REPORT_PATH}" ]]; then
      if [[ "${VERIFICATION_STATUS}" == "passed" ]]; then
        VERIFICATION_STATUS="missing_report"
      fi
      REPORT_PATH=""
      echo "report.json not found for ${instance_id}" >&2
    fi

    "${RUNNER_PYTHON}" - <<PY
import json
import re
from pathlib import Path

instance_id = ${instance_id@Q}
output_root = Path(${instance_output_root@Q})
report_path_raw = ${REPORT_PATH@Q}
report_path = Path(report_path_raw) if report_path_raw else None
verification_status = ${VERIFICATION_STATUS@Q}
verification_log_path = Path(${VERIFICATION_LOG_PATH@Q})

base_summary_path = output_root / "task_summary.json"
base_summary = {}
if base_summary_path.exists():
    try:
        base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    except Exception:
        base_summary = {}

resolved = False
verification_source = None
verification_error = None
if report_path and report_path.is_file():
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    resolved = bool(payload[instance_id]["resolved"])
    verification_source = "report.json"
elif verification_log_path.is_file():
    verification_error = verification_log_path.read_text(encoding="utf-8")[-4000:]
elif verification_status != "passed":
    verification_error = "verifier failed before producing report output"

if verification_status == "missing_report" and not verification_error:
    verification_error = "report.json not found after swebench.harness.run_evaluation"

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
    "submitted": bool(base_summary.get("submitted", True)),
    "completed": bool(base_summary.get("completed", True)),
    "unresolved": False if resolved else True,
    "empty_patch": bool(base_summary.get("empty_patch", False)),
    "error": bool(base_summary.get("error", False)) or verification_status != "passed",
    "run_id": ${instance_run_id@Q},
    "output_root": str(output_root),
    "report_path": str(report_path) if report_path else None,
    "agent_exit_status": agent_exit_status,
    "early_stop_triggered": early_stop_triggered,
    "early_stop_reason": early_stop_reason,
    "confidence_scores": confidence_scores,
    "last_confidence_score": confidence_scores[-1] if confidence_scores else None,
    "experience_retrieved": experience_retrieved,
    "traj_path": str(traj_candidates[0]) if traj_candidates else None,
    "verification_status": verification_status,
    "verification_supported": True,
    "verification_error": verification_error,
    "verification_log_path": str(verification_log_path),
    "verification_source": verification_source,
}
for key in [
    "phase",
    "cached_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "api_calls",
    "model",
    "updated_at",
    "message_count",
]:
    if key in base_summary and key not in task_summary:
        task_summary[key] = base_summary[key]

merged_summary = dict(base_summary)
merged_summary.update(task_summary)
(output_root / "task_summary.json").write_text(json.dumps(merged_summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
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
