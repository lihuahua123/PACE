#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-${REPO_ROOT}/.tmp}"
#django__django-11206 是deepseek最需要轮次的
#django__django-15098
#django__django-15563 15957 是gpt-5.4-mini最需要轮次的
#django__django-15973  deepseek 124次调用
# django__django-13417  deepseek 最终 resolved=true

INSTANCE_ID="${INSTANCE_ID:-django__django-11734}" #astropy__astropy-14508
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-Bench_Verified}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}" # deepseek-v4-flash gpt-5.4-mini
MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS:-0}"
MODEL_TAG="${MODEL##*/}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
sanitize_name() {
  local value="${1:-}"
  printf '%s' "${value}" | tr '/:@' '___' | tr -cd '[:alnum:]_.-'
}

model_log_name() {
  local value="${1:-}"
  value="${value//\//__}"
  printf '%s' "${value}" | tr ':@' '__' | tr -cd '[:alnum:]_.-'
}

SAFE_MODEL_TAG="$(sanitize_name "${MODEL_TAG}")"
RUN_ID="${RUN_ID:-my-${INSTANCE_ID}-${TIMESTAMP}-${SAFE_MODEL_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/my/${RUN_ID}}"
RUNNER_PYTHON="${RUNNER_PYTHON:-${WORKSPACE_ROOT}/skillsbench/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/mini-swe-agent/src/minisweagent/config/extra/swebench_org.yaml}"
MINISWEAGENT_SRC="${REPO_ROOT}/mini-swe-agent/src"
MINIMAX_API_KEY_FILE="${MINIMAX_API_KEY_FILE:-${WORKSPACE_ROOT}/.minimaxapikey}"
MINIMAX_ANTHROPIC_BASE_URL="${MINIMAX_ANTHROPIC_BASE_URL:-https://api.minimaxi.com/anthropic}"
AUTODL_KEY_FILE="${AUTODL_KEY_FILE:-${REPO_ROOT}/.autodlkey}"
AUTODL_BASE_DEFAULT="${AUTODL_BASE_DEFAULT:-https://www.autodl.art/api/v1}"
DEEPSEEK_KEY_FILE="${DEEPSEEK_KEY_FILE:-${WORKSPACE_ROOT}/.deepseekkey}"
DEEPSEEK_BASE_DEFAULT="${DEEPSEEK_BASE_DEFAULT:-https://api.deepseek.com}"
OPENROUTER_KEY_FILE="${OPENROUTER_KEY_FILE:-${WORKSPACE_ROOT}/.openrouterapikey}"
OPENROUTER_BASE_DEFAULT="${OPENROUTER_BASE_DEFAULT:-https://openrouter.ai/api/v1}"
VLLM_BASE_DEFAULT="${VLLM_BASE_URL:-${VLLM_API_BASE:-http://0.0.0.0:8900/v1}}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${LOCAL_TMP_ROOT}/minisweagent-config}"
HF_HOME="${HF_HOME:-${LOCAL_TMP_ROOT}/hf_cache}"
REDO_EXISTING="${REDO_EXISTING:-true}"
INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT="${INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT:-false}"
INJECT_MIN_TOKEN_PLAN_FIRST_PROMPT="${INJECT_MIN_TOKEN_PLAN_FIRST_PROMPT:-false}"
# Persist only high-confidence shell capability failures between matching SWE-bench
# environments. Disabled unless explicitly requested for an experiment.
CROSS_TASK_CAPABILITY_LEDGER_ENABLED="${CROSS_TASK_CAPABILITY_LEDGER_ENABLED:-false}"
CAPABILITY_LEDGER_PATH="${CAPABILITY_LEDGER_PATH:-${REPO_ROOT}/capability_ledger.json}"
CAPABILITY_LEDGER_BENCHMARK="${CAPABILITY_LEDGER_BENCHMARK:-SWE-bench Verified}"
CAPABILITY_LEDGER_ENVIRONMENT_FAMILY="${CAPABILITY_LEDGER_ENVIRONMENT_FAMILY:-sweb.eval.x86_64.${INSTANCE_ID%%__*}}"
# 直接提交会降低成功率
TRACE_REPETITION_NUDGE_ENABLED="${TRACE_REPETITION_NUDGE_ENABLED:-false}"
# 这个读限制不起作用
TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED="${TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED:-false}"
# Force 是反作用的，降低准确度
TRACE_REPETITION_NUDGE_ENABLED_FORCE="${TRACE_REPETITION_NUDGE_ENABLED_FORCE:-false}"
TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM="${TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM:-3}"
CONFIDENCE_CHECK_ENABLED="${CONFIDENCE_CHECK_ENABLED:-false}"
SUBMISSION_REVIEW_ENABLED="${SUBMISSION_REVIEW_ENABLED:-false}"
# 有些任务有用，只对弱模型有用
TRACE_STATUS_BAR_ENABLED="${TRACE_STATUS_BAR_ENABLED:-false}"
TRACE_STATUS_BAR_MAX_ITEMS="${TRACE_STATUS_BAR_MAX_ITEMS:-8}"
TRACE_STATUS_BAR_INTERVAL="${TRACE_STATUS_BAR_INTERVAL:-3}"
TRACE_STATUS_BAR_NEXT_ACTION_GUIDANCE_ENABLED="${TRACE_STATUS_BAR_NEXT_ACTION_GUIDANCE_ENABLED:-false}"
# Blocks repeated reads/searches/edits/tests without injecting the status bar into the agent context.
# 这个比较有用
TRACE_STATUS_REPEAT_BLOCK_ENABLED="${TRACE_STATUS_REPEAT_BLOCK_ENABLED:-false}"
TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS="${TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS:-10}"
TRACE_STATUS_REPEAT_BLOCK_MAX_TOTAL_INSPECTIONS="${TRACE_STATUS_REPEAT_BLOCK_MAX_TOTAL_INSPECTIONS:-35}"
TRACE_STATUS_REPEAT_BLOCK_MAX_SCRATCH_EXPERIMENTS="${TRACE_STATUS_REPEAT_BLOCK_MAX_SCRATCH_EXPERIMENTS:-7}"
TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS="${TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS:-18}"
TRACE_STATUS_REPEAT_BLOCK_SOFT_WARNING_ENABLED="${TRACE_STATUS_REPEAT_BLOCK_SOFT_WARNING_ENABLED:-false}"
TRACE_STATUS_REPEAT_BLOCK_WARN_CONSECUTIVE_INSPECTIONS="${TRACE_STATUS_REPEAT_BLOCK_WARN_CONSECUTIVE_INSPECTIONS:-8}"
TRACE_STATUS_REPEAT_BLOCK_WARN_TOTAL_INSPECTIONS="${TRACE_STATUS_REPEAT_BLOCK_WARN_TOTAL_INSPECTIONS:-11}"
TRACE_STATUS_REPEAT_BLOCK_WARN_SCRATCH_EXPERIMENTS="${TRACE_STATUS_REPEAT_BLOCK_WARN_SCRATCH_EXPERIMENTS:-1}"
TRACE_STATUS_REPEAT_BLOCK_WARN_TEST_PHASE_TESTS="${TRACE_STATUS_REPEAT_BLOCK_WARN_TEST_PHASE_TESTS:-4}"

# 这个没测过
TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED="${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED:-false}"
TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS="${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS:-30}"
# Blocks a single source-file read whose requested line window exceeds this limit.
# Add # TRACE_STATUS_LARGE_READ_BLOCK_CONFIRMED to explicitly allow that one read.
TRACE_STATUS_LARGE_READ_BLOCK_ENABLED="${TRACE_STATUS_LARGE_READ_BLOCK_ENABLED:-false}"
TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES="${TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES:-1000}"
# Independently limit a successful command's observation before it enters model context.
TRACE_STATUS_LONG_OUTPUT_TRUNCATION_ENABLED="${TRACE_STATUS_LONG_OUTPUT_TRUNCATION_ENABLED:-false}"
TRACE_STATUS_LONG_OUTPUT_TRUNCATION_MAX_CHARS="${TRACE_STATUS_LONG_OUTPUT_TRUNCATION_MAX_CHARS:-4000}"
TRACE_STATUS_LONG_OUTPUT_TRUNCATION_HEAD_TAIL_CHARS="${TRACE_STATUS_LONG_OUTPUT_TRUNCATION_HEAD_TAIL_CHARS:-1500}"
# Independently turn explicit oversized sed windows into their first chunk (no block/replan turn).
TRACE_STATUS_PROGRESSIVE_READ_ENABLED="${TRACE_STATUS_PROGRESSIVE_READ_ENABLED:-false}"
TRACE_STATUS_PROGRESSIVE_READ_CHUNK_LINES="${TRACE_STATUS_PROGRESSIVE_READ_CHUNK_LINES:-200}"
# 它会在较长的源码读取 observation 被下一轮思考“消费”后，将旧 observation 替换为结构化摘要，只保留新结论，避免上下文中重复保留原始内容。
TRACE_STATUS_OLD_OBSERVATION_COMPACTION_ENABLED="${TRACE_STATUS_OLD_OBSERVATION_COMPACTION_ENABLED:-false}"
TRACE_STATUS_OLD_OBSERVATION_COMPACTION_MAX_CHARS="${TRACE_STATUS_OLD_OBSERVATION_COMPACTION_MAX_CHARS:-4000}"
TRACE_STATUS_OLD_OBSERVATION_COMPACTION_CONCLUSION_CHARS="${TRACE_STATUS_OLD_OBSERVATION_COMPACTION_CONCLUSION_CHARS:-1200}"
TRACE_STATUS_POST_VERIFY_GUARD_ENABLED="${TRACE_STATUS_POST_VERIFY_GUARD_ENABLED:-false}"
# Fine-grained action budgets. When enabled, search -> edit -> test -> submit phases are
# enforced and the agent receives the remaining budget after every action.
TRACE_ACTION_BUDGET_CONTROLLER_ENABLED="${TRACE_ACTION_BUDGET_CONTROLLER_ENABLED:-false}"
TRACE_ACTION_BUDGET_SEARCH="${TRACE_ACTION_BUDGET_SEARCH:-8}"
TRACE_ACTION_BUDGET_EDIT="${TRACE_ACTION_BUDGET_EDIT:-5}"
TRACE_ACTION_BUDGET_TEST="${TRACE_ACTION_BUDGET_TEST:-3}"
# Counterfactual replay: block the first would-block action from this baseline
# trace, then continue with fresh model sampling. Empty means run normally.
REPLAY_TRACE="${REPLAY_TRACE:-}"
REPLAY_BLOCK_OCCURRENCE="${REPLAY_BLOCK_OCCURRENCE:-1}"

INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_MD="${INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_MD:-${REPO_ROOT}/skill_distillation/skills/localize-by-anchor/SKILL.md}"
USE_DYNAMIC_TURN_CONTROLLER="${USE_DYNAMIC_TURN_CONTROLLER:-false}"
TURN_CONTROL_INITIAL_LIMIT="${TURN_CONTROL_INITIAL_LIMIT:-27}"
TURN_CONTROL_EXTENSION_LIMIT="${TURN_CONTROL_EXTENSION_LIMIT:-44}"
TURN_CONTROL_REMINDER="${TURN_CONTROL_REMINDER:-false}"
MODEL_CLASS="${MODEL_CLASS:-}"
USE_WARMUP_MODEL_SWITCH="${USE_WARMUP_MODEL_SWITCH:-false}"
WARMUP_MODEL_STEPS="${WARMUP_MODEL_STEPS:-2}"
WARMUP_MODEL_NAME="${WARMUP_MODEL_NAME:-autodl/gpt-5.4-mini}"
MSWEA_MODEL_TIMEOUT_THRESHOLD="${MSWEA_MODEL_TIMEOUT_THRESHOLD:-120}"
MSWEA_COMMAND_TIMEOUT_THRESHOLD="${MSWEA_COMMAND_TIMEOUT_THRESHOLD:-0}"
MSWEA_STUCK_REPORT_INTERVAL="${MSWEA_STUCK_REPORT_INTERVAL:-60}"

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

configure_deepseek_openai_env() {
  local model_id="${1:-}"
  case "${model_id}" in
    deepseek/*)
      ;;
    *)
      return 0
      ;;
  esac

  local deepseek_key="${DEEPSEEK_API_KEY:-}"
  if [[ -z "${deepseek_key}" && -f "${DEEPSEEK_KEY_FILE}" ]]; then
    deepseek_key="$(tr -d '\r\n' < "${DEEPSEEK_KEY_FILE}")"
  fi
  if [[ -z "${deepseek_key}" ]]; then
    echo "Missing DeepSeek API key. Set DEEPSEEK_API_KEY or create ${DEEPSEEK_KEY_FILE}." >&2
    exit 2
  fi

  export DEEPSEEK_API_KEY="${deepseek_key}"
  export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-${DEEPSEEK_API_BASE_URL:-${DEEPSEEK_BASE_DEFAULT}}}"
  MODEL_CLASS="litellm"
}

configure_openrouter_env() {
  local model_id="${1:-}"
  case "${model_id}" in
    openrouter/*|openai/gpt-oss-*|gpt-oss-*)
      ;;
    *)
      return 0
      ;;
  esac

  local openrouter_key="${OPENROUTER_API_KEY:-}"
  if [[ -z "${openrouter_key}" && -f "${OPENROUTER_KEY_FILE}" ]]; then
    openrouter_key="$(tr -d '\r\n' < "${OPENROUTER_KEY_FILE}")"
  fi
  if [[ -z "${openrouter_key}" ]]; then
    echo "Missing OpenRouter API key. Set OPENROUTER_API_KEY or create ${OPENROUTER_KEY_FILE}." >&2
    exit 2
  fi

  export OPENROUTER_API_KEY="${openrouter_key}"
  export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-${OPENROUTER_BASE_DEFAULT}}"
  MODEL_CLASS="openrouter"
}

configure_vllm_env() {
  local model_id="${1:-}"
  case "${model_id}" in
    vllm/*)
      ;;
    *)
      return 0
      ;;
  esac

  # vLLM exposes an OpenAI-compatible API and does not require authentication.
  # LiteLLM/OpenAI still expects a non-empty client-side key, so use a local
  # placeholder and never inherit a real OPENAI_API_KEY for this provider.
  export OPENAI_API_KEY="${VLLM_API_KEY:-EMPTY}"
  export OPENAI_BASE_URL="${VLLM_BASE_DEFAULT}"
  export OPENAI_API_BASE="${VLLM_BASE_DEFAULT}"
  export VLLM_API_BASE="${VLLM_BASE_DEFAULT}"
  export NO_PROXY="${NO_PROXY:+${NO_PROXY},}0.0.0.0,127.0.0.1,localhost"
  export no_proxy="${no_proxy:+${no_proxy},}0.0.0.0,127.0.0.1,localhost"
  MODEL_CLASS="litellm"
}

normalize_model_id() {
  local model_id="${1:-}"
  case "${model_id}" in
    autodl/*)
      printf 'openai/%s\n' "${model_id#autodl/}"
      ;;
    vllm/*)
      printf 'openai/%s\n' "${model_id#vllm/}"
      ;;
    openrouter/*)
      printf '%s\n' "${model_id#openrouter/}"
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
  configure_deepseek_openai_env "${model_id}"
  configure_vllm_env "${model_id}"
}

is_true() {
  local value="${1:-}"
  case "${value,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ ! -x "${RUNNER_PYTHON}" ]]; then
  echo "Runner python is not executable: ${RUNNER_PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 2
fi

if is_true "${USE_DYNAMIC_TURN_CONTROLLER}" && (( TURN_CONTROL_EXTENSION_LIMIT <= TURN_CONTROL_INITIAL_LIMIT )); then
  echo "TURN_CONTROL_EXTENSION_LIMIT must be greater than TURN_CONTROL_INITIAL_LIMIT when USE_DYNAMIC_TURN_CONTROLLER=true." >&2
  echo "Got initial=${TURN_CONTROL_INITIAL_LIMIT}, extension=${TURN_CONTROL_EXTENSION_LIMIT}" >&2
  exit 2
fi

if is_true "${TRACE_REPETITION_NUDGE_ENABLED_FORCE}"; then
  if ! [[ "${TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM}" =~ ^[0-9]+$ ]] || (( TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM <= 0 )); then
    echo "TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM must be a positive integer when TRACE_REPETITION_NUDGE_ENABLED_FORCE=true." >&2
    echo "Got force_num=${TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM}" >&2
    exit 2
  fi
fi

if is_true "${TRACE_STATUS_BAR_ENABLED}"; then
  if ! [[ "${TRACE_STATUS_BAR_MAX_ITEMS}" =~ ^[0-9]+$ ]] || (( TRACE_STATUS_BAR_MAX_ITEMS <= 0 )); then
    echo "TRACE_STATUS_BAR_MAX_ITEMS must be a positive integer when TRACE_STATUS_BAR_ENABLED=true." >&2
    echo "Got TRACE_STATUS_BAR_MAX_ITEMS=${TRACE_STATUS_BAR_MAX_ITEMS}" >&2
    exit 2
  fi
  if ! [[ "${TRACE_STATUS_BAR_INTERVAL}" =~ ^[0-9]+$ ]] || (( TRACE_STATUS_BAR_INTERVAL <= 0 )); then
    echo "TRACE_STATUS_BAR_INTERVAL must be a positive integer when TRACE_STATUS_BAR_ENABLED=true." >&2
    echo "Got TRACE_STATUS_BAR_INTERVAL=${TRACE_STATUS_BAR_INTERVAL}" >&2
    exit 2
  fi
fi

if is_true "${TRACE_STATUS_REPEAT_BLOCK_ENABLED}"; then
  for budget_name in \
    TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS \
    TRACE_STATUS_REPEAT_BLOCK_MAX_TOTAL_INSPECTIONS \
    TRACE_STATUS_REPEAT_BLOCK_MAX_SCRATCH_EXPERIMENTS \
    TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS; do
    budget_value="${!budget_name}"
    if ! [[ "${budget_value}" =~ ^[0-9]+$ ]] || (( budget_value <= 0 )); then
      echo "${budget_name} must be a positive integer when TRACE_STATUS_REPEAT_BLOCK_ENABLED=true." >&2
      echo "Got ${budget_name}=${budget_value}" >&2
      exit 2
    fi
  done
fi

if is_true "${TRACE_STATUS_LARGE_READ_BLOCK_ENABLED}"; then
  if ! [[ "${TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES}" =~ ^[0-9]+$ ]] || (( TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES <= 0 )); then
    echo "TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES must be a positive integer when TRACE_STATUS_LARGE_READ_BLOCK_ENABLED=true." >&2
    echo "Got TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES=${TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES}" >&2
    exit 2
  fi
fi

for numeric_name in \
  TRACE_STATUS_LONG_OUTPUT_TRUNCATION_MAX_CHARS \
  TRACE_STATUS_LONG_OUTPUT_TRUNCATION_HEAD_TAIL_CHARS \
  TRACE_STATUS_PROGRESSIVE_READ_CHUNK_LINES \
  TRACE_STATUS_OLD_OBSERVATION_COMPACTION_MAX_CHARS \
  TRACE_STATUS_OLD_OBSERVATION_COMPACTION_CONCLUSION_CHARS; do
  numeric_value="${!numeric_name}"
  if ! [[ "${numeric_value}" =~ ^[0-9]+$ ]] || (( numeric_value <= 0 )); then
    echo "${numeric_name} must be a positive integer." >&2
    echo "Got ${numeric_name}=${numeric_value}" >&2
    exit 2
  fi
done

if is_true "${TRACE_ACTION_BUDGET_CONTROLLER_ENABLED}"; then
  for budget_name in TRACE_ACTION_BUDGET_SEARCH TRACE_ACTION_BUDGET_EDIT TRACE_ACTION_BUDGET_TEST; do
    budget_value="${!budget_name}"
    if ! [[ "${budget_value}" =~ ^[0-9]+$ ]] || (( budget_value <= 0 )); then
      echo "${budget_name} must be a positive integer when TRACE_ACTION_BUDGET_CONTROLLER_ENABLED=true." >&2
      echo "Got ${budget_name}=${budget_value}" >&2
      exit 2
    fi
  done
fi

if is_true "${USE_WARMUP_MODEL_SWITCH}"; then
  if ! [[ "${WARMUP_MODEL_STEPS}" =~ ^[0-9]+$ ]] || (( WARMUP_MODEL_STEPS <= 0 )); then
    echo "WARMUP_MODEL_STEPS must be a positive integer when USE_WARMUP_MODEL_SWITCH=true." >&2
    echo "Got WARMUP_MODEL_STEPS=${WARMUP_MODEL_STEPS}" >&2
    exit 2
  fi
fi

if ! [[ "${MODEL_MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "MODEL_MAX_TOKENS must be a non-negative integer." >&2
  echo "Got MODEL_MAX_TOKENS=${MODEL_MAX_TOKENS}" >&2
  exit 2
fi

configure_clash_proxy_env
configure_model_provider_env "${MODEL}"
configure_openrouter_env "${MODEL}"
if is_true "${USE_WARMUP_MODEL_SWITCH}"; then
  configure_model_provider_env "${WARMUP_MODEL_NAME}"
  configure_openrouter_env "${WARMUP_MODEL_NAME}"
  export MSWEA_WARMUP_MODEL_ENABLED=true
  export MSWEA_WARMUP_STEPS="${WARMUP_MODEL_STEPS}"
  export MSWEA_WARMUP_MODEL_NAME="$(normalize_model_id "${WARMUP_MODEL_NAME}")"
else
  unset MSWEA_WARMUP_MODEL_ENABLED MSWEA_WARMUP_STEPS MSWEA_WARMUP_MODEL_NAME || true
fi
MODEL="$(normalize_model_id "${MODEL}")"

export XDG_CONFIG_HOME
export HF_HOME
export MSWEA_MODEL_TIMEOUT_THRESHOLD
export MSWEA_COMMAND_TIMEOUT_THRESHOLD
export MSWEA_STUCK_REPORT_INTERVAL
export PYTHONPATH="${MINISWEAGENT_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOCAL_TMP_ROOT}"
mkdir -p "${XDG_CONFIG_HOME}"
mkdir -p "${HF_HOME}"
mkdir -p "${OUTPUT_ROOT}"

FILTER_REGEX="^${INSTANCE_ID}$"
EFFECTIVE_CONFIG_PATH="${CONFIG_PATH}"

REDO_ARGS=()
if [[ "${REDO_EXISTING}" == "true" ]]; then
  REDO_ARGS+=(--redo-existing)
fi

MODEL_ARGS=(--model "${MODEL}")
if [[ -n "${MODEL_CLASS}" ]]; then
  MODEL_ARGS+=(--model-class "${MODEL_CLASS}")
fi

if ! is_true "${USE_EXACT_CONFIG:-false}" && { is_true "${INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT}" || is_true "${INJECT_MIN_TOKEN_PLAN_FIRST_PROMPT}" || is_true "${CROSS_TASK_CAPABILITY_LEDGER_ENABLED}" || [[ "${TRACE_REPETITION_NUDGE_ENABLED}" != "" ]] || is_true "${USE_DYNAMIC_TURN_CONTROLLER}" || is_true "${SUBMISSION_REVIEW_ENABLED}" || is_true "${TRACE_STATUS_BAR_ENABLED}" || is_true "${TRACE_STATUS_REPEAT_BLOCK_ENABLED}" || is_true "${TRACE_STATUS_LARGE_READ_BLOCK_ENABLED}" || is_true "${TRACE_STATUS_LONG_OUTPUT_TRUNCATION_ENABLED}" || is_true "${TRACE_STATUS_PROGRESSIVE_READ_ENABLED}" || is_true "${TRACE_STATUS_OLD_OBSERVATION_COMPACTION_ENABLED}" || is_true "${TRACE_ACTION_BUDGET_CONTROLLER_ENABLED}" || (( MODEL_MAX_TOKENS > 0 )); }; then
  EFFECTIVE_CONFIG_PATH="$(mktemp "${LOCAL_TMP_ROOT}/eet_run_my_config_XXXXXX.yaml")"
  CONFIG_PATH_ENV="${CONFIG_PATH}" \
  EFFECTIVE_CONFIG_PATH_ENV="${EFFECTIVE_CONFIG_PATH}" \
  TRACE_REPETITION_NUDGE_ENABLED_ENV="${TRACE_REPETITION_NUDGE_ENABLED}" \
  TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED_ENV="${TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED}" \
  TRACE_REPETITION_NUDGE_ENABLED_FORCE_ENV="${TRACE_REPETITION_NUDGE_ENABLED_FORCE}" \
  TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM_ENV="${TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM}" \
  INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_ENV="${INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT}" \
  INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_MD_ENV="${INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_MD}" \
  INJECT_MIN_TOKEN_PLAN_FIRST_PROMPT_ENV="${INJECT_MIN_TOKEN_PLAN_FIRST_PROMPT}" \
  CROSS_TASK_CAPABILITY_LEDGER_ENABLED_ENV="${CROSS_TASK_CAPABILITY_LEDGER_ENABLED}" \
  CAPABILITY_LEDGER_PATH_ENV="${CAPABILITY_LEDGER_PATH}" \
  CAPABILITY_LEDGER_BENCHMARK_ENV="${CAPABILITY_LEDGER_BENCHMARK}" \
  CAPABILITY_LEDGER_ENVIRONMENT_FAMILY_ENV="${CAPABILITY_LEDGER_ENVIRONMENT_FAMILY}" \
  USE_DYNAMIC_TURN_CONTROLLER_ENV="${USE_DYNAMIC_TURN_CONTROLLER}" \
  CONFIDENCE_CHECK_ENABLED_ENV="${CONFIDENCE_CHECK_ENABLED}" \
  SUBMISSION_REVIEW_ENABLED_ENV="${SUBMISSION_REVIEW_ENABLED}" \
  TRACE_STATUS_BAR_ENABLED_ENV="${TRACE_STATUS_BAR_ENABLED}" \
  TRACE_STATUS_BAR_NEXT_ACTION_GUIDANCE_ENABLED_ENV="${TRACE_STATUS_BAR_NEXT_ACTION_GUIDANCE_ENABLED}" \
  TRACE_STATUS_REPEAT_BLOCK_ENABLED_ENV="${TRACE_STATUS_REPEAT_BLOCK_ENABLED}" \
  TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS_ENV="${TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS}" \
  TRACE_STATUS_REPEAT_BLOCK_MAX_TOTAL_INSPECTIONS_ENV="${TRACE_STATUS_REPEAT_BLOCK_MAX_TOTAL_INSPECTIONS}" \
  TRACE_STATUS_REPEAT_BLOCK_MAX_SCRATCH_EXPERIMENTS_ENV="${TRACE_STATUS_REPEAT_BLOCK_MAX_SCRATCH_EXPERIMENTS}" \
  TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS_ENV="${TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS}" \
  TRACE_STATUS_REPEAT_BLOCK_SOFT_WARNING_ENABLED_ENV="${TRACE_STATUS_REPEAT_BLOCK_SOFT_WARNING_ENABLED}" \
  TRACE_STATUS_REPEAT_BLOCK_WARN_CONSECUTIVE_INSPECTIONS_ENV="${TRACE_STATUS_REPEAT_BLOCK_WARN_CONSECUTIVE_INSPECTIONS}" \
  TRACE_STATUS_REPEAT_BLOCK_WARN_TOTAL_INSPECTIONS_ENV="${TRACE_STATUS_REPEAT_BLOCK_WARN_TOTAL_INSPECTIONS}" \
  TRACE_STATUS_REPEAT_BLOCK_WARN_SCRATCH_EXPERIMENTS_ENV="${TRACE_STATUS_REPEAT_BLOCK_WARN_SCRATCH_EXPERIMENTS}" \
  TRACE_STATUS_REPEAT_BLOCK_WARN_TEST_PHASE_TESTS_ENV="${TRACE_STATUS_REPEAT_BLOCK_WARN_TEST_PHASE_TESTS}" \
  TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED_ENV="${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED}" \
  TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS_ENV="${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS}" \
  TRACE_STATUS_LARGE_READ_BLOCK_ENABLED_ENV="${TRACE_STATUS_LARGE_READ_BLOCK_ENABLED}" \
  TRACE_STATUS_LONG_OUTPUT_TRUNCATION_ENABLED_ENV="${TRACE_STATUS_LONG_OUTPUT_TRUNCATION_ENABLED}" \
  TRACE_STATUS_PROGRESSIVE_READ_ENABLED_ENV="${TRACE_STATUS_PROGRESSIVE_READ_ENABLED}" \
  TRACE_STATUS_OLD_OBSERVATION_COMPACTION_ENABLED_ENV="${TRACE_STATUS_OLD_OBSERVATION_COMPACTION_ENABLED}" \
  TRACE_STATUS_POST_VERIFY_GUARD_ENABLED_ENV="${TRACE_STATUS_POST_VERIFY_GUARD_ENABLED}" \
  TRACE_ACTION_BUDGET_CONTROLLER_ENABLED_ENV="${TRACE_ACTION_BUDGET_CONTROLLER_ENABLED}" \
  MODEL_MAX_TOKENS_ENV="${MODEL_MAX_TOKENS}" \
  TRACE_STATUS_BAR_MAX_ITEMS_ENV="${TRACE_STATUS_BAR_MAX_ITEMS}" \
  TRACE_STATUS_BAR_INTERVAL_ENV="${TRACE_STATUS_BAR_INTERVAL}" \
  TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES_ENV="${TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES}" \
  TRACE_STATUS_LONG_OUTPUT_TRUNCATION_MAX_CHARS_ENV="${TRACE_STATUS_LONG_OUTPUT_TRUNCATION_MAX_CHARS}" \
  TRACE_STATUS_LONG_OUTPUT_TRUNCATION_HEAD_TAIL_CHARS_ENV="${TRACE_STATUS_LONG_OUTPUT_TRUNCATION_HEAD_TAIL_CHARS}" \
  TRACE_STATUS_PROGRESSIVE_READ_CHUNK_LINES_ENV="${TRACE_STATUS_PROGRESSIVE_READ_CHUNK_LINES}" \
  TRACE_STATUS_OLD_OBSERVATION_COMPACTION_MAX_CHARS_ENV="${TRACE_STATUS_OLD_OBSERVATION_COMPACTION_MAX_CHARS}" \
  TRACE_STATUS_OLD_OBSERVATION_COMPACTION_CONCLUSION_CHARS_ENV="${TRACE_STATUS_OLD_OBSERVATION_COMPACTION_CONCLUSION_CHARS}" \
  TRACE_ACTION_BUDGET_SEARCH_ENV="${TRACE_ACTION_BUDGET_SEARCH}" \
  TRACE_ACTION_BUDGET_EDIT_ENV="${TRACE_ACTION_BUDGET_EDIT}" \
  TRACE_ACTION_BUDGET_TEST_ENV="${TRACE_ACTION_BUDGET_TEST}" \
  TURN_CONTROL_INITIAL_LIMIT_ENV="${TURN_CONTROL_INITIAL_LIMIT}" \
  TURN_CONTROL_EXTENSION_LIMIT_ENV="${TURN_CONTROL_EXTENSION_LIMIT}" \
  TURN_CONTROL_REMINDER_ENV="${TURN_CONTROL_REMINDER}" \
  "${RUNNER_PYTHON}" - <<'PY'
import pathlib
import json
import os
import yaml

config_path = pathlib.Path(os.environ["CONFIG_PATH_ENV"])
output_path = pathlib.Path(os.environ["EFFECTIVE_CONFIG_PATH_ENV"])
trace_enabled = os.environ["TRACE_REPETITION_NUDGE_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
read_loop_trace_enabled = os.environ["TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_force_enabled = os.environ["TRACE_REPETITION_NUDGE_ENABLED_FORCE_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_force_num = int(os.environ["TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM_ENV"])
inject_enabled = os.environ["INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_ENV"].strip().lower() in {"1", "true", "yes", "on"}
inject_md = pathlib.Path(os.environ["INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT_MD_ENV"])
min_token_plan_enabled = os.environ["INJECT_MIN_TOKEN_PLAN_FIRST_PROMPT_ENV"].strip().lower() in {"1", "true", "yes", "on"}
ledger_enabled = os.environ["CROSS_TASK_CAPABILITY_LEDGER_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
ledger_path = pathlib.Path(os.environ["CAPABILITY_LEDGER_PATH_ENV"])
ledger_scope = {"benchmark": os.environ["CAPABILITY_LEDGER_BENCHMARK_ENV"], "environment_family": os.environ["CAPABILITY_LEDGER_ENVIRONMENT_FAMILY_ENV"]}
turn_control_enabled = os.environ["USE_DYNAMIC_TURN_CONTROLLER_ENV"].strip().lower() in {"1", "true", "yes", "on"}
confidence_check_enabled = os.environ["CONFIDENCE_CHECK_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
submission_review_enabled = os.environ["SUBMISSION_REVIEW_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_status_bar_enabled = os.environ["TRACE_STATUS_BAR_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_status_bar_next_action_guidance_enabled = os.environ["TRACE_STATUS_BAR_NEXT_ACTION_GUIDANCE_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_status_repeat_block_enabled = os.environ["TRACE_STATUS_REPEAT_BLOCK_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_status_large_read_block_enabled = os.environ["TRACE_STATUS_LARGE_READ_BLOCK_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_status_long_output_truncation_enabled = os.environ["TRACE_STATUS_LONG_OUTPUT_TRUNCATION_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_status_progressive_read_enabled = os.environ["TRACE_STATUS_PROGRESSIVE_READ_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_status_old_observation_compaction_enabled = os.environ["TRACE_STATUS_OLD_OBSERVATION_COMPACTION_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_status_post_verify_guard_enabled = os.environ["TRACE_STATUS_POST_VERIFY_GUARD_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
trace_action_budget_controller_enabled = os.environ["TRACE_ACTION_BUDGET_CONTROLLER_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
model_max_tokens = int(os.environ["MODEL_MAX_TOKENS_ENV"])
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
agent = config.setdefault("agent", {})
agent["trace_repetition_nudge_enabled"] = trace_enabled
agent["trace_repetition_read_loop_nudge_enabled"] = read_loop_trace_enabled
agent["trace_repetition_nudge_force_enabled"] = trace_force_enabled
agent["trace_repetition_nudge_force_num"] = trace_force_num
agent["confidence_check_enabled"] = confidence_check_enabled
agent["submission_review_enabled"] = submission_review_enabled
agent["trace_status_bar_enabled"] = trace_status_bar_enabled
agent["trace_status_bar_max_items"] = int(os.environ["TRACE_STATUS_BAR_MAX_ITEMS_ENV"])
agent["trace_status_bar_interval"] = int(os.environ["TRACE_STATUS_BAR_INTERVAL_ENV"])
agent["trace_status_bar_next_action_guidance_enabled"] = trace_status_bar_next_action_guidance_enabled
agent["trace_status_repeat_block_enabled"] = trace_status_repeat_block_enabled
agent["trace_status_repeat_block_max_consecutive_inspections"] = int(os.environ["TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS_ENV"])
agent["trace_status_repeat_block_max_total_inspections_since_edit"] = int(os.environ["TRACE_STATUS_REPEAT_BLOCK_MAX_TOTAL_INSPECTIONS_ENV"])
agent["trace_status_repeat_block_max_scratch_experiments_since_edit"] = int(os.environ["TRACE_STATUS_REPEAT_BLOCK_MAX_SCRATCH_EXPERIMENTS_ENV"])
agent["trace_status_repeat_block_max_test_phase_tests"] = int(os.environ["TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS_ENV"])
agent["trace_status_repeat_block_soft_warning_enabled"] = os.environ["TRACE_STATUS_REPEAT_BLOCK_SOFT_WARNING_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
agent["trace_status_repeat_block_warn_consecutive_inspections"] = int(os.environ["TRACE_STATUS_REPEAT_BLOCK_WARN_CONSECUTIVE_INSPECTIONS_ENV"])
agent["trace_status_repeat_block_warn_total_inspections"] = int(os.environ["TRACE_STATUS_REPEAT_BLOCK_WARN_TOTAL_INSPECTIONS_ENV"])
agent["trace_status_repeat_block_warn_scratch_experiments"] = int(os.environ["TRACE_STATUS_REPEAT_BLOCK_WARN_SCRATCH_EXPERIMENTS_ENV"])
agent["trace_status_repeat_block_warn_test_phase_tests"] = int(os.environ["TRACE_STATUS_REPEAT_BLOCK_WARN_TEST_PHASE_TESTS_ENV"])
agent["trace_status_repeat_block_dynamic_inspection_enabled"] = os.environ["TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED_ENV"].strip().lower() in {"1", "true", "yes", "on"}
agent["trace_status_repeat_block_dynamic_inspection_max_bonus"] = int(os.environ["TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS_ENV"])
agent["trace_status_large_read_block_enabled"] = trace_status_large_read_block_enabled
agent["trace_status_large_read_block_max_lines"] = int(os.environ["TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES_ENV"])
agent["trace_status_long_output_truncation_enabled"] = trace_status_long_output_truncation_enabled
agent["trace_status_long_output_truncation_max_chars"] = int(os.environ["TRACE_STATUS_LONG_OUTPUT_TRUNCATION_MAX_CHARS_ENV"])
agent["trace_status_long_output_truncation_head_tail_chars"] = int(os.environ["TRACE_STATUS_LONG_OUTPUT_TRUNCATION_HEAD_TAIL_CHARS_ENV"])
agent["trace_status_progressive_read_enabled"] = trace_status_progressive_read_enabled
agent["trace_status_progressive_read_chunk_lines"] = int(os.environ["TRACE_STATUS_PROGRESSIVE_READ_CHUNK_LINES_ENV"])
agent["trace_status_old_observation_compaction_enabled"] = trace_status_old_observation_compaction_enabled
agent["trace_status_old_observation_compaction_max_chars"] = int(os.environ["TRACE_STATUS_OLD_OBSERVATION_COMPACTION_MAX_CHARS_ENV"])
agent["trace_status_old_observation_compaction_conclusion_chars"] = int(os.environ["TRACE_STATUS_OLD_OBSERVATION_COMPACTION_CONCLUSION_CHARS_ENV"])
agent["trace_status_post_verify_guard_enabled"] = trace_status_post_verify_guard_enabled
agent["trace_action_budget_controller_enabled"] = trace_action_budget_controller_enabled
agent["trace_action_budget_search"] = int(os.environ["TRACE_ACTION_BUDGET_SEARCH_ENV"])
agent["trace_action_budget_edit"] = int(os.environ["TRACE_ACTION_BUDGET_EDIT_ENV"])
agent["trace_action_budget_test"] = int(os.environ["TRACE_ACTION_BUDGET_TEST_ENV"])
if model_max_tokens:
    model = config.setdefault("model", {})
    model_kwargs = model.setdefault("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise TypeError("model.model_kwargs must be a mapping")
    model_kwargs["max_tokens"] = model_max_tokens

if turn_control_enabled:
    agent["turn_control"] = True
    agent["turn_control_mode"] = "dynamic"
    agent["turn_control_initial_limit"] = int(os.environ["TURN_CONTROL_INITIAL_LIMIT_ENV"])
    agent["turn_control_extension_limit"] = int(os.environ["TURN_CONTROL_EXTENSION_LIMIT_ENV"])
    agent["turn_control_reminder"] = os.environ["TURN_CONTROL_REMINDER_ENV"].strip().lower() in {"1", "true", "yes", "on"}
    agent["confidence_check_enabled"] = False
    agent["turn_control_fixed_limit"] = 0

if inject_enabled:
    if not inject_md.is_file():
        raise FileNotFoundError(f"Injected markdown not found: {inject_md}")
    guidance = inject_md.read_text(encoding="utf-8").strip()
    template = str(agent.get("instance_template") or "")
    section = "\n\n    ## Token Efficient Triage First Prompt\n    " + guidance.replace("\n", "\n    ")
    marker = "\n    ## Command Execution Rules"
    if marker in template:
        before, after = template.split(marker, 1)
        agent["instance_template"] = before + section + marker + after
    else:
        agent["instance_template"] = template + section

if min_token_plan_enabled:
    template = str(agent.get("instance_template") or "")
    section = "\n\n    First, make a plan for this task using as few tokens as possible, then execute according to that plan."
    marker = "\n    ## Command Execution Rules"
    if marker in template:
        before, after = template.split(marker, 1)
        agent["instance_template"] = before + section + marker + after
    else:
        agent["instance_template"] = template + section

if ledger_enabled and ledger_path.is_file():
    records = json.loads(ledger_path.read_text(encoding="utf-8"))
    hints = [record for record in records if record.get("scope") == ledger_scope and record.get("status") == "unavailable"]
    if hints:
        lines = ["## Cross-task runtime experience", ""]
        for record in hints:
            command = record["command"]
            fallback = (record.get("recommended_fallback") or {}).get("command")
            lines.append(f"In previous matching environments, `{command}` was unavailable.")
            if fallback:
                lines.append(f"A previously successful fallback was: `{fallback}`")
        lines.extend(["", "This is an environment hint, not a prohibition: adapt if command output contradicts it."])
        template = str(agent.get("instance_template") or "")
        section = "\n\n    " + "\n    ".join(lines)
        marker = "\n    ## Command Execution Rules"
        agent["instance_template"] = template.replace(marker, section + marker, 1) if marker in template else template + section

output_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
PY
  trap 'rm -f "${EFFECTIVE_CONFIG_PATH}"' EXIT
fi

echo "Running EET_org mini-swe-agent on ${INSTANCE_ID}"
echo "model=${MODEL}"
echo "model_max_tokens=${MODEL_MAX_TOKENS}"
echo "model_class=${MODEL_CLASS}"
if [[ "${MODEL_CLASS}" == "litellm" && "${OPENAI_BASE_URL:-}" == "${VLLM_BASE_DEFAULT}" ]]; then
  echo "vllm_base_url=${VLLM_BASE_DEFAULT}"
fi
echo "output_root=${OUTPUT_ROOT}"
echo "cross_task_capability_ledger_enabled=${CROSS_TASK_CAPABILITY_LEDGER_ENABLED}"
echo "capability_ledger_path=${CAPABILITY_LEDGER_PATH}"
echo "inject_token_efficient_triage_first_prompt=${INJECT_TOKEN_EFFICIENT_TRIAGE_FIRST_PROMPT}"
echo "inject_min_token_plan_first_prompt=${INJECT_MIN_TOKEN_PLAN_FIRST_PROMPT}"
echo "trace_repetition_nudge_enabled=${TRACE_REPETITION_NUDGE_ENABLED}"
echo "trace_repetition_read_loop_nudge_enabled=${TRACE_REPETITION_READ_LOOP_NUDGE_ENABLED}"
echo "trace_repetition_nudge_enabled_force=${TRACE_REPETITION_NUDGE_ENABLED_FORCE}"
echo "trace_repetition_nudge_enabled_force_num=${TRACE_REPETITION_NUDGE_ENABLED_FORCE_NUM}"
echo "confidence_check_enabled=${CONFIDENCE_CHECK_ENABLED}"
echo "submission_review_enabled=${SUBMISSION_REVIEW_ENABLED}"
echo "trace_status_bar_enabled=${TRACE_STATUS_BAR_ENABLED}"
echo "trace_status_bar_max_items=${TRACE_STATUS_BAR_MAX_ITEMS}"
echo "trace_status_bar_interval=${TRACE_STATUS_BAR_INTERVAL}"
echo "trace_status_bar_next_action_guidance_enabled=${TRACE_STATUS_BAR_NEXT_ACTION_GUIDANCE_ENABLED}"
echo "trace_status_repeat_block_enabled=${TRACE_STATUS_REPEAT_BLOCK_ENABLED}"
echo "trace_status_repeat_block_max_consecutive_inspections=${TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS}"
echo "trace_status_repeat_block_max_total_inspections=${TRACE_STATUS_REPEAT_BLOCK_MAX_TOTAL_INSPECTIONS}"
echo "trace_status_repeat_block_max_scratch_experiments=${TRACE_STATUS_REPEAT_BLOCK_MAX_SCRATCH_EXPERIMENTS}"
echo "trace_status_repeat_block_max_test_phase_tests=${TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS}"
echo "trace_status_repeat_block_soft_warning_enabled=${TRACE_STATUS_REPEAT_BLOCK_SOFT_WARNING_ENABLED}"
echo "trace_status_repeat_block_warn_consecutive_inspections=${TRACE_STATUS_REPEAT_BLOCK_WARN_CONSECUTIVE_INSPECTIONS}"
echo "trace_status_repeat_block_warn_total_inspections=${TRACE_STATUS_REPEAT_BLOCK_WARN_TOTAL_INSPECTIONS}"
echo "trace_status_repeat_block_warn_scratch_experiments=${TRACE_STATUS_REPEAT_BLOCK_WARN_SCRATCH_EXPERIMENTS}"
echo "trace_status_repeat_block_warn_test_phase_tests=${TRACE_STATUS_REPEAT_BLOCK_WARN_TEST_PHASE_TESTS}"
echo "trace_status_repeat_block_dynamic_inspection_enabled=${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_ENABLED}"
echo "trace_status_repeat_block_dynamic_inspection_max_bonus=${TRACE_STATUS_REPEAT_BLOCK_DYNAMIC_INSPECTION_MAX_BONUS}"
echo "trace_status_large_read_block_enabled=${TRACE_STATUS_LARGE_READ_BLOCK_ENABLED}"
echo "trace_status_large_read_block_max_lines=${TRACE_STATUS_LARGE_READ_BLOCK_MAX_LINES}"
echo "trace_status_long_output_truncation_enabled=${TRACE_STATUS_LONG_OUTPUT_TRUNCATION_ENABLED}"
echo "trace_status_long_output_truncation_max_chars=${TRACE_STATUS_LONG_OUTPUT_TRUNCATION_MAX_CHARS}"
echo "trace_status_long_output_truncation_head_tail_chars=${TRACE_STATUS_LONG_OUTPUT_TRUNCATION_HEAD_TAIL_CHARS}"
echo "trace_status_progressive_read_enabled=${TRACE_STATUS_PROGRESSIVE_READ_ENABLED}"
echo "trace_status_progressive_read_chunk_lines=${TRACE_STATUS_PROGRESSIVE_READ_CHUNK_LINES}"
echo "trace_status_old_observation_compaction_enabled=${TRACE_STATUS_OLD_OBSERVATION_COMPACTION_ENABLED}"
echo "trace_status_old_observation_compaction_max_chars=${TRACE_STATUS_OLD_OBSERVATION_COMPACTION_MAX_CHARS}"
echo "trace_status_old_observation_compaction_conclusion_chars=${TRACE_STATUS_OLD_OBSERVATION_COMPACTION_CONCLUSION_CHARS}"
echo "trace_status_post_verify_guard_enabled=${TRACE_STATUS_POST_VERIFY_GUARD_ENABLED}"
echo "trace_action_budget_controller_enabled=${TRACE_ACTION_BUDGET_CONTROLLER_ENABLED}"
echo "replay_trace=${REPLAY_TRACE}"
echo "replay_block_occurrence=${REPLAY_BLOCK_OCCURRENCE}"
if is_true "${TRACE_ACTION_BUDGET_CONTROLLER_ENABLED}"; then
  echo "trace_action_budget_search=${TRACE_ACTION_BUDGET_SEARCH}"
  echo "trace_action_budget_edit=${TRACE_ACTION_BUDGET_EDIT}"
  echo "trace_action_budget_test=${TRACE_ACTION_BUDGET_TEST}"
fi
echo "use_dynamic_turn_controller=${USE_DYNAMIC_TURN_CONTROLLER}"
echo "use_warmup_model_switch=${USE_WARMUP_MODEL_SWITCH}"
echo "model_stuck_report_threshold=${MSWEA_MODEL_TIMEOUT_THRESHOLD}s"
echo "command_stuck_report_threshold=${MSWEA_COMMAND_TIMEOUT_THRESHOLD}s"
echo "stuck_report_interval=${MSWEA_STUCK_REPORT_INTERVAL}s"
if is_true "${USE_WARMUP_MODEL_SWITCH}"; then
  echo "warmup_model_name=${WARMUP_MODEL_NAME}"
  echo "warmup_model_steps=${WARMUP_MODEL_STEPS}"
fi
if is_true "${USE_DYNAMIC_TURN_CONTROLLER}"; then
  echo "turn_control_initial_limit=${TURN_CONTROL_INITIAL_LIMIT}"
  echo "turn_control_extension_limit=${TURN_CONTROL_EXTENSION_LIMIT}"
  echo "turn_control_reminder=${TURN_CONTROL_REMINDER}"
fi

if [[ -n "${REPLAY_TRACE}" ]]; then
  if [[ ! -f "${REPLAY_TRACE}" ]]; then
    echo "REPLAY_TRACE does not exist: ${REPLAY_TRACE}" >&2
    exit 2
  fi
  if ! [[ "${REPLAY_BLOCK_OCCURRENCE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPLAY_BLOCK_OCCURRENCE must be a positive integer." >&2
    exit 2
  fi
  "${RUNNER_PYTHON}" -m minisweagent.run.extra.swebench_replay_from_block \
    --trace "${REPLAY_TRACE}" \
    --block-occurrence "${REPLAY_BLOCK_OCCURRENCE}" \
    --subset verified \
    --split "${DATASET_SPLIT}" \
    --filter "${FILTER_REGEX}" \
    --output "${OUTPUT_ROOT}" \
    --config "${EFFECTIVE_CONFIG_PATH}" \
    "${MODEL_ARGS[@]}"
else
  "${RUNNER_PYTHON}" -m minisweagent.run.extra.swebench \
    --subset verified \
    --split "${DATASET_SPLIT}" \
    --filter "${FILTER_REGEX}" \
    --workers 1 \
    --output "${OUTPUT_ROOT}" \
    --config "${EFFECTIVE_CONFIG_PATH}" \
    "${MODEL_ARGS[@]}" \
    "${REDO_ARGS[@]}"
fi

PREDICTIONS_PATH="${OUTPUT_ROOT}/preds.json"
TASK_SUMMARY_PATH="${OUTPUT_ROOT}/task_summary.json"
if [[ ! -f "${PREDICTIONS_PATH}" ]]; then
  echo "preds.json not found after agent run: ${PREDICTIONS_PATH}" >&2
  exit 3
fi

TRAJ_PATH="${OUTPUT_ROOT}/${INSTANCE_ID}/${INSTANCE_ID}.traj.json"
if is_true "${CROSS_TASK_CAPABILITY_LEDGER_ENABLED}" && [[ -f "${TRAJ_PATH}" ]]; then
  CAPABILITY_LEDGER_PATH_ENV="${CAPABILITY_LEDGER_PATH}" \
  CAPABILITY_LEDGER_BENCHMARK_ENV="${CAPABILITY_LEDGER_BENCHMARK}" \
  CAPABILITY_LEDGER_ENVIRONMENT_FAMILY_ENV="${CAPABILITY_LEDGER_ENVIRONMENT_FAMILY}" \
  INSTANCE_ID_ENV="${INSTANCE_ID}" \
  TRAJ_PATH_ENV="${TRAJ_PATH}" \
  "${RUNNER_PYTHON}" - <<'PY'
import json
import os
import re
from pathlib import Path

ledger_path = Path(os.environ["CAPABILITY_LEDGER_PATH_ENV"])
scope = {"benchmark": os.environ["CAPABILITY_LEDGER_BENCHMARK_ENV"], "environment_family": os.environ["CAPABILITY_LEDGER_ENVIRONMENT_FAMILY_ENV"]}
task = os.environ["INSTANCE_ID_ENV"]
messages = json.loads(Path(os.environ["TRAJ_PATH_ENV"]).read_text(encoding="utf-8")).get("messages", [])
records = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else []

for index, message in enumerate(messages[:-1]):
    if message.get("role") != "assistant" or messages[index + 1].get("role") != "user":
        continue
    command_match = re.findall(r"```(?:bash)?\s*\n(.*?)```", str(message.get("content") or ""), re.S)
    if not command_match:
        continue
    command = command_match[-1].strip()
    observation = str(messages[index + 1].get("content") or "")
    returncode_match = re.search(r"<returncode>\s*(\d+)\s*</returncode>", observation)
    if not returncode_match:
        continue
    returncode = int(returncode_match.group(1))
    missing = re.search(r"(?:^|\n).*?([A-Za-z0-9_.+-]+): command not found", observation)
    if returncode in {126, 127} and missing:
        name = missing.group(1)
        record = next((item for item in records if item.get("scope") == scope and item.get("command") == name), None)
        if record is None:
            record = {"scope": scope, "capability": "command", "command": name, "status": "unavailable", "evidence_type": "command_not_found", "returncode": returncode, "observed_count": 0, "first_seen_task": task}
            records.append(record)
        record["observed_count"] += 1
    elif returncode == 0 and command.startswith(("grep ", "find ")):
        for record in records:
            if record.get("scope") == scope and record.get("command") == "rg":
                fallback = record.setdefault("recommended_fallback", {"command": command, "success_count": 0})
                fallback["success_count"] += 1

ledger_path.parent.mkdir(parents=True, exist_ok=True)
tmp = ledger_path.with_name(f".{ledger_path.name}.tmp")
tmp.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
tmp.replace(ledger_path)
PY
fi

EVAL_RUN_ID="$(sanitize_name "${RUN_ID}")__verify"
MODEL_LOG_NAME="$(model_log_name "${MODEL}")"
REPORT_PATH="${OUTPUT_ROOT}/logs/run_evaluation/${EVAL_RUN_ID}/${MODEL_LOG_NAME}/${INSTANCE_ID}/report.json"
RUN_INSTANCE_LOG_PATH="${OUTPUT_ROOT}/logs/run_evaluation/${EVAL_RUN_ID}/${MODEL_LOG_NAME}/${INSTANCE_ID}/run_instance.log"
TEST_OUTPUT_PATH="${OUTPUT_ROOT}/logs/run_evaluation/${EVAL_RUN_ID}/${MODEL_LOG_NAME}/${INSTANCE_ID}/test_output.txt"
VERIFY_SUMMARY_PATH="$("${RUNNER_PYTHON}" - <<PY
from pathlib import Path

output_root = Path(${OUTPUT_ROOT@Q})
eval_run_id = ${EVAL_RUN_ID@Q}
candidates = sorted(output_root.glob(f"*{eval_run_id}.json"))
print(candidates[0] if candidates else "")
PY
)"

echo "Running SWE-bench verifier for ${INSTANCE_ID}"
(
  cd "${OUTPUT_ROOT}"
  "${RUNNER_PYTHON}" -m swebench.harness.run_evaluation \
    --dataset_name "${DATASET_NAME}" \
    --predictions_path "${PREDICTIONS_PATH}" \
    --max_workers 1 \
    --run_id "${EVAL_RUN_ID}"
)

# The verifier creates its summary during the command above.  Resolve it only
# afterwards; doing this before evaluation makes successful root-level summary
# files invisible to the remainder of this script.
VERIFY_SUMMARY_PATH="$("${RUNNER_PYTHON}" - <<PY
from pathlib import Path

output_root = Path(${OUTPUT_ROOT@Q})
eval_run_id = ${EVAL_RUN_ID@Q}
candidates = sorted(output_root.glob(f"*{eval_run_id}.json"))
print(candidates[0] if candidates else "")
PY
)"

if [[ ! -f "${REPORT_PATH}" && ! -f "${VERIFY_SUMMARY_PATH}" ]]; then
  echo "Verifier outputs not found: report=${REPORT_PATH} verify_summary=${VERIFY_SUMMARY_PATH}" >&2
  exit 4
fi

"${RUNNER_PYTHON}" - <<PY
import json
from pathlib import Path

from minisweagent.pricing import sum_traj_local_pricing_cost

instance_id = ${INSTANCE_ID@Q}
report_path = Path(${REPORT_PATH@Q})
verify_summary_raw = ${VERIFY_SUMMARY_PATH@Q}
verify_summary_path = Path(verify_summary_raw) if verify_summary_raw else None
traj_path = Path(${TRAJ_PATH@Q})
predictions_path = Path(${PREDICTIONS_PATH@Q})


def cached_from_usage(usage):
    usage = usage or {}
    return max(
        int(usage.get("cached_tokens", 0) or 0),
        int(usage.get("cache_tokens", 0) or 0),
        int(usage.get("cache_read_input_tokens", 0) or 0),
        int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0),
        int(((usage.get("input_tokens_details") or {}).get("cached_tokens", 0)) or 0),
    )


def io_from_usage(usage):
    usage = usage or {}
    cached = cached_from_usage(usage)
    if "input_tokens" in usage:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
    else:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        input_tokens = max(0, prompt_tokens - cached)
    if "output_tokens" in usage:
        output_tokens = int(usage.get("output_tokens", 0) or 0)
    else:
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
    return cached, input_tokens, output_tokens


summary_source = None
resolved = False
submitted = False
completed = False
empty_patch = False
error = False
unresolved = False
cached_tokens = 0
input_tokens = 0
output_tokens = 0
prompt_tokens = 0
completion_tokens = 0
total_tokens = 0
api_calls = 0
cost_usd = None
local_pricing_cost_usd = None
agent_exit_status = None
experience_retrieved = False
confidence_scores = []

if traj_path.is_file():
    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    info = traj.get("info") or {}
    model_stats = info.get("model_stats") or {}
    messages = traj.get("messages") or []
    api_calls = int((model_stats.get("api_calls", 0)) or 0)
    cost_usd = model_stats.get("cost_usd", model_stats.get("instance_cost"))
    local_pricing_cost_usd = sum_traj_local_pricing_cost(traj, default_model_name=${MODEL@Q})
    if cost_usd in (None, "", 0) and local_pricing_cost_usd is not None:
        cost_usd = local_pricing_cost_usd
    agent_exit_status = info.get("exit_status")
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "")
        if "## Experience" in content and "<retrieved_experiences>" in content:
            experience_retrieved = True
        extra = msg.get("extra") or {}
        response = extra.get("response") or {}
        usage = response.get("usage") or {}
        c, i, o = io_from_usage(usage)
        cached_tokens += c
        input_tokens += i
        output_tokens += o
        prompt_tokens += i + c
        completion_tokens += o
    import re
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "")
        for m in re.finditer(r'\{[^{}]*"confidence_score"\s*:\s*(\d{1,3})[^{}]*\}', content):
            try:
                score = int(m.group(1))
            except Exception:
                continue
            if 0 <= score <= 100:
                confidence_scores.append(score)
    total_tokens = input_tokens + cached_tokens + output_tokens

if report_path.is_file():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved = bool(report[instance_id]["resolved"])
    submitted = True
    completed = True
    summary_source = "report.json"
elif verify_summary_path and verify_summary_path.is_file():
    payload = json.loads(verify_summary_path.read_text(encoding="utf-8"))

    def ids(key):
        return set(payload.get(key, []))

    submitted = instance_id in ids("submitted_ids")
    completed = instance_id in ids("completed_ids")
    resolved = instance_id in ids("resolved_ids")
    unresolved = instance_id in ids("unresolved_ids")
    empty_patch = instance_id in ids("empty_patch_ids")
    error = instance_id in ids("error_ids")
    summary_source = "verify_summary"
else:
    raise FileNotFoundError(
        f"Missing verifier outputs: report={report_path} verify_summary={verify_summary_path}"
    )

task_summary = {
    "instance_id": instance_id,
    "success_within_budget": resolved,
    "resolved": resolved,
    "submitted": submitted,
    "completed": completed,
    "unresolved": unresolved,
    "empty_patch": empty_patch,
    "error": error,
    "cached_tokens": cached_tokens,
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "total_tokens": total_tokens,
    "cost_usd": cost_usd,
    "total_cost_usd": cost_usd,
    "local_pricing_cost_usd": local_pricing_cost_usd,
    "api_calls": api_calls,
    "agent_exit_status": agent_exit_status,
    "experience_retrieved": experience_retrieved,
    "confidence_scores": confidence_scores,
    "last_confidence_score": confidence_scores[-1] if confidence_scores else None,
    "run_id": ${RUN_ID@Q},
    "output_root": ${OUTPUT_ROOT@Q},
    "model": ${MODEL@Q},
    "model_class": ${MODEL_CLASS@Q},
    "predictions_path": str(predictions_path),
    "traj_path": str(traj_path),
    "report_path": str(report_path),
    "verify_summary_path": str(verify_summary_path) if verify_summary_path else None,
    "verification_source": summary_source,
    "run_instance_log_path": ${RUN_INSTANCE_LOG_PATH@Q},
    "test_output_path": ${TEST_OUTPUT_PATH@Q},
}

task_summary_path = Path(${TASK_SUMMARY_PATH@Q})
task_summary_path.write_text(json.dumps(task_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"run_id={${RUN_ID@Q}}")
print(f"output_root={Path(${OUTPUT_ROOT@Q})}")
print(f"instance_id={instance_id}")
print(f"success={str(bool(resolved)).lower()}")
print(f"resolved={str(resolved).lower()}")
print(f"verification_source={summary_source}")
print(f"cached_tokens={cached_tokens}")
print(f"input_tokens={input_tokens}")
print(f"output_tokens={output_tokens}")
print(f"total_tokens={total_tokens}")
print(f"api_calls={api_calls}")
print(f"cost_usd={cost_usd}")
print(f"total_cost_usd={cost_usd}")
print(f"predictions_path={predictions_path}")
print(f"traj_path={traj_path}")
print(f"report_path={report_path}")
print(f"run_instance_log_path={Path(${RUN_INSTANCE_LOG_PATH@Q})}")
print(f"test_output_path={Path(${TEST_OUTPUT_PATH@Q})}")
print(f"task_summary={task_summary_path}")
PY
