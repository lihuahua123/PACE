from __future__ import annotations

from typing import Any

LOCAL_PRICING: dict[str, dict[str, Any]] = {
    "gpt-5.3-codex": {"input": 5.513, "cache": 0.554, "output": 44.100},
    "MiniMax-M2.5": {"input": 1.440, "cache": 0.288, "output": 5.760},
    "MiniMax-M2.7": {"input": 1.680, "cache": 0.336, "output": 6.720},
    "gpt-5.4-mini": {"input": 2.363, "cache": 0.239, "output": 14.175},
    "DeepSeek-V3.2": {"input": 1.200, "cache": 0.240, "output": 1.800},
    "DeepSeek-V4-Flash": {"input": 2.100, "cache": 0.070, "output": 6.300},
    "DeepSeek-V4-Pro": {"input": 3.000, "cache": 0.300, "output": 6.000},
    "Kimi-K2.5": {"input": 2.400, "cache": 0.480, "output": 12.600},
    "GLM-5.1": {
        "tiers": [
            {"max_input_tokens": 32_000, "input": 4.800, "cache": 0.960, "output": 19.200},
            {"max_input_tokens": None, "input": 6.400, "cache": 1.280, "output": 22.400},
        ]
    },
    "qwen3.6-plus": {
        "tiers": [
            {"max_input_tokens": 256_000, "input": 1.600, "cache": 0.320, "output": 9.600},
            {"max_input_tokens": None, "input": 6.400, "cache": 1.280, "output": 38.400},
        ]
    },
    "Qwen3.5-397B-A17B": {
        "tiers": [
            {"max_input_tokens": 128_000, "input": 0.720, "cache": 0.144, "output": 4.320},
            {"max_input_tokens": None, "input": 1.800, "cache": 0.360, "output": 10.800},
        ]
    },
    "gpt-5.4": {
        "tiers": [
            {"max_input_tokens": 272_000, "input": 10.150, "cache": 1.015, "output": 60.900},
            {"max_input_tokens": None, "input": 20.300, "cache": 2.030, "output": 91.350},
        ]
    },
    "gpt-5.6-luna": {
        "tiers": [
            {
                "max_input_tokens": 272_000,
                "input": 0.812,
                "cache": 0.081,
                "cache_write": 1.015,
                "output": 4.872,
            },
            {
                "max_input_tokens": None,
                "input": 1.624,
                "cache": 0.162,
                "cache_write": 2.030,
                "output": 7.308,
            },
        ]
    },
    "claude-sonnet-5": {
        "input": 8.400,
        "cache": 0.840,
        "output": 42.000,
        "cache_write": 10.500,
        "cache_write_5m": 10.500,
        "cache_write_1h": 16.800,
    },
}

LOCAL_MODEL_ALIASES = {
    "gpt-5.3-codex": "gpt-5.3-codex",
    "openai/gpt-5.3-codex": "gpt-5.3-codex",
    "anthropic/minimax-m2.5": "MiniMax-M2.5",
    "minimax-m2.5": "MiniMax-M2.5",
    "anthropic/minimax-m2.7": "MiniMax-M2.7",
    "minimax-m2.7": "MiniMax-M2.7",
    "autodl/gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "openai/deepseek-v3.2": "DeepSeek-V3.2",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "autodl/deepseek-v4-flash": "DeepSeek-V4-Flash",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "autodl/deepseek-v4-pro": "DeepSeek-V4-Pro",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "kimi-k2.5": "Kimi-K2.5",
    "glm-5.1": "GLM-5.1",
    "openai/qwen3.6-plus": "qwen3.6-plus",
    "qwen3.6-plus": "qwen3.6-plus",
    "openai/qwen3.5-397b-a17b": "Qwen3.5-397B-A17B",
    "qwen3.5-397b-a17b": "Qwen3.5-397B-A17B",
    "autodl/gpt-5.4": "gpt-5.4",
    "openai/gpt-5.4": "gpt-5.4",
    "gpt-5.4": "gpt-5.4",
    "autodl/gpt-5.6-luna": "gpt-5.6-luna",
    "openai/gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
    "claude-sonnet-5": "claude-sonnet-5",
}


def normalize_pricing_model_name(model_name: str) -> str:
    normalized = (model_name or "").strip()
    lowered = normalized.lower()
    if lowered in LOCAL_MODEL_ALIASES:
        return LOCAL_MODEL_ALIASES[lowered]
    tail = normalized.split("/", 1)[-1]
    tail_lowered = tail.lower()
    if tail_lowered in LOCAL_MODEL_ALIASES:
        return LOCAL_MODEL_ALIASES[tail_lowered]
    for pricing_key in LOCAL_PRICING:
        pricing_key_lower = pricing_key.lower()
        if tail_lowered == pricing_key_lower or tail_lowered.startswith(f"{pricing_key_lower}-"):
            return pricing_key
    return tail or normalized


def select_pricing(model_name: str, prompt_tokens: int):
    canonical_name = normalize_pricing_model_name(model_name)
    pricing = LOCAL_PRICING.get(canonical_name)
    if pricing is None:
        return canonical_name, None
    tiers = pricing.get("tiers")
    if tiers:
        for tier_index, tier in enumerate(tiers):
            max_input_tokens = tier.get("max_input_tokens")
            if max_input_tokens is None or prompt_tokens <= int(max_input_tokens):
                return canonical_name, dict(tier, tier_index=tier_index)
        return canonical_name, dict(tiers[-1], tier_index=len(tiers) - 1)
    return canonical_name, dict(pricing)


def cached_from_usage(usage):
    usage = usage or {}
    return max(
        int(usage.get("cached_tokens", 0) or 0),
        int(usage.get("cache_tokens", 0) or 0),
        int(usage.get("cache_read_input_tokens", 0) or 0),
        int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0),
        int(((usage.get("input_tokens_details") or {}).get("cached_tokens", 0)) or 0),
    )


def cache_creation_from_usage(usage):
    usage = usage or {}
    return max(
        int(usage.get("cache_creation_input_tokens", 0) or 0),
        int(usage.get("cache_creation_tokens", 0) or 0),
        int(((usage.get("prompt_tokens_details") or {}).get("cache_creation_tokens", 0)) or 0),
        int(((usage.get("input_tokens_details") or {}).get("cache_creation_tokens", 0)) or 0),
    )


def usage_breakdown(usage):
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    cached_tokens = min(prompt_tokens, cached_from_usage(usage))
    cache_creation_tokens = max(0, cache_creation_from_usage(usage))
    # Provider responses used by this runner expose ``input_tokens`` as the
    # complete prompt, including cache reads.  Charge only the remainder here;
    # cache reads and cache creation are priced separately below.
    input_tokens = max(prompt_tokens - cached_tokens - cache_creation_tokens, 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    completion_tokens = int(usage.get("completion_tokens", output_tokens) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + cached_tokens + cache_creation_tokens + output_tokens) or 0)
    reasoning_tokens = int((((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0))
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def compute_usage_cost_usd(model_name: str, usage):
    usage_stats = usage_breakdown(usage)
    _canonical_name, pricing = select_pricing(model_name, usage_stats["prompt_tokens"])
    if pricing is None:
        return _canonical_name, None, None
    cache_write_rate = float(pricing.get("cache_write", pricing.get("cache", 0.0)))
    if "cache_write_1h" in pricing and str(model_name or "").lower().endswith("1h"):
        cache_write_rate = float(pricing["cache_write_1h"])
    cost_usd = round(
        (
            usage_stats["input_tokens"] * float(pricing["input"]) / 1_000_000
            + usage_stats["cached_tokens"] * float(pricing["cache"]) / 1_000_000
            + usage_stats["cache_creation_tokens"] * cache_write_rate / 1_000_000
            + usage_stats["output_tokens"] * float(pricing["output"]) / 1_000_000
        ),
        9,
    )
    return _canonical_name, pricing, cost_usd


def sum_traj_local_pricing_cost(traj_data: dict, default_model_name: str | None = None):
    local_cost_steps = []
    for message in traj_data.get("messages", []):
        if not isinstance(message, dict):
            continue
        extra = message.get("extra") or {}
        response = extra.get("response") or {}
        usage = response.get("usage") or {}
        if not usage:
            continue
        response_model_name = str(response.get("model") or default_model_name or "")
        _, _, step_cost_usd = compute_usage_cost_usd(response_model_name, usage)
        if step_cost_usd is not None:
            local_cost_steps.append(step_cost_usd)
    return round(sum(local_cost_steps), 9) if local_cost_steps else None


def estimate_summary_cost(data: dict):
    model_name = data.get("model_name") or data.get("model")
    if not model_name:
        return None

    prompt_tokens = int(data.get("prompt_tokens", data.get("input_tokens", 0)) or 0)
    cached_tokens = int(data.get("cached_tokens", 0) or 0)
    input_tokens = int(data.get("input_tokens", max(prompt_tokens - cached_tokens, 0)) or 0)
    output_tokens = int(data.get("output_tokens", data.get("completion_tokens", 0)) or 0)
    cache_creation_tokens = int(
        data.get(
            "cache_creation_tokens",
            data.get("cache_write_tokens", data.get("cache_write_input_tokens", 0)),
        )
        or 0
    )

    _canonical_name, pricing = select_pricing(model_name, prompt_tokens)
    if pricing is None:
        return None

    cache_write_rate = float(pricing.get("cache_write", pricing.get("cache", 0.0)))
    if "cache_write_1h" in pricing and str(model_name or "").lower().endswith("1h"):
        cache_write_rate = float(pricing["cache_write_1h"])

    return round(
        (
            input_tokens * float(pricing["input"]) / 1_000_000
            + cached_tokens * float(pricing["cache"]) / 1_000_000
            + cache_creation_tokens * cache_write_rate / 1_000_000
            + output_tokens * float(pricing["output"]) / 1_000_000
        ),
        9,
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _get_usage_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    extra = message.get("extra")
    if not isinstance(extra, dict):
        return None
    response = extra.get("response")
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if isinstance(usage, dict):
        return usage
    return None


def _get_response_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    extra = message.get("extra")
    if not isinstance(extra, dict):
        return None
    response = extra.get("response")
    if isinstance(response, dict):
        return response
    return None


def _calculate_usage_cost_with_model_name(model_name: str, usage: dict[str, Any]) -> float | None:
    prompt_tokens = _safe_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion_tokens = _safe_int(usage.get("completion_tokens") or usage.get("output_tokens"))

    prompt_details = usage.get("prompt_tokens_details")
    input_details = usage.get("input_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    if not isinstance(input_details, dict):
        input_details = {}

    cached_tokens = max(
        _safe_int(usage.get("cached_tokens")),
        _safe_int(usage.get("cache_tokens")),
        _safe_int(usage.get("cache_read_input_tokens")),
        _safe_int(prompt_details.get("cached_tokens")),
        _safe_int(input_details.get("cached_tokens")),
    )
    uncached_prompt_tokens = max(0, prompt_tokens - cached_tokens)

    tier = select_pricing(model_name, prompt_tokens)[1]
    if tier is None:
        return None

    return (
        uncached_prompt_tokens * float(tier["input"])
        + cached_tokens * float(tier["cache"])
        + completion_tokens * float(tier["output"])
    ) / 1_000_000


def _calculate_pricing_stats_for_messages(messages: list[dict[str, Any]], default_model_name: str) -> dict[str, Any] | None:
    total_cost = 0.0
    total_calls = 0
    has_usage = False
    per_model: dict[str, dict[str, Any]] = {}

    for message in messages:
        usage = _get_usage_payload(message)
        if not usage:
            continue

        response = _get_response_payload(message) or {}
        raw_model_name = response.get("model")
        if not isinstance(raw_model_name, str) or not raw_model_name.strip():
            raw_model_name = default_model_name
        normalized_model_name = normalize_pricing_model_name(raw_model_name)
        if normalized_model_name is None:
            return None

        cost = _calculate_usage_cost_with_model_name(raw_model_name, usage)
        if cost is None:
            return None

        entry = per_model.setdefault(
            normalized_model_name,
            {
                "model_name": normalized_model_name,
                "response_models": [],
                "api_calls": 0,
                "cost_usd": 0.0,
            },
        )
        response_models = entry["response_models"]
        if raw_model_name not in response_models:
            response_models.append(raw_model_name)
        entry["api_calls"] += 1
        entry["cost_usd"] += cost

        total_cost += cost
        total_calls += 1
        has_usage = True

    if not has_usage:
        return None

    for entry in per_model.values():
        entry["cost_usd"] = round(float(entry["cost_usd"]), 9)

    return {
        "cost_usd": round(total_cost, 9),
        "api_calls": total_calls,
        "per_model": list(per_model.values()),
    }


def _merge_per_model_stats(*stats_items: dict[str, Any] | None) -> list[dict[str, Any]]:
    per_model: dict[str, dict[str, Any]] = {}
    for stats in stats_items:
        if not stats:
            continue
        for item in stats.get("per_model", []):
            model_name = item.get("model_name")
            if not isinstance(model_name, str) or not model_name:
                continue
            entry = per_model.setdefault(
                model_name,
                {
                    "model_name": model_name,
                    "response_models": [],
                    "api_calls": 0,
                    "cost_usd": 0.0,
                },
            )
            for response_model in item.get("response_models", []):
                if response_model not in entry["response_models"]:
                    entry["response_models"].append(response_model)
            entry["api_calls"] += int(item.get("api_calls") or 0)
            entry["cost_usd"] += float(item.get("cost_usd") or 0.0)

    for entry in per_model.values():
        entry["cost_usd"] = round(float(entry["cost_usd"]), 9)
    return list(per_model.values())


def _calculate_model_pricing_stats(agent) -> dict[str, Any] | None:
    default_model_name = getattr(agent.model.config, "model_name", None)
    if not isinstance(default_model_name, str):
        return None

    extra_pricing_messages = []
    get_extra_pricing_messages = getattr(agent, "get_extra_pricing_messages", None)
    if callable(get_extra_pricing_messages):
        extra_pricing_messages = get_extra_pricing_messages() or []

    scout_stats = _calculate_pricing_stats_for_messages(list(extra_pricing_messages), default_model_name)
    main_stats = _calculate_pricing_stats_for_messages(list(agent.messages), default_model_name)
    if scout_stats is None and main_stats is None:
        return None

    total_cost = float((scout_stats or {}).get("cost_usd") or 0.0) + float((main_stats or {}).get("cost_usd") or 0.0)
    total_calls = int((scout_stats or {}).get("api_calls") or 0) + int((main_stats or {}).get("api_calls") or 0)

    return {
        "instance_cost": round(total_cost, 9),
        "cost_usd": round(total_cost, 9),
        "api_calls": total_calls,
        "per_model": _merge_per_model_stats(scout_stats, main_stats),
        "main_cost_usd": (main_stats or {}).get("cost_usd", 0.0),
        "main_api_calls": (main_stats or {}).get("api_calls", 0),
        "main_per_model": (main_stats or {}).get("per_model", []),
        "scout_handoff_cost_usd": (scout_stats or {}).get("cost_usd", 0.0),
        "scout_handoff_api_calls": (scout_stats or {}).get("api_calls", 0),
        "scout_handoff_per_model": (scout_stats or {}).get("per_model", []),
    }


def _calculate_instance_cost_from_local_pricing(agent) -> float | None:
    pricing_stats = _calculate_model_pricing_stats(agent)
    if pricing_stats is None:
        return None
    return float(pricing_stats["instance_cost"])
