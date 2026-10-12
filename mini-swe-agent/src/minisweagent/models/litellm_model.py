import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.utils.timing import elapsed_since, fail_on_threshold_exceeded, threshold_watchdog

logger = logging.getLogger("litellm_model")


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    request_timeout_threshold: float = float(os.getenv("MSWEA_MODEL_TIMEOUT_THRESHOLD", "0"))
    """Max wall-clock seconds for one model request. Defaults to 0, which disables the check."""


class LitellmModel:
    def __init__(self, *, config_class: type = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self._last_prompt_tokens: int | None = None
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _is_qwen_like_model(self) -> bool:
        model_name = str(self.config.model_name or "").lower()
        return "qwen" in model_name

    def _is_gpt5_like_model(self) -> bool:
        model_name = str(self.config.model_name or "").lower()
        return "gpt-5" in model_name

    def _sanitize_kwargs_for_model(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(kwargs)
        if self._is_gpt5_like_model() and sanitized.get("temperature") == 0.0:
            # GPT-5 endpoints reject temperature=0; use the provider-supported default.
            sanitized["temperature"] = 1
        return sanitized

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except Exception:
            return 0

    def _augment_cache_usage_for_qwen(self, usage: dict[str, Any]) -> dict[str, Any]:
        """Backfill cache usage when provider reports zero cached tokens for qwen-compatible endpoints."""
        if not self._is_qwen_like_model():
            prompt_tokens = self._safe_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
            if prompt_tokens > 0:
                self._last_prompt_tokens = prompt_tokens
            return usage

        prompt_tokens = self._safe_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
        if prompt_tokens <= 0:
            return usage

        prompt_details = usage.get("prompt_tokens_details")
        input_details = usage.get("input_tokens_details")
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        if not isinstance(input_details, dict):
            input_details = {}

        reported_candidates = [
            usage.get("cached_tokens"),
            usage.get("cache_tokens"),
            usage.get("cache_read_input_tokens"),
            prompt_details.get("cached_tokens"),
            input_details.get("cached_tokens"),
        ]
        reported_cached = max([self._safe_int(v) for v in reported_candidates] or [0])

        if reported_cached > 0:
            self._last_prompt_tokens = prompt_tokens
            return usage

        estimated_cached = 0
        if self._last_prompt_tokens is not None:
            # With growing chat history, repeated prompt prefix approximates cache hits.
            delta = max(0, prompt_tokens - self._last_prompt_tokens)
            estimated_cached = max(0, prompt_tokens - delta)
        self._last_prompt_tokens = prompt_tokens

        if estimated_cached <= 0:
            return usage

        usage["cached_tokens"] = estimated_cached
        usage["cache_tokens"] = estimated_cached
        usage["cache_read_input_tokens"] = estimated_cached
        prompt_details["cached_tokens"] = estimated_cached
        input_details["cached_tokens"] = estimated_cached
        usage["prompt_tokens_details"] = prompt_details
        usage["input_tokens_details"] = input_details
        usage["cache_tokens_source"] = "estimated_from_prompt_deltas"
        return usage

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
                SystemExit,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        start_time = time.perf_counter()
        threshold = self.config.request_timeout_threshold
        try:
            request_kwargs = self._sanitize_kwargs_for_model(self.config.model_kwargs | kwargs)
            with threshold_watchdog(
                operation="model request",
                start_time=start_time,
                threshold_sec=threshold,
                logger=logger,
                details={"model": self.config.model_name, "messages": len(messages)},
            ):
                return litellm.completion(
                    model=self.config.model_name, messages=messages, **request_kwargs
                )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        finally:
            fail_on_threshold_exceeded(
                operation="model request",
                elapsed_sec=elapsed_since(start_time),
                threshold_sec=threshold,
                logger=logger,
                details={"model": self.config.model_name},
            )

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        response = self._query(messages, **kwargs)
        try:
            cost = litellm.cost_calculator.completion_cost(response)
        except Exception as e:
           
            cost = 0.0
        self.n_calls += 1
        assert cost >= 0.0, f"Cost is negative: {cost}"
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        response_payload = response.model_dump()
        usage_payload = response_payload.get("usage")
        if isinstance(usage_payload, dict):
            response_payload["usage"] = self._augment_cache_usage_for_qwen(usage_payload)
        return {
            "content": response.choices[0].message.content or "",  # type: ignore
            "extra": {
                "response": response_payload,
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
