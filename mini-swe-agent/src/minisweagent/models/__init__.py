"""This file provides convenience functions for selecting models.
You can ignore this file completely if you explicitly set your model in your run script.
"""

import copy
import importlib
import os
import threading

from minisweagent import Model
from minisweagent.models.scout_handoff_model import ScoutHandoffModel
from minisweagent.models.step_switch_model import StepSwitchModel
from minisweagent.models.test_failure_switch_model import TestFailureSwitchModel


class GlobalModelStats:
    """Global model statistics tracker with optional limits."""

    def __init__(self):
        self._cost = 0.0
        self._n_calls = 0
        self._lock = threading.Lock()
        self.cost_limit = float(os.getenv("MSWEA_GLOBAL_COST_LIMIT", "0"))
        self.call_limit = int(os.getenv("MSWEA_GLOBAL_CALL_LIMIT", "0"))
        if (self.cost_limit > 0 or self.call_limit > 0) and not os.getenv("MSWEA_SILENT_STARTUP"):
            print(f"Global cost/call limit: ${self.cost_limit:.4f} / {self.call_limit}")

    def add(self, cost: float) -> None:
        """Add a model call with its cost, checking limits."""
        with self._lock:
            self._cost += cost
            self._n_calls += 1
        if 0 < self.cost_limit < self._cost or 0 < self.call_limit < self._n_calls + 1:
            raise RuntimeError(f"Global cost/call limit exceeded: ${self._cost:.4f} / {self._n_calls + 1}")

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def n_calls(self) -> int:
        return self._n_calls


GLOBAL_MODEL_STATS = GlobalModelStats()


def get_model(input_model_name: str | None = None, config: dict | None = None) -> Model:
    """Get an initialized model object from any kind of user input or settings."""
    resolved_model_name = get_model_name(input_model_name, config)
    if config is None:
        config = {}
    config = copy.deepcopy(config)
    warmup_model_name = os.getenv("MSWEA_WARMUP_MODEL_NAME", "").strip()
    warmup_steps = _get_warmup_steps()
    warmup_enabled = _is_warmup_enabled(warmup_model_name, warmup_steps, resolved_model_name)
    test_failure_switch_model_name = os.getenv("MSWEA_TEST_FAILURE_SWITCH_MODEL_NAME", "").strip()
    test_failure_switch_threshold = _get_test_failure_switch_threshold()
    test_failure_switch_enabled = _is_test_failure_switch_enabled(
        test_failure_switch_model_name,
        test_failure_switch_threshold,
        resolved_model_name,
    )
    scout_handoff_model_name = str(config.get("scout_handoff_model_name") or "").strip()
    scout_handoff_steps = _get_scout_handoff_steps(config)
    scout_handoff_enabled = _is_scout_handoff_enabled(
        config,
        scout_handoff_model_name,
        scout_handoff_steps,
        resolved_model_name,
    )

    if warmup_enabled:
        primary_model = _build_single_model(resolved_model_name, config)
        warmup_config = _build_warmup_config(config)
        warmup_model = _build_single_model(warmup_model_name, warmup_config)
        model = StepSwitchModel(primary_model, warmup_model, warmup_steps=warmup_steps)
    else:
        model = _build_single_model(resolved_model_name, config)

    if scout_handoff_enabled:
        scout_config = _build_scout_handoff_config(config)
        scout_model = _build_single_model(scout_handoff_model_name, scout_config)
        model = ScoutHandoffModel(
            model,
            scout_model,
            scout_handoff_steps=scout_handoff_steps,
        )

    if test_failure_switch_enabled:
        fallback_config = _build_test_failure_switch_config(config)
        fallback_model = _build_single_model(test_failure_switch_model_name, fallback_config)
        return TestFailureSwitchModel(
            model,
            fallback_model,
            failure_threshold=test_failure_switch_threshold,
        )

    return model


def _build_single_model(resolved_model_name: str, config: dict) -> Model:
    config = copy.deepcopy(config)
    config["model_name"] = resolved_model_name
    _strip_model_routing_keys(config)

    model_class = get_model_class(resolved_model_name, config.pop("model_class", ""))

    if (from_env := os.getenv("MSWEA_MODEL_API_KEY")) and not str(type(model_class)).endswith("DeterministicModel"):
        config.setdefault("model_kwargs", {})["api_key"] = from_env
    
    # Check for DashScope API key only when explicitly using DashScope provider.
    # Environment variable takes precedence over config file.
    model_kwargs = config.get("model_kwargs")
    if model_kwargs is None:
        model_kwargs = {}
    if (
        not str(type(model_class)).endswith("DeterministicModel")
        and (
            "dashscope" in resolved_model_name.lower()
            or model_kwargs.get("custom_llm_provider") == "dashscope"
        )
    ):
        if dashscope_key := os.getenv("DASHSCOPE_API_KEY"):
            config.setdefault("model_kwargs", {})
            model_kwargs["api_key"] = dashscope_key

    if (
        any(s in resolved_model_name.lower() for s in ["anthropic", "sonnet", "opus", "claude"])
        and "set_cache_control" not in config
    ):
        # Select cache control for Anthropic models by default
        config["set_cache_control"] = "default_end"

    return model_class(**config)


def _strip_model_routing_keys(config: dict) -> None:
    for key in (
        "scout_handoff_enabled",
        "scout_handoff_model_name",
        "scout_handoff_steps",
        "test_failure_switch_enabled",
        "test_failure_switch_model_name",
        "test_failure_switch_threshold",
    ):
        config.pop(key, None)


def _build_warmup_config(config: dict) -> dict:
    warmup_config = copy.deepcopy(config)
    warmup_config.pop("model_class", None)
    return warmup_config


def _build_test_failure_switch_config(config: dict) -> dict:
    fallback_config = copy.deepcopy(config)
    fallback_config.pop("model_class", None)
    _strip_model_routing_keys(fallback_config)
    return fallback_config


def _build_scout_handoff_config(config: dict) -> dict:
    scout_config = copy.deepcopy(config)
    scout_config.pop("model_class", None)
    _strip_model_routing_keys(scout_config)
    return scout_config


def _get_warmup_steps() -> int:
    raw_value = os.getenv("MSWEA_WARMUP_STEPS", "0").strip()
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 0


def _get_test_failure_switch_threshold() -> int:
    raw_value = os.getenv("MSWEA_TEST_FAILURE_SWITCH_THRESHOLD", "0").strip()
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 0


def _get_scout_handoff_steps(config: dict) -> int:
    raw_value = config.get("scout_handoff_steps", 0)
    try:
        return max(0, int(str(raw_value).strip()))
    except ValueError:
        return 0


def _is_enabled(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_warmup_enabled(warmup_model_name: str, warmup_steps: int, resolved_model_name: str) -> bool:
    flag = os.getenv("MSWEA_WARMUP_MODEL_ENABLED", "").strip().lower()
    if not _is_enabled(flag):
        return False
    if warmup_steps <= 0 or not warmup_model_name:
        return False
    return warmup_model_name != resolved_model_name


def _is_test_failure_switch_enabled(
    fallback_model_name: str,
    failure_threshold: int,
    resolved_model_name: str,
) -> bool:
    flag = os.getenv("MSWEA_TEST_FAILURE_SWITCH_ENABLED", "").strip().lower()
    if not _is_enabled(flag):
        return False
    if failure_threshold <= 0 or not fallback_model_name:
        return False
    return fallback_model_name != resolved_model_name


def _is_scout_handoff_enabled(
    config: dict,
    scout_model_name: str,
    scout_handoff_steps: int,
    resolved_model_name: str,
) -> bool:
    raw_flag = config.get("scout_handoff_enabled", False)
    if not _is_enabled(raw_flag):
        return False
    if scout_handoff_steps <= 0 or not scout_model_name:
        return False
    return True


def get_model_name(input_model_name: str | None = None, config: dict | None = None) -> str:
    """Get a model name from any kind of user input or settings."""
    if config is None:
        config = {}
    if input_model_name:
        return input_model_name
    if from_config := config.get("model_name"):
        return from_config
    if from_env := os.getenv("MSWEA_MODEL_NAME"):
        return from_env
    raise ValueError("No default model set. Please run `mini-extra config setup` to set one.")


_MODEL_CLASS_MAPPING = {
    "anthropic": "minisweagent.models.anthropic.AnthropicModel",
    "litellm": "minisweagent.models.litellm_model.LitellmModel",
    "openrouter": "minisweagent.models.openrouter_model.OpenRouterModel",
    "portkey": "minisweagent.models.portkey_model.PortkeyModel",
    "deterministic": "minisweagent.models.test_models.DeterministicModel",
    "dashscope": "minisweagent.models.dashscope_model.DashScopeModel",
}


def get_model_class(model_name: str, model_class: str = "") -> type:
    """Select the best model class.

    If a model_class is provided (as shortcut name, or as full import path,
    e.g., "anthropic" or "minisweagent.models.anthropic.AnthropicModel"),
    it takes precedence over the `model_name`.
    Otherwise, the model_name is used to select the best model class.
    
    Router logic: use DashScope only when explicitly requested.
    """
    if model_class:
        full_path = _MODEL_CLASS_MAPPING.get(model_class, model_class)
        try:
            module_name, class_name = full_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError):
            msg = f"Unknown model class: {model_class} (resolved to {full_path}, available: {_MODEL_CLASS_MAPPING})"
            raise ValueError(msg)

    # Route to DashScope only for explicit DashScope models/providers.
    lower_model_name = model_name.lower()
    if lower_model_name.startswith("dashscope/"):
        from minisweagent.models.dashscope_model import DashScopeModel
        return DashScopeModel

    # Default to LitellmModel
    from minisweagent.models.litellm_model import LitellmModel

    return LitellmModel
