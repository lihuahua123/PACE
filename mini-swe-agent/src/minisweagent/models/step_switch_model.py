from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StepSwitchModelConfig:
    model_name: str
    warmup_model_name: str
    warmup_steps: int


class StepSwitchModel:
    """Route the first N queries to a warmup model, then switch to the primary model."""

    def __init__(self, primary_model: Any, warmup_model: Any, *, warmup_steps: int):
        self.primary_model = primary_model
        self.warmup_model = warmup_model
        self.warmup_steps = max(0, int(warmup_steps))
        self.config = StepSwitchModelConfig(
            model_name=self.primary_model.config.model_name,
            warmup_model_name=self.warmup_model.config.model_name,
            warmup_steps=self.warmup_steps,
        )

    @property
    def cost(self) -> float:
        return float(self.primary_model.cost) + float(self.warmup_model.cost)

    @property
    def n_calls(self) -> int:
        return int(self.primary_model.n_calls) + int(self.warmup_model.n_calls)

    def _active_model(self):
        if self.n_calls < self.warmup_steps:
            return self.warmup_model
        return self.primary_model

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        return self._active_model().query(messages, **kwargs)

    def get_template_vars(self) -> dict[str, Any]:
        template_vars = dict(self.primary_model.get_template_vars())
        template_vars.update(
            {
                "model_name": self.primary_model.config.model_name,
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "warmup_model_name": self.warmup_model.config.model_name,
                "warmup_steps": self.warmup_steps,
                "warmup_model_calls": self.warmup_model.n_calls,
                "primary_model_calls": self.primary_model.n_calls,
            }
        )
        return template_vars
