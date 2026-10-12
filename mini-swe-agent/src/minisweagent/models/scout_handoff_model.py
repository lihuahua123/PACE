from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ScoutHandoffModelConfig:
    model_name: str
    scout_model_name: str
    scout_handoff_steps: int


class ScoutHandoffModel:
    """Use a scout model for an initial handoff, then route normal queries to the primary model."""

    def __init__(self, primary_model: Any, scout_model: Any, *, scout_handoff_steps: int):
        self.primary_model = primary_model
        self.scout_model = scout_model
        self.scout_handoff_steps = max(0, int(scout_handoff_steps))
        self.config = ScoutHandoffModelConfig(
            model_name=self.primary_model.config.model_name,
            scout_model_name=self.scout_model.config.model_name,
            scout_handoff_steps=self.scout_handoff_steps,
        )

    @property
    def cost(self) -> float:
        return float(self.primary_model.cost) + float(self.scout_model.cost)

    @property
    def n_calls(self) -> int:
        return int(self.primary_model.n_calls) + int(self.scout_model.n_calls)

    @property
    def scout_handoff_enabled(self) -> bool:
        return self.scout_handoff_steps > 0

    def scout_query(self, messages: list[dict[str, Any]], **kwargs) -> dict:
        return self.scout_model.query(messages, **kwargs)

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict:
        return self.primary_model.query(messages, **kwargs)

    def get_template_vars(self) -> dict[str, Any]:
        template_vars = dict(self.primary_model.get_template_vars())
        template_vars.update(
            {
                "model_name": self.primary_model.config.model_name,
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "scout_handoff_enabled": self.scout_handoff_enabled,
                "scout_handoff_model_name": self.scout_model.config.model_name,
                "scout_handoff_steps": self.scout_handoff_steps,
                "scout_handoff_model_calls": self.scout_model.n_calls,
                "primary_model_calls": self.primary_model.n_calls,
            }
        )
        return template_vars
