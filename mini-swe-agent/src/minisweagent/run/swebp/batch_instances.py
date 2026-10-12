from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class BatchInstance:
    instance_id: str
    problem_statement: str
    base_commit: str
    image_name: str
    repo_name: str
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw_instance: dict[str, Any]) -> "BatchInstance":
        known_fields = {"instance_id", "problem_statement", "base_commit", "image_name", "repo_name", "extra_fields"}
        payload = dict(raw_instance)
        explicit_extra = payload.pop("extra_fields", {}) or {}
        extra_fields = {k: v for k, v in payload.items() if k not in known_fields}
        extra_fields = {**extra_fields, **explicit_extra}
        return cls(
            instance_id=str(payload["instance_id"]),
            problem_statement=str(payload["problem_statement"]),
            base_commit=str(payload["base_commit"]),
            image_name=str(payload["image_name"]),
            repo_name=str(payload["repo_name"]),
            extra_fields=extra_fields,
        )

    def to_template_context(self) -> dict[str, Any]:
        working_dir = self.extra_fields.get("working_dir")
        if not working_dir:
            working_dir = f"/{self.repo_name}"
        return {
            "instance_id": self.instance_id,
            "problem_statement": self.problem_statement,
            "base_commit": self.base_commit,
            "image_name": self.image_name,
            "repo_name": self.repo_name,
            "working_dir": str(working_dir),
            **self.extra_fields,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "problem_statement": self.problem_statement,
            "base_commit": self.base_commit,
            "image_name": self.image_name,
            "repo_name": self.repo_name,
            "extra_fields": self.extra_fields,
        }


def load_batch_instances(instances_path: str | Path) -> list[BatchInstance]:
    path = Path(instances_path)
    raw_data = yaml.safe_load(path.read_text()) or []
    if isinstance(raw_data, dict):
        raw_instances = raw_data.get("instances", [])
    else:
        raw_instances = raw_data
    if not isinstance(raw_instances, list):
        raise TypeError(f"Expected a list of instances in {path}, got {type(raw_instances).__name__}")
    return [BatchInstance.from_mapping(instance) for instance in raw_instances]


def dump_batch_instances(instances: list[BatchInstance], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [instance.to_dict() for instance in instances]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
