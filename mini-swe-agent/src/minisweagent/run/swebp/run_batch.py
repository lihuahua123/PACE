#!/usr/bin/env python3

"""Run mini-SWE-agent on generic batch instances with SWE-bench Pro compatible outputs."""

from __future__ import annotations

import concurrent.futures
import copy
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger

from .batch_instances import BatchInstance, load_batch_instances
from .common import process_batch_instance

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_HELP_TEXT = """Run mini-SWE-agent on generic batch instances.

[not dim]
Use this path for SWE-bench Pro style instance files that already contain image names,
problem statements, and base commits.
[/not dim]
"""

_DEFAULT_SWEAGENT_CONFIG = Path(__file__).resolve().parents[5] / "SWE-agent" / "config" / "default.yaml"


def _normalize_sweagent_config_for_mini(config: dict[str, Any]) -> dict[str, Any]:
    """Adapt a SWE-agent config file to mini-SWE-agent's schema.

    SWE-agent stores prompt templates under ``agent.templates`` plus tool/runtime
    settings under ``agent.tools``. mini-SWE-agent expects a flatter structure with
    ``agent.system_template`` / ``instance_template`` and top-level ``environment`` /
    ``model`` keys. When we detect the SWE-agent schema, start from mini's builtin
    default config and overlay the compatible fields from the SWE-agent config.
    """

    agent_config = config.get("agent")
    if not isinstance(agent_config, dict) or "templates" not in agent_config:
        return config

    templates = agent_config.get("templates")
    if not isinstance(templates, dict):
        return config

    tools = agent_config.get("tools") if isinstance(agent_config.get("tools"), dict) else {}
    raw_model = config.get("model") if isinstance(config.get("model"), dict) else {}

    base_config = yaml.safe_load((builtin_config_dir / "default.yaml").read_text(encoding="utf-8")) or {}
    normalized = copy.deepcopy(base_config)

    normalized_agent = normalized.setdefault("agent", {})
    for key in ("system_template", "instance_template", "format_error_template", "action_observation_template"):
        value = templates.get(key)
        if value is not None:
            normalized_agent[key] = value

    if "per_instance_cost_limit" in raw_model and raw_model["per_instance_cost_limit"] is not None:
        normalized_agent["cost_limit"] = raw_model["per_instance_cost_limit"]
    if "per_instance_call_limit" in raw_model and raw_model["per_instance_call_limit"] is not None:
        normalized_agent["step_limit"] = raw_model["per_instance_call_limit"]

    normalized_env = normalized.setdefault("environment", {})
    normalized_env.setdefault("env", {})
    if isinstance(tools.get("env_variables"), dict):
        normalized_env["env"].update(tools["env_variables"])
    if tools.get("execution_timeout") is not None:
        normalized_env["timeout"] = tools["execution_timeout"]

    normalized_model = normalized.setdefault("model", {})
    if raw_model.get("name") is not None:
        normalized_model["model_name"] = raw_model["name"]
    model_kwargs = normalized_model.setdefault("model_kwargs", {})
    for key in ("temperature", "top_p"):
        if raw_model.get(key) is not None:
            model_kwargs[key] = raw_model[key]
    if isinstance(raw_model.get("completion_kwargs"), dict):
        model_kwargs.update(raw_model["completion_kwargs"])

    return normalized


def load_agent_config(config_path: Path) -> dict[str, Any]:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _normalize_sweagent_config_for_mini(raw_config)


def filter_batch_instances(
    instances: list[BatchInstance], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[BatchInstance]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda instance: instance.instance_id)
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    if filter_spec:
        instances = [instance for instance in instances if re.match(filter_spec, instance.instance_id)]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != after_filter:
            logger.info(f"Instance slice: {after_filter} -> {after_slice} instances")
    return instances


def load_existing_prediction_ids(output_path: Path) -> set[str]:
    preds_path = output_path / "preds.json"
    if not preds_path.exists():
        return set()
    return set(json.loads(preds_path.read_text()).keys())


def process_futures(
    futures: dict[concurrent.futures.Future[Any], str], progress_manager: RunBatchProgressManager
) -> None:
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except concurrent.futures.CancelledError:
            pass
        except Exception as e:
            instance_id = futures[future]
            logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
            progress_manager.on_uncaught_exception(instance_id, e)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    instances_path: Path = typer.Option(..., "--instances", exists=True, file_okay=True, dir_okay=False, readable=True, help="YAML file containing batch instances", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g. '0:5')", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class override", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option(_DEFAULT_SWEAGENT_CONFIG, "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use", rich_help_panel="Advanced"),
    experience_min_similarity: float | None = typer.Option(None, "--experience-min-similarity", help="Minimum similarity threshold for experience retrieval", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    instances = load_batch_instances(instances_path)
    instances = filter_batch_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing:
        existing_instances = load_existing_prediction_ids(output_path)
        if existing_instances:
            logger.info(f"Skipping {len(existing_instances)} existing instances")
            instances = [instance for instance in instances if instance.instance_id not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = load_agent_config(config_path)
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    agent_config = config.setdefault("agent", {})
    use_experience = agent_config.get("experience_store_path") is not None
    if use_experience and experience_min_similarity is not None:
        agent_config["experience_min_similarity"] = experience_min_similarity

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_batch_instance, instance, output_path, config, progress_manager): instance.instance_id
                for instance in instances
            }
            try:
                process_futures(futures, progress_manager)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures, progress_manager)


if __name__ == "__main__":
    app()
