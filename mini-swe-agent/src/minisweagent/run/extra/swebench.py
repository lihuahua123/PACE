#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import random
import re
import threading
import time
import traceback
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import AgentConfig, DefaultAgent, NonTerminatingException, TerminatingException
from minisweagent.agents.experience_retrieval import ExperienceRetrievalAgent, ExperienceRetrievalAgentConfig
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import _calculate_instance_cost_from_local_pricing, save_traj
from minisweagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


_OUTPUT_FILE_LOCK = threading.Lock()


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _usage_totals_from_messages(messages: list[dict]) -> tuple[int, int, int, list[int]]:
    cached_tokens = 0
    input_tokens = 0
    output_tokens = 0
    confidence_scores: list[int] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        extra = msg.get("extra") or {}
        response = extra.get("response") or {}
        usage = response.get("usage") or {}
        cached = max(
            int(usage.get("cached_tokens", 0) or 0),
            int(usage.get("cache_tokens", 0) or 0),
            int(usage.get("cache_read_input_tokens", 0) or 0),
            int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0),
            int(((usage.get("input_tokens_details") or {}).get("cached_tokens", 0)) or 0),
        )
        if "input_tokens" in usage:
            current_input_tokens = int(usage.get("input_tokens", 0) or 0)
        else:
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            current_input_tokens = max(0, prompt_tokens - cached)
        if "output_tokens" in usage:
            current_output_tokens = int(usage.get("output_tokens", 0) or 0)
        else:
            current_output_tokens = int(usage.get("completion_tokens", 0) or 0)
        cached_tokens += cached
        input_tokens += current_input_tokens
        output_tokens += current_output_tokens

        content = str(msg.get("content") or "")
        for match in re.finditer(r'\{[^{}]*"confidence_score"\s*:\s*(\d{1,3})[^{}]*\}', content):
            try:
                score = int(match.group(1))
            except Exception:
                continue
            if 0 <= score <= 100:
                confidence_scores.append(score)

    return cached_tokens, input_tokens, output_tokens, confidence_scores


def _write_live_task_summary(
    *,
    output_dir: Path,
    instance_id: str,
    traj_path: Path,
    model_name: str,
    agent: DefaultAgent | None,
    phase: str,
    exit_status: str | None = None,
) -> None:
    messages = agent.messages if agent is not None else []
    cached_tokens, input_tokens, output_tokens, confidence_scores = _usage_totals_from_messages(messages)
    cost_usd = _calculate_instance_cost_from_local_pricing(agent) if agent is not None else None
    experience_retrieved = any(
        isinstance(msg, dict)
        and "## Experience" in str(msg.get("content") or "")
        and "<retrieved_experiences>" in str(msg.get("content") or "")
        for msg in messages
    )
    payload = {
        "instance_id": instance_id,
        "phase": phase,
        "success_within_budget": False,
        "resolved": False,
        "submitted": False,
        "completed": phase == "finished",
        "unresolved": False,
        "empty_patch": False,
        "error": phase == "error",
        "cached_tokens": cached_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens": cached_tokens + input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": cached_tokens + input_tokens + output_tokens,
        "cost_usd": cost_usd,
        "api_calls": getattr(getattr(agent, "model", None), "n_calls", 0),
        "agent_exit_status": exit_status if phase == "finished" else "RUNNING",
        "experience_retrieved": experience_retrieved,
        "confidence_scores": confidence_scores,
        "last_confidence_score": confidence_scores[-1] if confidence_scores else None,
        "model": model_name,
        "traj_path": str(traj_path),
        "verification_source": None,
        "updated_at": datetime.now(UTC).isoformat(),
        "message_count": len(messages),
    }
    _write_json_atomic(output_dir / "task_summary.json", payload)


class _AutosaveRunMixin:
    def __init__(self, *args, autosave_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._autosave_callback = autosave_callback

    def _autosave(self, exit_status: str | None = None, phase: str = "running") -> None:
        if callable(self._autosave_callback):
            self._autosave_callback(self, exit_status=exit_status, phase=phase)

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        self._has_seen_test_command = False
        self._has_successful_test_command = False
        self._initialize_run(task, **kwargs)
        self._autosave()
        while True:
            try:
                self.step()
                self._autosave()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
                self._autosave()
            except TerminatingException as e:
                self.add_message("user", str(e))
                self._autosave(exit_status=type(e).__name__, phase="finished")
                return type(e).__name__, str(e)


class ProgressTrackingAgent(_AutosaveRunMixin, DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(
        self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", autosave_callback=None, **kwargs
    ):
        super().__init__(*args, autosave_callback=autosave_callback, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


class ProgressTrackingExperienceAgent(_AutosaveRunMixin, ExperienceRetrievalAgent):
    """Wrapper around ExperienceRetrievalAgent that provides progress updates."""

    def __init__(
        self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", autosave_callback=None, **kwargs
    ):
        super().__init__(*args, autosave_callback=autosave_callback, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def _filter_agent_config_for(agent_cls, agent_config: dict) -> dict:
    """Keep only config keys accepted by the target agent's config dataclass."""
    config_cls = ExperienceRetrievalAgentConfig if issubclass(agent_cls, ExperienceRetrievalAgent) else AgentConfig
    valid_keys = {f.name for f in fields(config_cls)}
    return {k: v for k, v in agent_config.items() if k in valid_keys}


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent = None
    extra_info = None
    traj_path = instance_dir / f"{instance_id}.traj.json"

    def autosave_callback(current_agent: DefaultAgent, *, exit_status: str | None = None, phase: str = "running") -> None:
        save_traj(
            current_agent,
            traj_path,
            print_path=False,
            exit_status=exit_status or "RUNNING",
            result=None,
            instance_id=instance_id,
        )
        _write_live_task_summary(
            output_dir=output_dir,
            instance_id=instance_id,
            traj_path=traj_path,
            model_name=model.config.model_name,
            agent=current_agent,
            phase=phase,
            exit_status=exit_status,
        )

    try:
        env = get_sb_environment(config, instance)
        agent_config = config.get("agent", {})
        
        # Check if experience retrieval is enabled
        use_experience = agent_config.get("experience_store_path") is not None
        if use_experience:
            filtered_agent_config = _filter_agent_config_for(ProgressTrackingExperienceAgent, agent_config)
            # Use ExperienceRetrievalAgent
            agent = ProgressTrackingExperienceAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                autosave_callback=autosave_callback,
                **filtered_agent_config,
            )
            # Pass issue_id for targeted experience retrieval
            exit_status, result = agent.run(task, issue_id=instance_id)
        else:
            filtered_agent_config = _filter_agent_config_for(ProgressTrackingAgent, agent_config)
            # Use default agent
            agent = ProgressTrackingAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                autosave_callback=autosave_callback,
                **filtered_agent_config,
            )
            exit_status, result = agent.run(task)
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            traj_path,
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        _write_live_task_summary(
            output_dir=output_dir,
            instance_id=instance_id,
            traj_path=traj_path,
            model_name=model.config.model_name,
            agent=agent,
            phase="error" if extra_info else "finished",
            exit_status=exit_status,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    experience_min_similarity: float | None = typer.Option(None, "--experience-min-similarity", help="Minimum similarity threshold for experience retrieval", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
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

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
