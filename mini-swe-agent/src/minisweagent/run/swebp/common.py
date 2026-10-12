from __future__ import annotations

import copy
import json
import re
import traceback
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from jinja2 import StrictUndefined, Template

from minisweagent import Environment
from minisweagent.agents.default import AgentConfig, DefaultAgent, NonTerminatingException, TerminatingException
from minisweagent.agents.experience_retrieval import ExperienceRetrievalAgent, ExperienceRetrievalAgentConfig
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import _calculate_instance_cost_from_local_pricing, save_traj
from minisweagent.utils.log import logger

from .batch_instances import BatchInstance
from .deterministic_retrieval import build_retrieval_command

_OUTPUT_FILE_LOCK = Lock()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def usage_totals_from_messages(messages: list[dict]) -> tuple[int, int, int, list[int]]:
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

def has_non_empty_patch(result: str | None) -> bool:
    return bool(result and result.strip())


def write_live_task_summary(
    *,
    output_dir: Path,
    instance_id: str,
    traj_path: Path,
    model_name: str,
    agent: DefaultAgent | None,
    phase: str,
    exit_status: str | None = None,
    result: str | None = None,
) -> None:
    messages = agent.messages if agent is not None else []
    cached_tokens, input_tokens, output_tokens, confidence_scores = usage_totals_from_messages(messages)
    cost_usd = _calculate_instance_cost_from_local_pricing(agent) if agent is not None else None
    experience_retrieved = any(
        isinstance(msg, dict)
        and "## Experience" in str(msg.get("content") or "")
        and "<retrieved_experiences>" in str(msg.get("content") or "")
        for msg in messages
    )
    submitted = phase == "finished" and exit_status == "Submitted"
    empty_patch = submitted and not has_non_empty_patch(result)
    payload = {
        "instance_id": instance_id,
        "phase": phase,
        "success_within_budget": False,
        "resolved": False,
        "submitted": submitted,
        "completed": phase == "finished",
        "unresolved": False,
        "empty_patch": empty_patch,
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
    write_json_atomic(output_dir / "task_summary.json", payload)


class AutosaveRunMixin:
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


class ProgressTrackingAgent(AutosaveRunMixin, DefaultAgent):
    def __init__(
        self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", autosave_callback=None, **kwargs
    ):
        super().__init__(*args, autosave_callback=autosave_callback, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


class ProgressTrackingExperienceAgent(AutosaveRunMixin, ExperienceRetrievalAgent):
    def __init__(
        self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", autosave_callback=None, **kwargs
    ):
        super().__init__(*args, autosave_callback=autosave_callback, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def filter_agent_config_for(agent_cls, agent_config: dict[str, Any]) -> dict[str, Any]:
    config_cls = ExperienceRetrievalAgentConfig if issubclass(agent_cls, ExperienceRetrievalAgent) else AgentConfig
    valid_keys = {field.name for field in fields(config_cls)}
    return {key: value for key, value in agent_config.items() if key in valid_keys}


def get_batch_environment(config: dict[str, Any], instance: BatchInstance) -> Environment:
    env_config = copy.deepcopy(config.get("environment", {}))
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    if env_config["environment_class"] == "docker":
        env_config["image"] = instance.image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + instance.image_name
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        rendered_command = Template(startup_command, undefined=StrictUndefined).render(**instance.to_template_context())
        out = env.execute(rendered_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def maybe_build_deterministic_retrieval_context(
    env: Environment,
    task: str,
    config: dict[str, Any],
) -> str:
    retrieval_config = config.get("run", {}).get("deterministic_retrieval", {})
    if not isinstance(retrieval_config, dict) or not retrieval_config.get("enabled"):
        return ""

    command = build_retrieval_command(
        task,
        max_symbols=int(retrieval_config.get("max_symbols", 24)),
        max_files=int(retrieval_config.get("max_files", 10)),
        max_matches_per_file=int(retrieval_config.get("max_matches_per_file", 4)),
    )
    output = env.execute(command, timeout=int(retrieval_config.get("timeout", 120)))
    if output["returncode"] == 0:
        return str(output["output"]).strip()
    return (
        "## Deterministic Retrieval Context\n\n"
        "The deterministic repository scan failed before the model started.\n\n"
        f"<returncode>{output['returncode']}</returncode>\n"
        "<output>\n"
        f"{output['output']}"
        "\n</output>"
    )


def maybe_run_pre_prompt_command(env: Environment, config: dict[str, Any]) -> str:
    command = str(config.get("run", {}).get("pre_prompt_command") or "").strip()
    if not command:
        return ""

    output = env.execute(command)
    return (
        "## Pre-Prompt Command Output\n\n"
        "The following command was executed before the first prompt was sent to the model.\n\n"
        "<command>\n"
        f"{command}\n"
        "</command>\n"
        f"<returncode>{output['returncode']}</returncode>\n"
        "<output>\n"
        f"{output['output']}"
        "\n</output>"
    )


def prediction_record(instance_id: str, model_name: str, result: str | None) -> dict[str, str | None]:
    return {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": result,
    }


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str | None) -> None:
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = prediction_record(instance_id, model_name, result)
        output_path.write_text(json.dumps(output_data, indent=2) + "\n", encoding="utf-8")


def remove_from_preds_file(output_path: Path, instance_id: str) -> None:
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2) + "\n", encoding="utf-8")


def write_prediction_artifacts(output_dir: Path, instance_id: str, model_name: str, result: str | None) -> None:
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / f"{instance_id}.pred").write_text(result or "", encoding="utf-8")
    update_preds_file(output_dir / "preds.json", instance_id, model_name, result)


def process_batch_instance(
    instance: BatchInstance,
    output_dir: Path,
    config: dict[str, Any],
    progress_manager: RunBatchProgressManager,
) -> None:
    config = copy.deepcopy(config)
    instance_id = instance.instance_id
    instance_dir = output_dir / instance_id
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    (instance_dir / f"{instance_id}.pred").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance.problem_statement
    template_context = instance.to_template_context()

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
        write_live_task_summary(
            output_dir=instance_dir,
            instance_id=instance_id,
            traj_path=traj_path,
            model_name=model.config.model_name,
            agent=current_agent,
            phase=phase,
            exit_status=exit_status,
        )

    try:
        env = get_batch_environment(config, instance)
        if pre_prompt_command_output := maybe_run_pre_prompt_command(env, config):
            template_context["pre_prompt_command_output"] = pre_prompt_command_output
        if retrieval_context := maybe_build_deterministic_retrieval_context(env, task, config):
            template_context["deterministic_retrieval_context"] = retrieval_context
        agent_config = config.get("agent", {})
        use_experience = agent_config.get("experience_store_path") is not None
        if use_experience:
            filtered_agent_config = filter_agent_config_for(ProgressTrackingExperienceAgent, agent_config)
            agent = ProgressTrackingExperienceAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                autosave_callback=autosave_callback,
                **filtered_agent_config,
            )
            exit_status, result = agent.run(task, issue_id=instance_id, **template_context)
        else:
            filtered_agent_config = filter_agent_config_for(ProgressTrackingAgent, agent_config)
            agent = ProgressTrackingAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                autosave_callback=autosave_callback,
                **filtered_agent_config,
            )
            exit_status, result = agent.run(task, **template_context)
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
        write_live_task_summary(
            output_dir=instance_dir,
            instance_id=instance_id,
            traj_path=traj_path,
            model_name=model.config.model_name,
            agent=agent,
            phase="error" if extra_info else "finished",
            exit_status=exit_status,
            result=result,
        )
        if exit_status == "Submitted" and not has_non_empty_patch(result):
            logger.warning("Submitted instance %s without an exported patch payload", instance_id)
        write_prediction_artifacts(output_dir, instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)
