"""Resume one SWE-bench run with a trace-status guard enabled.

The archived prefix is executed in a fresh task environment to recreate its
working tree, while the original prefix messages are restored verbatim before
the new model is queried.  The selected would-block action is *not* executed:
the agent receives the normal repeat-block observation and then continues.
"""

from __future__ import annotations

import copy
import json
import traceback
from pathlib import Path

import typer
import yaml
from datasets import load_dataset

from minisweagent import global_config_dir
from minisweagent.agents.default import DefaultAgent, NonTerminatingException, TerminatingException
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.models import get_model
from minisweagent.run.extra.swebench import (
    DATASET_MAPPING,
    _filter_agent_config_for,
    get_sb_environment,
    update_preds_file,
)
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger


app = typer.Typer(add_completion=False)
DEFAULT_OUTPUT = global_config_dir / "last_swebench_replay_from_block"


def _would_trace_status_block(
    agent: DefaultAgent,
    action_text: str,
    *,
    large_read_block_enabled: bool,
    repeat_block_enabled: bool,
) -> bool:
    """Use trace-status guard predicates without returning their block output."""
    large_read_block = large_read_block_enabled and (
        "TRACE_STATUS_LARGE_READ_BLOCK_CONFIRMED" not in action_text
        and bool(agent._trace_status_large_read_block_message(action_text))
    )
    if large_read_block:
        return True
    if not repeat_block_enabled:
        return False
    if "TRACE_STATUS_REPEAT_CONFIRMED" in action_text:
        return False
    return bool(
        agent._trace_status_inspection_budget_message(action_text)
        or agent._trace_status_repeat_block_message(action_text)
    )


def _assistant_action(message: dict) -> str | None:
    if message.get("role") != "assistant":
        return None
    try:
        return DefaultAgent.parse_action(None, {"content": message.get("content", "")}).get("action")
    except NonTerminatingException:
        return None


def _rehydrate_and_block(agent: DefaultAgent, trace: dict, block_occurrence: int) -> tuple[int, str]:
    """Replay prefix actions, then block the selected historical action.

    ``_trace_status_bar_enabled`` is temporarily enabled only to rebuild the
    repeat detector's internal evidence ledger.  Generated replay observations
    are discarded in favor of the archived messages before resuming.
    """
    messages = trace.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Trace does not contain a messages list")

    original_status_bar = agent._trace_status_bar_enabled
    original_repeat_block = agent._trace_status_repeat_block_enabled
    original_large_read_block = agent._trace_status_large_read_block_enabled
    agent._trace_status_bar_enabled = True
    agent._trace_status_repeat_block_enabled = False
    agent._trace_status_large_read_block_enabled = False
    matched = 0

    for index, message in enumerate(messages):
        action_text = _assistant_action(message)
        if not action_text:
            continue
        if _would_trace_status_block(
            agent,
            action_text,
            large_read_block_enabled=original_large_read_block,
            repeat_block_enabled=original_repeat_block,
        ):
            matched += 1
            if matched != block_occurrence:
                # The historical command was actually executed, so it must be
                # included while advancing to a later hypothetical block.
                agent.get_observation({"content": message.get("content", "")})
                agent._turn_control_used += 1
                continue

            # Preserve the historical prefix, including the assistant action
            # that is now blocked. Its observation is newly produced by the
            # real DefaultAgent block path, rather than copied from the trace.
            agent.messages = copy.deepcopy(messages[: index + 1])
            agent._trace_status_bar_enabled = original_status_bar
            agent._trace_status_repeat_block_enabled = original_repeat_block
            agent._trace_status_large_read_block_enabled = original_large_read_block
            output = agent.get_observation({"content": message.get("content", "")})
            if not (output.get("trace_status_repeat_blocked") or output.get("trace_status_large_read_blocked")):
                raise RuntimeError(
                    "Replay candidate did not pass DefaultAgent.execute_action's trace-status block path; "
                    "refusing to continue with an unverified intervention."
                )
            agent._turn_control_used += 1
            agent._trace_status_repeat_block_enabled = original_repeat_block
            return index, str(output.get("output", ""))

        agent.get_observation({"content": message.get("content", "")})
        agent._turn_control_used += 1

    agent._trace_status_bar_enabled = original_status_bar
    agent._trace_status_repeat_block_enabled = original_repeat_block
    agent._trace_status_large_read_block_enabled = original_large_read_block
    raise ValueError(f"Only found {matched} repeat-block candidates; requested occurrence {block_occurrence}")


@app.command()
def main(
    trace_path: Path = typer.Option(..., "--trace", exists=True, dir_okay=False),
    block_occurrence: int = typer.Option(1, "--block-occurrence", min=1),
    subset: str = typer.Option("verified", "--subset"),
    split: str = typer.Option("test", "--split"),
    filter_spec: str = typer.Option("", "--filter"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "--output"),
    model: str | None = typer.Option(None, "--model"),
    model_class: str | None = typer.Option(None, "--model-class"),
    config_spec: Path = typer.Option(builtin_config_dir / "extra" / "swebench.yaml", "--config"),
) -> None:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    instance_id = str(trace.get("instance_id") or "")
    if not instance_id:
        raise typer.BadParameter("Trace has no instance_id", param_hint="--trace")
    if filter_spec and filter_spec != f"^{instance_id}$":
        raise typer.BadParameter("--filter must match the trace instance exactly", param_hint="--filter")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    instance = next(
        (row for row in load_dataset(dataset_path, split=split) if row["instance_id"] == instance_id),
        None,
    )
    if instance is None:
        raise typer.BadParameter(f"{instance_id} not found in {dataset_path}:{split}", param_hint="--trace")

    config = yaml.safe_load(get_config_path(config_spec).read_text()) or {}
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    # This experiment isolates the configured intervention. The large-read
    # guard is injected by the caller; do not add the independent repeat guard.
    config.setdefault("agent", {})["trace_status_repeat_block_enabled"] = False
    config["agent"]["scout_handoff_enabled"] = False

    output.mkdir(parents=True, exist_ok=True)
    instance_dir = output / instance_id
    trajectory_path = instance_dir / f"{instance_id}.traj.json"
    agent = None
    exit_status = result = None
    extra_info = None
    replay_error: Exception | None = None
    try:
        env = get_sb_environment(config, instance)
        agent_config = _filter_agent_config_for(DefaultAgent, config["agent"])
        agent = DefaultAgent(get_model(config=config.get("model", {})), env, **agent_config)
        agent._initialize_run(instance["problem_statement"])
        block_message_index, block_output = _rehydrate_and_block(agent, trace, block_occurrence)
        logger.info("Blocked trace candidate at message %s", block_message_index)
        while True:
            try:
                agent.step()
            except NonTerminatingException as error:
                agent.add_message("user", str(error))
            except TerminatingException as error:
                agent.add_message("user", str(error))
                exit_status, result = type(error).__name__, str(error)
                break
        extra_info = {
            "replay_from_trace": str(trace_path),
            "replay_block_occurrence": block_occurrence,
            "replay_block_message_index": block_message_index,
            "replay_block_enabled": True,
            "replay_block_observation": block_output,
        }
    except Exception as error:
        logger.error("Replay failed: %s", error, exc_info=True)
        exit_status, result = type(error).__name__, str(error)
        replay_error = error
        extra_info = {"traceback": traceback.format_exc(), "replay_from_trace": str(trace_path)}
    finally:
        save_traj(
            agent,
            trajectory_path,
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
        )
        if replay_error is None:
            model_name = getattr(getattr(agent, "model", None), "config", None)
            update_preds_file(output / "preds.json", instance_id, getattr(model_name, "model_name", "unknown"), result or "")

    if replay_error is not None:
        raise replay_error


if __name__ == "__main__":
    app()
