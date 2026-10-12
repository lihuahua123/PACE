#!/usr/bin/env python3
"""Audit archived repeat-block trajectories without rerunning any experiment.

It pairs a guard-on run with its Org baseline, reconstructs CNY cost from
stored token usage, records every actual rc=126 block and its next agent
action, and writes a task ledger, block episodes, and a reviewable audit
sample.  It never invokes an LLM, Docker, or the SWE-bench evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_GUARD_RUN = Path(
    "/workspace/pace_source/swebench/log/my-2026-08-30__20-06-34-MiniMax-M2.7"
)
DEFAULT_ORG_RUN = Path(
    "/workspace/pace_source/swebench/log/my-2026-08-16__08-43-07-MiniMax-M2.7_org"
)
DEFAULT_OUTPUT = Path(
    "/workspace/pace_source/scripts/outputs/repeat_block_pair_audit_minimax_m27"
)

# MiniMax-M2.7 local price schedule, CNY per million tokens.
INPUT_RATE, CACHE_RATE, OUTPUT_RATE = 1.680, 0.336, 6.720


def int_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def usage_metrics(messages: list[dict]) -> tuple[int, int, int, int, float]:
    prompt = cached = cache_write = output = 0
    for message in messages:
        usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
        if not usage:
            continue
        prompt_tokens = int_value(usage.get("prompt_tokens", usage.get("input_tokens")))
        cached_tokens = max(
            int_value(usage.get("cache_read_input_tokens")),
            int_value(usage.get("cached_tokens")),
            int_value((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
        )
        cache_creation_tokens = max(
            int_value(usage.get("cache_creation_input_tokens")),
            int_value(usage.get("cache_creation_tokens")),
            int_value((usage.get("prompt_tokens_details") or {}).get("cache_creation_tokens")),
        )
        prompt += prompt_tokens
        cached += min(prompt_tokens, cached_tokens)
        cache_write += cache_creation_tokens
        output += int_value(usage.get("completion_tokens", usage.get("output_tokens")))
    uncached = max(0, prompt - cached - cache_write)
    cost = (uncached * INPUT_RATE + (cached + cache_write) * CACHE_RATE + output * OUTPUT_RATE) / 1_000_000
    return prompt, cached, cache_write, output, cost


def action_from_message(message: dict) -> str:
    match = re.search(r"```bash\s*\n(.*?)\n```", message.get("content", ""), re.DOTALL)
    return " ".join(match.group(1).split()) if match else ""


def action_kind(action: str) -> str:
    lowered = action.lower()
    if "complete_task_and_submit" in lowered or "submit" in lowered:
        return "submit"
    if any(token in lowered for token in ("pytest", "runtests.py", "manage.py test", "python -m django test", "unittest")):
        return "test"
    if any(token in lowered for token in ("sed -i", "apply_patch", "write_text", "cat <<", " tee ", "> ")):
        return "edit"
    if any(token in lowered for token in ("grep ", "rg ", "sed -n", "cat ", "head ", "tail ", "find ")):
        return "inspect"
    return "other"


def block_kind(text: str) -> str:
    lowered = text.lower()
    if "consecutive source read/search" in lowered:
        return "consecutive_inspection"
    if "test actions since the last edit" in lowered:
        return "test_phase"
    if "already read" in lowered or "already searched" in lowered or "already executed" in lowered:
        return "repeated_evidence"
    return "other"


def load_run(run_dir: Path) -> tuple[dict[str, dict], list[dict]]:
    tasks: dict[str, dict] = {}
    episodes: list[dict] = []
    for summary_path in sorted(run_dir.glob("*/task_summary.json")):
        summary = json.loads(summary_path.read_text())
        task_id = summary.get("instance_id") or summary_path.parent.name
        trajectory_paths = list(summary_path.parent.glob(f"{task_id}/{task_id}.traj.json"))
        if not trajectory_paths:
            trajectory_paths = list(summary_path.parent.glob("**/*.traj.json"))
        if not trajectory_paths:
            continue
        trajectory_path = trajectory_paths[0]
        trajectory = json.loads(trajectory_path.read_text())
        messages = trajectory.get("messages") or []
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        prompt, cached, cache_write, output, cost = usage_metrics(messages)
        tasks[task_id] = {
            "task_id": task_id,
            "resolved": bool(summary.get("resolved")),
            "calls": len(assistant_messages),
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "cache_write_tokens": cache_write,
            "output_tokens": output,
            "cost_cny": cost,
            "trajectory": str(trajectory_path),
        }
        for index, message in enumerate(messages):
            content = message.get("content", "")
            if message.get("role") != "user" or "<returncode>126</returncode>" not in content or "BLOCKED:" not in content:
                continue
            blocked_action = ""
            block_turn = 0
            for prior in reversed(messages[:index]):
                if prior.get("role") == "assistant":
                    blocked_action = action_from_message(prior)
                    break
            block_turn = sum(1 for prior in messages[:index] if prior.get("role") == "assistant")
            next_action = ""
            for later in messages[index + 1 :]:
                if later.get("role") == "assistant":
                    next_action = action_from_message(later)
                    break
            episodes.append(
                {
                    "task_id": task_id,
                    "resolved": bool(summary.get("resolved")),
                    "block_turn": block_turn,
                    "block_kind": block_kind(content),
                    "blocked_action": blocked_action,
                    "next_action_kind": action_kind(next_action),
                    "next_action": next_action,
                    "block_message": " ".join(content.split()),
                    "trajectory": str(trajectory_path),
                }
            )
    return tasks, episodes


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard-run", type=Path, default=DEFAULT_GUARD_RUN)
    parser.add_argument("--org-run", type=Path, default=DEFAULT_ORG_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-sample-size", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    guard, episodes = load_run(args.guard_run)
    org, _ = load_run(args.org_run)
    common = sorted(set(guard) & set(org))
    ledger = []
    for task_id in common:
        guarded, baseline = guard[task_id], org[task_id]
        task_episodes = [episode for episode in episodes if episode["task_id"] == task_id]
        ledger.append(
            {
                "task_id": task_id,
                "guard_resolved": guarded["resolved"],
                "org_resolved": baseline["resolved"],
                "outcome_cell": (
                    "both_resolved" if guarded["resolved"] and baseline["resolved"] else
                    "guard_only" if guarded["resolved"] else
                    "org_only" if baseline["resolved"] else "both_unresolved"
                ),
                "guard_calls": guarded["calls"],
                "org_calls": baseline["calls"],
                "guard_cost_cny": round(guarded["cost_cny"], 6),
                "org_cost_cny": round(baseline["cost_cny"], 6),
                "cost_delta_cny": round(guarded["cost_cny"] - baseline["cost_cny"], 6),
                "cost_saving_pct": round(100 * (1 - guarded["cost_cny"] / baseline["cost_cny"]), 2) if baseline["cost_cny"] else "",
                "guard_prompt_tokens": guarded["prompt_tokens"],
                "org_prompt_tokens": baseline["prompt_tokens"],
                "actual_blocks": len(task_episodes),
                "first_block_turn": min((episode["block_turn"] for episode in task_episodes), default=""),
                "block_types": ";".join(sorted({episode["block_kind"] for episode in task_episodes})),
                "guard_trajectory": guarded["trajectory"],
                "org_trajectory": baseline["trajectory"],
            }
        )
    write_csv(args.output / "task_pair_ledger.csv", ledger, list(ledger[0]) if ledger else [])
    write_csv(args.output / "block_episodes.csv", episodes, list(episodes[0]) if episodes else [])

    # Stratify review rows by final paired outcome and triggering rule.
    outcome = {row["task_id"]: row["outcome_cell"] for row in ledger}
    rng = random.Random(20260911)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for episode in episodes:
        grouped.setdefault((outcome.get(episode["task_id"], "unknown"), episode["block_kind"]), []).append(episode)
    sample = []
    while len(sample) < min(args.audit_sample_size, len(episodes)):
        choices = [key for key, rows in grouped.items() if rows]
        if not choices:
            break
        key = min(choices, key=lambda item: len([row for row in sample if (row["outcome_cell"], row["block_kind"]) == item]))
        episode = grouped[key].pop(rng.randrange(len(grouped[key])))
        sample.append({
            **episode,
            "outcome_cell": outcome.get(episode["task_id"], "unknown"),
            "human_judgment": "",  # redundant / deferrable / necessary / unclear
            "human_note": "",
        })
    sample_fields = list(sample[0]) if sample else []
    write_csv(args.output / "manual_block_audit_sample.csv", sample, sample_fields)

    outcome_counts = Counter(row["outcome_cell"] for row in ledger)
    block_counts = Counter(episode["block_kind"] for episode in episodes)
    next_counts = Counter(episode["next_action_kind"] for episode in episodes)
    tasks_blocked = len({episode["task_id"] for episode in episodes})
    guarded_cost = sum(guard[task_id]["cost_cny"] for task_id in common)
    org_cost = sum(org[task_id]["cost_cny"] for task_id in common)
    summary = f"""# MiniMax-M2.7 repeat-block paired audit

Source runs: guard-on `{args.guard_run.name}` and Org `{args.org_run.name}`.
All values are reconstructed offline from archived trajectories; no model or evaluator was run.

## Paired outcomes ({len(common)} common tasks)

| both resolved | guard only | Org only | both unresolved |
| ---: | ---: | ---: | ---: |
| {outcome_counts['both_resolved']} | {outcome_counts['guard_only']} | {outcome_counts['org_only']} | {outcome_counts['both_unresolved']} |

Guard cost: {guarded_cost:.3f} CNY; Org cost: {org_cost:.3f} CNY; paired saving: {100 * (1 - guarded_cost / org_cost):.2f}%.

## Actual blocks

{len(episodes)} actual blocks occurred in {tasks_blocked}/{len(common)} guarded tasks.

| repeated evidence | consecutive inspection | test phase | other |
| ---: | ---: | ---: | ---: |
| {block_counts['repeated_evidence']} | {block_counts['consecutive_inspection']} | {block_counts['test_phase']} | {block_counts['other']} |

## First action after a block

| edit | test | inspect | submit | other |
| ---: | ---: | ---: | ---: | ---: |
| {next_counts['edit']} | {next_counts['test']} | {next_counts['inspect']} | {next_counts['submit']} | {next_counts['other']} |

Files: `task_pair_ledger.csv`, `block_episodes.csv`, and the stratified `manual_block_audit_sample.csv`.
"""
    (args.output / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
