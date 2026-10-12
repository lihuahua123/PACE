#!/usr/bin/env python3
"""Classify Mini SWE Agent bash actions into deterministic command intents."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRACE_ROOT = Path(
    "/workspace/skill/swebench/log/org/"
    "batch-core12-swebp-20260709_160007-DeepSeek-V4-Flash"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/workspace/skill/analysis/outputs/minisweagent_intent_classification"
)
DEFAULT_TRAJ_GLOB = "**/*.traj.json"

TEST_RUNNER_PATTERNS = (
    r"\bpytest\b",
    r"\bpython3?\s+-m\s+pytest\b",
    r"\bpython3?\s+-m\s+unittest\b",
    r"\bnpx\s+jest\b",
    r"\bjest\b",
    r"\bmocha\b",
    r"\bnodebb\s+test\b",
    r"\bgo\s+test\b",
)

READ_COMMANDS = {"cat", "nl", "head", "tail", "awk", "more", "less", "sed"}
SEARCH_COMMANDS = {"grep", "rg", "ag", "find"}
LIST_COMMANDS = {"ls", "tree", "pwd"}
METADATA_COMMANDS = {"wc", "stat", "file"}
CLEANUP_COMMANDS = {"rm", "rmdir", "mv", "cp", "chmod"}
TOOL_CHECK_COMMANDS = {"which", "type"}
SERVICE_COMMANDS = {"redis-server", "redis-cli", "mongod", "mysql", "postgres"}
INSTALL_HINTS = (
    "pip install",
    "pip3 install",
    "npm install",
    "yarn install",
    "pnpm install",
    "apt-get install",
    "apt install",
)
EDIT_HINTS = (
    "sed -i",
    "perl -0pi",
    "perl -pi",
    "python -c",
    "python3 -c",
    "node -e",
)
CREATE_HINTS = (
    "cat <<",
    "cat >",
    "cat >>",
    "tee ",
)
SUBMIT_HINTS = (
    "complete_task_and_submit_final_output",
    "git diff --cached",
)
SCRIPT_EXTENSIONS = (".py", ".js", ".ts", ".sh", ".bash")

INTENT_TO_HIGH_LEVEL = {
    # read
    "read-file": "read",
    "run-inline-script": "read",
    # search
    "list-directory": "search",
    "search-keyword": "search",
    "search-files-by-name": "search",
    "search-files-by-content": "search",
    "inspect-metadata": "search",
    # edit
    "edit-file": "edit",
    "create-file": "edit",
    # verify
    "syntax-check": "verify",
    "compile-build": "verify",
    "run-test-suite": "verify",
    "run-test-specific": "verify",
    "run-script": "verify",
    "run-repro-or-verify-script": "verify",
    "run-local-executable": "verify",
    # git
    "git-diff": "git",
    "git-inspect": "git",
    "git-modify": "git",
    "git-other": "git",
    # housekeeping
    "file-cleanup": "housekeeping",
    "check-tool-exists": "housekeeping",
    "service-control": "housekeeping",
    "install-deps": "housekeeping",
    "mkdir": "housekeeping",
    # failed
    "read(failed)": "failed",
    "search(failed)": "failed",
    "run-test(failed)": "failed",
    "bash-command(failed)": "failed",
    # other
    "submit": "other",
    "echo": "other",
    "bash-other": "other",
    "comment-only": "other",
    "malformed-response": "other",
    "empty": "other",
}


@dataclass
class StepRecord:
    trace_id: str
    traj_path: Path
    step_index: int
    message_index: int
    thought: str
    raw_command: str
    observation_raw: str
    returncode: int | None


@dataclass
class IntentResult:
    intent: str
    rule: str
    normalized_command: str
    command_head: str


def to_proportions(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--traj-glob", default=DEFAULT_TRAJ_GLOB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extract_bash_block(text: str) -> str:
    match = re.search(r"```bash\s*(.*?)```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_thought(text: str) -> str:
    if "THOUGHT:" not in text:
        return ""
    after = text.split("THOUGHT:", 1)[1]
    return after.split("```bash", 1)[0].strip()


def parse_returncode(text: str) -> int | None:
    match = re.search(r"<returncode>(-?\d+)</returncode>", text)
    return int(match.group(1)) if match else None


def extract_observation_text(text: str) -> str:
    match = re.search(r"<output>\n?(.*?)</output>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_steps(trace_id: str, traj_path: Path, messages: list[dict[str, Any]]) -> list[StepRecord]:
    steps: list[StepRecord] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        command = extract_bash_block(content)
        if not command:
            continue
        next_message = messages[message_index + 1] if message_index + 1 < len(messages) else {}
        observation_raw = next_message.get("content", "") if next_message.get("role") == "user" else ""
        steps.append(
            StepRecord(
                trace_id=trace_id,
                traj_path=traj_path,
                step_index=len(steps),
                message_index=message_index,
                thought=extract_thought(content),
                raw_command=command,
                observation_raw=observation_raw,
                returncode=parse_returncode(observation_raw),
            )
        )
    return steps


def strip_env_prefixes(command: str) -> str:
    text = command.strip()
    while True:
        match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*=[^\s]+\s+", text)
        if not match:
            return text
        text = text[match.end():].lstrip()


def strip_outer_shell_quotes(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    match = re.match(r"^\$?([\"'])([\s\S]*)\1\}?$", text)
    if match:
        return match.group(2)

    if text[0] in {"'", '"'} and text.endswith("}"):
        return text[1:-1]

    return text


def split_shell_chain(command: str) -> str:
    for token in ("&&", "||", ";"):
        if token in command:
            return command.split(token, 1)[0].strip()
    return command.strip()


def unwrap_command(command: str) -> str:
    text = command.strip()
    if not text:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    while lines and lines[0].lstrip().startswith("#"):
        lines.pop(0)
    text = "\n".join(lines).strip()
    if not text:
        return ""
    for _ in range(6):
        changed = False
        current = text.strip()
        current_lower = current.lower()

        if current_lower.startswith("bash -lc"):
            inner = current[8:].strip()
            text = strip_outer_shell_quotes(inner)
            changed = True

        current = text.strip()
        if (current.startswith("cd ") or current.startswith("source ")) and "&&" in current:
            text = current.split("&&", 1)[1].strip()
            changed = True

        if not changed:
            break

    return text


def strip_leading_env_and_timeout(command: str) -> str:
    text = command.strip()
    for _ in range(8):
        prev = text
        text = re.sub(r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)+", "", text)
        text = re.sub(r"^env\s+", "", text)
        text = re.sub(r"^timeout\s+\S+\s+", "", text)
        text = re.sub(r"^set\s+-[A-Za-z]+(?:\s+-[A-Za-z]+)*\s*;\s*", "", text)
        text = re.sub(r"^set\s+-o\s+\S+\s*;\s*", "", text)
        if text == prev:
            break
    return text.strip()


def first_command_name(command: str) -> str:
    head = split_shell_chain(strip_leading_env_and_timeout(unwrap_command(command)))
    if not head:
        return ""
    try:
        parts = shlex.split(head)
    except ValueError:
        parts = head.split()
    if not parts:
        return ""
    return Path(parts[0]).name


def has_shell_error(observation_text: str) -> bool:
    lowered = (observation_text or "")[:500].lower()
    return any(
        token in lowered
        for token in (
            "syntax error",
            "unexpected token",
            "command not found",
            "here-document at line",
            "unexpected `}'",
            "invalid number of lines",
            "invalid option",
            "broken pipe",
            "no such file or directory",
        )
    )


def looks_like_test_command(normalized: str) -> bool:
    return any(re.search(pattern, normalized) for pattern in TEST_RUNNER_PATTERNS)


def is_search_command(head_lower: str) -> bool:
    return head_lower.startswith(("grep", "rg ", "ag ", "find "))


def is_read_command(head_lower: str) -> bool:
    return head_lower.startswith(("cat ", "head ", "tail ", "sed -n", "nl ", "awk ", "ls "))


def is_script_command(head_lower: str) -> bool:
    return head_lower.startswith(("python ", "python3 ", "node ", "go run "))


def extract_script_target(head: str) -> str:
    try:
        parts = shlex.split(head)
    except ValueError:
        parts = head.split()
    for token in parts[1:]:
        if token.startswith("-"):
            continue
        lowered = token.lower()
        if lowered.endswith(SCRIPT_EXTENSIONS):
            return Path(lowered).name
    return ""


def classify_command(raw_command: str, observation_text: str) -> IntentResult:
    normalized = unwrap_command(raw_command)
    cmd_for_match = strip_leading_env_and_timeout(normalized)
    head = split_shell_chain(cmd_for_match)
    lowered = cmd_for_match.lower()
    head_lower = head.lower()
    command_name = first_command_name(raw_command).lower()

    if not raw_command.strip():
        return IntentResult("empty", "blank bash block", normalized, head)

    if command_name == "thought:" or head_lower.startswith("thought:"):
        return IntentResult("malformed-response", "assistant leaked THOUGHT text into bash block", normalized, head)

    if head.startswith("#") or not command_name:
        return IntentResult("comment-only", "bash block contains only comments or no runnable command", normalized, head)

    if any(hint in lowered for hint in SUBMIT_HINTS):
        return IntentResult("submit", "command contains the Mini SWE Agent submit marker", normalized, head)

    if has_shell_error(observation_text):
        if is_search_command(head_lower):
            return IntentResult("search(failed)", "search command with shell-level error in observation", normalized, head)
        if looks_like_test_command(head_lower):
            return IntentResult("run-test(failed)", "test runner command with shell-level error in observation", normalized, head)
        if is_script_command(head_lower):
            return IntentResult("bash-command(failed)", "script command with shell-level error in observation", normalized, head)
        if is_read_command(head_lower):
            return IntentResult("read(failed)", "read/list command with shell-level error in observation", normalized, head)
        return IntentResult("bash-command(failed)", "other shell command with shell-level error in observation", normalized, head)

    if command_name == "git":
        if re.search(r"\bgit\s+diff\b", lowered):
            return IntentResult("git-diff", "git diff command", normalized, head)
        if re.search(r"\bgit\s+(status|show|log)\b", lowered):
            return IntentResult("git-inspect", "git status/show/log command", normalized, head)
        if re.search(r"\bgit\s+(add|commit|checkout|restore|reset|stash)\b", lowered):
            return IntentResult("git-modify", "git write-side operation", normalized, head)
        return IntentResult("git-other", "other git command", normalized, head)

    if looks_like_test_command(normalized):
        if "::" in normalized or " -k " in lowered:
            return IntentResult("run-test-specific", "test runner narrowed to specific cases", normalized, head)
        return IntentResult("run-test-suite", "test runner command", normalized, head)

    if any(token in head_lower for token in ("py_compile", "compileall", "node -c ", "node --check")):
        return IntentResult("syntax-check", "python/node syntax validation command", normalized, head)

    if head_lower.startswith(("go build", "go vet", "make ", "npx tsc", "tsc ", "./node_modules/.bin/tsc")):
        return IntentResult("compile-build", "build/typecheck command", normalized, head)

    if any(token in head_lower for token in ("npm run build", "yarn build", "check-types", "lint:types", "npm run types")):
        return IntentResult("compile-build", "build/typecheck command", normalized, head)

    if command_name in SEARCH_COMMANDS:
        if command_name == "find" and ("grep" in lowered or "xargs" in lowered):
            return IntentResult("search-files-by-content", "find piped into grep/xargs grep", normalized, head)
        if command_name == "find":
            return IntentResult("search-files-by-name", "find used to locate files/paths", normalized, head)
        return IntentResult("search-keyword", "grep/rg/ag keyword search", normalized, head)

    if command_name == "sed" and head_lower.startswith("sed -n"):
        return IntentResult("read-file", "sed -n style file read", normalized, head)

    if any(hint in lowered for hint in CREATE_HINTS):
        return IntentResult("create-file", "heredoc or tee used to create file content", normalized, head)

    if command_name in READ_COMMANDS:
        return IntentResult("read-file", "cat/nl/head/tail/awk style file read", normalized, head)

    if command_name in LIST_COMMANDS:
        return IntentResult("list-directory", "ls/tree/pwd directory inspection", normalized, head)

    if command_name in METADATA_COMMANDS:
        return IntentResult("inspect-metadata", "wc/stat/file metadata inspection", normalized, head)

    if command_name in TOOL_CHECK_COMMANDS:
        return IntentResult("check-tool-exists", "which/type tool existence check", normalized, head)

    if command_name in SERVICE_COMMANDS:
        return IntentResult("service-control", "service startup or service ping command", normalized, head)

    if any(hint in lowered for hint in INSTALL_HINTS):
        return IntentResult("install-deps", "dependency installation command", normalized, head)

    if command_name in {"sed", "perl"} and "-i" in lowered:
        return IntentResult("edit-file", "in-place sed/perl file edit", normalized, head)

    if command_name in {"python", "python3", "node"} and re.search(r"\s(?:-c|-e)\b", head_lower):
        return IntentResult("run-inline-script", "inline python/node snippet", normalized, head)

    if command_name in {"python", "python3", "node", "bash", "sh"}:
        script_name = extract_script_target(head)
        if script_name:
            lowered_name = script_name.lower()
            if any(token in lowered_name for token in ("repro", "debug", "check", "verify", "test")):
                return IntentResult("run-repro-or-verify-script", "named script with repro/verify/test keywords", normalized, head)
            return IntentResult("run-script", "named python/node/bash script", normalized, head)
        if any(hint in lowered for hint in EDIT_HINTS):
            return IntentResult("edit-file", "inline python/node used as file-edit helper", normalized, head)
        return IntentResult("run-script", "python/node/bash execution", normalized, head)

    if command_name in {"echo", "printf"}:
        return IntentResult("echo", "echo/printf command", normalized, head)

    if command_name in CLEANUP_COMMANDS:
        return IntentResult("file-cleanup", "rm/rmdir/mv/cp/chmod filesystem operation", normalized, head)

    if command_name == "mkdir":
        return IntentResult("mkdir", "directory creation command", normalized, head)

    if head.startswith("./"):
        return IntentResult("run-local-executable", "directly running a local executable/script", normalized, head)

    return IntentResult("bash-other", "fell through all deterministic command rules", normalized, head)


def iter_traces(trace_root: Path, traj_glob: str) -> list[Path]:
    return sorted(
        p
        for p in trace_root.glob(traj_glob)
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(trace_root).parts)
    )


def trace_id_for(traj_path: Path, trace_root: Path) -> str:
    rel = traj_path.relative_to(trace_root)
    return rel.parts[0] if rel.parts else traj_path.stem


def classify_trace(
    traj_path: Path,
    trace_root: Path,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    payload = json.loads(traj_path.read_text())
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError(f"unsupported trajectory format in {traj_path}")

    trace_id = trace_id_for(traj_path, trace_root)
    steps = extract_steps(trace_id, traj_path, messages)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    high_counts: Counter[str] = Counter()
    for step in steps:
        observation_text = extract_observation_text(step.observation_raw)
        result = classify_command(step.raw_command, observation_text)
        high_level = INTENT_TO_HIGH_LEVEL.get(result.intent, "other")
        counts[result.intent] += 1
        high_counts[high_level] += 1
        rows.append(
            {
                "trace_id": trace_id,
                "traj_path": str(traj_path),
                "step_index": step.step_index,
                "message_index": step.message_index,
                "intent": result.intent,
                "high_level": high_level,
                "rule": result.rule,
                "returncode": step.returncode,
                "command_head": result.command_head,
                "raw_command": step.raw_command,
                "normalized_command": result.normalized_command,
                "thought": step.thought,
                "observation_excerpt": observation_text[:300],
            }
        )
    return rows, counts, high_counts


def main() -> None:
    args = parse_args()
    traces = iter_traces(args.trace_root, args.traj_glob)
    if not traces:
        raise SystemExit(f"No trajectory files found under {args.trace_root} with glob {args.traj_glob!r}")

    batch_name = args.trace_root.name
    output_dir = args.output_root / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    per_trace_rows: list[dict[str, Any]] = []
    global_counts: Counter[str] = Counter()
    global_high_counts: Counter[str] = Counter()
    per_trace_high_rows: list[dict[str, Any]] = []

    for traj_path in traces:
        rows, counts, high_counts = classify_trace(traj_path, args.trace_root)
        all_rows.extend(rows)
        global_counts.update(counts)
        global_high_counts.update(high_counts)
        trace_id = trace_id_for(traj_path, args.trace_root)
        per_trace_rows.append(
            {
                "trace_id": trace_id,
                "traj_path": str(traj_path),
                "step_count": sum(counts.values()),
                "unique_intents": len(counts),
                "top_intent": counts.most_common(1)[0][0] if counts else "",
                "top_intent_count": counts.most_common(1)[0][1] if counts else 0,
                "intent_counts": json.dumps(dict(counts), ensure_ascii=False, sort_keys=True),
            }
        )
        per_trace_high_rows.append(
            {
                "trace_id": trace_id,
                "traj_path": str(traj_path),
                "step_count": sum(high_counts.values()),
                "high_level_counts": json.dumps(dict(high_counts), ensure_ascii=False, sort_keys=True),
                "high_level_proportions": json.dumps(
                    {k: round(v, 6) for k, v in to_proportions(high_counts).items()},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    step_csv = output_dir / "step_intent_rows.csv"
    trace_csv = output_dir / "trace_intent_summary.csv"
    global_csv = output_dir / "intent_counts.csv"
    high_global_csv = output_dir / "high_level_counts.csv"
    high_trace_csv = output_dir / "trace_high_level_summary.csv"
    summary_json = output_dir / "summary.json"

    write_csv(
        step_csv,
        all_rows,
        [
            "trace_id",
            "traj_path",
            "step_index",
            "message_index",
            "intent",
            "high_level",
            "rule",
            "returncode",
            "command_head",
            "raw_command",
            "normalized_command",
            "thought",
            "observation_excerpt",
        ],
    )
    write_csv(
        trace_csv,
        per_trace_rows,
        [
            "trace_id",
            "traj_path",
            "step_count",
            "unique_intents",
            "top_intent",
            "top_intent_count",
            "intent_counts",
        ],
    )
    write_csv(
        global_csv,
        [
            {
                "intent": intent,
                "count": count,
                "share": round(count / len(all_rows), 6) if all_rows else 0.0,
            }
            for intent, count in global_counts.most_common()
        ],
        ["intent", "count", "share"],
    )
    write_csv(
        high_global_csv,
        [
            {
                "category": category,
                "count": count,
                "share": round(count / len(all_rows), 6) if all_rows else 0.0,
            }
            for category, count in global_high_counts.most_common()
        ],
        ["category", "count", "share"],
    )
    write_csv(
        high_trace_csv,
        per_trace_high_rows,
        [
            "trace_id",
            "traj_path",
            "step_count",
            "high_level_counts",
            "high_level_proportions",
        ],
    )
    write_json(
        summary_json,
        {
            "trace_root": str(args.trace_root),
            "traj_glob": args.traj_glob,
            "trace_count": len(traces),
            "step_count": len(all_rows),
            "intent_counts": dict(global_counts),
            "high_counts": dict(global_high_counts),
            "high_proportions": {k: round(v, 6) for k, v in to_proportions(global_high_counts).items()},
            "intent_to_category": INTENT_TO_HIGH_LEVEL,
            "outputs": {
                "step_intent_rows_csv": str(step_csv),
                "trace_intent_summary_csv": str(trace_csv),
                "intent_counts_csv": str(global_csv),
                "trace_high_level_summary_csv": str(high_trace_csv),
                "high_level_counts_csv": str(high_global_csv),
            },
        },
    )

    print(json.dumps(json.loads(summary_json.read_text()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
