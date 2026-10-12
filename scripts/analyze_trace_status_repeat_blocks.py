#!/usr/bin/env python3
"""Replay GPT trajectories against TRACE_STATUS_REPEAT_BLOCK_ENABLED.

The flag was off for the historical runs, so this tool reports *counterfactual*
blocks: commands that the current repeat-block implementation would have
returned with rc=126.  A blocked command does not update the simulated trace,
matching ``DefaultAgent.execute_action``.

By default the newest trajectory is selected for each instance.  Use
``--all-traces`` when every saved rerun is wanted instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TraceResult:
    instance_id: str
    trajectory: str
    actions: int
    executable_actions: int
    would_block: int
    repeated_evidence_blocks: int
    consecutive_inspection_blocks: int
    test_phase_blocks: int
    override_actions: int


def assistant_actions(trajectory: dict) -> list[str]:
    """Extract the one bash action generated on each agent turn."""
    actions = []
    for message in trajectory.get("messages", []):
        if message.get("role") != "assistant":
            continue
        matches = re.findall(r"```bash\s*\n(.*?)\n```", message.get("content", ""), re.DOTALL)
        if len(matches) == 1:
            actions.append(matches[0].strip())
    return actions


# These helpers deliberately mirror the current DefaultAgent implementation.
# Keeping this script dependency-free lets it run against archived traces on a
# machine that does not have mini-swe-agent's optional runtime dependencies.
def normalize_path(raw_path: str) -> str:
    return raw_path.strip().strip("'\"").rstrip(");:,")


def looks_like_source_file(path: str) -> bool:
    path_obj = Path(path)
    return bool(path_obj.name and "." in path_obj.name and not any(part in {"node_modules", ".git", "__pycache__"} for part in path_obj.parts))


def extract_read_targets(action: str) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    normalized = action.replace("\\\n", " ")

    def add(path: str, line_range: str) -> None:
        item = (normalize_path(path), line_range)
        if item not in seen:
            seen.add(item)
            targets.append(item)

    patterns = [
        (r"(?:nl|cat)\s+-n\s+([^\s|;&]+)\s*\|\s*sed\s+-n\s+'(\d+),(\d+)p'", (1, 2, 3)),
        (r"nl\s+-ba\s+([^\s|;&]+)\s*\|\s*sed\s+-n\s+'(\d+),(\d+)p'", (1, 2, 3)),
        (r"sed\s+-n\s+'(\d+),(\d+)p'\s+([^\s|;&]+)", (3, 1, 2)),
        (r"awk\s+'NR>=(\d+)\s*&&\s*NR<=(\d+).*?'\s+([^\s|;&]+)", (3, 1, 2)),
    ]
    for pattern, groups in patterns:
        for match in re.finditer(pattern, normalized):
            add(match.group(groups[0]), f"{match.group(groups[1])}-{match.group(groups[2])}")
    for match in re.finditer(r"head\s+-n\s+(\d+)\s+([^\s|;&]+)\s*\|\s*tail\s+-n\s+(\d+)", normalized):
        add(match.group(2), f"{max(1, int(match.group(1)) - int(match.group(3)) + 1)}-{match.group(1)}")
    for match in re.finditer(r"(?:cat|cat\s+-n|nl\s+-ba)\s+([^\s|;&]+)\s*\|\s*head\s+-n\s+(\d+)", normalized):
        add(match.group(1), f"1-{match.group(2)}")
    for match in re.finditer(r"head\s+-n\s+(\d+)\s+([^\s|;&]+)", normalized):
        add(match.group(2), f"1-{match.group(1)}")
    for match in re.finditer(r"(?:cat|cat\s+-n|nl\s+-ba)\s+([^\s|;&]+)(?:\s+2?>&?1)?\s*\|\s*head\s+(?:-n\s+|-)(\d+)", normalized):
        add(match.group(1), f"1-{match.group(2)}")
    for match in re.finditer(r"(?:cat|cat\s+-n|nl\s+-ba)\s+([^\s|;&]+)\s*\|\s*tail\s+-n\s+\+?(\d+)", normalized):
        add(match.group(1), f"{match.group(2)}-tail")
    for match in re.finditer(r"tail\s+-n\s+(\d+)\s+([^\s|;&]+)", normalized):
        add(match.group(2), "tail")
    for pattern, kind in [(r"nl\s+-ba\s+([^\s|;&]+)(?=\s*(?:$|[;]|2?>&?1\s*(?:$|[;])))", "full"), (r"cat\s+([^\s|;&]+)(?=\s*(?:$|[;]|2?>&?1\s*(?:$|[;])))", "full")]:
        for match in re.finditer(pattern, normalized):
            add(match.group(1), kind)
    for match in re.finditer(r"cat\s+<\s*([^\s|;&]+)", normalized):
        add(match.group(1), "full")
    for match in re.finditer(r"(?:python|python3)\s+-c\s+['\"].*?open\((['\"])([^'\"]+)\1.*?['\"]", normalized):
        add(match.group(2), "full")
    for pattern in [r"(?:grep|rg)\s+[^|;&\n]*?\s+([^\s|;&]+)\s*$", r"(?:grep|rg)\s+[^|;&\n]*?\s+([^\s|;&]+)\s*\|\s*head\s+-\d+"]:
        for match in re.finditer(pattern, normalized):
            add(match.group(1), "search")
    return [(path, line_range) for path, line_range in targets if looks_like_source_file(path)]


def extract_search_signature(action: str) -> str:
    normalized = " ".join(action.replace("\\\n", " ").split())
    match = re.search(r"\b(?:grep|rg)\b\s+[^'\"]*['\"]([^'\"]+)['\"]\s+([^|;&]+)", normalized)
    if match:
        return shorten(f"{match.group(1)} @ {match.group(2)}", 90)
    match = re.search(r"\bfind\b\s+([^|;&]+)", normalized)
    return shorten(f"find {match.group(1)}", 90) if match else ""


def shorten(command: str, limit: int) -> str:
    command = " ".join(command.split())
    return command if len(command) <= limit else command[: limit - 3] + "..."


def command_signature(command: str) -> str:
    return " ".join(command.split())


def is_test_command(action: str) -> bool:
    lowered = action.lower()
    markers = ("pytest", "py.test", "unittest", "doctest", "nosetests", "tox", "nox", "manage.py test", "django test", "runtests.py", "make test", "make check", "cargo test", "go test", "npm test", "pnpm test", "yarn test", "bazel test", "ctest", "mvn test", "gradle test", "rspec", "phpunit", "dotnet test", "prove ", "verify", "verification", "smoke", "regression")
    return any(marker in lowered for marker in markers)


def action_may_change_code(action: str) -> bool:
    patterns = (r"\bsed\s+-i\b", r"\bperl\s+-\S*i\b", r"\bapply_patch\b", r"\bpython3?\b.*\bwrite_text\b", r"\bpython3?\b.*\bopen\([^)]*,\s*['\"]w", r"\bcat\s+<<.*>", r">\s*[^&\s]+", r"\btee\s+", r"\bnpm\s+install\b", r"\byarn\s+add\b", r"\bpip\s+install\b")
    return any(re.search(pattern, action, re.S) for pattern in patterns)


def extract_write_targets(action: str) -> set[str]:
    targets = set()
    normalized = action.replace("\\\n", " ")
    patterns = (r"sed\s+-i(?:\s+['\"][\s\S]*?['\"])+\s+([^\s|;&]+)", r"(?:cp|mv)\s+[^\s|;&]+\s+([^\s|;&]+)", r"tee\s+([^\s|;&]+)", r">\s*([^\s&;]+)", r"(?:python|python3)\s+-c\s+['\"].*?open\((['\"])([^'\"]+)\1,\s*['\"](?:w|a)", r"(?:Path|pathlib\.Path)\(\s*(['\"])([^'\"]+)\1\s*\)\.write_text\(")
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            path = normalize_path(match.group(match.lastindex or 1))
            if looks_like_source_file(path):
                targets.add(path)
    return targets


def is_inspection(action: str) -> bool:
    return not action_may_change_code(action) and not is_test_command(action) and bool(extract_read_targets(action) or extract_search_signature(action))


def ranges_overlap(left: str, right: str) -> bool:
    if "search" in {left, right}:
        return False
    if left in {"full", "tail"}:
        return True
    if right in {"full", "tail"}:
        return False
    left_match, right_match = re.fullmatch(r"(\d+)-(\d+)", left), re.fullmatch(r"(\d+)-(\d+)", right)
    if not left_match or not right_match:
        return left == right
    left_start, left_end = map(int, left_match.groups())
    right_start, right_end = map(int, right_match.groups())
    return (min(left_end, right_end) - max(left_start, right_start) + 1) / (right_end - right_start + 1) >= 0.8


def has_repeated_evidence(
    action: str,
    reads: dict[tuple[str, str], dict],
    searches: dict[str, dict],
    tests: list[dict],
    edits: dict[str, dict],
) -> bool:
    """Mirror _trace_status_repeat_block_message without formatting its prose."""
    last_edit_step = max((entry["last_step"] for entry in edits.values()), default=0)
    for path, line_range in extract_read_targets(action):
        for entry in reads.values():
            if (
                entry["last_step"] > last_edit_step
                and entry["file"] == path
                and ranges_overlap(entry["line_range"], line_range)
            ):
                return True
    signature = extract_search_signature(action)
    if signature and signature in searches and searches[signature]["last_step"] > last_edit_step:
        return True
    if is_test_command(action):
        signature = command_signature(action)
        return any(test["signature"] == signature and test["step"] > last_edit_step for test in tests)
    return False


def record_executed_action(
    action: str,
    step: int,
    reads: dict[tuple[str, str], dict],
    searches: dict[str, dict],
    edits: dict[str, dict],
    tests: list[dict],
) -> None:
    """Mirror the trace-state portions of _update_trace_status_bar."""
    if action_may_change_code(action):
        for target in extract_write_targets(action) or {"<unknown source edit>"}:
            entry = edits.setdefault(target, {"target": target, "first_step": step, "last_step": step, "count": 0})
            entry["last_step"] = step
            entry["count"] += 1
        return

    for path, line_range in extract_read_targets(action):
        key = (path, line_range)
        entry = reads.setdefault(
            key, {"file": path, "line_range": line_range, "first_step": step, "last_step": step, "count": 0}
        )
        entry["last_step"] = step
        entry["count"] += 1
    signature = extract_search_signature(action)
    if signature:
        entry = searches.setdefault(
            signature, {"signature": signature, "first_step": step, "last_step": step, "count": 0}
        )
        entry["last_step"] = step
        entry["count"] += 1
    if is_test_command(action):
        tests.append({"step": step, "signature": command_signature(action)})
        del tests[:-8]  # AgentConfig.trace_status_bar_max_items defaults to 8.


def replay(path: Path, max_consecutive_inspections: int, max_test_phase_tests: int) -> TraceResult:
    trajectory = json.loads(path.read_text())
    actions = assistant_actions(trajectory)
    reads: dict[tuple[str, str], dict] = {}
    searches: dict[str, dict] = {}
    edits: dict[str, dict] = {}
    tests: list[dict] = []
    consecutive_inspections = 0
    test_phase_tests = 0
    block_kinds: Counter[str] = Counter()
    executable_actions = 0
    overrides = 0

    for step, action in enumerate(actions, start=1):
        override = "TRACE_STATUS_REPEAT_CONFIRMED" in action
        if override:
            overrides += 1
        inspection = is_inspection(action)
        if not override and inspection and consecutive_inspections >= max_consecutive_inspections:
            block_kinds["consecutive_inspection"] += 1
            continue
        if not override and is_test_command(action) and test_phase_tests >= max_test_phase_tests:
            block_kinds["test_phase"] += 1
            continue
        if not override and has_repeated_evidence(action, reads, searches, tests, edits):
            block_kinds["repeated_evidence"] += 1
            continue

        executable_actions += 1
        if inspection:
            consecutive_inspections += 1
        else:
            consecutive_inspections = 0
        if action_may_change_code(action):
            test_phase_tests = 0
        elif is_test_command(action):
            test_phase_tests += 1
        record_executed_action(action, step, reads, searches, edits, tests)

    return TraceResult(
        instance_id=trajectory.get("instance_id", path.stem),
        trajectory=str(path),
        actions=len(actions),
        executable_actions=executable_actions,
        would_block=sum(block_kinds.values()),
        repeated_evidence_blocks=block_kinds["repeated_evidence"],
        consecutive_inspection_blocks=block_kinds["consecutive_inspection"],
        test_phase_blocks=block_kinds["test_phase"],
        override_actions=overrides,
    )


def trajectories(root: Path, all_traces: bool) -> list[Path]:
    paths = sorted(
        (path for path in root.glob("**/*.traj.json")),
        key=lambda path: path.stat().st_mtime,
    )
    if all_traces:
        return paths
    newest: dict[str, Path] = {}
    for path in paths:
        try:
            instance_id = json.loads(path.read_text()).get("instance_id", path.stem)
        except json.JSONDecodeError:
            continue
        newest[instance_id] = path
    return sorted(newest.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", nargs="?", type=Path, default=REPO_ROOT / "logs" / "org")
    parser.add_argument("--all-traces", action="store_true", help="analyze all saved reruns instead of latest trace per task")
    parser.add_argument("--max-consecutive-inspections", type=int, default=10)
    parser.add_argument("--max-test-phase-tests", type=int, default=18)
    parser.add_argument("--csv", type=Path, help="write the per-task report to this CSV path")
    args = parser.parse_args()
    if args.max_consecutive_inspections < 1:
        parser.error("--max-consecutive-inspections must be at least 1")
    if args.max_test_phase_tests < 1:
        parser.error("--max-test-phase-tests must be at least 1")

    paths = trajectories(args.trace_root, args.all_traces)
    if not paths:
        parser.error(f"no trajectory files found under {args.trace_root}")
    results = [replay(path, args.max_consecutive_inspections, args.max_test_phase_tests) for path in paths]
    results.sort(key=lambda row: (-row.would_block, row.instance_id))

    print(
        f"traces={len(results)} mode={'all' if args.all_traces else 'latest-per-task'} "
        f"inspection_threshold={args.max_consecutive_inspections} test_phase_threshold={args.max_test_phase_tests}"
    )
    print("instance_id, actions, would_block, repeated_evidence, consecutive_inspection, test_phase, overrides")
    for row in results:
        print(
            f"{row.instance_id}, {row.actions}, {row.would_block}, {row.repeated_evidence_blocks}, "
            f"{row.consecutive_inspection_blocks}, {row.test_phase_blocks}, {row.override_actions}"
        )
    totals = {
        field: sum(getattr(row, field) for row in results)
        for field in (
            "actions",
            "would_block",
            "repeated_evidence_blocks",
            "consecutive_inspection_blocks",
            "test_phase_blocks",
            "override_actions",
        )
    }
    print(
        "TOTAL, "
        f"actions={totals['actions']}, would_block={totals['would_block']}, "
        f"repeated_evidence={totals['repeated_evidence_blocks']}, "
        f"consecutive_inspection={totals['consecutive_inspection_blocks']}, "
        f"test_phase={totals['test_phase_blocks']}, overrides={totals['override_actions']}"
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=TraceResult.__dataclass_fields__.keys())
            writer.writeheader()
            writer.writerows(asdict(row) for row in results)


if __name__ == "__main__":
    main()
