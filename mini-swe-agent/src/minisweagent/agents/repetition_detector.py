"""Rule-based trace repetition detector for convergence nudges."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Deque, Optional


ROOT_CAUSE_PATTERNS = [
    r"\bthe problem is\b",
    r"\broot cause\b",
    r"\bplaceholder\b",
    r"\bshould use\b",
    r"\bshould instead use\b",
    r"\bactual world value\b",
    r"\bactual world coordinate\b",
    r"\bline\s+\d+\b",
]


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _extract_primary_file_and_range(command: str) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    cmd = command or ""
    file_match = re.search(r"(/[\w\-/\.]+\.py)", cmd)
    if not file_match:
        return None, None
    file_path = file_match.group(1)

    range_match = re.search(r"sed\s+-n\s+'(\d+),(\d+)p'", cmd)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if start > end:
            start, end = end, start
        return file_path, (start, end)

    single_line_match = re.search(r":(\d+)(?::\d+)?", cmd)
    if single_line_match:
        line_no = int(single_line_match.group(1))
        return file_path, (line_no, line_no)
    return file_path, None


def _range_overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0) + 1
    if union <= 0:
        return 0.0
    return inter / union


def _classify_action(command: str) -> str:
    cmd = (command or "").lower()
    if "complete_task_and_submit_final_output" in cmd:
        return "submit"
    if _looks_like_test_command(cmd):
        return "test"
    if _looks_like_patch_command(cmd):
        return "patch"
    if _looks_like_repeated_read_command(cmd):
        return "inspect_read"
    return "inspect"


def _looks_like_test_command(cmd: str) -> bool:
    runner_markers = [
        "pytest",
        "py.test",
        "python -m pytest",
        "python -m unittest",
        "python -m doctest",
        "python -m django test",
        "manage.py test",
        "tests/runtests.py",
        "runtests.py",
        "nosetests",
        "tox",
        "nox",
        "cargo test",
        "go test",
        "npm test",
        "pnpm test",
        "yarn test",
        "bazel test",
        "ctest",
        "mvn test",
        "gradle test",
        "rspec",
        "bundle exec rspec",
        "phpunit",
        "dotnet test",
        "prove ",
    ]
    if any(marker in cmd for marker in runner_markers):
        return True

    if "verify" in cmd or "verification" in cmd or "regression" in cmd or "smoke" in cmd:
        return True

    python_runner_markers = ["python ", "python3 ", "py ", "ipython ", "pypy "]
    if any(marker in cmd for marker in python_runner_markers):
        verification_markers = [
            "assert ",
            "assert(",
            "raise systemexit",
            "sys.exit(",
                "doctest",
                "unittest",
                "import pytest",
                "from pytest",
            ]
        if any(marker in cmd for marker in verification_markers):
            return True
    return False


def _looks_like_patch_command(cmd: str) -> bool:
    direct_patch_markers = [
        "sed -i",
        "apply_patch",
        "git add",
        "perl -0pi",
        "perl -pi",
    ]
    if any(marker in cmd for marker in direct_patch_markers):
        return True

    python_runner_markers = ["python ", "python3 ", "py "]
    if any(marker in cmd for marker in python_runner_markers):
        write_markers = [
            ".write(",
            ".writelines(",
            "write_text(",
            "write_bytes(",
            "open(",
        ]
        write_modes = ["'w'", '"w"', "'a'", '"a"', "'r+'", '"r+"']
        source_targets = [".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb", ".php", ".c", ".cc", ".cpp"]
        if any(marker in cmd for marker in write_markers) and any(mode in cmd for mode in write_modes) and any(
            target in cmd for target in source_targets
        ):
            return True

    redirect_match = re.search(r">+\s*([./\w\-/]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|php|c|cc|cpp))\b", cmd)
    if redirect_match:
        return True

    return False


def _looks_like_repeated_read_command(cmd: str) -> bool:
    patterns = [
        r"\bsed\s+-n\b",
        r"\bcat\b",
        r"\bgrep\b",
        r"\brg\b",
        r"\bhead\b",
        r"\btail\b",
        r"\bnl\s+-ba\b",
    ]
    return any(re.search(p, cmd) for p in patterns)


def _extract_generic_symbol_focus(text: str) -> set[str]:
    normalized = _normalize_text(text)
    snake = set(re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", normalized))
    camel = set(re.findall(r"\b[a-z]+(?:[A-Z][a-z0-9]+){1,}\b", text or ""))
    path_parts = set(re.findall(r"\b[a-z0-9_\-]+(?:/[a-z0-9_\-]+){1,}\b", normalized))
    keep = set()
    for token in snake | camel:
        if len(token) >= 4:
            keep.add(token.lower())
    for part in path_parts:
        for seg in part.split("/"):
            if len(seg) >= 4 and re.match(r"^[a-z0-9_\-]+$", seg):
                keep.add(seg.lower())
    return keep


def _extract_symbol_focus(text: str) -> set[str]:
    normalized = _normalize_text(text)
    symbols = set(re.findall(r"\b[a-z_][a-z0-9_]{2,}\b", normalized))
    keep = {
        token
        for token in symbols
        if (
            token.endswith("_values")
            or token.endswith("_wcs")
            or token in {"slicedlowlevelwcs", "slicedlowlevelwcs", "world_to_pixel_values", "pixel_to_world_values"}
            or "_" in token
        )
    }
    # Keep existing domain-specific heuristics and add generic symbols.
    return keep | _extract_generic_symbol_focus(text)


def _normalize_test_command(command: str) -> Optional[str]:
    cmd = (command or "").strip()
    if not cmd:
        return None
    if _classify_action(cmd) != "test":
        return None
    lowered = cmd.lower()
    lowered = re.sub(r"^\s*cd\s+[^&;]+&&\s*", "", lowered)
    lowered = re.sub(r"\|\s*tail\b[^|;&]*", " ", lowered)
    lowered = re.sub(r"\|\s*head\b[^|;&]*", " ", lowered)
    lowered = re.sub(r"\s+\d?>{1,2}\s*\S+", " ", lowered)
    lowered = re.sub(r"\s*2>&1\s*", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered or None


def _extract_failure_signature(output_text: str, returncode: int) -> Optional[str]:
    if returncode == 0:
        return None
    sig = _normalize_text(output_text)[:240]
    return sig or f"returncode:{returncode}"


def _extract_returncode(output: dict) -> int:
    try:
        return int((output or {}).get("returncode", 1))
    except (TypeError, ValueError):
        return 1


def _test_output_looks_failed(output_text: str) -> bool:
    normalized = _normalize_text(output_text)
    if not normalized:
        return False
    failure_patterns = [
        r"\btraceback\b",
        r"\bno module named\b",
        r"\bimporterror\b",
        r"\bruntimeerror\b",
        r"\battributeerror\b",
        r"\bassertionerror\b",
        r"\bfailed\s*\(",
        r"\b[1-9]\d*\s+failed\b",
        r"\b(errors?|failures?)=[1-9]\d*\b",
        r"\berror:\s",
    ]
    return any(re.search(pattern, normalized) for pattern in failure_patterns)


def _jaccard_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _extract_line_hint(text: str) -> Optional[int]:
    match = re.search(r"\bline\s+(\d+)\b", _normalize_text(text))
    if not match:
        return None
    return int(match.group(1))


def _looks_like_root_cause_locked(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in ROOT_CAUSE_PATTERNS)


@dataclass
class RoundSignal:
    reasoning: str
    action: str
    action_class: str
    output_signature: str
    returncode: int
    output_len: int
    file_path: Optional[str]
    line_range: Optional[tuple[int, int]]
    symbol_focus: set[str]
    line_hint: Optional[int]
    root_cause_locked: bool
    normalized_test_cmd: Optional[str]
    failure_signature: Optional[str]


class RuleBasedRepetitionDetector:
    """Detect repeated low-information rounds using a state machine and sliding window."""

    def __init__(self, history_size: int = 4, enable_read_loop_nudge: bool = False) -> None:
        self._history: Deque[RoundSignal] = deque(maxlen=max(4, history_size))
        self._window_hits: Deque[dict] = deque(maxlen=max(4, history_size))
        self._recent_action_classes: Deque[str] = deque(maxlen=8)
        self._enable_read_loop_nudge = bool(enable_read_loop_nudge)
        self._root_cause_locked = False
        self._rounds_since_root_cause_lock = 0
        self._test_budget_used = 0
        self._patch_seen = False
        self._test_cmd_success_counts: dict[str, int] = {}
        self._seen_failure_signatures: set[str] = set()
        self._new_failure_since_patch = False
        self._round_index = 0
        self._last_nudge_round = -10_000
        self._last_nudge_reason: Optional[str] = None
        self._verified_passed = False
        self._post_verify_test_count = 0
        self._post_verify_inspection_count = 0
        self._post_patch_action_window: Deque[str] = deque(maxlen=6)

    def reset(self) -> None:
        self._history.clear()
        self._window_hits.clear()
        self._recent_action_classes.clear()
        self._root_cause_locked = False
        self._rounds_since_root_cause_lock = 0
        self._test_budget_used = 0
        self._patch_seen = False
        self._test_cmd_success_counts.clear()
        self._seen_failure_signatures.clear()
        self._new_failure_since_patch = False
        self._round_index = 0
        self._last_nudge_round = -10_000
        self._last_nudge_reason = None
        self._verified_passed = False
        self._post_verify_test_count = 0
        self._post_verify_inspection_count = 0
        self._post_patch_action_window.clear()

    def _allow_nudge(self, reason: str) -> bool:
        min_global_gap = 3
        if self._round_index - self._last_nudge_round < min_global_gap:
            return False
        min_same_reason_gap = 6
        if reason == self._last_nudge_reason and self._round_index - self._last_nudge_round < min_same_reason_gap:
            return False
        return True

    def _mark_nudge_fired(self, reason: str) -> None:
        self._last_nudge_round = self._round_index
        self._last_nudge_reason = reason

    def _compare_with_previous(self, signal: RoundSignal) -> dict:
        if not self._history:
            return {
                "same_root_cause": False,
                "same_code_region": False,
                "same_symbol_focus": False,
                "bigger_test_no_new_info": False,
                "repeat_score": 0,
                "repeat_hit": False,
            }

        prev = self._history[-1]
        same_root_cause = _text_similarity(signal.reasoning, prev.reasoning) >= 0.84

        same_code_region = False
        if signal.file_path and prev.file_path and signal.file_path == prev.file_path:
            if signal.line_range and prev.line_range:
                same_code_region = _range_overlap_ratio(signal.line_range, prev.line_range) >= 0.55
            else:
                same_code_region = True

        same_symbol_focus = False
        if signal.symbol_focus and prev.symbol_focus:
            overlap = signal.symbol_focus & prev.symbol_focus
            same_symbol_focus = len(overlap) >= 1 or _jaccard_overlap(signal.symbol_focus, prev.symbol_focus) >= 0.30

        heavier_test = (
            signal.action_class == "test"
            and prev.action_class == "test"
            and signal.output_len >= int(prev.output_len * 1.05)
        )
        no_new_info = (
            signal.returncode == prev.returncode
            and _text_similarity(signal.output_signature, prev.output_signature) >= 0.88
        )
        bigger_test_no_new_info = heavier_test and no_new_info

        repeat_score = sum(
            [
                same_root_cause,
                same_code_region or same_symbol_focus,
                bigger_test_no_new_info,
            ]
        )
        return {
            "same_root_cause": same_root_cause,
            "same_code_region": same_code_region,
            "same_symbol_focus": same_symbol_focus,
            "bigger_test_no_new_info": bigger_test_no_new_info,
            "repeat_score": repeat_score,
            "repeat_hit": repeat_score >= 2,
        }

    def _is_low_information_patch_test_loop(self, signal: RoundSignal, comparison: dict) -> bool:
        if signal.action_class not in {"patch", "test"}:
            return False
        window = list(self._post_patch_action_window)
        if len(window) < 4:
            return False
        patch_count = sum(1 for action in window if action == "patch")
        test_count = sum(1 for action in window if action == "test")
        alternating_tail = window[-4:] in (["patch", "test", "patch", "test"], ["test", "patch", "test", "patch"])
        low_info_test = (
            signal.action_class == "test"
            and not self._new_failure_since_patch
            and (
                comparison.get("bigger_test_no_new_info")
                or comparison.get("same_code_region")
                or comparison.get("same_symbol_focus")
            )
        )
        return patch_count >= 2 and test_count >= 2 and alternating_tail and low_info_test

    def _should_trigger_nudge(self, signal: RoundSignal, comparison: dict) -> tuple[bool, str]:
        window = list(self._window_hits)
        recent_root_repeats = sum(1 for item in window if item.get("same_root_cause"))
        recent_symbol_repeats = sum(
            1 for item in window if item.get("same_code_region") or item.get("same_symbol_focus")
        )
        recent_repeat_hits = sum(1 for item in window if item.get("repeat_hit"))
        if self._root_cause_locked and signal.action_class in {"inspect", "test"}:
            if self._rounds_since_root_cause_lock >= 4 and recent_repeat_hits >= 1:
                return True, "root_cause_locked_without_patch"

        if recent_root_repeats >= 3 and recent_symbol_repeats >= 2:
            return True, "sliding_window_repeated_root_cause"

        if recent_repeat_hits >= 3:
            return True, "sliding_window_repeat_hits"

        if (
            self._patch_seen
            and self._verified_passed
            and signal.action_class in {"inspect", "inspect_read"}
            and signal.returncode == 0
            and self._post_verify_inspection_count >= 2
            and not self._new_failure_since_patch
        ):
            return True, "post_verify_low_info_inspection"

        if (
            self._patch_seen
            and self._verified_passed
            and signal.action_class == "test"
            and signal.returncode == 0
            and self._post_verify_test_count > 2
            and not self._new_failure_since_patch
        ):
            return True, "post_verify_over_testing"

        if self._patch_seen and self._is_low_information_patch_test_loop(signal, comparison):
            return True, "patch_test_loop_low_info"

        if self._enable_read_loop_nudge and self._is_read_loop_detected(signal):
            return True, "read_loop_detected"

        return False, ""

    def _is_read_loop_detected(self, signal: RoundSignal) -> bool:
        if signal.action_class != "inspect_read":
            return False
        if not self._history:
            return False
        # Avoid false positives right after meaningful progress.
        if any(a in {"patch", "test", "submit"} for a in self._recent_action_classes):
            return False

        read_signals = [h for h in self._history if h.action_class == "inspect_read"]
        if len(read_signals) < 5:
            return False

        same_file_reads = 0
        overlap_reads = 0
        for prev in read_signals[-6:]:
            if signal.file_path and prev.file_path and signal.file_path == prev.file_path:
                same_file_reads += 1
                if signal.line_range and prev.line_range and _range_overlap_ratio(signal.line_range, prev.line_range) >= 0.60:
                    overlap_reads += 1
        return same_file_reads >= 4 and overlap_reads >= 2

    def update(self, *, response_content: str, action: str, output: dict) -> dict:
        self._round_index += 1
        reasoning = _normalize_text(response_content)
        output_text = (output or {}).get("output", "") or ""
        output_signature = _normalize_text(output_text)[:600]
        returncode = _extract_returncode(output or {})
        action_class = _classify_action(action or "")
        if action_class == "test" and returncode == 0 and _test_output_looks_failed(output_text):
            returncode = 1
        file_path, line_range = _extract_primary_file_and_range(action)
        signal = RoundSignal(
            reasoning=reasoning,
            action=action or "",
            action_class=action_class,
            output_signature=output_signature,
            returncode=returncode,
            output_len=len(output_text),
            file_path=file_path,
            line_range=line_range,
            symbol_focus=_extract_symbol_focus(f"{response_content}\n{action}"),
            line_hint=_extract_line_hint(response_content),
            root_cause_locked=_looks_like_root_cause_locked(response_content),
            normalized_test_cmd=_normalize_test_command(action or ""),
            failure_signature=_extract_failure_signature(output_text, returncode),
        )

        if signal.root_cause_locked:
            self._root_cause_locked = True
            self._rounds_since_root_cause_lock = 0
        else:
            if self._root_cause_locked and signal.action_class != "patch":
                self._rounds_since_root_cause_lock += 1

        if signal.action_class == "patch":
            self._patch_seen = True
            self._root_cause_locked = False
            self._rounds_since_root_cause_lock = 0
            self._test_budget_used = 0
            self._test_cmd_success_counts.clear()
            self._new_failure_since_patch = False
            self._verified_passed = False
            self._post_verify_test_count = 0
            self._post_verify_inspection_count = 0
            self._post_patch_action_window.append("patch")
        elif self._patch_seen and signal.action_class == "test":
            self._test_budget_used += 1
            self._post_patch_action_window.append("test")
            if signal.normalized_test_cmd:
                if signal.returncode == 0:
                    self._test_cmd_success_counts[signal.normalized_test_cmd] = (
                        self._test_cmd_success_counts.get(signal.normalized_test_cmd, 0) + 1
                    )
                else:
                    self._test_cmd_success_counts[signal.normalized_test_cmd] = 0
            if signal.failure_signature:
                if signal.failure_signature not in self._seen_failure_signatures:
                    self._new_failure_since_patch = True
                self._seen_failure_signatures.add(signal.failure_signature)
            if signal.returncode == 0:
                self._new_failure_since_patch = False
                if self._verified_passed:
                    self._post_verify_test_count += 1
                else:
                    self._verified_passed = True
                    self._post_verify_test_count = 0
                self._post_verify_inspection_count = 0
            elif signal.returncode != 0:
                self._verified_passed = False
                self._post_verify_test_count = 0
                self._post_verify_inspection_count = 0
        elif (
            self._patch_seen
            and self._verified_passed
            and signal.action_class in {"inspect", "inspect_read"}
            and signal.returncode == 0
            and not self._new_failure_since_patch
        ):
            self._post_verify_inspection_count += 1
        elif self._patch_seen and self._verified_passed and signal.action_class in {"inspect", "inspect_read"}:
            self._post_verify_inspection_count = 0

        comparison = self._compare_with_previous(signal)
        self._window_hits.append(comparison)
        triggered, trigger_reason = self._should_trigger_nudge(signal, comparison)
        if triggered and not self._allow_nudge(trigger_reason):
            triggered, trigger_reason = False, ""
        if triggered and trigger_reason:
            self._mark_nudge_fired(trigger_reason)

        self._recent_action_classes.append(signal.action_class)
        self._history.append(signal)
        return {
            "repeat_score": comparison["repeat_score"],
            "repeat_hit": comparison["repeat_hit"],
            "triggered": triggered,
            "trigger_reason": trigger_reason,
            "same_root_cause": comparison["same_root_cause"],
            "same_code_region": comparison["same_code_region"],
            "same_symbol_focus": comparison["same_symbol_focus"],
            "bigger_test_no_new_info": comparison["bigger_test_no_new_info"],
            "root_cause_locked": self._root_cause_locked,
            "rounds_since_root_cause_lock": self._rounds_since_root_cause_lock,
            "test_budget_used": self._test_budget_used,
            "verified_passed": self._verified_passed,
            "post_verify_test_count": self._post_verify_test_count,
            "post_verify_inspection_count": self._post_verify_inspection_count,
            "post_patch_action_window": list(self._post_patch_action_window),
            "normalized_test_cmd": signal.normalized_test_cmd,
            "new_failure_since_patch": self._new_failure_since_patch,
            "round_index": self._round_index,
        }
