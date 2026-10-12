"""Label-inspired online repetition detector for convergence nudges."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Deque, Optional


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


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
    return any(re.search(pattern, cmd) for pattern in patterns)


def _classify_action(command: str) -> str:
    cmd = (command or "").lower()
    if "pytest" in cmd or " test" in cmd or "verify" in cmd:
        return "test"
    if "sed -i" in cmd or "apply_patch" in cmd or "git add" in cmd:
        return "patch"
    if "complete_task_and_submit_final_output" in cmd:
        return "submit"
    if _looks_like_repeated_read_command(cmd):
        return "inspect_read"
    return "inspect"


def _extract_primary_file_and_range(command: str) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    cmd = command or ""
    file_match = re.search(r"([./\w\-/]+\.py)", cmd)
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


def _normalize_test_command(command: str) -> Optional[str]:
    cmd = (command or "").strip()
    if not cmd or _classify_action(cmd) != "test":
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
    signature = _normalize_text(output_text)[:240]
    return signature or f"returncode:{returncode}"


@dataclass
class RoundSignal:
    action: str
    action_class: str
    output_signature: str
    returncode: int
    output_len: int
    file_path: Optional[str]
    line_range: Optional[tuple[int, int]]
    normalized_test_cmd: Optional[str]
    failure_signature: Optional[str]


class LabelBasedRepetitionDetector:
    """Approximate label-based repetition patterns online and trigger nudges."""

    def __init__(self, history_size: int = 4, enable_read_loop_nudge: bool = False) -> None:
        self._history: Deque[RoundSignal] = deque(maxlen=max(6, history_size + 2))
        self._recent_action_classes: Deque[str] = deque(maxlen=8)
        self._enable_read_loop_nudge = bool(enable_read_loop_nudge)
        self._round_index = 0
        self._last_nudge_round = -10_000
        self._last_nudge_reason: Optional[str] = None
        self._patch_seen = False
        self._test_budget_used = 0
        self._test_cmd_success_counts: dict[str, int] = {}
        self._seen_failure_signatures: set[str] = set()
        self._new_failure_since_patch = False
        self._low_info_streak = 0
        self._same_focus_streak = 0
        self._repeat_read_streak = 0
        self._verified_passed = False
        self._post_verify_test_count = 0
        self._post_patch_action_window: Deque[str] = deque(maxlen=6)

    def reset(self) -> None:
        self._history.clear()
        self._recent_action_classes.clear()
        self._round_index = 0
        self._last_nudge_round = -10_000
        self._last_nudge_reason = None
        self._patch_seen = False
        self._test_budget_used = 0
        self._test_cmd_success_counts.clear()
        self._seen_failure_signatures.clear()
        self._new_failure_since_patch = False
        self._low_info_streak = 0
        self._same_focus_streak = 0
        self._repeat_read_streak = 0
        self._verified_passed = False
        self._post_verify_test_count = 0
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

    def _same_focus(self, current: RoundSignal, previous: RoundSignal) -> bool:
        if current.file_path and previous.file_path and current.file_path == previous.file_path:
            if current.line_range and previous.line_range:
                return _range_overlap_ratio(current.line_range, previous.line_range) >= 0.55
            return True
        if current.normalized_test_cmd and previous.normalized_test_cmd:
            return _text_similarity(current.normalized_test_cmd, previous.normalized_test_cmd) >= 0.88
        return False

    def _has_new_evidence(self, current: RoundSignal, previous: Optional[RoundSignal]) -> bool:
        if previous is None:
            return True
        if current.returncode != previous.returncode:
            return True
        if current.failure_signature and current.failure_signature not in self._seen_failure_signatures:
            return True
        if current.file_path and current.file_path != previous.file_path:
            return True
        return False

    def _is_low_info_step(self, current: RoundSignal, previous: Optional[RoundSignal]) -> bool:
        if previous is None:
            return False
        if current.action_class in {"patch", "submit"}:
            return False
        if not self._same_focus(current, previous):
            return False
        if self._has_new_evidence(current, previous):
            return False
        similar_output = _text_similarity(current.output_signature, previous.output_signature) >= 0.88
        same_test = (
            current.action_class == "test"
            and previous.action_class == "test"
            and current.normalized_test_cmd
            and current.normalized_test_cmd == previous.normalized_test_cmd
        )
        repeated_read = current.action_class == "inspect_read" and previous.action_class == "inspect_read"
        return similar_output or same_test or repeated_read

    def _is_read_loop_detected(self, signal: RoundSignal, previous: Optional[RoundSignal]) -> bool:
        if not self._enable_read_loop_nudge or signal.action_class != "inspect_read":
            return False
        if any(action in {"patch", "test", "submit"} for action in self._recent_action_classes):
            return False
        if previous is None or previous.action_class != "inspect_read":
            return False
        read_signals = [item for item in self._history if item.action_class == "inspect_read"]
        if len(read_signals) < 4 or not signal.file_path:
            return False
        same_file_reads = 0
        overlap_reads = 0
        for prev in read_signals[-5:]:
            if prev.file_path == signal.file_path:
                same_file_reads += 1
                if signal.line_range and prev.line_range and _range_overlap_ratio(signal.line_range, prev.line_range) >= 0.60:
                    overlap_reads += 1
        return same_file_reads >= 3 and overlap_reads >= 2

    def _is_low_information_patch_test_loop(self, signal: RoundSignal, previous: Optional[RoundSignal]) -> bool:
        if signal.action_class != "test" or previous is None:
            return False
        window = list(self._post_patch_action_window)
        if len(window) < 4:
            return False
        patch_count = sum(1 for action in window if action == "patch")
        test_count = sum(1 for action in window if action == "test")
        alternating_tail = window[-4:] in (["patch", "test", "patch", "test"], ["test", "patch", "test", "patch"])
        similar_output = previous.action_class == "test" and _text_similarity(signal.output_signature, previous.output_signature) >= 0.88
        return patch_count >= 2 and test_count >= 2 and alternating_tail and similar_output and not self._new_failure_since_patch

    def update(self, *, response_content: str, action: str, output: dict) -> dict:
        del response_content  # This detector intentionally relies on behavior rather than phrasing.
        self._round_index += 1
        output_text = (output or {}).get("output", "") or ""
        returncode = int((output or {}).get("returncode", 1) or 1)
        file_path, line_range = _extract_primary_file_and_range(action or "")
        signal = RoundSignal(
            action=action or "",
            action_class=_classify_action(action or ""),
            output_signature=_normalize_text(output_text)[:600],
            returncode=returncode,
            output_len=len(output_text),
            file_path=file_path,
            line_range=line_range,
            normalized_test_cmd=_normalize_test_command(action or ""),
            failure_signature=_extract_failure_signature(output_text, returncode),
        )

        previous = self._history[-1] if self._history else None
        low_info_step = self._is_low_info_step(signal, previous)
        same_focus_step = bool(previous is not None and self._same_focus(signal, previous))
        read_loop_detected = self._is_read_loop_detected(signal, previous)

        self._low_info_streak = self._low_info_streak + 1 if low_info_step else 0
        self._same_focus_streak = self._same_focus_streak + 1 if same_focus_step else 0
        self._repeat_read_streak = (
            self._repeat_read_streak + 1
            if signal.action_class == "inspect_read" and same_focus_step
            else 0
        )

        if signal.action_class == "patch":
            self._patch_seen = True
            self._test_budget_used = 0
            self._test_cmd_success_counts.clear()
            self._new_failure_since_patch = False
            self._low_info_streak = 0
            self._verified_passed = False
            self._post_verify_test_count = 0
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
            if signal.returncode == 0 and not self._new_failure_since_patch:
                if self._verified_passed:
                    self._post_verify_test_count += 1
                else:
                    self._verified_passed = True
                    self._post_verify_test_count = 0
            elif signal.returncode != 0:
                self._verified_passed = False
                self._post_verify_test_count = 0

        triggered = False
        trigger_reason = ""
        if read_loop_detected:
            triggered = True
            trigger_reason = "read_loop_detected"
        elif (
            self._same_focus_streak >= 4
            and self._low_info_streak >= 3
            and signal.action_class in {"inspect", "inspect_read", "test"}
            and not self._patch_seen
        ):
            triggered = True
            trigger_reason = "root_cause_locked_without_patch"
        elif (
            self._same_focus_streak >= 5
            and self._low_info_streak >= 3
            and signal.action_class in {"inspect", "inspect_read", "test"}
        ):
            triggered = True
            trigger_reason = "sliding_window_repeat_hits"
        elif (
            self._patch_seen
            and self._verified_passed
            and signal.action_class == "test"
            and signal.returncode == 0
            and self._post_verify_test_count > 2
            and not self._new_failure_since_patch
        ):
            triggered = True
            trigger_reason = "post_verify_over_testing"
        elif self._patch_seen and self._is_low_information_patch_test_loop(signal, previous):
            triggered = True
            trigger_reason = "patch_test_loop_low_info"

        if triggered and not self._allow_nudge(trigger_reason):
            triggered = False
            trigger_reason = ""
        if triggered and trigger_reason:
            self._mark_nudge_fired(trigger_reason)

        self._recent_action_classes.append(signal.action_class)
        self._history.append(signal)
        return {
            "repeat_score": int(same_focus_step) + int(low_info_step),
            "repeat_hit": low_info_step,
            "triggered": triggered,
            "trigger_reason": trigger_reason,
            "same_root_cause": False,
            "same_code_region": same_focus_step,
            "same_symbol_focus": False,
            "bigger_test_no_new_info": (
                signal.action_class == "test"
                and previous is not None
                and previous.action_class == "test"
                and _text_similarity(signal.output_signature, previous.output_signature) >= 0.88
            ),
            "root_cause_locked": False,
            "rounds_since_root_cause_lock": self._same_focus_streak,
            "test_budget_used": self._test_budget_used,
            "verified_passed": self._verified_passed,
            "post_verify_test_count": self._post_verify_test_count,
            "post_patch_action_window": list(self._post_patch_action_window),
            "normalized_test_cmd": signal.normalized_test_cmd,
            "new_failure_since_patch": self._new_failure_since_patch,
            "round_index": self._round_index,
            "low_info_streak": self._low_info_streak,
            "same_focus_streak": self._same_focus_streak,
        }
