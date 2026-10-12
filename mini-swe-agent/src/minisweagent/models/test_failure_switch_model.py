from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class TestFailureSwitchModelConfig:
    model_name: str
    fallback_model_name: str
    failure_threshold: int


class TestFailureSwitchModel:
    """Switch to a fallback model after repeated post-edit run failures."""

    _COMMAND_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\s*\n(?P<command>.*?)```", re.DOTALL | re.IGNORECASE)
    _RETURNCODE_RE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.IGNORECASE)
    _OUTPUT_RE = re.compile(r"<output(?:_head|_tail)?>\s*(?P<output>.*?)\s*</output(?:_head|_tail)?>", re.DOTALL | re.IGNORECASE)

    _RUN_COMMAND_RE = re.compile(
        r"(^|[\s;&|()])("
        r"python(?:\d+(?:\.\d+)?)?|pytest|py\.test|tox|nox|unittest|nosetests|"
        r"npm\s+test|npm\s+run\s+test|yarn\s+test|pnpm\s+test|"
        r"node|ruby|php|perl|bash|sh|"
        r"go\s+test|cargo\s+test|mvn\s+test|gradle\s+test|\./gradlew\s+test|"
        r"rspec|bundle\s+exec\s+rspec|phpunit|"
        r"\./[^\s;&|()]+"
        r")($|[\s;&|()])",
        re.IGNORECASE,
    )
    _ADHOC_TEST_COMMAND_RE = re.compile(
        r"(pytest|py\.test|tox|nox|unittest|nosetests|/tmp/test|test_[\w.-]+\.py|\bassert\b)",
        re.IGNORECASE,
    )
    _READ_SEARCH_COMMAND_RE = re.compile(
        r"^\s*(cd\s+\S+\s*&&\s*)?"
        r"(grep|rg|find|ls|pwd|cat|head|tail|nl|sed\s+-n|wc|git\s+(diff|status|show|log))"
        r"\b",
        re.IGNORECASE,
    )
    _EDIT_COMMAND_RE = re.compile(
        r"("
        r"\bsed\s+-i\b|\bperl\s+-pi\b|\bapply_patch\b|\bgit\s+apply\b|"
        r"\bcat\s+<<['\"]?\w+['\"]?\s*>|\btee\s+[-\w\s]*\s+[\w./-]+|"
        r">\s*[\w./-]+|>>\s*[\w./-]+|"
        r"\.write_text\s*\(|open\s*\([^)]*,\s*['\"][wa]"
        r")",
        re.IGNORECASE,
    )
    _FAILURE_OUTPUT_RE = re.compile(
        r"(FAILED \[|=+ FAILURES =+|Traceback \(most recent call last\)|"
        r"AssertionError|SyntaxError|ModuleNotFoundError|ImportError|"
        r"tests? failed|failed tests?|[1-9]\d*\s+(failed|errors?))",
        re.IGNORECASE,
    )

    def __init__(self, primary_model: Any, fallback_model: Any, *, failure_threshold: int):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.failure_threshold = max(1, int(failure_threshold))
        self.config = TestFailureSwitchModelConfig(
            model_name=self.primary_model.config.model_name,
            fallback_model_name=self.fallback_model.config.model_name,
            failure_threshold=self.failure_threshold,
        )
        self.last_failure_count = 0

    @property
    def cost(self) -> float:
        return float(self.primary_model.cost) + float(self.fallback_model.cost)

    @property
    def n_calls(self) -> int:
        return int(self.primary_model.n_calls) + int(self.fallback_model.n_calls)

    @property
    def switch_triggered(self) -> bool:
        return self.fallback_model.n_calls > 0

    def _active_model(self, messages: list[dict[str, Any]]):
        self.last_failure_count = self._post_edit_test_failure_count(messages)
        if self.last_failure_count >= self.failure_threshold:
            return self.fallback_model
        return self.primary_model

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict:
        active_model = self._active_model(messages)
        routed_messages = messages
        if active_model is self.fallback_model and self.fallback_model.n_calls == 0:
            routed_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "MODEL HANDOFF NOTICE: The previous model has modified code and then observed "
                        f"{self.failure_threshold} failing run command(s). Continue from the current "
                        "repository state, diagnose the remaining failure, and avoid repeating the same "
                        "unsuccessful fix."
                    ),
                }
            ]
        response = active_model.query(routed_messages, **kwargs)
        extra = response.setdefault("extra", {})
        if isinstance(extra, dict):
            extra["routed_model_name"] = active_model.config.model_name
            response_payload = extra.get("response")
            if isinstance(response_payload, dict):
                raw_model_name = response_payload.get("model")
                if not isinstance(raw_model_name, str) or not raw_model_name.strip():
                    response_payload["model"] = active_model.config.model_name
        return response

    def get_template_vars(self) -> dict[str, Any]:
        template_vars = dict(self.primary_model.get_template_vars())
        template_vars.update(
            {
                "model_name": self.primary_model.config.model_name,
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "test_failure_switch_enabled": True,
                "test_failure_switch_model_name": self.fallback_model.config.model_name,
                "test_failure_switch_threshold": self.failure_threshold,
                "test_failure_switch_triggered": self.switch_triggered,
                "test_failure_switch_failure_count": self.last_failure_count,
                "test_failure_switch_primary_calls": self.primary_model.n_calls,
                "test_failure_switch_fallback_calls": self.fallback_model.n_calls,
            }
        )
        return template_vars

    @classmethod
    def _extract_command(cls, message: dict[str, Any]) -> str:
        content = str(message.get("content") or "")
        match = cls._COMMAND_BLOCK_RE.search(content)
        if match:
            return match.group("command").strip()
        return content.strip()

    @classmethod
    def _extract_returncode(cls, message: dict[str, Any]) -> int | None:
        content = str(message.get("content") or "")
        match = cls._RETURNCODE_RE.search(content)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @classmethod
    def _extract_output(cls, message: dict[str, Any]) -> str:
        content = str(message.get("content") or "")
        return "\n".join(match.group("output") for match in cls._OUTPUT_RE.finditer(content))

    @classmethod
    def _looks_like_edit_command(cls, command: str) -> bool:
        return bool(cls._EDIT_COMMAND_RE.search(command))

    @classmethod
    def _looks_like_read_search_command(cls, command: str) -> bool:
        return bool(cls._READ_SEARCH_COMMAND_RE.search(command))

    @classmethod
    def _looks_like_run_command(cls, command: str) -> bool:
        if cls._ADHOC_TEST_COMMAND_RE.search(command):
            return bool(cls._RUN_COMMAND_RE.search(command))
        if cls._looks_like_read_search_command(command):
            return False
        if cls._looks_like_edit_command(command):
            return False
        return bool(cls._RUN_COMMAND_RE.search(command))

    @classmethod
    def _looks_like_failed_run(cls, observation: dict[str, Any]) -> bool:
        returncode = cls._extract_returncode(observation)
        if returncode is not None and returncode != 0:
            return True
        output = cls._extract_output(observation)
        return bool(output and cls._FAILURE_OUTPUT_RE.search(output))

    @classmethod
    def _post_edit_test_failure_count(cls, messages: list[dict[str, Any]]) -> int:
        edited = False
        failures = 0
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            command = cls._extract_command(message)
            if cls._looks_like_edit_command(command):
                edited = True
            if not edited or not cls._looks_like_run_command(command):
                continue
            if index + 1 >= len(messages):
                continue
            observation = messages[index + 1]
            if observation.get("role") == "user" and cls._looks_like_failed_run(observation):
                failures += 1
        return failures
