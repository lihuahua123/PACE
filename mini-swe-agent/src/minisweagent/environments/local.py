import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from minisweagent.utils.timing import elapsed_since, fail_on_threshold_exceeded, threshold_watchdog


@dataclass
class LocalEnvironmentConfig:
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    command_timeout_threshold: float = float(os.getenv("MSWEA_COMMAND_TIMEOUT_THRESHOLD", "0"))


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None):
        """Execute a command in the local environment and return the result as a dict."""
        cwd = cwd or self.config.cwd or os.getcwd()
        start_time = time.perf_counter()
        effective_timeout = timeout or self.config.timeout
        try:
            with threshold_watchdog(
                operation="command execution",
                start_time=start_time,
                threshold_sec=self.config.command_timeout_threshold,
                details={"cwd": cwd, "command": command},
            ):
                result = subprocess.run(
                    command,
                    shell=True,
                    text=True,
                    cwd=cwd,
                    env=os.environ | self.config.env,
                    timeout=effective_timeout,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.TimeoutExpired:
            fail_on_threshold_exceeded(
                operation="command execution",
                elapsed_sec=elapsed_since(start_time),
                threshold_sec=self.config.command_timeout_threshold,
                details={"cwd": cwd, "command": command},
            )
            raise
        fail_on_threshold_exceeded(
            operation="command execution",
            elapsed_sec=elapsed_since(start_time),
            threshold_sec=self.config.command_timeout_threshold,
            details={"cwd": cwd, "command": command},
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | platform.uname()._asdict() | os.environ
