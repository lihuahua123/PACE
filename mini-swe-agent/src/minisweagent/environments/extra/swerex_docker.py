import asyncio
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from swerex.deployment.docker import DockerDeployment
from swerex.runtime.abstract import Command as RexCommand

from minisweagent.utils.timing import elapsed_since, fail_on_threshold_exceeded, threshold_watchdog


@dataclass
class SwerexDockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    command_timeout_threshold: float = float(os.getenv("MSWEA_COMMAND_TIMEOUT_THRESHOLD", "0"))
    """Max wall-clock seconds for one command execution. Set to 0 to disable."""
    deployment_extra_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra kwargs to pass to DockerDeployment."""


class SwerexDockerEnvironment:
    def __init__(self, **kwargs):
        """This class executes bash commands in a Docker container using SWE-ReX for sandboxing."""
        self.config = SwerexDockerEnvironmentConfig(**kwargs)
        self.deployment = DockerDeployment(image=self.config.image, **self.config.deployment_extra_kwargs)
        asyncio.run(self.deployment.start())

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the environment and return the raw output."""
        start_time = time.perf_counter()
        work_dir = cwd or self.config.cwd
        effective_timeout = timeout or self.config.timeout
        try:
            with threshold_watchdog(
                operation="command execution",
                start_time=start_time,
                threshold_sec=self.config.command_timeout_threshold,
                details={"cwd": work_dir, "command": command},
            ):
                output = asyncio.run(
                    self.deployment.runtime.execute(
                        RexCommand(
                            command=command,
                            shell=True,
                            check=False,
                            cwd=work_dir,
                            timeout=effective_timeout,
                            merge_output_streams=True,
                        )
                    )
                )
        except TimeoutError:
            fail_on_threshold_exceeded(
                operation="command execution",
                elapsed_sec=elapsed_since(start_time),
                threshold_sec=self.config.command_timeout_threshold,
                details={"cwd": work_dir, "command": command},
            )
            raise
        fail_on_threshold_exceeded(
            operation="command execution",
            elapsed_sec=elapsed_since(start_time),
            threshold_sec=self.config.command_timeout_threshold,
            details={"cwd": work_dir, "command": command},
        )
        return {
            "output": output.stdout,
            "returncode": output.exit_code,
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)
