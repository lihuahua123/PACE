"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model
from minisweagent.agents.repetition_detector import RuleBasedRepetitionDetector
from minisweagent.utils.log import logger


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0
    turn_control: bool = False
    turn_control_mode: str = "unrestricted"
    turn_control_fixed_limit: int = 0
    turn_control_initial_limit: int = 0
    turn_control_extension_limit: int = 0
    turn_control_reminder: bool = False
    trace_repetition_nudge_enabled: bool | str = False
    trace_repetition_read_loop_nudge_enabled: bool | str = False
    trace_repetition_nudge_force_enabled: bool | str = False
    trace_repetition_nudge_force_num: int | str = 0
    trace_status_bar_enabled: bool | str = False
    trace_status_bar_max_items: int | str = 8
    trace_status_bar_interval: int | str = 3
    trace_status_bar_next_action_guidance_enabled: bool | str = True
    trace_status_repeat_block_enabled: bool | str = False
    trace_status_repeat_block_max_consecutive_inspections: int | str = 10
    trace_status_repeat_block_max_total_inspections_since_edit: int | str = 35
    trace_status_repeat_block_max_scratch_experiments_since_edit: int | str = 7
    trace_status_repeat_block_max_test_phase_tests: int | str = 18
    trace_status_repeat_block_soft_warning_enabled: bool | str = False
    trace_status_repeat_block_warn_consecutive_inspections: int | str = 8
    trace_status_repeat_block_warn_total_inspections: int | str = 11
    trace_status_repeat_block_warn_scratch_experiments: int | str = 1
    trace_status_repeat_block_warn_test_phase_tests: int | str = 4
    trace_status_repeat_block_dynamic_inspection_enabled: bool | str = False
    # P75 total-inspection limit is 35; a 30-action cap reaches the observed P95 (65).
    trace_status_repeat_block_dynamic_inspection_max_bonus: int | str = 30
    trace_status_large_read_block_enabled: bool | str = False
    trace_status_large_read_block_max_lines: int | str = 1000
    # Independently truncate a single command observation before it enters model context.
    trace_status_long_output_truncation_enabled: bool | str = False
    trace_status_long_output_truncation_max_chars: int | str = 4000
    trace_status_long_output_truncation_head_tail_chars: int | str = 1500
    # Independently turn oversized explicit sed windows into their first chunk.
    trace_status_progressive_read_enabled: bool | str = False
    trace_status_progressive_read_chunk_lines: int | str = 200
    # Independently compact an old, already-consumed source observation.
    trace_status_old_observation_compaction_enabled: bool | str = False
    trace_status_old_observation_compaction_max_chars: int | str = 4000
    trace_status_old_observation_compaction_conclusion_chars: int | str = 1200
    trace_status_post_verify_guard_enabled: bool | str = False
    trace_status_post_verify_max_extra_tests: int | str = 1
    trace_status_post_verify_max_extra_inspections: int | str = 3
    trace_action_budget_controller_enabled: bool | str = False
    trace_action_budget_search: int | str = 0
    trace_action_budget_edit: int | str = 0
    trace_action_budget_test: int | str = 0
    consecutive_unproductive_failure_threshold: int | str = 3
    repeated_submit_force_threshold: int | str = 3
    require_successful_test_before_submit: bool | str = False
    submission_review_enabled: bool | str = False
    deterministic_retrieval_first_action_guard: bool | str = False
    whole_file_read_hint_enabled: bool | str = False
    whole_file_read_block_enabled: bool | str = False
    read_cache_enable: bool | str = False
    scout_handoff_enabled: bool | str = False
    scout_handoff_constraints_enabled: bool | str = False
    scout_handoff_system_template: str = (
        "You are a scout agent. Your job is to cheaply explore the repository before a stronger model "
        "takes over. Do not edit files and do not submit. Either respond with exactly one bash command "
        "in one fenced bash block to gather more evidence, or output the final handoff in the requested "
        "Issue/Observed/Candidate files/Uncertainty format when you have enough information. "
        "Once a plausible production file and function are found, prefer producing the handoff instead of "
        "broad exploration. Before the final handoff, run one focused read/search for existing tests or call "
        "sites of the candidate function when possible, and capture exact call signatures, keyword arguments, "
        "or selected test commands; never list tests as candidate files."
    )
    scout_handoff_instance_template: str = (
        "Scout this issue for the main model:\n\n{{task}}\n\n"
        "Run only read-only inspection commands or focused reproduction commands. Do not modify files. "
        "Prefer concise commands whose output can support a handoff. When you have enough evidence, "
        "especially after identifying the likely production source file and relevant line range, "
        "inspect focused existing tests or call sites for the candidate symbols if possible, then stop running "
        "commands and output exactly this format without a code block. Include expected call signatures, "
        "keyword arguments, and test command names under Observed, not under Candidate files. If you could not "
        "find focused tests or call sites, say which command you tried under Uncertainty:\n\n"
        "Issue:\n"
        "- 用户报告/失败现象一句话\n\n"
        "Observed:\n"
        "- 复现命令和关键输出\n\n"
        "Candidate files:\n"
        "- path:line，为什么相关。Only list production source files here; do not list tests as candidate files.\n\n\n\n"
        "Uncertainty:\n"
        "- 哪些地方还不确定"
    )
    scout_handoff_constraint_template: str = (
        "\n\nAdditional constraints for this run:\n"
        "- If an expected direct path is missing, first identify the repository's source root instead of assuming "
        "the current directory is the import/build root. Common layouts include src/, lib/, package directories, "
        "and packages/*, but infer from this repo.\n"
        "- When validating, make sure commands exercise the current checkout, not a globally installed package, "
        "dependency cache, virtualenv dependency, or system copy. If imports resolve outside the workspace, adjust "
        "the command/environment rather than editing outside the workspace.\n"
        "- Do not edit external dependencies, installed packages, package caches, virtualenv dependencies, or "
        "system files. Final patches must come from the repository workspace.\n"
        "- In addition to any focused existing test, run or describe one minimal issue-driven verification that "
        "directly covers the user's reported behavior and API shape."
    )
    scout_handoff_summary_template: str = (
        "Now produce the handoff for the main model. Do not include a code block. Use exactly this format:\n\n"
        "Issue:\n"
        "- 用户报告/失败现象一句话\n\n"
        "Observed:\n"
        "- 复现命令和关键输出。Include exact production signatures and any focused test/call-site signatures or kwargs observed.\n\n"
        "Candidate files:\n"
        "- path:line，为什么相关。Only list production source files here; do not list tests as candidate files.\n\n\n\n"
        "Uncertainty:\n"
        "- 哪些地方还不确定"
    )


@dataclass
class ReadCacheEntry:
    file: str
    line_range: str
    step: int
    state_epoch: int


@dataclass
class TraceStatusReadEntry:
    file: str
    line_range: str
    first_step: int
    last_step: int
    count: int = 1


@dataclass
class TraceStatusSearchEntry:
    signature: str
    first_step: int
    last_step: int
    count: int = 1


@dataclass
class TraceStatusEditEntry:
    target: str
    first_step: int
    last_step: int
    count: int = 1


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class RepetitionNudgeLimitExceeded(TerminatingException):
    """Raised when repetition nudges have triggered too many times."""


class ConsecutiveUnproductiveFailuresExceeded(TerminatingException):
    """Raised when repeated execution failures are unlikely to provide new information."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self._turn_control_used = 0
        self._turn_control_extension_used = False
        self._turn_control_extension_notice_pending = False
        self._turn_control_effective_limit = 0
        self._consecutive_submit_attempts = 0
        self._consecutive_verification_blocks = 0
        self._last_submit_action = ""
        self._trace_repetition_nudge_enabled = self._parse_boolish(
            getattr(self.config, "trace_repetition_nudge_enabled", False)
        )
        self._trace_repetition_read_loop_nudge_enabled = self._parse_boolish(
            getattr(self.config, "trace_repetition_read_loop_nudge_enabled", False)
        )
        self._trace_repetition_nudge_force_enabled = self._parse_boolish(
            getattr(self.config, "trace_repetition_nudge_force_enabled", False)
        )
        self._trace_repetition_nudge_force_num = self._parse_intish(
            getattr(self.config, "trace_repetition_nudge_force_num", 0)
        )
        self._trace_status_bar_enabled = self._parse_boolish(getattr(self.config, "trace_status_bar_enabled", False))
        self._trace_status_bar_max_items = max(
            1, self._parse_intish(getattr(self.config, "trace_status_bar_max_items", 8))
        )
        self._trace_status_bar_interval = max(
            1, self._parse_intish(getattr(self.config, "trace_status_bar_interval", 3))
        )
        self._trace_status_bar_next_action_guidance_enabled = self._parse_boolish(
            getattr(self.config, "trace_status_bar_next_action_guidance_enabled", True)
        )
        self._trace_status_repeat_block_enabled = self._parse_boolish(
            getattr(self.config, "trace_status_repeat_block_enabled", False)
        )
        self._trace_status_repeat_block_max_consecutive_inspections = max(
            1,
            self._parse_intish(
                getattr(self.config, "trace_status_repeat_block_max_consecutive_inspections", 10)
            ),
        )
        self._trace_status_repeat_block_max_total_inspections_since_edit = max(
            1,
            self._parse_intish(
                getattr(self.config, "trace_status_repeat_block_max_total_inspections_since_edit", 35)
            ),
        )
        self._trace_status_repeat_block_max_scratch_experiments_since_edit = max(
            1,
            self._parse_intish(
                getattr(self.config, "trace_status_repeat_block_max_scratch_experiments_since_edit", 7)
            ),
        )
        self._trace_status_repeat_block_max_test_phase_tests = max(
            1,
            self._parse_intish(
                getattr(self.config, "trace_status_repeat_block_max_test_phase_tests", 18)
            ),
        )
        self._trace_status_repeat_block_soft_warning_enabled = self._parse_boolish(
            getattr(self.config, "trace_status_repeat_block_soft_warning_enabled", False)
        )
        self._trace_status_repeat_block_warning_limits = {
            "consecutive inspections": self._parse_intish(self.config.trace_status_repeat_block_warn_consecutive_inspections),
            "total inspections": self._parse_intish(self.config.trace_status_repeat_block_warn_total_inspections),
            "scratch experiments": self._parse_intish(self.config.trace_status_repeat_block_warn_scratch_experiments),
            "tests": self._parse_intish(self.config.trace_status_repeat_block_warn_test_phase_tests),
        }
        self._trace_status_repeat_block_dynamic_inspection_enabled = self._parse_boolish(
            getattr(self.config, "trace_status_repeat_block_dynamic_inspection_enabled", False)
        )
        self._trace_status_repeat_block_dynamic_inspection_max_bonus = max(
            0,
            self._parse_intish(
                getattr(self.config, "trace_status_repeat_block_dynamic_inspection_max_bonus", 30)
            ),
        )
        self._trace_status_large_read_block_enabled = self._parse_boolish(
            getattr(self.config, "trace_status_large_read_block_enabled", False)
        )
        self._trace_status_large_read_block_max_lines = max(
            1, self._parse_intish(getattr(self.config, "trace_status_large_read_block_max_lines", 1000))
        )
        self._trace_status_long_output_truncation_enabled = self._parse_boolish(
            getattr(self.config, "trace_status_long_output_truncation_enabled", False)
        )
        self._trace_status_long_output_truncation_max_chars = max(
            1, self._parse_intish(getattr(self.config, "trace_status_long_output_truncation_max_chars", 4000))
        )
        self._trace_status_long_output_truncation_head_tail_chars = max(
            1,
            self._parse_intish(getattr(self.config, "trace_status_long_output_truncation_head_tail_chars", 1500)),
        )
        self._trace_status_progressive_read_enabled = self._parse_boolish(
            getattr(self.config, "trace_status_progressive_read_enabled", False)
        )
        self._trace_status_progressive_read_chunk_lines = max(
            1, self._parse_intish(getattr(self.config, "trace_status_progressive_read_chunk_lines", 200))
        )
        self._trace_status_old_observation_compaction_enabled = self._parse_boolish(
            getattr(self.config, "trace_status_old_observation_compaction_enabled", False)
        )
        self._trace_status_old_observation_compaction_max_chars = max(
            1,
            self._parse_intish(getattr(self.config, "trace_status_old_observation_compaction_max_chars", 4000)),
        )
        self._trace_status_old_observation_compaction_conclusion_chars = max(
            1,
            self._parse_intish(
                getattr(self.config, "trace_status_old_observation_compaction_conclusion_chars", 1200)
            ),
        )
        self._trace_status_post_verify_guard_enabled = self._parse_boolish(
            getattr(self.config, "trace_status_post_verify_guard_enabled", True)
        )
        self._trace_status_post_verify_max_extra_tests = max(
            0, self._parse_intish(getattr(self.config, "trace_status_post_verify_max_extra_tests", 1))
        )
        self._trace_status_post_verify_max_extra_inspections = max(
            0, self._parse_intish(getattr(self.config, "trace_status_post_verify_max_extra_inspections", 3))
        )
        self._trace_action_budget_controller_enabled = self._parse_boolish(
            getattr(self.config, "trace_action_budget_controller_enabled", False)
        )
        self._trace_action_budget_remaining = {
            "search": max(0, self._parse_intish(getattr(self.config, "trace_action_budget_search", 0))),
            "edit": max(0, self._parse_intish(getattr(self.config, "trace_action_budget_edit", 0))),
            "test": max(0, self._parse_intish(getattr(self.config, "trace_action_budget_test", 0))),
        }
        self._trace_action_budget_phase = "search"
        self._trace_action_budget_transitions: list[str] = []
        self._consecutive_unproductive_failure_threshold = max(
            0, self._parse_intish(getattr(self.config, "consecutive_unproductive_failure_threshold", 3))
        )
        self._repeated_submit_force_threshold = max(
            0, self._parse_intish(getattr(self.config, "repeated_submit_force_threshold", 3))
        )
        self._require_successful_test_before_submit = self._parse_boolish(
            getattr(self.config, "require_successful_test_before_submit", False)
        )
        self._submission_review_enabled = self._parse_boolish(
            getattr(self.config, "submission_review_enabled", False)
        )
        self._whole_file_read_hint_enabled = self._parse_boolish(
            getattr(self.config, "whole_file_read_hint_enabled", False)
        )
        self._whole_file_read_block_enabled = self._parse_boolish(
            getattr(self.config, "whole_file_read_block_enabled", False)
        )
        self._trace_repetition_nudge_trigger_count = 0
        self._repetition_detector = RuleBasedRepetitionDetector(
            enable_read_loop_nudge=self._trace_repetition_read_loop_nudge_enabled
        )
        self._read_cache_enabled = self._parse_boolish(getattr(self.config, "read_cache_enable", False))
        self._read_cache: dict[tuple[str, str], ReadCacheEntry] = {}
        self._read_cache_hits = 0
        self._read_cache_state_epoch = 0
        self._trace_status_reads: dict[tuple[str, str], TraceStatusReadEntry] = {}
        self._trace_status_searches: dict[str, TraceStatusSearchEntry] = {}
        self._trace_status_edits: dict[str, TraceStatusEditEntry] = {}
        self._trace_status_tests: list[dict[str, object]] = []
        self._trace_status_repeat_notes: list[str] = []
        self._trace_status_repeat_block_counts: dict[str, int] = {}
        self._trace_status_repeat_confirmations_since_source_edit = 0
        self._trace_status_repeat_warned_since_source_edit: set[str] = set()
        self._trace_status_post_pass_edit_guard_hits = 0
        self._trace_status_consecutive_inspections = 0
        self._trace_status_dynamic_inspection_bonus = 0
        self._trace_status_novel_evidence_count = 0
        self._trace_status_read_coverage: dict[str, list[tuple[int, int]]] = {}
        self._trace_status_seen_search_hits: set[str] = set()
        self._trace_status_seen_test_commands: set[str] = set()
        self._trace_status_seen_test_failures: set[str] = set()
        self._trace_status_seen_scratch_probes: set[str] = set()
        self._trace_status_tracked_files: set[str] = set()
        self._trace_status_total_inspections_since_source_edit = 0
        self._trace_status_scratch_experiments_since_source_edit = 0
        self._trace_status_test_phase_tests = 0
        self._trace_status_last_source_edit_step = 0
        self._trace_status_last_rendered_step = 0
        self._trace_status_raw_observation_cache: dict[str, str] = {}
        self._trace_status_pending_compaction: dict[str, object] | None = None
        self._trace_status_compaction_counter = 0
        self._unproductive_failure_streak = 0
        self._last_unproductive_failure_signature = ""
        self._has_seen_test_command = False
        self._has_successful_test_command = False
        self._post_verify_extra_tests = 0
        self._post_verify_extra_inspections = 0
        self._action_sequence = 0
        self._last_source_edit_action = 0
        self._last_successful_test_action = 0
        self._scout_handoff_pricing_messages: list[dict] = []
        self._scout_handoff_transcript: list[dict] = []
        self._scout_handoff_status = "not_run"

    @staticmethod
    def _parse_boolish(value: bool | str) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_intish(value: int | str) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return 0

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def _reset_run_state(self) -> None:
        self.messages = []
        self._turn_control_used = 0
        self._turn_control_extension_used = False
        self._turn_control_extension_notice_pending = False
        self._turn_control_effective_limit = self._get_turn_control_initial_limit()
        self._trace_action_budget_remaining = {
            "search": max(0, self._parse_intish(getattr(self.config, "trace_action_budget_search", 0))),
            "edit": max(0, self._parse_intish(getattr(self.config, "trace_action_budget_edit", 0))),
            "test": max(0, self._parse_intish(getattr(self.config, "trace_action_budget_test", 0))),
        }
        self._trace_action_budget_phase = "search"
        self._trace_action_budget_transitions = []
        self._advance_trace_action_budget_phase()
        self._consecutive_submit_attempts = 0
        self._consecutive_verification_blocks = 0
        self._last_submit_action = ""
        self._trace_repetition_nudge_trigger_count = 0
        self._repetition_detector.reset()
        self._read_cache = {}
        self._read_cache_hits = 0
        self._read_cache_state_epoch = 0
        self._trace_status_reads = {}
        self._trace_status_searches = {}
        self._trace_status_edits = {}
        self._trace_status_tests = []
        self._trace_status_repeat_notes = []
        self._trace_status_repeat_block_counts = {}
        self._trace_status_repeat_confirmations_since_source_edit = 0
        self._trace_status_repeat_warned_since_source_edit = set()
        self._trace_status_post_pass_edit_guard_hits = 0
        self._trace_status_consecutive_inspections = 0
        self._trace_status_dynamic_inspection_bonus = 0
        self._trace_status_novel_evidence_count = 0
        self._trace_status_read_coverage = {}
        self._trace_status_seen_search_hits = set()
        self._trace_status_seen_test_commands = set()
        self._trace_status_seen_test_failures = set()
        self._trace_status_seen_scratch_probes = set()
        self._trace_status_total_inspections_since_source_edit = 0
        self._trace_status_scratch_experiments_since_source_edit = 0
        self._trace_status_test_phase_tests = 0
        self._trace_status_last_source_edit_step = 0
        self._trace_status_last_rendered_step = 0
        self._trace_status_raw_observation_cache = {}
        self._trace_status_pending_compaction = None
        self._trace_status_compaction_counter = 0
        self._unproductive_failure_streak = 0
        self._last_unproductive_failure_signature = ""
        self._has_seen_test_command = False
        self._has_successful_test_command = False
        self._post_verify_extra_tests = 0
        self._post_verify_extra_inspections = 0
        self._action_sequence = 0
        self._last_source_edit_action = 0
        self._last_successful_test_action = 0
        self._scout_handoff_pricing_messages = []
        self._scout_handoff_transcript = []
        self._scout_handoff_status = "not_run"

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self._initialize_run(task, **kwargs)
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def _initialize_run(self, task: str, **kwargs) -> None:
        self.extra_template_vars |= {"task": task, **kwargs}
        self._reset_run_state()
        self._trace_status_tracked_files = self._load_trace_status_tracked_files()
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        self._maybe_run_scout_handoff()

    def _maybe_run_scout_handoff(self) -> None:
        if not self._parse_boolish(getattr(self.config, "scout_handoff_enabled", True)):
            return
        if not hasattr(self.model, "scout_query"):
            return
        scout_max_steps = max(0, self._parse_intish(getattr(self.model.config, "scout_handoff_steps", 0)))
        if scout_max_steps <= 0:
            return

        scout_messages = [
            {
                "role": "system",
                "content": self.render_template(self.config.scout_handoff_system_template)
                + self._render_scout_constraints_if_enabled(),
            },
            {
                "role": "user",
                "content": self.render_template(self.config.scout_handoff_instance_template)
                + self._render_scout_constraints_if_enabled(),
            },
        ]
        scout_transcript: list[dict] = []
        self._scout_handoff_status = "running"
        summary: dict | None = None
        for _ in range(scout_max_steps):
            response = self.model.scout_query(scout_messages)
            scout_messages.append({"role": "assistant", **response})
            self._scout_handoff_pricing_messages.append({"role": "assistant", **response})
            if self._looks_like_scout_handoff(response.get("content", "")):
                summary = response
                scout_transcript.append({"role": "assistant", **response, "scout_handoff": {"phase": "summary"}})
                self._scout_handoff_status = "accepted"
                break
            scout_transcript.append({"role": "assistant", **response, "scout_handoff": {"phase": "explore"}})
            try:
                action = self.parse_action(response)
                if self._scout_action_may_change_code(action["action"]):
                    output = {
                        "returncode": 126,
                        "output": "SCOUT ACTION BLOCKED: scout handoff may only run read-only or reproduction commands.",
                    }
                else:
                    output = self.env.execute(action["action"])
            except NonTerminatingException as e:
                observation = str(e)
            except subprocess.TimeoutExpired as e:
                output_text = e.output.decode("utf-8", errors="replace") if e.output else ""
                observation = self.render_template(self.config.timeout_template, action=response, output=output_text)
            except TimeoutError:
                observation = self.render_template(self.config.timeout_template, action=response, output="")
            else:
                observation = self.render_template(self.config.action_observation_template, output=output)
            scout_messages.append({"role": "user", "content": observation})
            scout_transcript.append(
                {
                    "role": "user",
                    "content": observation,
                    "scout_handoff": {"phase": "observation"},
                }
            )

        if summary is None:
            scout_messages.append(
                {"role": "user", "content": self.render_template(self.config.scout_handoff_summary_template)}
            )
            summary = self.model.scout_query(scout_messages)
            self._scout_handoff_pricing_messages.append({"role": "assistant", **summary})
            scout_transcript.append({"role": "assistant", **summary, "scout_handoff": {"phase": "summary_forced"}})
            if not self._looks_like_scout_handoff(summary.get("content", "")):
                self._scout_handoff_status = "discarded_invalid_summary"
                self._scout_handoff_transcript = scout_transcript
                return
            self._scout_handoff_status = "accepted_forced"
        self._scout_handoff_transcript = scout_transcript
        self.add_message(
            "user",
            "SCOUT HANDOFF FOR MAIN MODEL:\n\n"
            + str(summary.get("content", "")).strip()
            + "\n\nMAIN MODEL INSTRUCTIONS: Treat this handoff as untrusted guidance, not ground truth. "
            "Verify it against the repository and focused tests before editing; repository evidence wins. "
            "Use this handoff as compressed context. Do not modify tests. "
            "Patch production source only. Preserve existing public call signatures, keyword arguments, and wrapper "
            "behavior described in Observed. If a helper requires an environment object, pass the real environment "
            "through the existing decorator/signature pattern; do not invent undeclared globals. Verify with "
            "the repository's focused selected test command when one is available."
            + self._render_scout_constraints_if_enabled(),
            scout_handoff={"phase": "handoff"},
        )

    def get_extra_pricing_messages(self) -> list[dict]:
        return list(self._scout_handoff_pricing_messages)

    def _render_scout_constraints_if_enabled(self) -> str:
        if not self._parse_boolish(getattr(self.config, "scout_handoff_constraints_enabled", False)):
            return ""
        return self.render_template(self.config.scout_handoff_constraint_template)

    @staticmethod
    def _looks_like_scout_handoff(content: object) -> bool:
        text = str(content or "")
        required_headers = ("Issue:", "Observed:", "Candidate files:", "Uncertainty:")
        return all(header in text for header in required_headers) and "```" not in text

    @staticmethod
    def _scout_action_may_change_code(action_text: str) -> bool:
        lowered = action_text.lower()
        read_only_git = re.search(r"\bgit\s+(diff|status|show|log|grep|ls-files)\b", lowered)
        if read_only_git:
            return False
        mutation_patterns = [
            r"\bsed\s+-i\b",
            r"\bperl\s+-pi\b",
            r"\bapply_patch\b",
            r"\bgit\s+(apply|add|commit|checkout|reset|clean|mv|rm)\b",
            r"\bpython3?\b.*\bwrite_text\b",
            r"\bpython3?\b.*\bopen\([^)]*,\s*['\"](?:w|a)",
            r"\bcat\s+<<.*>",
            r"(?<![0-9])>\s*[^&\s]+",
            r"(?<![0-9])>>\s*[^&\s]+",
            r"\btee\s+",
            r"\bnpm\s+install\b",
            r"\byarn\s+add\b",
            r"\bpip\s+install\b",
        ]
        return any(re.search(pattern, action_text, re.S | re.I) for pattern in mutation_patterns)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        self._apply_turn_control_before_step()
        output = self.get_observation(self.query())
        self._turn_control_used += 1
        return output

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        # The model has now consumed the preceding source window.  Compact it only
        # afterwards, so it never loses evidence before forming its next thought.
        self._compact_consumed_source_observation(str(response.get("content", "")))
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        action = self.parse_action(response)
        output = self.execute_action(action)
        unproductive_failure_signature = self._record_unproductive_failure(action=action, output=output)
        if unproductive_failure_signature:
            observation = self.render_template(self.config.action_observation_template, output=output)
            observation = self._truncate_long_observation(
                observation, raw_output_chars=len(str(output.get("output", "")))
            )
            self._finalize_observation(observation, repetition_meta=None)
            self._raise_if_unproductive_failures_exceeded(unproductive_failure_signature)
            return output
        self._update_read_cache(action=action, output=output)
        trace_status_bar = ""
        if not output.get("trace_status_repeat_blocked") and not output.get("trace_action_budget_blocked"):
            trace_status_bar = self._update_trace_status_bar(action=action, output=output)
        observation = self.render_template(self.config.action_observation_template, output=output)
        observation = self._truncate_long_observation(
            observation, raw_output_chars=len(str(output.get("output", "")))
        )
        if trace_status_bar:
            observation += "\n\n" + trace_status_bar
        if action_budget_status := self._trace_action_budget_status_message():
            observation += "\n\n" + action_budget_status
        observation, repetition_meta = self._maybe_append_repetition_nudge(
            observation=observation,
            response=response,
            action=action,
            output=output,
        )
        self._finalize_observation(observation, repetition_meta=repetition_meta)
        self._remember_compactable_source_observation(action=action, output=output)
        return output

    def _truncate_long_observation(self, observation: str, *, raw_output_chars: int) -> str:
        if not self._trace_status_long_output_truncation_enabled:
            return observation
        if raw_output_chars <= self._trace_status_long_output_truncation_max_chars:
            return observation
        keep = self._trace_status_long_output_truncation_head_tail_chars
        if len(observation) <= keep * 2:
            return observation
        return (
            observation[:keep]
            + "\n\n[OUTPUT TRUNCATED: original output was "
            + f"{raw_output_chars:,} characters; kept first and last {keep:,} characters. "
            + "Use `rg -n <symbol> <file>` or `sed -n 'start,endp' <file>` for a focused read.]\n\n"
            + observation[-keep:]
        )

    def _remember_compactable_source_observation(self, *, action: dict, output: dict) -> None:
        if not self._trace_status_old_observation_compaction_enabled:
            return
        raw_output = str(output.get("output", ""))
        if len(raw_output) <= self._trace_status_old_observation_compaction_max_chars:
            return
        targets = self._extract_read_targets(str(action.get("action", "")))
        if not targets or not self.messages or self.messages[-1].get("role") != "user":
            return
        self._trace_status_compaction_counter += 1
        cache_id = f"source_observation_{self._trace_status_compaction_counter}"
        self._trace_status_raw_observation_cache[cache_id] = raw_output
        self._trace_status_pending_compaction = {
            "message_index": len(self.messages) - 1,
            "cache_id": cache_id,
            "targets": targets,
            "raw_output_chars": len(raw_output),
        }

    def _compact_consumed_source_observation(self, next_agent_content: str) -> None:
        pending = self._trace_status_pending_compaction
        self._trace_status_pending_compaction = None
        if not self._trace_status_old_observation_compaction_enabled or not pending:
            return
        message_index = int(pending["message_index"])
        if message_index >= len(self.messages):
            return
        thought = next_agent_content.split("```", 1)[0].strip()
        thought = thought[: self._trace_status_old_observation_compaction_conclusion_chars].strip()
        targets = pending["targets"]
        read_lines = ", ".join(f"{path}:{line_range}" for path, line_range in targets)
        content_lines = [
            "<structured_source_memory>",
            f"Read: {read_lines}",
            f"Raw output reference: {pending['cache_id']} ({int(pending['raw_output_chars']):,} characters; retained in agent cache).",
        ]
        if thought:
            content_lines.append(f"Agent conclusion (unverified): {thought}")
        content_lines.extend(
            [
                "To inspect details again, use `rg -n <symbol> <file>` or `sed -n 'start,endp' <file>`.",
                "</structured_source_memory>",
            ]
        )
        self.messages[message_index]["content"] = "\n".join(content_lines)

    def _record_unproductive_failure(self, *, action: dict, output: dict) -> str:
        if not self._trace_status_bar_enabled:
            return ""
        signature = self._unproductive_failure_signature(action=action, output=output)
        if not signature:
            self._unproductive_failure_streak = 0
            self._last_unproductive_failure_signature = ""
            return ""
        if signature == self._last_unproductive_failure_signature:
            self._unproductive_failure_streak += 1
        else:
            self._unproductive_failure_streak = 1
            self._last_unproductive_failure_signature = signature
        return signature

    def _raise_if_unproductive_failures_exceeded(self, signature: str) -> None:
        if not self._consecutive_unproductive_failure_threshold:
            return
        if self._unproductive_failure_streak < self._consecutive_unproductive_failure_threshold:
            return
        raise ConsecutiveUnproductiveFailuresExceeded(
            "Stopping after "
            f"{self._unproductive_failure_streak} consecutive unproductive execution failures "
            f"({signature})."
        )

    def _unproductive_failure_signature(self, *, action: dict, output: dict) -> str:
        if int(output.get("returncode", 0) or 0) == 0:
            return ""
        action_text = str(action.get("action", ""))
        output_text = str(output.get("output", ""))
        if self._action_contains_submit_marker(action_text):
            return "submit-command-failed"
        if self._output_indicates_environment_unavailable(output_text):
            return "environment-unavailable"
        return ""

    @staticmethod
    def _action_contains_submit_marker(action_text: str) -> bool:
        return (
            "MINI_SWE_AGENT_FINAL_OUTPUT" in action_text
            or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in action_text
        )

    @staticmethod
    def _output_indicates_environment_unavailable(output_text: str) -> bool:
        lowered = output_text.lower()
        patterns = [
            "no such container",
            "cannot connect to the docker daemon",
            "error response from daemon",
            "container environment is unavailable",
            "container has been terminated",
        ]
        return any(pattern in lowered for pattern in patterns)

    @staticmethod
    def _extract_submit_payload(output: dict[str, str]) -> str | None:
        """Return any text that appears after the submit marker."""
        lines = output.get("output", "").splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
                return "".join(lines[i + 1 :])
        return None

    def _maybe_append_repetition_nudge(
        self, *, observation: str, response: dict, action: dict, output: dict
    ) -> tuple[str, dict | None]:
        if not self._trace_repetition_nudge_enabled:
            return observation, None
        repeat_eval = self._repetition_detector.update(
            response_content=response.get("content", ""),
            action=action.get("action", ""),
            output=output,
        )
        if repeat_eval.get("triggered"):
            self._trace_repetition_nudge_trigger_count += 1
            repetition_meta = {
                "triggered": True,
                "trigger_reason": repeat_eval.get("trigger_reason", ""),
                "trigger_count": self._trace_repetition_nudge_trigger_count,
            }
            if (
                self._trace_repetition_nudge_force_enabled
                and self._trace_repetition_nudge_force_num > 0
                and self._trace_repetition_nudge_trigger_count >= self._trace_repetition_nudge_force_num
            ):
                repetition_meta["force_stop"] = True
                repetition_meta["force_stop_message"] = (
                    "Stopping because repetition nudge was triggered "
                    f"{self._trace_repetition_nudge_trigger_count} times "
                    f"(limit={self._trace_repetition_nudge_force_num}) without task completion."
                )
            if repetition_meta["trigger_reason"] == "read_loop_detected":
                force_stop_suffix = ""
                if repetition_meta.get("force_stop"):
                    force_stop_suffix = (
                        "\n\nFORCE STOP: repetition nudge limit reached. "
                        "This run will terminate now."
                    )
                return (
                    observation
                    + "\n\nYou are repeating code reads with little new info."
                    + "\n\nNext step must be one of:"
                    + "\n1) apply a minimal patch, or"
                    + "\n2) run one focused test to choose between hypotheses, or"
                    + "\n3) submit if ready."
                    + "\n\nDo not keep reading the same file regions."
                    + force_stop_suffix
                ), repetition_meta
            if repetition_meta["trigger_reason"] == "root_cause_locked_without_patch":
                force_stop_suffix = ""
                if repetition_meta.get("force_stop"):
                    force_stop_suffix = (
                        "\n\nFORCE STOP: repetition nudge limit reached. "
                        "This run will terminate now."
                    )
                return (
                    observation
                    + "\n\nYou appear stuck after identifying a likely root cause."
                    + "\n\nBefore any submission, run one minimal verification experiment:"
                    + "\n1) execute exactly one focused test that can falsify your current hypothesis, or"
                    + "\n2) apply one minimal patch and run one focused test."
                    + "\n\nDo not broaden scope. Do not create extra debug files."
                    + "\nOnly proceed to submission after that focused verification passes."
                    + force_stop_suffix
                ), repetition_meta
            if repetition_meta["trigger_reason"] == "post_verify_over_testing":
                force_stop_suffix = ""
                if repetition_meta.get("force_stop"):
                    force_stop_suffix = (
                        "\n\nFORCE STOP: repetition nudge limit reached. "
                        "This run will terminate now."
                    )
                return (
                    observation
                    + "\n\nYou already have a passing post-patch verification."
                    + "\nYou have continued running extra tests more than two times without a new failure."
                    + "\n\nNext step must be one of:"
                    + "\n1) submit if the fix is ready, or"
                    + "\n2) apply another patch if you found a concrete defect, or"
                    + "\n3) run one test only if it can falsify a new hypothesis."
                    + "\n\nDo not keep expanding successful validation."
                    + force_stop_suffix
                ), repetition_meta
            if repetition_meta["trigger_reason"] == "post_verify_low_info_inspection":
                force_stop_suffix = ""
                if repetition_meta.get("force_stop"):
                    force_stop_suffix = (
                        "\n\nFORCE STOP: repetition nudge limit reached. "
                        "This run will terminate now."
                    )
                return (
                    observation
                    + "\n\nYou already have a passing post-patch verification."
                    + "\nYou are continuing low-information inspection without a new failure or patch."
                    + "\n\nNext step must be one of:"
                    + "\n1) submit if the fix is ready, or"
                    + "\n2) apply another patch if this inspection found a concrete defect, or"
                    + "\n3) run one focused test only if it can falsify a new hypothesis."
                    + "\n\nDo not keep re-reading code after successful verification."
                    + force_stop_suffix
                ), repetition_meta
            if repetition_meta["trigger_reason"] == "patch_test_loop_low_info":
                force_stop_suffix = ""
                if repetition_meta.get("force_stop"):
                    force_stop_suffix = (
                        "\n\nFORCE STOP: repetition nudge limit reached. "
                        "This run will terminate now."
                    )
                return (
                    observation
                    + "\n\nYou are in a low-information patch/test loop around the same hypothesis."
                    + "\nRecent patches and tests are repeating without a clearly new failure signal."
                    + "\n\nNext step must be one of:"
                    + "\n1) run one test that can falsify a different hypothesis, or"
                    + "\n2) make one meaningfully different patch, or"
                    + "\n3) stop expanding this loop and submit if the current fix is already sufficient."
                    + "\n\nDo not keep making small same-focus edits followed by the same test."
                    + force_stop_suffix
                ), repetition_meta
            force_stop_suffix = ""
            if repetition_meta.get("force_stop"):
                force_stop_suffix = (
                    "\n\nFORCE STOP: repetition nudge limit reached. "
                    "This run will terminate now."
                )
            # Legacy repetition nudge wording (kept for reference, do not delete):
            # return (
            #     observation
            #     + "\n\nYou already have enough evidence. Avoid further exploration, avoid creating new test files, "
            #     + "and avoid broader validation unless a test fails."
            #     + "\n\nIf you are ready to submit, use this command so only staged source changes are submitted:"
            #     + "\n```bash\n"
            #     + "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
            #     + "git status --porcelain | grep -E \"^\\?\\?|^ M|^A \" | "
            #     + "grep -vE \"(test_|_test\\.py|reproduce_issue|debug_)\" | "
            #     + "awk '{print $2}' | xargs -r git add && git diff --cached\n"
            #     + "```"
            #     + "\nSubmit only when `git diff --cached` is non-empty."
            #     + force_stop_suffix
            # ), repetition_meta
            confidence_score = "N/A"
            files_warning = ""
            prompt = f"""
<submission_prompt>
## Ready to Submit?

Based on detected repetitive patterns in your recent workflow, you appear to be ready for submission.

**Before submitting, please verify:**

1. **Only source code files are being submitted** (No test files, debug scripts, or temporary files)
   - Test files: `test_*.py`, `*_test.py`, `reproduce_issue*.py`
   - Debug files: `debug_*.py`
   - These files should be created in temporary locations or excluded from git

2. **Your fix logic is correct**
   - Function parameters are correct
   - Return types match expectations
   - Logic flow handles all cases

**To submit, execute:**
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git status --porcelain | grep -E "^\\?\\?|^ M|^A " | grep -vE "(test_|_test\\.py|reproduce_issue|debug_)" | awk '{{print $2}}' | xargs -r git add && git diff --cached
```

Or manually add only source files:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add <source_file1.py> <source_file2.py> && git diff --cached
```

If you need to make more changes, continue working as usual.
{files_warning}
</submission_prompt>
"""
            return (observation + "\n\n" + prompt + force_stop_suffix), repetition_meta
        return observation, None

    @staticmethod
    def _normalize_cache_path(raw_path: str) -> str:
        path = raw_path.strip().strip("'\"")
        path = path.rstrip(");:,")
        return path

    def _load_trace_status_tracked_files(self) -> set[str]:
        """Snapshot the repository before the agent can create scratch files."""
        if not (self._trace_status_repeat_block_enabled or self._trace_status_bar_enabled):
            return set()
        try:
            output = self.env.execute("git ls-files -z", timeout=10)
        except (OSError, TimeoutError, subprocess.TimeoutExpired):
            return set()
        if output.get("returncode", 1) != 0:
            return set()
        return {
            path.replace("\\", "/").removeprefix("./")
            for path in str(output.get("output", "")).split("\0")
            if path
        }

    @classmethod
    def _extract_read_targets(cls, action_text: str) -> list[tuple[str, str]]:
        targets: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        normalized = action_text.replace("\\\n", " ")

        def add_target(raw_path: str, line_range: str) -> None:
            path = cls._normalize_cache_path(raw_path)
            item = (path, line_range)
            if item not in seen:
                seen.add(item)
                targets.append(item)

        direct_range_patterns = [
            re.compile(r"(?:nl|cat)\s+-n\s+([^\s|;&]+)\s*\|\s*sed\s+-n\s+'(\d+),(\d+)p'"),
            re.compile(r"nl\s+-ba\s+([^\s|;&]+)\s*\|\s*sed\s+-n\s+'(\d+),(\d+)p'"),
            re.compile(r"sed\s+-n\s+'(\d+),(\d+)p'\s+([^\s|;&]+)"),
            re.compile(r"awk\s+'NR>=(\d+)\s*&&\s*NR<=(\d+).*?'\s+([^\s|;&]+)"),
        ]
        for match in direct_range_patterns[0].finditer(normalized):
            add_target(match.group(1), f"{match.group(2)}-{match.group(3)}")
        for match in direct_range_patterns[1].finditer(normalized):
            add_target(match.group(1), f"{match.group(2)}-{match.group(3)}")
        for match in direct_range_patterns[2].finditer(normalized):
            add_target(match.group(3), f"{match.group(1)}-{match.group(2)}")
        for match in direct_range_patterns[3].finditer(normalized):
            add_target(match.group(3), f"{match.group(1)}-{match.group(2)}")

        for match in re.finditer(r"head\s+-n\s+(\d+)\s+([^\s|;&]+)\s*\|\s*tail\s+-n\s+(\d+)", normalized):
            head_n = int(match.group(1))
            tail_n = int(match.group(3))
            start = max(1, head_n - tail_n + 1)
            end = head_n
            add_target(match.group(2), f"{start}-{end}")

        for match in re.finditer(r"(?:cat|cat\s+-n|nl\s+-ba)\s+([^\s|;&]+)\s*\|\s*head\s+-n\s+(\d+)", normalized):
            add_target(match.group(1), f"1-{match.group(2)}")

        for match in re.finditer(r"head\s+-n\s+(\d+)\s+([^\s|;&]+)", normalized):
            add_target(match.group(2), f"1-{match.group(1)}")

        for match in re.finditer(
            r"(?:cat|cat\s+-n|nl\s+-ba)\s+([^\s|;&]+)(?:\s+2?>&?1)?\s*\|\s*head\s+(?:-n\s+|-)(\d+)",
            normalized,
        ):
            add_target(match.group(1), f"1-{match.group(2)}")

        for match in re.finditer(r"(?:cat|cat\s+-n|nl\s+-ba)\s+([^\s|;&]+)\s*\|\s*tail\s+-n\s+\+?(\d+)", normalized):
            add_target(match.group(1), f"{match.group(2)}-tail")

        for match in re.finditer(r"tail\s+-n\s+(\d+)\s+([^\s|;&]+)", normalized):
            add_target(match.group(2), "tail")

        path_end = r"(?=\s*(?:$|[;]|2?>&?1\s*(?:$|[;])))"

        for match in re.finditer(r"nl\s+-ba\s+([^\s|;&]+)" + path_end, normalized):
            add_target(match.group(1), "full")

        for match in re.finditer(r"cat\s+-n\s+([^\s|;&]+)" + path_end, normalized):
            add_target(match.group(1), "full")

        for match in re.finditer(r"cat\s+<\s*([^\s|;&]+)", normalized):
            add_target(match.group(1), "full")

        for match in re.finditer(r"cat\s+([^\s|;&]+)" + path_end, normalized):
            add_target(match.group(1), "full")

        for match in re.finditer(r"(?:python|python3)\s+-c\s+['\"].*?open\((['\"])([^'\"]+)\1.*?['\"]", normalized):
            add_target(match.group(2), "full")

        for match in re.finditer(r"(?:python|python3)\s+-c\s+['\"].*?open\((['\"])([^'\"]+)\1.*?['\"]\s*\|\s*head\s+-n\s+(\d+)", normalized):
            add_target(match.group(2), f"1-{match.group(3)}")

        search_patterns = [
            re.compile(r"(?:grep|rg)\s+[^|;&\n]*?\s+([^\s|;&]+)\s*$"),
            re.compile(r"(?:grep|rg)\s+[^|;&\n]*?\s+([^\s|;&]+)\s*\|\s*head\s+-\d+"),
        ]
        for pattern in search_patterns:
            for match in pattern.finditer(normalized):
                add_target(match.group(1), "search")

        return [(path, line_range) for path, line_range in targets if cls._looks_like_source_file(path)]

    @staticmethod
    def _looks_like_source_file(path: str) -> bool:
        path_obj = Path(path)
        if not path_obj.name:
            return False
        if path != path.strip("'\"") or any(char in path for char in "()<>="):
            return False
        if any(part in {"node_modules", ".git", "__pycache__"} for part in path_obj.parts):
            return False
        return "." in path_obj.name

    @staticmethod
    def _action_may_change_code(action_text: str) -> bool:
        lowered = action_text.lower()
        mutation_patterns = [
            r"\bsed\s+-i\b",
            r"\bperl\s+-\S*i\b",
            r"\bapply_patch\b",
            r"\bgit\s+checkout\b",
            r"\bpython3?\b.*\bwrite_text\b",
            r"\bpython3?\b.*\bopen\([^)]*,\s*['\"]w",
            r"\bcat\s+<<.*>",
            r"\btee\s+",
            r"\bnpm\s+install\b",
            r"\byarn\s+add\b",
            r"\bpip\s+install\b",
        ]
        if any(re.search(pattern, action_text, re.S) for pattern in mutation_patterns):
            return True
        # A generic `>` also matches stderr redirects and Python comparisons
        # in heredocs. Restrict redirects to common shell writer commands.
        if re.search(
            r"\b(?:cat|printf|echo)\b[^\n]*(?<![<>=!])\s>\s*[^\s&;]+",
            action_text,
        ):
            return True
        if any(token in lowered for token in ["git diff", "git status", "grep ", "rg ", "sed -n", "cat ", "nl -ba"]):
            return False
        return False

    @classmethod
    def _extract_write_targets(cls, action_text: str) -> set[str]:
        targets: set[str] = set()
        normalized = action_text.replace("\\\n", " ")

        patterns = [
            re.compile(r"sed\s+-i(?:\s+['\"][\s\S]*?['\"])+\s+([^\s|;&]+)"),
            re.compile(r"(?:cp|mv)\s+[^\s|;&]+\s+([^\s|;&]+)"),
            re.compile(r"tee\s+([^\s|;&]+)"),
            re.compile(
                r"\b(?:cat|printf|echo)\b[^\n]*(?<![<>=!])\s>\s*([^\s&;]+)"
            ),
            re.compile(r"\bopen\(\s*(['\"])([^'\"]+)\1\s*,\s*['\"](?:w|a)"),
            re.compile(r"(?:Path|pathlib\.Path)\(\s*(['\"])([^'\"]+)\1\s*\)\.write_text\("),
        ]
        for pattern in patterns:
            for match in pattern.finditer(normalized):
                path = match.group(match.lastindex or 1)
                normalized_path = cls._normalize_cache_path(path)
                if cls._looks_like_source_file(normalized_path):
                    targets.add(normalized_path)

        path_vars: dict[str, str] = {}
        for match in re.finditer(
            r"\b([A-Za-z_]\w*)\s*=\s*(?:Path|pathlib\.Path)\(\s*(['\"])([^'\"]+)\2\s*\)",
            normalized,
        ):
            path_vars[match.group(1)] = match.group(3)
        for var_name, path in path_vars.items():
            if re.search(rf"\b{re.escape(var_name)}\.write_text\(", normalized):
                normalized_path = cls._normalize_cache_path(path)
                if cls._looks_like_source_file(normalized_path):
                    targets.add(normalized_path)
            if re.search(
                rf"\bopen\(\s*{re.escape(var_name)}\s*,\s*['\"](?:w|a)",
                normalized,
            ):
                normalized_path = cls._normalize_cache_path(path)
                if cls._looks_like_source_file(normalized_path):
                    targets.add(normalized_path)

        string_vars: dict[str, str] = {}
        for match in re.finditer(
            r"\b([A-Za-z_]\w*)\s*=\s*(['\"])([^'\"]+)\2",
            normalized,
        ):
            string_vars[match.group(1)] = match.group(3)
        for var_name, path in string_vars.items():
            if re.search(
                rf"\bopen\(\s*{re.escape(var_name)}\s*,\s*['\"](?:w|a)",
                normalized,
            ):
                normalized_path = cls._normalize_cache_path(path)
                if cls._looks_like_source_file(normalized_path):
                    targets.add(normalized_path)
        return targets

    @staticmethod
    def _is_python_heredoc_action(action_text: str) -> bool:
        return bool(re.search(r"\bpython3?\s+-\s*<<", action_text))

    @staticmethod
    def _is_scratch_path(path: str) -> bool:
        path_obj = Path(path)
        parts = path_obj.parts
        if not parts:
            return False
        if path.startswith(("/tmp/", "/var/tmp/")):
            return True
        return any(part in {"tmp", ".tmp"} for part in parts[:-1])

    @classmethod
    def _extract_scratch_write_targets(cls, action_text: str) -> set[str]:
        return {target for target in cls._extract_write_targets(action_text) if cls._is_scratch_path(target)}

    @classmethod
    def _action_changes_source_code(cls, action_text: str) -> bool:
        if not cls._action_may_change_code(action_text):
            return False
        write_targets = cls._extract_write_targets(action_text)
        if write_targets:
            return any(not cls._is_scratch_path(target) for target in write_targets)
        if cls._is_python_heredoc_action(action_text):
            # A verification heredoc with an unresolvable write target should
            # not silently reset the source-edit budgets.
            return False
        # Dependency/environment mutations can affect subsequent test behavior
        # even when a source file target is unavailable. Other ambiguous shell
        # commands must not reset the source-edit budgets.
        return bool(
            re.search(r"\b(?:apply_patch|npm\s+install|yarn\s+add|pip\s+install)\b", action_text)
        )

    @classmethod
    def _is_scratch_experiment_action(cls, action_text: str) -> bool:
        if cls._extract_scratch_write_targets(action_text):
            return True
        for match in re.finditer(r"\bpython3?\s+([^\s|;&]+)", action_text):
            command_path = cls._normalize_cache_path(match.group(1).strip("'\""))
            if command_path.endswith(".py") and cls._is_scratch_path(command_path):
                return True
        return False

    @staticmethod
    def _trace_status_relative_path(path: str) -> str:
        normalized = path.replace("\\", "/").removeprefix("./")
        return normalized.removeprefix("/testbed/")

    @classmethod
    def _is_test_path(cls, path: str) -> bool:
        relative = cls._trace_status_relative_path(path)
        name = Path(relative).name
        return (
            relative.startswith(("test/", "tests/"))
            or name.startswith("test_")
            or name.endswith("_test.py")
        )

    def _is_tracked_production_path(self, path: str) -> bool:
        relative = self._trace_status_relative_path(path)
        return relative in self._trace_status_tracked_files and not self._is_test_path(relative)

    def _action_changes_tracked_source_code(self, action_text: str) -> bool:
        if not self._action_may_change_code(action_text):
            return False
        # Checkout can replace tracked source or switch branches without an explicit write target.
        if re.search(r"\bgit\s+checkout\b", action_text):
            return True
        return any(self._is_tracked_production_path(path) for path in self._extract_write_targets(action_text))

    def _is_trace_status_scratch_experiment(self, action_text: str) -> bool:
        write_targets = self._extract_write_targets(action_text)
        if write_targets:
            return not any(self._is_tracked_production_path(path) for path in write_targets)
        if self._is_test_command(action_text):
            return False
        for match in re.finditer(r"\bpython3?\s+([^\s|;&]+)", action_text):
            command_path = self._normalize_cache_path(match.group(1).strip("'\""))
            if command_path.endswith(".py") and not self._is_tracked_production_path(command_path):
                return True
        return False

    def _find_cached_read_entry(self, action_text: str) -> ReadCacheEntry | None:
        for path, line_range in self._extract_read_targets(action_text):
            entry = self._read_cache.get((path, line_range))
            if entry and entry.state_epoch == self._read_cache_state_epoch:
                return entry
        return None

    def _maybe_return_cached_read(self, action_text: str) -> dict[str, str] | None:
        if not self._read_cache_enabled:
            return None
        cached = self._find_cached_read_entry(action_text)
        if cached is None:
            return None
        self._read_cache_hits += 1
        message = (
            "READ CACHE: 该段未变化，已读过。\n"
            f"上次读取 step={cached.step}\n"
            f"file={cached.file}\n"
            f"range={cached.line_range}"
        )
        return {"output": message, "returncode": 0}

    def _invalidate_read_cache(self, action_text: str) -> None:
        changed_paths = self._extract_write_targets(action_text)
        if changed_paths:
            for path in changed_paths:
                for key in list(self._read_cache.keys()):
                    if key[0] == path:
                        del self._read_cache[key]
        else:
            self._read_cache.clear()
        self._read_cache_state_epoch += 1

    def _update_read_cache(self, *, action: dict, output: dict) -> None:
        if not self._read_cache_enabled:
            return None
        action_text = action.get("action", "")
        try:
            returncode = int(output.get("returncode", 1))
        except (TypeError, ValueError):
            returncode = 1
        if returncode != 0:
            return
        output_text = str(output.get("output", "")).lstrip()
        if output_text.startswith("READ CACHE:"):
            return
        if self._action_may_change_code(action_text):
            self._invalidate_read_cache(action_text)
            return
        current_step = self._turn_control_used + 1
        for path, line_range in self._extract_read_targets(action_text):
            self._read_cache[(path, line_range)] = ReadCacheEntry(
                file=path,
                line_range=line_range,
                step=current_step,
                state_epoch=self._read_cache_state_epoch,
            )

    def get_traj_metadata(self) -> dict:
        metadata = {}
        if self._trace_status_old_observation_compaction_enabled:
            # This metadata is written to the trajectory, not sent back through
            # self.messages.  It makes every compacted observation recoverable
            # after the process exits while keeping it out of future prompts.
            metadata["trace_status_old_observation_compaction"] = {
                "enabled": True,
                "cached_raw_outputs": dict(self._trace_status_raw_observation_cache),
            }
        if self._read_cache_enabled:
            metadata.update(
                {
                    "read_cache_enabled": True,
                    "read_cache_hits": self._read_cache_hits,
                    "read_cache_entries": [asdict(entry) for entry in self._read_cache.values()],
                }
            )
        if self._trace_status_bar_enabled or self._trace_status_repeat_block_enabled:
            metadata["trace_status_repeat_block_enabled"] = self._trace_status_repeat_block_enabled
        if self._trace_status_repeat_block_enabled:
            metadata["trace_status_repeat_block_thresholds"] = {
                "max_consecutive_inspections": self._trace_status_repeat_block_max_consecutive_inspections,
                "max_total_inspections_since_edit": (
                    self._trace_status_repeat_block_max_total_inspections_since_edit
                ),
                "max_scratch_experiments_since_edit": (
                    self._trace_status_repeat_block_max_scratch_experiments_since_edit
                ),
                "max_test_phase_tests": self._trace_status_repeat_block_max_test_phase_tests,
                "dynamic_inspection_max_bonus": self._trace_status_repeat_block_dynamic_inspection_max_bonus,
            }
            metadata["trace_status_repeat_block"] = {
                "reads": [asdict(entry) for entry in self._trace_status_reads.values()],
                "searches": [asdict(entry) for entry in self._trace_status_searches.values()],
                "edits": [asdict(entry) for entry in self._trace_status_edits.values()],
                "tests": list(self._trace_status_tests),
                "total_inspections_since_source_edit": self._trace_status_total_inspections_since_source_edit,
                "novel_evidence_count": self._trace_status_novel_evidence_count,
                "dynamic_inspection_bonus": self._trace_status_dynamic_inspection_bonus,
                "effective_total_inspection_limit": (
                    self._trace_status_repeat_block_max_total_inspections_since_edit
                    + (
                        self._trace_status_dynamic_inspection_bonus
                        if self._trace_status_repeat_block_dynamic_inspection_enabled
                        else 0
                    )
                ),
                "scratch_experiments_since_source_edit": self._trace_status_scratch_experiments_since_source_edit,
                "repeat_confirmations_since_source_edit": (
                    self._trace_status_repeat_confirmations_since_source_edit
                ),
                "post_pass_edit_guard_hits": self._trace_status_post_pass_edit_guard_hits,
                "last_source_edit_step": self._trace_status_last_source_edit_step,
            }
        if self._trace_status_bar_enabled:
            metadata["trace_status_bar_enabled"] = True
            metadata["trace_status_bar_interval"] = self._trace_status_bar_interval
            metadata["trace_status_bar_next_action_guidance_enabled"] = (
                self._trace_status_bar_next_action_guidance_enabled
            )
            metadata["trace_status_bar"] = {
                "reads": [asdict(entry) for entry in self._trace_status_reads.values()],
                "searches": [asdict(entry) for entry in self._trace_status_searches.values()],
                "edits": [asdict(entry) for entry in self._trace_status_edits.values()],
                "tests": list(self._trace_status_tests),
                "repeat_notes": list(self._trace_status_repeat_notes),
            }
        if self._trace_action_budget_controller_enabled:
            metadata["trace_action_budget_controller"] = {
                "phase": self._trace_action_budget_phase,
                "remaining": dict(self._trace_action_budget_remaining),
                "transitions": list(self._trace_action_budget_transitions),
            }
        if self._scout_handoff_pricing_messages:
            metadata["scout_handoff_api_calls"] = len(self._scout_handoff_pricing_messages)
            metadata["scout_handoff_status"] = self._scout_handoff_status
            metadata["scout_handoff_transcript"] = self._scout_handoff_transcript
        return metadata

    def _is_turn_control_enabled(self) -> bool:
        return self.config.turn_control

    def _get_turn_control_initial_limit(self) -> int:
        if not self._is_turn_control_enabled():
            return 0
        if self.config.turn_control_mode == "fixed":
            return max(0, self.config.turn_control_fixed_limit)
        if self.config.turn_control_mode == "dynamic":
            return max(0, self.config.turn_control_initial_limit)
        return 0

    def _get_turn_control_extension_limit(self) -> int:
        if self.config.turn_control_mode != "dynamic":
            return 0
        return max(0, self.config.turn_control_extension_limit)

    def _apply_turn_control_before_step(self) -> None:
        if not self._is_turn_control_enabled():
            return
        if self._turn_control_effective_limit <= 0:
            return
        if self._turn_control_used < self._turn_control_effective_limit:
            return
        if (
            self.config.turn_control_mode == "dynamic"
            and not self._turn_control_extension_used
            and self._turn_control_effective_limit == self._get_turn_control_initial_limit()
            and self._get_turn_control_extension_limit() > self._turn_control_effective_limit
        ):
            self._turn_control_effective_limit = self._get_turn_control_extension_limit()
            self._turn_control_extension_used = True
            self._turn_control_extension_notice_pending = True
            return
        raise LimitsExceeded()

    def _append_turn_control_reminder(self, observation: str) -> str:
        if not self._is_turn_control_enabled() or not self.config.turn_control_reminder:
            return observation
        if self._turn_control_effective_limit <= 0:
            return observation

        next_turn_used = self._turn_control_used + 1
        turns_left = max(0, self._turn_control_effective_limit - next_turn_used)
        reminder_parts = []

        if self._turn_control_extension_notice_pending:
            extension_turns = self._turn_control_effective_limit - self._get_turn_control_initial_limit()
            reminder_parts.append(
                "ENVIRONMENT REMINDER: You have used up all turns but have not yet completed the task. "
                f"You are granted an additional {extension_turns} turns to continue and complete the task."
            )
            self._turn_control_extension_notice_pending = False

        if self.config.turn_control_mode == "dynamic" and self._turn_control_extension_used:
            reminder_parts.append(
                f"ENVIRONMENT REMINDER: Extension active. You have {turns_left} turns left. "
                "Focus on final verification and submission."
            )
        else:
            reminder_parts.append(
                f"ENVIRONMENT REMINDER: You have {turns_left} turns left to finish the task. "
                "Prioritize the minimal set of actions needed for a correct patch and verification."
            )

        return observation + "\n\n" + "\n".join(reminder_parts)

    def _trace_action_budget_category(self, action_text: str) -> str | None:
        """Classify an action into the one budget it primarily consumes.

        Edits take precedence over tests so a combined "patch && pytest" action
        cannot bypass the edit budget.  Unclassified commands (for example
        ``git diff`` and the final submit marker) remain available in every phase.
        """
        if self._action_may_change_code(action_text):
            return "edit"
        if self._is_test_command(action_text):
            return "test"
        if self._is_trace_status_inspection(action_text):
            return "search"
        return None

    def _advance_trace_action_budget_phase(self) -> None:
        if not self._trace_action_budget_controller_enabled:
            return
        phases = ("search", "edit", "test", "submit")
        while self._trace_action_budget_phase != "submit":
            phase_index = phases.index(self._trace_action_budget_phase)
            if self._trace_action_budget_remaining[self._trace_action_budget_phase] > 0:
                return
            previous = self._trace_action_budget_phase
            self._trace_action_budget_phase = phases[phase_index + 1]
            transition = (
                f"{previous} budget exhausted; forced into {self._trace_action_budget_phase} phase."
            )
            self._trace_action_budget_transitions.append(transition)

    def _trace_action_budget_block_message(self, action_text: str) -> str:
        if not self._trace_action_budget_controller_enabled:
            return ""
        category = self._trace_action_budget_category(action_text)
        if not category:
            return ""
        phases = ("search", "edit", "test", "submit")
        category_index = phases.index(category)
        phase_index = phases.index(self._trace_action_budget_phase)
        if self._trace_action_budget_remaining[category] > 0 and category_index >= phase_index:
            return ""
        next_action = {
            "edit": "make the minimal patch from the evidence already gathered",
            "test": "run the most targeted verification, then submit if it supports the patch",
            "submit": "inspect the diff if needed, then submit the best patch now",
        }[self._trace_action_budget_phase]
        return (
            f"ACTION BUDGET BLOCKED: {category} actions are unavailable in the "
            f"{self._trace_action_budget_phase} phase. Do not execute this command. "
            f"Next: {next_action}.\n\n{self._trace_action_budget_status_message()}"
        )

    def _consume_trace_action_budget(self, action_text: str) -> None:
        if not self._trace_action_budget_controller_enabled:
            return
        category = self._trace_action_budget_category(action_text)
        if category and self._trace_action_budget_remaining[category] > 0:
            self._trace_action_budget_remaining[category] -= 1
        self._advance_trace_action_budget_phase()

    def _trace_action_budget_status_message(self) -> str:
        if not self._trace_action_budget_controller_enabled:
            return ""
        remaining = self._trace_action_budget_remaining
        transition = self._trace_action_budget_transitions[-1] if self._trace_action_budget_transitions else ""
        message = (
            "ACTION BUDGET: "
            f"phase={self._trace_action_budget_phase}; "
            f"search={remaining['search']}; edit={remaining['edit']}; test={remaining['test']}."
        )
        if transition:
            message += f" ENVIRONMENT REMINDER: {transition}"
        return message

    def _update_trace_status_bar(self, *, action: dict, output: dict) -> str:
        if not (self._trace_status_bar_enabled or self._trace_status_repeat_block_enabled):
            return ""
        action_text = action.get("action", "")
        current_step = self._turn_control_used + 1
        observation_succeeded = output.get("returncode", 0) == 0
        action_may_change_code = self._action_may_change_code(action_text)
        action_changes_source_code = self._action_changes_tracked_source_code(action_text)
        novel_evidence = self._trace_status_observes_novel_evidence(
            action_text=action_text, output=output, observation_succeeded=observation_succeeded
        )

        if self._trace_status_repeat_block_enabled:
            if novel_evidence:
                self._trace_status_novel_evidence_count += 1
                if self._trace_status_repeat_block_dynamic_inspection_enabled:
                    self._trace_status_dynamic_inspection_bonus = min(
                        self._trace_status_repeat_block_dynamic_inspection_max_bonus,
                        self._trace_status_dynamic_inspection_bonus + 1,
                    )
            # Failed reads/searches yield no repository evidence. They must not
            # make a fallback command look like repeated work, nor consume the
            # consecutive-inspection budget.
            if self._is_trace_status_inspection(action_text) and observation_succeeded:
                self._trace_status_consecutive_inspections += 1
                self._trace_status_total_inspections_since_source_edit += 1
            else:
                self._trace_status_consecutive_inspections = 0
            if (
                not action_changes_source_code
                and self._is_trace_status_scratch_experiment(action_text)
                and observation_succeeded
            ):
                self._trace_status_scratch_experiments_since_source_edit += 1

        if action_changes_source_code:
            self._trace_status_last_source_edit_step = current_step

        if action_may_change_code:
            if self._trace_status_repeat_block_enabled:
                if action_changes_source_code:
                    self._trace_status_repeat_block_counts = {}
                    self._trace_status_repeat_confirmations_since_source_edit = 0
                    self._trace_status_repeat_warned_since_source_edit = set()
                    self._trace_status_total_inspections_since_source_edit = 0
                    self._trace_status_dynamic_inspection_bonus = 0
                    self._trace_status_novel_evidence_count = 0
                    self._trace_status_read_coverage = {}
                    self._trace_status_seen_search_hits = set()
                    self._trace_status_seen_test_commands = set()
                    self._trace_status_seen_test_failures = set()
                    self._trace_status_seen_scratch_probes = set()
                    self._trace_status_scratch_experiments_since_source_edit = 0
            write_targets = self._extract_write_targets(action_text)
            if not write_targets:
                write_targets = {"<unknown source edit>"}
            for target in write_targets:
                entry = self._trace_status_edits.get(target)
                if entry:
                    entry.last_step = current_step
                    entry.count += 1
                else:
                    self._trace_status_edits[target] = TraceStatusEditEntry(
                        target=target,
                        first_step=current_step,
                        last_step=current_step,
                    )
        elif observation_succeeded:
            for path, line_range in self._extract_read_targets(action_text):
                key = (path, line_range)
                if key in self._trace_status_reads:
                    entry = self._trace_status_reads[key]
                    entry.last_step = current_step
                    entry.count += 1
                    self._remember_trace_status_repeat_note(
                        f"Already read `{path}` ({line_range}) at step {entry.first_step}; avoid rereading unless it changed."
                    )
                else:
                    self._trace_status_reads[key] = TraceStatusReadEntry(
                        file=path,
                        line_range=line_range,
                        first_step=current_step,
                        last_step=current_step,
                    )

            search_signature = self._extract_search_signature(action_text)
            if search_signature:
                entry = self._trace_status_searches.get(search_signature)
                if entry:
                    entry.last_step = current_step
                    entry.count += 1
                    self._remember_trace_status_repeat_note(
                        f"Search `{search_signature}` has already been tried; use its result instead of another variant."
                    )
                else:
                    self._trace_status_searches[search_signature] = TraceStatusSearchEntry(
                        signature=search_signature,
                        first_step=current_step,
                        last_step=current_step,
                    )

        if self._is_test_command(action_text):
            self._trace_status_tests.append(
                {
                    "step": current_step,
                    "returncode": output.get("returncode"),
                    "command": self._shorten_command(action_text, 90),
                    "signature": self._command_signature(action_text),
                }
            )
            self._trace_status_tests = self._trace_status_tests[-self._trace_status_bar_max_items :]

        warning = ""
        if self._trace_status_repeat_block_enabled and self._trace_status_repeat_block_soft_warning_enabled:
            counts = {
                "consecutive inspections": self._trace_status_consecutive_inspections,
                "total inspections": self._trace_status_total_inspections_since_source_edit,
                "scratch experiments": self._trace_status_scratch_experiments_since_source_edit,
                "tests": self._trace_status_test_phase_tests,
            }
            reached = [
                (name, counts[name], limit)
                for name, limit in self._trace_status_repeat_block_warning_limits.items()
                if counts[name] >= limit and name not in self._trace_status_repeat_warned_since_source_edit
            ]
            if reached:
                self._trace_status_repeat_warned_since_source_edit.update(name for name, _, _ in reached)
                warning = (
                    "PACE WARNING: "
                    + ", ".join(f"{name} {count}/{limit}" for name, count, limit in reached)
                    + ". These actions still ran; plan the next edit or focused verification before the hard limit."
                )
        if self._trace_status_bar_enabled and self._should_render_trace_status_bar_checkpoint(current_step):
            self._trace_status_last_rendered_step = current_step
            return "\n\n".join(filter(None, (warning, self._render_trace_status_bar())))
        return warning

    def _should_render_trace_status_bar_checkpoint(self, current_step: int) -> bool:
        return current_step - self._trace_status_last_rendered_step >= self._trace_status_bar_interval

    def _trace_status_repeat_block_message(self, action_text: str) -> str:
        # A compound edit-and-inspect command may mention an already-read file,
        # but its primary action changes source and must never be treated as a
        # repeated inspection. The edit invalidates the earlier evidence.
        if self._action_changes_tracked_source_code(action_text):
            return ""
        repeated: list[str] = []
        last_edit_step = self._trace_status_last_source_edit_step
        search_signature = self._extract_search_signature(action_text)
        if not search_signature:
            for path, line_range in self._extract_read_targets(action_text):
                if not self._trace_status_read_has_new_coverage(action_text):
                    repeated.append(f"read `{path}` ({line_range})")
        if (
            search_signature
            and (entry := self._trace_status_searches.get(search_signature))
            and entry.last_step > last_edit_step
        ):
            repeated.append(f"searched `{search_signature}` (step {entry.first_step})")
        if not repeated:
            return ""
        details = "; ".join(dict.fromkeys(repeated))
        return (
            "BLOCKED: You have already "
            f"{details} earlier. Treat that prior result as evidence instead of rereading it. "
            "Replan before issuing another inspection: state one root-cause hypothesis, then choose "
            "exactly one focused next action -- reproduce/run the target test, inspect the production "
            "module that owns the failing behavior, or make a minimal patch. Do not broaden the search "
            "or add defensive edits until that hypothesis is tested. "
            "If this evidence is essential, rerun it with the comment # TRACE_STATUS_REPEAT_CONFIRMED."
        )

    def _is_trace_status_inspection(self, action_text: str) -> bool:
        """Return whether an action only gathers source evidence.

        Tests and edits are progress and reset the inspection streak. This is kept
        inside the repeat-block feature so disabled runs retain their prior flow.
        """
        return (
            not self._action_may_change_code(action_text)
            and not self._is_test_command(action_text)
            and bool(self._extract_read_targets(action_text) or self._extract_search_signature(action_text))
        )

    def _trace_status_inspection_budget_message(self, action_text: str) -> str:
        if not self._is_trace_status_inspection(action_text):
            return ""
        limit = self._trace_status_repeat_block_max_consecutive_inspections
        if self._trace_status_consecutive_inspections < limit:
            return ""
        return (
            "BLOCKED: You have already executed "
            f"{self._trace_status_consecutive_inspections} consecutive source read/search actions without "
            f"an edit or test (current limit: {limit}). Replan from the gathered evidence before inspecting more: state one root-cause "
            "hypothesis, then reproduce/run the target test, inspect the production module that owns the "
            "failing behavior, or make a minimal patch. Do not broaden the search or add defensive edits "
            "until that hypothesis is tested. If one more inspection is essential, rerun it with the comment "
            "# TRACE_STATUS_REPEAT_CONFIRMED."
        )

    def _trace_status_total_inspection_budget_message(self, action_text: str) -> str:
        if not self._is_trace_status_inspection(action_text):
            return ""
        limit = self._trace_status_repeat_block_max_total_inspections_since_edit
        if self._trace_status_repeat_block_dynamic_inspection_enabled:
            limit += self._trace_status_dynamic_inspection_bonus
        if self._trace_status_total_inspections_since_source_edit < limit:
            return ""
        return (
            "BLOCKED: You have already executed "
            f"{self._trace_status_total_inspections_since_source_edit} total source read/search actions since "
            f"the last source edit (current limit: {limit}). Tests or scratch experiments may have been interleaved, but this is still "
            "an inspection-heavy loop without changing repository code. Replan before inspecting more: state "
            "one root-cause hypothesis and either make a minimal source edit, run one targeted test, or submit "
            "if the patch is already verified. If one more inspection is essential, rerun it with the comment "
            "# TRACE_STATUS_REPEAT_CONFIRMED."
        )

    def _trace_status_scratch_experiment_budget_message(self, action_text: str) -> str:
        if not self._is_trace_status_scratch_experiment(action_text):
            return ""
        if (
            self._trace_status_scratch_experiments_since_source_edit
            < self._trace_status_repeat_block_max_scratch_experiments_since_edit
        ):
            return ""
        return (
            "BLOCKED: You have already executed "
            f"{self._trace_status_scratch_experiments_since_source_edit} scratch experiment actions since "
            "the last source edit. Writing or running temporary scripts is no longer moving the repository "
            "forward. Convert the strongest observation into one minimal source edit, then run exactly one "
            "targeted verification. If one more scratch experiment is essential, rerun it with the comment "
            "# TRACE_STATUS_REPEAT_CONFIRMED."
        )

    def _trace_status_post_pass_edit_guard_message(self, action_text: str) -> tuple[str, bool]:
        if not self._action_changes_tracked_source_code(action_text):
            return "", False
        if self._last_successful_test_action <= self._last_source_edit_action:
            return "", False
        if self._trace_status_post_pass_edit_guard_hits:
            return (
                "POST-PASS EDIT GUARD: A source edit already followed a successful verification and no "
                "new test failure has been recorded. This edit is allowed, but it must be justified by "
                "concrete failure output already observed in the trace. Do not make speculative changes; "
                "run the smallest relevant test immediately afterward.",
                True,
            )
        return (
            "POST-PASS EDIT GUARD: The current patch has passed verification and no new failure has been "
            "observed. This edit is allowed once, but only if it addresses a concrete defect from the prior "
            "output. After it, run the smallest relevant test; do not continue speculative optimization.",
            True,
        )

    def _trace_status_test_phase_budget_message(self, action_text: str) -> str:
        if not self._is_test_command(action_text):
            return ""
        if self._trace_status_test_phase_tests < self._trace_status_repeat_block_max_test_phase_tests:
            return ""
        return (
            "BLOCKED: You have already executed "
            f"{self._trace_status_test_phase_tests} test actions since the last edit. "
            "Read/search actions may have been interleaved, but this is still the same test-debug phase. "
            "Replan before running another test: state the failing behavior, the concrete hypothesis, and "
            "the next edit or final submission criterion. If one more test is essential, rerun it with the "
            "comment # TRACE_STATUS_REPEAT_CONFIRMED."
        )

    def _trace_status_large_read_block_message(self, action_text: str) -> str:
        """Return a confirmation gate for explicit source reads over the configured limit."""
        limit = self._trace_status_large_read_block_max_lines
        oversized: list[tuple[str, int, str]] = []
        total_requested_lines = 0
        for path, line_range in self._extract_read_targets(action_text):
            match = re.fullmatch(r"(\d+)-(\d+)", line_range)
            if not match:
                continue
            start, end = map(int, match.groups())
            requested_lines = max(0, end - start + 1)
            total_requested_lines += requested_lines
            if requested_lines > limit:
                oversized.append((path, requested_lines, line_range))
        if not oversized and total_requested_lines <= limit:
            return ""

        details = "; ".join(
            f"`{path}` ({line_range}: {requested_lines} lines)"
            for path, requested_lines, line_range in oversized
        )
        if not details:
            details = f"the combined source-read windows ({total_requested_lines} lines)"
        return (
            "BLOCKED: This file-read operation exceeds the configured limit of "
            f"{limit} lines: {details}. Total requested source-read length: {total_requested_lines} lines. "
            "The requested read length is shown above (for example, "
            "a `1-2400` window reads 2400 lines). Narrow the range to at most "
            f"{limit} lines or search for the relevant symbol first. If the full window is essential, "
            "rerun the exact command with the comment # TRACE_STATUS_LARGE_READ_BLOCK_CONFIRMED."
        )

    def _progressive_read_action(self, action_text: str) -> tuple[str, list[tuple[str, int, int, int]]]:
        """Shrink explicit sed windows to their first chunk without adding a turn."""
        if (
            not self._trace_status_progressive_read_enabled
            or "TRACE_STATUS_PROGRESSIVE_READ_CONFIRMED" in action_text
        ):
            return action_text, []
        chunk = self._trace_status_progressive_read_chunk_lines
        notes: list[tuple[str, int, int, int]] = []

        def replace(match: re.Match) -> str:
            start, end = int(match.group(1)), int(match.group(2))
            if end - start + 1 <= chunk:
                return match.group(0)
            first_end = start + chunk - 1
            notes.append((match.group(3), start, first_end, end))
            return f"sed -n '{start},{first_end}p' {match.group(3)}"

        # Deliberately constrain this to direct source reads.  Rewriting a sed
        # expression embedded in arbitrary shell/pipeline syntax could change its meaning.
        rewritten = re.sub(
            r"sed\s+-n\s+'(\d+),(\d+)p'\s+([^\s|;&]+)", replace, action_text
        )
        return rewritten, notes

    @staticmethod
    def _progressive_read_message(notes: list[tuple[str, int, int, int]]) -> str:
        details = "; ".join(
            f"`{path}` requested {start}-{end}, returned {start}-{first_end}; remaining {first_end + 1}-{end}"
            for path, start, first_end, end in notes
        )
        return (
            "PROGRESSIVE SOURCE READ: oversized source window was read in its first chunk; "
            + details
            + ". Do not continue reading the remainder sequentially by default. First narrow the target with a "
            "more precise search such as `rg -n <symbol> <file>`, then read only the smallest relevant window. "
            "Continue into the remaining range only if the needed definition or control flow clearly crosses this "
            "chunk boundary. If the full window is truly required, rerun the exact full command with "
            "# TRACE_STATUS_PROGRESSIVE_READ_CONFIRMED."
        )

    @staticmethod
    def _trace_status_ranges_overlap(left: str, right: str) -> bool:
        if "search" in {left, right}:
            return False
        if left in {"full", "tail"}:
            return True
        if right in {"full", "tail"}:
            return False
        left_match = re.fullmatch(r"(\d+)-(\d+)", left)
        right_match = re.fullmatch(r"(\d+)-(\d+)", right)
        if not left_match or not right_match:
            return left == right
        left_start, left_end = map(int, left_match.groups())
        right_start, right_end = map(int, right_match.groups())
        overlap = min(left_end, right_end) - max(left_start, right_start) + 1
        requested_size = right_end - right_start + 1
        return overlap / requested_size >= 0.8

    @staticmethod
    def _numeric_range(line_range: str) -> tuple[int, int] | None:
        match = re.fullmatch(r"(\d+)-(\d+)", line_range)
        return tuple(map(int, match.groups())) if match else None

    def _trace_status_read_has_new_coverage(self, action_text: str) -> bool:
        for path, line_range in self._extract_read_targets(action_text):
            requested = self._numeric_range(line_range)
            if requested is None:
                if (path, line_range) not in self._trace_status_reads:
                    return True
                continue
            start, end = requested
            cursor = start
            for left, right in self._trace_status_read_coverage.get(path, []):
                if right < cursor:
                    continue
                if left > cursor:
                    return True
                cursor = right + 1
                if cursor > end:
                    break
            if cursor <= end:
                return True
        return False

    def _trace_status_record_read_coverage(self, action_text: str) -> None:
        for path, line_range in self._extract_read_targets(action_text):
            requested = self._numeric_range(line_range)
            if requested is None:
                continue
            ranges = sorted([*self._trace_status_read_coverage.get(path, []), requested])
            merged: list[tuple[int, int]] = []
            for start, end in ranges:
                if merged and start <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            self._trace_status_read_coverage[path] = merged

    @staticmethod
    def _trace_status_search_hits(output: dict) -> set[str]:
        hits = set()
        for line in str(output.get("output", "")).splitlines():
            match = re.match(r"(.+?):(\d+)(?::|-)", line)
            if match:
                hits.add(f"{match.group(1)}:{match.group(2)}")
        return hits

    def _trace_status_observes_novel_evidence(
        self, *, action_text: str, output: dict, observation_succeeded: bool
    ) -> bool:
        if self._action_changes_tracked_source_code(action_text):
            return False
        if observation_succeeded and (search_signature := self._extract_search_signature(action_text)):
            hits = self._trace_status_search_hits(output) or {f"query:{search_signature}"}
            novel = bool(hits - self._trace_status_seen_search_hits)
            self._trace_status_seen_search_hits.update(hits)
            return novel
        if observation_succeeded and self._extract_read_targets(action_text):
            novel = self._trace_status_read_has_new_coverage(action_text)
            self._trace_status_record_read_coverage(action_text)
            return novel
        if self._is_test_command(action_text):
            command = self._command_signature(action_text)
            novel = command not in self._trace_status_seen_test_commands
            self._trace_status_seen_test_commands.add(command)
            if not observation_succeeded:
                failure = self._trace_status_failure_signature(output)
                novel = novel or failure not in self._trace_status_seen_test_failures
                self._trace_status_seen_test_failures.add(failure)
            return novel
        if observation_succeeded and self._is_trace_status_scratch_experiment(action_text):
            probe = self._command_signature(action_text)
            novel = probe not in self._trace_status_seen_scratch_probes
            self._trace_status_seen_scratch_probes.add(probe)
            return novel
        return False

    @staticmethod
    def _trace_status_failure_signature(output: dict) -> str:
        text = str(output.get("output", ""))
        exception = re.search(r"\b[A-Za-z_][\w.]*(?:Error|Exception)\b", text)
        failed_test = re.search(r"(?:FAILED\s+)?([\w./-]+::[\w\[\]-]+)", text)
        frame = re.search(r'File "([^"]+)", line (\d+)', text)
        return "|".join(
            [
                exception.group(0) if exception else f"returncode={output.get('returncode')}",
                failed_test.group(1) if failed_test else "",
                f"{frame.group(1)}:{frame.group(2)}" if frame else "",
            ]
        )

    def _remember_trace_status_repeat_note(self, note: str) -> None:
        if note in self._trace_status_repeat_notes:
            return
        self._trace_status_repeat_notes.append(note)
        self._trace_status_repeat_notes = self._trace_status_repeat_notes[-self._trace_status_bar_max_items :]

    @staticmethod
    def _shorten_command(command: str, limit: int) -> str:
        normalized = " ".join(str(command).split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)] + "..."

    @staticmethod
    def _command_signature(command: str) -> str:
        return " ".join(str(command).split())

    @staticmethod
    def _format_trace_status_step(first_step: int, last_step: int | None = None, count: int = 1) -> str:
        last_step = first_step if last_step is None else last_step
        if count > 1 and last_step != first_step:
            return f"s{first_step}->s{last_step} x{count}"
        if count > 1:
            return f"s{first_step} x{count}"
        return f"s{first_step}"

    @classmethod
    def _extract_search_signature(cls, action_text: str) -> str:
        normalized = " ".join(action_text.replace("\\\n", " ").split())
        match = re.search(r"\b(?:grep|rg)\b\s+[^'\"]*['\"]([^'\"]+)['\"]\s+([^|;&]+)", normalized)
        if match:
            return cls._shorten_command(f"{match.group(1)} @ {match.group(2)}", 90)
        match = re.search(r"\bfind\b\s+([^|;&]+)", normalized)
        if match:
            return cls._shorten_command(f"find {match.group(1)}", 90)
        return ""

    def _render_trace_status_bar(self) -> str:
        max_items = self._trace_status_bar_max_items
        read_entries = sorted(self._trace_status_reads.values(), key=lambda entry: entry.last_step, reverse=True)[
            :max_items
        ]
        searches = sorted(self._trace_status_searches.values(), key=lambda entry: entry.last_step, reverse=True)[
            :max_items
        ]
        edits = sorted(self._trace_status_edits.values(), key=lambda entry: entry.last_step, reverse=True)[:max_items]
        tests = self._trace_status_tests[-max_items:]

        lines = ["<trace_status>", "Known context from this run:"]
        if read_entries:
            lines.append("Read:")
            for entry in read_entries:
                step = self._format_trace_status_step(entry.first_step, entry.last_step, entry.count)
                lines.append(f"- {step} `{entry.file}` {entry.line_range}")
        if searches:
            lines.append("Searched:")
            for entry in searches:
                step = self._format_trace_status_step(entry.first_step, entry.last_step, entry.count)
                lines.append(f"- {step} `{entry.signature}`")
        if edits:
            lines.append("Edited:")
            for entry in edits:
                step = self._format_trace_status_step(entry.first_step, entry.last_step, entry.count)
                lines.append(f"- {step} `{entry.target}`")
        if tests:
            lines.append("Tests:")
            for test in tests:
                lines.append(f"- s{test['step']} rc={test['returncode']} `{test['command']}`")
        if self._trace_status_repeat_notes:
            lines.append("Avoid repeating:")
            lines.extend(f"- {note}" for note in self._trace_status_repeat_notes[-3:])
        if self._trace_status_bar_next_action_guidance_enabled:
            lines.append("Next action guidance:")
            lines.append(f"- {self._trace_status_next_action_guidance()}")
        lines.append("</trace_status>")
        return "\n".join(lines)

    def _trace_status_next_action_guidance(self) -> str:
        if self._last_successful_test_action:
            return "A test has passed after a source edit; run the issue's exact regression test before submitting."
        if self._last_source_edit_action:
            return "Source changed; run the smallest relevant test or inspect the diff, not more broad exploration."
        if self._trace_status_reads or self._trace_status_searches:
            return "Use the recorded context to choose one concrete hypothesis, then reproduce or patch; avoid rereading the same windows."
        return "Start with one focused search or targeted file read."

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\s*\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        if len(actions) > 1:
            logger.warning(
                "Model response contained %d bash actions; executing only the first action.",
                len(actions),
            )
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    @staticmethod
    def _is_test_command(action_text: str) -> bool:
        command_lower = action_text.lower()
        test_markers = [
            "pytest",
            "py.test",
            "python -m pytest",
            "python -m unittest",
            "python -m doctest",
            "unittest",
            "doctest",
            "nosetests",
            "tox",
            "nox",
            "manage.py test",
            "django test",
            "tests/runtests.py",
            "runtests.py",
            "make test",
            "make check",
            "make unittest",
            "make pytest",
            "make integration-test",
            "make smoke-test",
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
            "verify",
            "verification",
            "smoke",
            "regression",
        ]
        if any(marker in command_lower for marker in test_markers):
            return True

        if re.search(r"(?<![\w./-])(?:python|python3|py|pytest|ipython|pypy)\s", command_lower):
            verification_markers = [
                "assert ",
                "assert(",
                "raise systemexit",
                "sys.exit(",
                "doctest",
                "unittest",
                "import pytest",
                "from pytest",
                "check=True",
                "check = true",
                "verify",
                "verification",
                "regression",
            ]
            if any(marker in command_lower for marker in verification_markers):
                return True
            test_path_patterns = [
                r"(?:^|[/'\" ])test[^/'\" ]*\.py(?:$|[ '\"&;])",
                r"(?:^|[/'\" ])[^/'\" ]*_test\.py(?:$|[ '\"&;])",
                r"(?:^|[/'\" ])check[^/'\" ]*\.py(?:$|[ '\"&;])",
                r"(?:^|[/'\" ])verify[^/'\" ]*\.py(?:$|[ '\"&;])",
                r"(?:^|[/'\" ])repro[^/'\" ]*\.py(?:$|[ '\"&;])",
            ]
            if any(re.search(pattern, command_lower) for pattern in test_path_patterns):
                return True

        return False

    @staticmethod
    def _is_submit_attempt(output: dict[str, str]) -> bool:
        lines = output.get("output", "").splitlines()
        return any(
            line.strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
            for line in lines
        )

    @staticmethod
    def _submission_review_message() -> str:
        return """SUBMISSION REVIEW REQUIRED: the proposed submission was not executed.

Before submitting, verify all of the following:
1. Only production source files will be submitted; exclude tests, debug scripts, and temporary files.
2. Function parameters and return types are correct.
3. Logic flow covers the issue and relevant edge cases.

If any item is uncertain, continue editing or run a focused test. If ready, resubmit the same command prefixed with:
`echo SUBMISSION_REVIEW_CONFIRMED &&`
"""

    @staticmethod
    def _test_output_indicates_no_real_run(output: dict[str, str]) -> bool:
        output_text = output.get("output", "").lower()
        return "no tests ran" in output_text

    def _test_command_succeeded(self, output: dict[str, str]) -> bool:
        return output.get("returncode", 0) == 0 and not self._test_output_indicates_no_real_run(output)

    def _maybe_submit_after_trace_status_post_verify_budget(self, *, action_text: str, output: dict[str, str]) -> None:
        if not (
            self._trace_status_bar_enabled
            and self._trace_status_post_verify_guard_enabled
            and self._last_source_edit_action
            and self._last_successful_test_action > self._last_source_edit_action
            and self._action_sequence > self._last_successful_test_action
            and not self._action_contains_submit_marker(action_text)
        ):
            return
        if output.get("returncode", 0) != 0:
            return

        if self._is_test_command(action_text):
            self._post_verify_extra_tests += 1
            if self._post_verify_extra_tests <= self._trace_status_post_verify_max_extra_tests:
                return
            reason = "extra successful tests"
        elif self._is_trace_status_inspection_action(action_text):
            self._post_verify_extra_inspections += 1
            if self._post_verify_extra_inspections <= self._trace_status_post_verify_max_extra_inspections:
                return
            reason = "extra inspections"
        else:
            return

        submit_command = self._framework_submit_command()
        submit_output = self.env.execute(submit_command)
        self.add_message(
            "user",
            "FRAMEWORK ACTION: trace status post-verification budget exceeded "
            f"after {reason}. The framework submitted the current diff with `{submit_command}`.",
            framework_submit={
                "command": submit_command,
                "trigger_reason": "trace_status_post_verify_budget",
                "extra_tests": self._post_verify_extra_tests,
                "extra_inspections": self._post_verify_extra_inspections,
            },
        )
        raise Submitted(str(submit_output.get("output", "")))

    def _is_trace_status_inspection_action(self, action_text: str) -> bool:
        lowered = action_text.lower()
        return bool(
            self._extract_read_targets(action_text)
            or self._extract_search_signature(action_text)
            or re.search(r"\bgit\s+(diff|status|show|log)\b", lowered)
        )

    def _block_or_force_submit(self, message: str) -> dict[str, str]:
        self._consecutive_verification_blocks += 1
        if (
            self._repeated_submit_force_threshold > 0
            and self._consecutive_verification_blocks > self._repeated_submit_force_threshold
        ):
            submit_command = self._framework_submit_command()
            submit_output = self.env.execute(submit_command)
            self.add_message(
                "user",
                "FRAMEWORK FORCED SUBMISSION: verification gate blocked submission "
                f"{self._consecutive_verification_blocks} times; executing `{submit_command}`.",
                framework_submit={
                    "command": submit_command,
                    "trigger_reason": "repeated_verification_blocks",
                    "trigger_count": self._consecutive_verification_blocks,
                },
            )
            raise Submitted(str(submit_output.get("output", "")))
        return {"returncode": 126, "output": message}

    @staticmethod
    def _framework_submit_command() -> str:
        return "git add -A && git diff --cached"

    def _deterministic_retrieval_candidates(self) -> list[str]:
        context = str(self.extra_template_vars.get("deterministic_retrieval_context") or "")
        window_candidates = []
        file_candidates = []
        in_candidate_section = False
        for line in context.splitlines():
            if line.strip() in {
                "### Recommended Reading Windows",
                "### Likely Files",
                "### Suggested First Reads",
                "### Pre-searched Candidate Hits",
            }:
                in_candidate_section = True
                continue
            if in_candidate_section and line.startswith("### "):
                break
            if in_candidate_section:
                match = re.match(r"(?:\d+\.|-)\s+`([^`]+)`", line.strip())
                if match:
                    candidate = match.group(1)
                    target = window_candidates if re.search(r":\d+-\d+$", candidate) else file_candidates
                    target.append(candidate)
                    file_match = re.match(r"(.+?):\d+-\d+$", match.group(1))
                    if file_match:
                        target.append(file_match.group(1))
        return window_candidates or file_candidates

    @staticmethod
    def _looks_like_broad_first_search(action_text: str) -> bool:
        lowered = action_text.lower()
        broad_patterns = [
            r"find\s+/\s",
            r"find\s+\.\s+-type\s+f",
            r"find\s+\.\s+.*-name\s+['\"]?\*\.",
            r"ls\s+-r",
            r"grep\s+-r[^a-z]",
            r"grep\s+-rn[^a-z]",
            r"rg\s+--files",
        ]
        return any(re.search(pattern, lowered) for pattern in broad_patterns)

    @staticmethod
    def _looks_like_whole_file_first_read(action_text: str) -> bool:
        stripped = action_text.strip()
        lowered = stripped.lower()
        if "\n" in stripped:
            return False
        if "<<" in stripped:
            return False
        whole_file_patterns = [
            r"^\s*cat\s+[^|;&><]+$",
            r"^\s*nl\s+-ba\s+[^|;&><]+$",
            r"^\s*sed\s+-n\s+['\"]?1,\$p['\"]?\s+[^|;&><]+$",
        ]
        return any(re.search(pattern, lowered) for pattern in whole_file_patterns)

    @staticmethod
    def _extract_primary_read_target(action_text: str) -> str:
        stripped = action_text.strip()
        if "\n" in stripped or "<<" in stripped:
            return "<file>"
        patterns = [
            r"^cat\s+([^\s|;&><]+)",
            r"^nl\s+-ba\s+([^\s|;&><]+)",
            r"^sed\s+-n\s+['\"]?1,\$p['\"]?\s+([^\s|;&><]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, stripped, re.IGNORECASE)
            if match:
                return match.group(1)
        return "<file>"

    @staticmethod
    def _looks_like_large_diagnostic_file(path_text: str) -> bool:
        normalized = path_text.strip().strip("'\"").lower()
        diagnostic_suffixes = (
            ".traj.json",
            ".traj.json.txt",
            ".log",
            ".out",
            ".stderr",
            ".stdout",
        )
        diagnostic_markers = (
            "/logs/",
            "trajectory",
            "traj",
            "task_summary",
            "report.json",
        )
        return normalized.endswith(diagnostic_suffixes) or any(marker in normalized for marker in diagnostic_markers)

    def _whole_file_read_block_message(self, action_text: str) -> str:
        target = self._extract_primary_read_target(action_text)
        stripped_target = target.strip().strip("'\"")
        suggested_window = (
            f"sed -n 'start,endp' {target}"
            if target != "<file>"
            else "sed -n '120,180p' <file>"
        )
        grep_target = target if target != "<file>" else "<file>"
        head_target = target if target != "<file>" else "<file>"
        rationale = "Whole-file reads are disabled for this run."
        if self._looks_like_large_diagnostic_file(stripped_target):
            rationale += " This looks like a trajectory/log-style artifact where full reads are especially wasteful."
        return (
            "ACTION BLOCKED: "
            + rationale
            + "\nUse a focused search first, for example:\n"
            + f"- `{suggested_window}`\n"
            + f"- `head -n 80 {head_target}` or `tail -n 80 {head_target}` if you need initial structure"
        )

    @staticmethod
    def _looks_like_test_contract_chase(action_text: str) -> bool:
        lowered = action_text.lower()
        return (
            ("pytest" in lowered or "tox" in lowered or "runtests" in lowered)
            and any(marker in lowered for marker in ["pexpect", "with subprocess", "old behavior", "legacy"])
        )

    def _action_discipline_hint(self, action_text: str) -> str:
        hints = []
        if self._whole_file_read_hint_enabled and self._looks_like_whole_file_first_read(action_text):
            hints.append("Whole-file read detected; prefer a narrow window first.")
        if self._looks_like_test_contract_chase(action_text):
            hints.append(
                "If an existing test asserts behavior that directly conflicts with the issue's Expected Behavior, do not tune "
                "the production patch to the stale test. Verify syntax, inspect the diff, and keep the issue contract as the source of truth."
            )
        return "\n".join(hints)

    @staticmethod
    def _legacy_test_contract_hint(output_text: str) -> str:
        lowered = output_text.lower()
        if "kerberos auth failure" not in lowered:
            return ""
        if not any(marker in lowered for marker in ["with subprocess", "with pexpect", "pexpect"]):
            return ""
        return (
            "The test output appears to assert the old WinRM Kerberos wording/path. If that contradicts the issue's "
            "Expected Behavior, do not tune the production patch to the old test contract. Keep the issue contract as "
            "the source of truth, verify syntax/diff, and avoid reintroducing legacy pexpect/subprocess wording."
        )

    def _first_action_guard_failure(self, action_text: str) -> str:
        if self._turn_control_used != 0:
            return ""
        if not self._parse_boolish(getattr(self.config, "deterministic_retrieval_first_action_guard", False)):
            return ""
        if "deterministic_retrieval_context" not in self.extra_template_vars:
            return ""

        candidates = self._deterministic_retrieval_candidates()
        normalized_action = action_text.replace("\\", "/")
        mentions_candidate = any(candidate.replace("\\", "/") in normalized_action for candidate in candidates)
        forbidden_workspace = re.search(r"cd\s+/(workspace|app|home/user)\b|/(workspace|home/user)/", action_text)
        whole_file_read = self._looks_like_whole_file_first_read(action_text)

        if candidates and mentions_candidate and not forbidden_workspace and not whole_file_read:
            return ""

        if not candidates and not self._looks_like_broad_first_search(action_text) and not forbidden_workspace and not whole_file_read:
            return ""

        candidate_list = "\n".join(f"- `{candidate}`" for candidate in candidates[:10]) or "- No candidates available"
        reason = "Deterministic retrieval suggested starting from a candidate before broad search."
        if forbidden_workspace:
            reason += " This action also used a guessed absolute workspace path."
        elif whole_file_read:
            reason += " This action looked like a whole-file read; prefer the provided line window or a focused symbol search."
        elif self._looks_like_broad_first_search(action_text):
            reason += " This action also looked like broad repository enumeration."
        return (
            f"{reason}\n\nSuggested deterministic retrieval candidates:\n{candidate_list}\n\n"
            "If one of them looks relevant, inspect it with a targeted read command such as "
            "`nl -ba <file> | sed -n 'start,endp'`. If they look noisy, continue with a focused symbol search instead."
        )

    def _should_framework_submit_after_repetition_nudge(self, repetition_meta: dict | None) -> bool:
        return bool(
            repetition_meta
            and repetition_meta.get("trigger_reason") == "post_verify_over_testing"
            and int(repetition_meta.get("trigger_count", 0) or 0) >= 2
        )

    def _execute_framework_submit_after_repetition_nudge(self, repetition_meta: dict) -> None:
        submit_command = self._framework_submit_command()
        submit_output = self.env.execute(submit_command)
        framework_observation = self.render_template(self.config.action_observation_template, output=submit_output)
        framework_observation += (
            "\n\nFRAMEWORK ACTION: Post-verification over-testing nudge triggered. "
            "The framework executed the submit command and is ending the run."
        )
        self.add_message(
            "user",
            framework_observation,
            framework_submit={
                "command": submit_command,
                "trigger_reason": repetition_meta.get("trigger_reason"),
                "trigger_count": repetition_meta.get("trigger_count"),
            },
        )
        if submit_output.get("returncode", 1) != 0:
            raise RepetitionNudgeLimitExceeded(
                "Framework submit command failed after verification repetition nudge."
            )
        try:
            self.has_finished(submit_output)
        except Submitted:
            raise
        raise Submitted(str(submit_output.get("output", "")))

    def _finalize_observation(self, observation: str, *, repetition_meta: dict | None) -> None:
        observation = self._append_turn_control_reminder(observation)
        if repetition_meta:
            self.add_message("user", observation, repetition_meta=repetition_meta)
            if self._should_framework_submit_after_repetition_nudge(repetition_meta):
                self._execute_framework_submit_after_repetition_nudge(repetition_meta)
            if repetition_meta.get("force_stop"):
                raise RepetitionNudgeLimitExceeded(repetition_meta["force_stop_message"])
            return
        self.add_message("user", observation)

    def execute_action(self, action: dict) -> dict:
        action_text = action["action"]
        self._action_sequence += 1
        current_action = self._action_sequence
        if (
            self._submission_review_enabled
            and self._action_contains_submit_marker(action_text)
            and "SUBMISSION_REVIEW_CONFIRMED" not in action_text
        ):
            return {"returncode": 126, "output": self._submission_review_message()}
        action_changes_source_code = self._action_changes_tracked_source_code(action_text)
        if action_budget_message := self._trace_action_budget_block_message(action_text):
            return {
                "output": action_budget_message,
                "returncode": 126,
                "trace_action_budget_blocked": True,
            }
        if (
            self._trace_status_large_read_block_enabled
            and "TRACE_STATUS_LARGE_READ_BLOCK_CONFIRMED" not in action_text
            and (large_read_block_message := self._trace_status_large_read_block_message(action_text))
        ):
            return {
                "output": large_read_block_message,
                "returncode": 126,
                "trace_status_large_read_blocked": True,
            }
        post_pass_edit_guard_message = ""
        post_pass_edit_guard_first_hit = False
        if self._trace_status_repeat_block_enabled:
            post_pass_edit_guard_message, post_pass_edit_guard_first_hit = (
                self._trace_status_post_pass_edit_guard_message(action_text)
            )
            if post_pass_edit_guard_message.startswith("BLOCKED:"):
                return {
                    "output": post_pass_edit_guard_message,
                    "returncode": 126,
                    "trace_status_repeat_blocked": True,
                }
        if self._trace_status_repeat_block_enabled:
            confirmation_requested = "TRACE_STATUS_REPEAT_CONFIRMED" in action_text
            budget_messages = (
                self._trace_status_inspection_budget_message(action_text),
                self._trace_status_total_inspection_budget_message(action_text),
                self._trace_status_test_phase_budget_message(action_text),
                self._trace_status_scratch_experiment_budget_message(action_text),
                self._trace_status_repeat_block_message(action_text),
            )
            for repeat_block_message in budget_messages:
                if not repeat_block_message:
                    continue
                if (
                    confirmation_requested
                    and self._trace_status_repeat_confirmations_since_source_edit == 0
                ):
                    # A confirmation is an escape hatch for one essential
                    # action, not a permanent way to disable the guard.
                    self._trace_status_repeat_confirmations_since_source_edit = 1
                    break
                if confirmation_requested:
                    repeat_block_message += (
                        " The single TRACE_STATUS_REPEAT_CONFIRMED exception since the last source "
                        "edit has already been used."
                    )
                repeat_count = self._trace_status_repeat_block_counts.get(repeat_block_message, 0) + 1
                self._trace_status_repeat_block_counts[repeat_block_message] = repeat_count
                return {
                    "output": repeat_block_message,
                    "returncode": 126,
                    "trace_status_repeat_blocked": True,
                }
        if blocked_output := self._maybe_return_cached_read(action["action"]):
            return blocked_output
        if self._whole_file_read_block_enabled and self._looks_like_whole_file_first_read(action_text):
            return {
                "output": self._whole_file_read_block_message(action_text),
                "returncode": 126,
            }
        soft_guard_hint = self._first_action_guard_failure(action["action"])
        discipline_hint = self._action_discipline_hint(action["action"])
        executed_action_text, progressive_notes = self._progressive_read_action(action_text)
        try:
            output = self.env.execute(executed_action_text)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        if progressive_notes:
            output["output"] = (
                str(output.get("output", ""))
                + "\n\n"
                + self._progressive_read_message(progressive_notes)
            )
            output["trace_status_progressive_read"] = {
                "executed_action": executed_action_text,
                "windows": progressive_notes,
            }
        if self._trace_status_repeat_block_enabled:
            if action_changes_source_code:
                self._trace_status_test_phase_tests = 0
            elif self._is_test_command(action_text):
                self._trace_status_test_phase_tests += 1
        self._consume_trace_action_budget(action_text)
        if action_changes_source_code:
            self._last_source_edit_action = current_action
            self._consecutive_verification_blocks = 0
            self._post_verify_extra_tests = 0
            self._post_verify_extra_inspections = 0
        if self._is_test_command(action_text):
            self._has_seen_test_command = True
            if self._test_command_succeeded(output):
                self._has_successful_test_command = True
                self._last_successful_test_action = current_action
                self._consecutive_verification_blocks = 0
                self._post_verify_extra_tests = 0
                self._post_verify_extra_inspections = 0
            else:
                # A real failure supplies new evidence and unlocks another
                # edit/test iteration.
                self._trace_status_post_pass_edit_guard_hits = 0
        if post_pass_edit_guard_first_hit and output.get("returncode", 0) == 0:
            self._trace_status_post_pass_edit_guard_hits = 1
            output["output"] = (
                post_pass_edit_guard_message
                + "\n\nAction still executed.\n\n"
                + output.get("output", "")
            )
        self._maybe_submit_after_trace_status_post_verify_budget(action_text=action_text, output=output)
        if soft_guard_hint:
            output["output"] = (
                "SOFT RETRIEVAL HINT:\n"
                + soft_guard_hint
                + "\n\nAction still executed.\n\n"
                + output.get("output", "")
            )
        if discipline_hint:
            output["output"] = (
                "SOFT ACTION DISCIPLINE HINT:\n"
                + discipline_hint
                + "\n\nAction still executed.\n\n"
                + output.get("output", "")
            )
        if legacy_test_hint := self._legacy_test_contract_hint(output.get("output", "")):
            output["output"] = (
                output.get("output", "")
                + "\n\nSOFT TEST CONTRACT HINT:\n"
                + legacy_test_hint
            )
        if self._is_submit_attempt(output):
            if self._require_successful_test_before_submit and not self._has_successful_test_command:
                self._consecutive_submit_attempts = 0
                return self._block_or_force_submit(
                        "SUBMISSION BLOCKED: this run requires at least one successful test or verification command "
                        "before submission. The completion marker was ignored. Run one focused test that exercises "
                        "the changed behavior, inspect a successful result, then submit again.\n\n"
                        "RECOGNIZED TEST MARKERS: pytest, unittest, doctest, nosetests, tox, nox, "
                        "runtests.py, make test/check, cargo/go/npm/pnpm/yarn/bazel/mvn/gradle test, "
                        "rspec, phpunit, dotnet test, prove, verify, verification, smoke, regression.\n"
                        "For an inline Python check, include an assertion so it is recognized, for example: "
                        "`python -c \"...; assert actual == expected\"`. A command that only prints values "
                        "is not recorded as a successful verification."
                )
            if self._require_successful_test_before_submit and self._last_successful_test_action <= self._last_source_edit_action:
                self._consecutive_submit_attempts = 0
                return self._block_or_force_submit(
                        "SUBMISSION BLOCKED: run a successful test in a separate action after the last source edit. "
                        "Do not combine editing, testing, and submission in one command."
                )
            if action["action"] == self._last_submit_action:
                self._consecutive_submit_attempts += 1
            else:
                self._consecutive_submit_attempts = 1
                self._last_submit_action = action["action"]
            submit_payload = self._extract_submit_payload(output)
            if submit_payload and submit_payload.strip():
                raise Submitted(submit_payload)
            submit_output = self.env.execute(self._framework_submit_command())
            raise Submitted(str(submit_output.get("output", "")))
        else:
            self._consecutive_submit_attempts = 0
            self._last_submit_action = ""
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
                raise Submitted("".join(lines[i + 1 :]))
