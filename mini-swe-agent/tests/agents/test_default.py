from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent, NonTerminatingException, Submitted
from minisweagent.agents.repetition_detector import RuleBasedRepetitionDetector
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


class StubEnvironment:
    def __init__(self, outputs, tracked_files=None):
        self.outputs = list(outputs)
        self.commands = []
        self.tracked_files = set(tracked_files or {"src/topics/tags.js", "src/app.py"})

    def execute(self, command: str, cwd: str = "", *, timeout=None):
        if command == "git ls-files -z":
            return {"output": "\0".join(sorted(self.tracked_files)), "returncode": 0}
        self.commands.append(command)
        assert self.outputs, f"No stub output left for command: {command}"
        return self.outputs.pop(0)

    def get_template_vars(self):
        return {}


class StubRepetitionDetector:
    def __init__(self, evaluations):
        self.evaluations = list(evaluations)

    def reset(self):
        return None

    def update(self, *, response_content: str, action: str, output: dict):
        del response_content, action, output
        assert self.evaluations, "No repetition evaluation left"
        return self.evaluations.pop(0)


@pytest.fixture
def default_config():
    """Load default agent config from config/default.yaml"""
    config_path = Path("src/minisweagent/config/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["agent"]


def test_successful_completion(default_config):
    """Test agent completes successfully when COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is encountered."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "I'll echo a message\n```bash\necho 'hello world'\n```",
                "Now finishing\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'Task completed successfully'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **default_config,
    )

    exit_status, result = agent.run("Echo hello world then finish")
    assert exit_status == "Submitted"
    assert result == "Task completed successfully\n"
    assert agent.model.n_calls == 2
    assert len(agent.messages) == 6  # system, user, assistant, user, assistant, user


def test_submit_is_blocked_without_a_successful_test(default_config):
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Submit too early\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
                "Run focused test\n```bash\npytest tests/test_widget.py\n```",
                "Submit after test\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=StubEnvironment(
            [
                {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", "returncode": 0},
                {"output": "1 passed\n", "returncode": 0},
                {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
            ]
        ),
        **{**default_config, "require_successful_test_before_submit": True},
    )

    exit_status, result = agent.run("Require a test before submit")

    assert exit_status == "Submitted"
    assert result == "done\n"
    blocked_message = next(message["content"] for message in agent.messages if "SUBMISSION BLOCKED" in message["content"])
    assert "RECOGNIZED TEST MARKERS: pytest" in blocked_message
    assert "include an assertion" in blocked_message


def test_submission_review_requires_confirmation(default_config):
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Submit\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
                "Confirm\n```bash\necho SUBMISSION_REVIEW_CONFIRMED && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo done\n```",
            ]
        ),
        env=StubEnvironment(
            [{"output": "SUBMISSION_REVIEW_CONFIRMED\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0}]
        ),
        **{
            **default_config,
            "system_template": "system",
            "instance_template": "{{task}}",
            "submission_review_enabled": True,
        },
    )

    exit_status, result = agent.run("Require submission review")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert agent.env.commands == [
        "echo SUBMISSION_REVIEW_CONFIRMED && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo done"
    ]
    assert any("SUBMISSION REVIEW REQUIRED" in message["content"] for message in agent.messages)


def test_repeated_verification_blocks_force_submission(default_config):
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Submit too early\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
                "Submit again\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=StubEnvironment(
            [
                {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", "returncode": 0},
                {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", "returncode": 0},
                {"output": "forced patch\n", "returncode": 0},
            ]
        ),
        **{
            **default_config,
            "require_successful_test_before_submit": True,
            "repeated_submit_force_threshold": 1,
        },
    )

    exit_status, result = agent.run("Force submission after repeated verification blocks")

    assert exit_status == "Submitted"
    assert result == "forced patch\n"
    assert any("FRAMEWORK FORCED SUBMISSION" in message["content"] for message in agent.messages)


def test_step_limit_enforcement(default_config):
    """Test agent stops when step limit is reached."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=["First command\n```bash\necho 'step1'\n```", "Second command\n```bash\necho 'step2'\n```"]
        ),
        env=LocalEnvironment(),
        **{**default_config, "step_limit": 1},
    )

    exit_status, _ = agent.run("Run multiple commands")
    assert exit_status == "LimitsExceeded"
    assert agent.model.n_calls == 1


def test_cost_limit_enforcement(default_config):
    """Test agent stops when cost limit is reached."""
    model = DeterministicModel(outputs=["```bash\necho 'test'\n```"])

    agent = DefaultAgent(
        model=model,
        env=LocalEnvironment(),
        **{**default_config, "cost_limit": 0.5},
    )

    exit_status, _ = agent.run("Test cost limit")
    assert exit_status == "LimitsExceeded"


def test_format_error_handling(default_config):
    """Test agent handles malformed action formats properly."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "No code blocks here",
                "Multiple blocks\n```bash\necho 'first'\n```\n```bash\necho 'second'\n```",
                "Now correct\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **default_config,
    )

    exit_status, result = agent.run("Test format errors")
    assert exit_status == "Submitted"
    assert result == "done\n"
    assert agent.model.n_calls == 3
    # Should have error messages in conversation
    assert (
        len([msg for msg in agent.messages if "Please always provide EXACTLY ONE action" in msg.get("content", "")])
        == 1
    )


def test_timeout_handling(default_config):
    """Test agent handles command timeouts properly."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Long sleep\n```bash\nsleep 5\n```",  # This will timeout
                "Quick finish\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'\n```",
            ]
        ),
        env=LocalEnvironment(timeout=1),  # Very short timeout
        **default_config,
    )

    exit_status, result = agent.run("Test timeout handling")
    assert exit_status == "Submitted"
    assert result == "recovered\n"
    # Should have timeout error message
    assert len([msg for msg in agent.messages if "timed out" in msg.get("content", "")]) == 1


def test_timeout_captures_partial_output(default_config):
    """Test that timeout error captures partial output from commands that produce output before timing out."""
    num1, num2 = 111, 9
    calculation_command = f"echo $(({num1}*{num2})); sleep 10"
    expected_output = str(num1 * num2)
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                f"Output then sleep\n```bash\n{calculation_command}\n```",
                "Quick finish\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'\n```",
            ]
        ),
        env=LocalEnvironment(timeout=1),
        **default_config,
    )
    exit_status, result = agent.run("Test timeout with partial output")
    assert exit_status == "Submitted"
    assert result == "recovered\n"  # final output should be `recovered` from the last command
    timed_out_messages = [msg for msg in agent.messages if "timed out" in msg.get("content", "")]
    assert len(timed_out_messages) == 1
    assert expected_output in timed_out_messages[0]["content"]  # ensure timed out output is still captured


def test_parse_action_success(default_config):
    """Test action parsing works correctly for valid formats."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        **default_config,
    )

    # Test different valid formats
    result = agent.parse_action({"content": "```bash\necho 'test'\n```"})
    assert result["action"] == "echo 'test'"
    assert result["content"] == "```bash\necho 'test'\n```"

    result = agent.parse_action({"content": "```bash\nls -la\n```"})
    assert result["action"] == "ls -la"
    assert result["content"] == "```bash\nls -la\n```"

    result = agent.parse_action({"content": "Some text\n```bash\necho 'hello'\n```\nMore text"})
    assert result["action"] == "echo 'hello'"
    assert result["content"] == "Some text\n```bash\necho 'hello'\n```\nMore text"


def test_parse_action_failures(default_config, caplog):
    """Test action parsing raises appropriate exceptions for invalid formats."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        **default_config,
    )

    # No code blocks
    with pytest.raises(NonTerminatingException):
        agent.parse_action({"content": "No code blocks here"})

    # Multiple code blocks: warn and execute the first action.
    result = agent.parse_action({"content": "```bash\necho 'first'\n```\n```bash\necho 'second'\n```"})
    assert result["action"] == "echo 'first'"
    assert "contained 2 bash actions; executing only the first action" in caplog.text

    # Code block without bash language specifier
    with pytest.raises(NonTerminatingException):
        agent.parse_action({"content": "```\nls -la\n```"})


def test_message_history_tracking(default_config):
    """Test that messages are properly added and tracked."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Response 1\n```bash\necho 'test1'\n```",
                "Response 2\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **default_config,
    )

    exit_status, result = agent.run("Track messages")
    assert exit_status == "Submitted"
    assert result == "done\n"

    # After completion should have full conversation
    assert len(agent.messages) == 6
    assert [msg["role"] for msg in agent.messages] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_deterministic_retrieval_candidates_parse_new_context_sections(default_config):
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        **default_config,
    )

    agent.extra_template_vars["deterministic_retrieval_context"] = """
## Deterministic Retrieval Context

### Pre-searched Candidate Hits
1. `lib/example/core.py`
   - likely core implementation path

### Suggested First Reads
- `lib/example/core.py:20-48`
- `tests/example/test_core.py:10-30`
""".strip()

    assert agent._deterministic_retrieval_candidates() == [
        "lib/example/core.py:20-48",
        "lib/example/core.py",
        "tests/example/test_core.py:10-30",
        "tests/example/test_core.py",
    ]


def test_multiple_steps_before_completion(default_config):
    """Test agent can handle multiple steps before finding completion signal."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Step 1\n```bash\necho 'first'\n```",
                "Step 2\n```bash\necho 'second'\n```",
                "Step 3\n```bash\necho 'third'\n```",
                "Final step\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed all steps'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **{**default_config, "cost_limit": 5.0},  # Increase cost limit to allow all 4 calls (4.0 total cost)
    )

    exit_status, result = agent.run("Multi-step task")
    assert exit_status == "Submitted"
    assert result == "completed all steps\n"
    assert agent.model.n_calls == 4

    # Check that all intermediate outputs are captured (final step doesn't get observation due to termination)
    observations = [
        msg["content"] for msg in agent.messages if msg["role"] == "user" and "<returncode>" in msg["content"]
    ]
    assert len(observations) == 3
    assert "first" in observations[0]
    assert "second" in observations[1]
    assert "third" in observations[2]


def test_custom_config(default_config):
    """Test agent works with custom configuration."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Test response\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'custom config works'\n```"
            ]
        ),
        env=LocalEnvironment(),
        **{
            **default_config,
            "system_template": "You are a test assistant.",
            "instance_template": "Task: {{task}}. Return bash command.",
            "step_limit": 2,
            "cost_limit": 1.0,
        },
    )

    exit_status, result = agent.run("Test custom config")
    assert exit_status == "Submitted"
    assert result == "custom config works\n"
    assert agent.messages[0]["content"] == "You are a test assistant."
    assert "Test custom config" in agent.messages[1]["content"]


def test_render_template_model_stats(default_config):
    """Test that render_template has access to n_model_calls and model_cost from model."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=["output1", "output2"]),
        env=LocalEnvironment(),
        **default_config,
    )

    # Make some model calls to generate stats
    agent.model.query([])
    agent.model.query([])

    # Test template rendering with model stats
    template = "Calls: {{n_model_calls}}, Cost: {{model_cost}}"
    result = agent.render_template(template)

    assert result == "Calls: 2, Cost: 2.0"


def test_messages_include_timestamps(default_config):
    """Test that all messages include timestamps."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Response 1\n```bash\necho 'test1'\n```",
                "Response 2\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **default_config,
    )

    agent.run("Test timestamps")

    # All messages should have timestamps
    assert all("timestamp" in msg for msg in agent.messages)
    # Timestamps should be numeric (floats from time.time())
    assert all(isinstance(msg["timestamp"], float) for msg in agent.messages)
    # Timestamps should be monotonically increasing
    timestamps = [msg["timestamp"] for msg in agent.messages]
    assert timestamps == sorted(timestamps)


def test_step_output_includes_action(default_config):
    """Test that step output includes the action that was executed."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=["Test command\n```bash\necho 'hello'\n```"]),
        env=LocalEnvironment(),
        **default_config,
    )

    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    output = agent.step()

    assert "action" in output
    assert output["action"] == "echo 'hello'"
    assert "output" in output


def test_failed_test_command_does_not_add_retest_prompt(default_config):
    """Test that failing test commands do not trigger the submit-time reminder."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=StubEnvironment([{"output": "tests failed\n", "returncode": 1}]),
        **default_config,
    )
    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    agent.get_observation({"content": "Run tests\n```bash\npython -m pytest tests/test_example.py -q\n```"})

    last_message = agent.messages[-1]["content"]
    assert "TESTING REQUIREMENT" not in last_message


def test_submit_without_any_test_command_still_completes(default_config):
    """Test that submit attempts complete even if no prior test command was run."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=StubEnvironment([{"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nsubmitted diff\n", "returncode": 0}]),
        **default_config,
    )
    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    with pytest.raises(Submitted) as exc_info:
        agent.get_observation({"content": "Submit now\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"})
    assert str(exc_info.value) == "submitted diff\n"


def test_submit_after_failed_test_still_completes(default_config):
    """Test that failed prior tests no longer block a later submit attempt."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=StubEnvironment(
            [
                {"output": "tests failed\n", "returncode": 1},
                {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nsubmitted diff\n", "returncode": 0},
            ]
        ),
        **default_config,
    )
    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    agent.get_observation({"content": "Run tests\n```bash\npython -m pytest tests/test_example.py -q\n```"})
    with pytest.raises(Submitted) as exc_info:
        agent.get_observation({"content": "Submit now\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"})
    assert str(exc_info.value) == "submitted diff\n"


def test_submit_after_no_tests_ran_still_completes(default_config):
    """Test that 'no tests ran' no longer blocks a later submit attempt."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=StubEnvironment(
            [
                {"output": "no tests ran in 0.00s\n", "returncode": 0},
                {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nsubmitted diff\n", "returncode": 0},
            ]
        ),
        **default_config,
    )
    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    agent.get_observation({"content": "Run tests\n```bash\npython -m pytest tests/test_example.py -q\n```"})
    with pytest.raises(Submitted) as exc_info:
        agent.get_observation({"content": "Submit now\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"})
    assert str(exc_info.value) == "submitted diff\n"


def test_submit_after_test_command_still_completes(default_config):
    """Test that submit attempts still complete when the immediately previous command was a successful test."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=StubEnvironment(
            [
                {"output": "tests passed\n", "returncode": 0},
                {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nsubmitted diff\n", "returncode": 0},
            ]
        ),
        **default_config,
    )
    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    agent.get_observation({"content": "Run tests\n```bash\npython tests/runtests.py expressions --verbosity 2\n```"})

    with pytest.raises(Submitted) as exc_info:
        agent.get_observation({"content": "Submit now\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"})
    assert str(exc_info.value) == "submitted diff\n"


def test_submit_after_successful_test_then_other_command_is_blocked(default_config):
    """Test that an intervening non-test command no longer blocks submit."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=StubEnvironment(
            [
                {"output": "tests passed\n", "returncode": 0},
                {"output": "readme output\n", "returncode": 0},
                {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nsubmitted diff\n", "returncode": 0},
            ]
        ),
        **default_config,
    )
    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    agent.get_observation({"content": "Run tests\n```bash\npython tests/runtests.py expressions --verbosity 2\n```"})
    agent.get_observation({"content": "Read code\n```bash\ncat README.md\n```"})
    with pytest.raises(Submitted) as exc_info:
        agent.get_observation({"content": "Submit now\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"})
    assert str(exc_info.value) == "submitted diff\n"


def test_submit_token_can_appear_after_other_output(default_config):
    """Test that the submit token is recognized even when it is not the first output line."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=StubEnvironment(
            [
                {"output": "tests passed\n", "returncode": 0},
                {
                    "output": "some preamble\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nsubmitted diff\n",
                    "returncode": 0,
                },
            ]
        ),
        **default_config,
    )
    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    agent.get_observation({"content": "Run tests\n```bash\npython tests/runtests.py expressions --verbosity 2\n```"})

    with pytest.raises(Submitted) as exc_info:
        agent.get_observation({"content": "Submit now\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"})

    assert str(exc_info.value) == "submitted diff\n"


def test_submit_token_triggers_framework_submit_when_no_payload(default_config):
    """Test that a bare submit token delegates the actual patch export to the framework."""
    env = StubEnvironment(
        [
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", "returncode": 0},
            {"output": "diff --git a/file.py b/file.py\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=env,
        **default_config,
    )
    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    with pytest.raises(Submitted) as exc_info:
        agent.get_observation({"content": "Submit now\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"})

    assert str(exc_info.value) == "diff --git a/file.py b/file.py\n"
    assert env.commands == ["echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'", "git add -A && git diff --cached"]


def test_post_verify_over_testing_nudge_auto_submits_and_stops(default_config):
    """Test that repeated post-verification nudges eventually trigger a framework submit."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Run focused tests\n```bash\npython tests/runtests.py queries.tests.QuerysetOrderedTests -v 2\n```",
                "Run one more focused test\n```bash\npython tests/runtests.py queries.tests.QuerysetOrderedTests -v 2\n```",
                "This output should never be used\n```bash\necho 'unexpected'\n```",
            ]
        ),
        env=StubEnvironment(
            [
                {"output": "tests passed\n", "returncode": 0},
                {"output": "tests still passed\n", "returncode": 0},
                {
                    "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nsubmitted diff\n",
                    "returncode": 0,
                },
            ]
        ),
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_repetition_nudge_enabled": True,
        },
    )
    agent._repetition_detector = StubRepetitionDetector(
        [
            {
                "triggered": True,
                "trigger_reason": "post_verify_over_testing",
                "repeat_score": 0,
                "repeat_hit": False,
                "same_root_cause": False,
                "same_code_region": False,
                "same_symbol_focus": False,
                "bigger_test_no_new_info": False,
                "root_cause_locked": False,
                "rounds_since_root_cause_lock": 0,
                "test_budget_used": 4,
                "verified_passed": True,
                "post_verify_test_count": 3,
                "normalized_test_cmd": "python tests/runtests.py queries.tests.querysetorderedtests -v 2",
                "new_failure_since_patch": False,
                "round_index": 1,
            },
            {
                "triggered": True,
                "trigger_reason": "post_verify_over_testing",
                "repeat_score": 0,
                "repeat_hit": False,
                "same_root_cause": False,
                "same_code_region": False,
                "same_symbol_focus": False,
                "bigger_test_no_new_info": False,
                "root_cause_locked": False,
                "rounds_since_root_cause_lock": 0,
                "test_budget_used": 5,
                "verified_passed": True,
                "post_verify_test_count": 4,
                "normalized_test_cmd": "python tests/runtests.py queries.tests.querysetorderedtests -v 2",
                "new_failure_since_patch": False,
                "round_index": 2,
            },
        ]
    )

    exit_status, result = agent.run("Test verification nudge auto submit")

    assert exit_status == "Submitted"
    assert result == "submitted diff\n"
    assert agent.model.n_calls == 2
    assert any(msg.get("repetition_meta", {}).get("trigger_reason") == "post_verify_over_testing" for msg in agent.messages)
    framework_submit_messages = [msg for msg in agent.messages if "FRAMEWORK ACTION:" in msg.get("content", "")]
    assert len(framework_submit_messages) == 1
    assert "executed the submit command" in framework_submit_messages[0]["content"]


def test_patch_test_loop_low_info_nudge_adds_guidance_without_auto_submit(default_config):
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Apply small patch\n```bash\nsed -i 's/foo/bar/' src/app.py\n```",
                "Run same test again\n```bash\npython tests/runtests.py queries.tests.QuerysetOrderedTests -v 2\n```",
                "Keep iterating\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=StubEnvironment(
            [
                {"output": "patched\n", "returncode": 0},
                {"output": "still failing the same way\n", "returncode": 1},
                {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
            ]
        ),
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_repetition_nudge_enabled": True,
        },
    )
    agent._repetition_detector = StubRepetitionDetector(
        [
            {
                "triggered": False,
                "trigger_reason": "",
            },
            {
                "triggered": True,
                "trigger_reason": "patch_test_loop_low_info",
                "trigger_count": 1,
                "repeat_score": 0,
                "repeat_hit": False,
                "same_root_cause": True,
                "same_code_region": True,
                "same_symbol_focus": True,
                "bigger_test_no_new_info": True,
                "root_cause_locked": False,
                "rounds_since_root_cause_lock": 0,
                "test_budget_used": 2,
                "verified_passed": False,
                "post_verify_test_count": 0,
                "post_patch_action_window": ["patch", "test", "patch", "test"],
                "normalized_test_cmd": "python tests/runtests.py queries.tests.querysetorderedtests -v 2",
                "new_failure_since_patch": False,
                "round_index": 2,
            },
            {
                "triggered": False,
                "trigger_reason": "",
            },
        ]
    )

    exit_status, result = agent.run("Test patch/test loop nudge")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert agent.model.n_calls == 3
    nudge_messages = [msg for msg in agent.messages if msg.get("repetition_meta", {}).get("trigger_reason") == "patch_test_loop_low_info"]
    assert len(nudge_messages) == 1
    assert "low-information patch/test loop" in nudge_messages[0]["content"]
    framework_submit_messages = [msg for msg in agent.messages if "FRAMEWORK ACTION:" in msg.get("content", "")]
    assert len(framework_submit_messages) == 0


def test_post_verify_low_info_inspection_detector_triggers_after_successful_verification():
    detector = RuleBasedRepetitionDetector()

    detector.update(
        response_content="Apply fix",
        action="sed -i 's/foo/bar/' src/app.py",
        output={"output": "patched\n", "returncode": 0},
    )
    failed_setup_attempt = detector.update(
        response_content="Try pytest first",
        action="python -m pytest tests/queries/tests.py::QuerySetOrderedTests -q",
        output={"output": "No module named pytest\n", "returncode": 1},
    )
    detector.update(
        response_content="Run focused verification",
        action="python tests/runtests.py queries.tests.QuerysetOrderedTests -v 2",
        output={"output": "OK\n", "returncode": 0},
    )
    first_inspection = detector.update(
        response_content="Check the edited code",
        action="sed -n '10,30p' src/app.py",
        output={"output": "code\n", "returncode": 0},
    )
    second_inspection = detector.update(
        response_content="Check related compiler behavior",
        action="grep -rn \"force_no_ordering\" src/app.py",
        output={"output": "src/app.py:20: force_no_ordering\n", "returncode": 0},
    )

    assert failed_setup_attempt["new_failure_since_patch"] is True
    assert first_inspection["triggered"] is False
    assert first_inspection["new_failure_since_patch"] is False
    assert first_inspection["post_verify_inspection_count"] == 1
    assert second_inspection["triggered"] is True
    assert second_inspection["trigger_reason"] == "post_verify_low_info_inspection"
    assert second_inspection["post_verify_inspection_count"] == 2


def test_post_verify_low_info_inspection_nudge_adds_guidance_without_auto_submit(default_config):
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Inspect after passing verification\n```bash\nsed -n '10,30p' src/app.py\n```",
                "Submit after nudge\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=StubEnvironment(
            [
                {"output": "code\n", "returncode": 0},
                {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
            ]
        ),
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_repetition_nudge_enabled": True,
        },
    )
    agent._repetition_detector = StubRepetitionDetector(
        [
            {
                "triggered": True,
                "trigger_reason": "post_verify_low_info_inspection",
                "trigger_count": 1,
                "repeat_score": 0,
                "repeat_hit": False,
                "same_root_cause": False,
                "same_code_region": True,
                "same_symbol_focus": False,
                "bigger_test_no_new_info": False,
                "root_cause_locked": False,
                "rounds_since_root_cause_lock": 0,
                "test_budget_used": 1,
                "verified_passed": True,
                "post_verify_test_count": 0,
                "post_verify_inspection_count": 2,
                "post_patch_action_window": ["patch", "test"],
                "normalized_test_cmd": None,
                "new_failure_since_patch": False,
                "round_index": 3,
            },
            {
                "triggered": False,
                "trigger_reason": "",
            },
        ]
    )

    exit_status, result = agent.run("Test post-verification inspection nudge")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert agent.model.n_calls == 2
    nudge_messages = [
        msg
        for msg in agent.messages
        if msg.get("repetition_meta", {}).get("trigger_reason") == "post_verify_low_info_inspection"
    ]
    assert len(nudge_messages) == 1
    assert "low-information inspection" in nudge_messages[0]["content"]
    framework_submit_messages = [msg for msg in agent.messages if "FRAMEWORK ACTION:" in msg.get("content", "")]
    assert len(framework_submit_messages) == 0


def test_read_cache_returns_cached_message_for_same_range(default_config):
    env = StubEnvironment(
        [
            {"output": "line 10\nline 11\nline 12\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "THOUGHT: parser normalization looks suspicious in this region.\n```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "THOUGHT: I want to reread the same region.\n```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "THOUGHT: finish.\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "read_cache_enable": True,
        },
    )

    exit_status, result = agent.run("Test read cache duplicate blocking")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert env.commands == ["sed -n '10,12p' src/topics/tags.js", "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"]
    cache_messages = [msg for msg in agent.messages if "READ CACHE:" in msg.get("content", "")]
    assert len(cache_messages) == 1
    assert "上次读取 step=1" in cache_messages[0]["content"]
    assert "file=src/topics/tags.js" in cache_messages[0]["content"]
    assert "range=10-12" in cache_messages[0]["content"]


def test_read_cache_metadata_export(default_config):
    env = StubEnvironment(
        [
            {"output": "print('hi')\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "THOUGHT: helper function here looks relevant.\n```bash\ncat src/app.py\n```",
                "THOUGHT: finish.\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{**default_config, "read_cache_enable": True},
    )

    exit_status, result = agent.run("Test read cache metadata")

    assert exit_status == "Submitted"
    assert result == "done\n"
    metadata = agent.get_traj_metadata()
    assert metadata["read_cache_enabled"] is True
    assert metadata["read_cache_hits"] == 0
    assert metadata["read_cache_entries"][0]["file"] == "src/app.py"
    assert metadata["read_cache_entries"][0]["line_range"] == "full"
    assert metadata["read_cache_entries"][0]["step"] == 1


def test_read_cache_invalidates_after_file_change(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n11 beta\n12 gamma\n", "returncode": 0},
            {"output": "patched\n", "returncode": 0},
            {"output": "10 alpha\n11 beta\n12 gamma\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "```bash\nsed -i 's/foo/bar/' src/topics/tags.js\n```",
                "```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{**default_config, "read_cache_enable": True},
    )

    exit_status, result = agent.run("Invalidate cache after file change")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert env.commands == [
        "sed -n '10,12p' src/topics/tags.js",
        "sed -i 's/foo/bar/' src/topics/tags.js",
        "sed -n '10,12p' src/topics/tags.js",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    cache_messages = [msg for msg in agent.messages if "READ CACHE:" in msg.get("content", "")]
    assert len(cache_messages) == 0


@pytest.mark.parametrize("checkout", ["git checkout -- src/topics/tags.js", "git checkout feature"])
def test_git_checkout_refreshes_edit_epoch_and_read_cache(default_config, checkout):
    read = "sed -n '10,12p' src/topics/tags.js"
    submit = "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"
    env = StubEnvironment(
        [
            {"output": "old source\n", "returncode": 0},
            {"output": "checkout complete\n", "returncode": 0},
            {"output": "new source\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[f"```bash\n{command}\n```" for command in (read, checkout, read, submit)]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "read_cache_enable": True,
            "trace_status_repeat_block_enabled": True,
        },
    )

    exit_status, result = agent.run("Read source again after checkout")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [read, checkout, read, submit]
    assert agent._trace_status_last_source_edit_step == 2
    assert not any("READ CACHE:" in message.get("content", "") for message in agent.messages)
    assert not any("BLOCKED:" in message.get("content", "") for message in agent.messages)


def test_trace_status_bar_summarizes_repeated_reads(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n11 beta\n12 gamma\n", "returncode": 0},
            {"output": "10 alpha\n11 beta\n12 gamma\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_bar_enabled": True,
            "trace_status_bar_interval": 2,
        },
    )

    exit_status, result = agent.run("Summarize repeated read")

    assert exit_status == "Submitted"
    assert result == "done\n"
    status_messages = [msg for msg in agent.messages if "<trace_status>" in msg.get("content", "")]
    assert len(status_messages) == 1
    assert "- s1->s2 x2 `src/topics/tags.js` 10-12" in status_messages[0]["content"]
    assert "Already read `src/topics/tags.js` (10-12)" in status_messages[0]["content"]
    assert "Next action guidance" in status_messages[0]["content"]
    metadata = agent.get_traj_metadata()
    assert metadata["trace_status_bar_enabled"] is True
    assert metadata["trace_status_bar"]["reads"][0]["count"] == 2


def test_trace_status_bar_can_omit_next_action_guidance(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,10p' src/topics/tags.js\n```",
                "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_bar_enabled": True,
            "trace_status_bar_interval": 1,
            "trace_status_bar_next_action_guidance_enabled": False,
        },
    )

    exit_status, _ = agent.run("Omit status-bar guidance")

    assert exit_status == "Submitted"
    status_messages = [msg for msg in agent.messages if "<trace_status>" in msg.get("content", "")]
    assert len(status_messages) == 1
    assert "Read:" in status_messages[0]["content"]
    assert "Next action guidance:" not in status_messages[0]["content"]
    assert agent.get_traj_metadata()["trace_status_bar_next_action_guidance_enabled"] is False


def test_trace_status_bar_requires_exact_regression_test_after_a_passing_test():
    agent = DefaultAgent.__new__(DefaultAgent)
    agent._last_successful_test_action = 1

    assert agent._trace_status_next_action_guidance() == (
        "A test has passed after a source edit; run the issue's exact regression test before submitting."
    )


def test_trace_status_bar_is_disabled_by_default(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "```bash\nsed -n '12,15p' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{**default_config, "instance_template": "{{task}}"},
    )

    exit_status, _ = agent.run("No trace status by default")

    assert exit_status == "Submitted"
    assert not any("<trace_status>" in msg.get("content", "") for msg in agent.messages)
    assert "trace_status_bar_enabled" not in agent.get_traj_metadata()


def test_trace_action_budget_controller_forces_ordered_phases(default_config):
    env = StubEnvironment(
        [
            {"output": "src/widget.py\n", "returncode": 0},
            {"output": "patched\n", "returncode": 0},
            {"output": "1 passed\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nrg -n 'widget' src\n```",
                "```bash\nprintf 'patched' > src/widget.py\n```",
                "```bash\npytest tests/test_widget.py\n```",
                "```bash\nrg -n 'widget' src\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_action_budget_controller_enabled": True,
            "trace_action_budget_search": 1,
            "trace_action_budget_edit": 1,
            "trace_action_budget_test": 1,
        },
    )

    exit_status, result = agent.run("Use phased budgets")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert env.commands == [
        "rg -n 'widget' src",
        "printf 'patched' > src/widget.py",
        "pytest tests/test_widget.py",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_message = next(message["content"] for message in agent.messages if "ACTION BUDGET BLOCKED" in message["content"])
    assert "submit phase" in blocked_message
    assert "search=0; edit=0; test=0" in blocked_message
    metadata = agent.get_traj_metadata()["trace_action_budget_controller"]
    assert metadata["phase"] == "submit"
    assert metadata["remaining"] == {"search": 0, "edit": 0, "test": 0}
    assert metadata["transitions"] == [
        "search budget exhausted; forced into edit phase.",
        "edit budget exhausted; forced into test phase.",
        "test budget exhausted; forced into submit phase.",
    ]


def test_trace_status_repeat_block_blocks_mostly_repeated_read_without_status_injection(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "12 gamma\n", "returncode": 0},
            {"output": "12 gamma\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,20p' src/topics/tags.js\n```",
                "```bash\nsed -n '10,21p' src/topics/tags.js\n```",
                "```bash\nsed -n '10,21p' src/topics/tags.js\n```",
                "```bash\nsed -n '10,21p' src/topics/tags.js # TRACE_STATUS_REPEAT_CONFIRMED\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
        },
    )

    exit_status, result = agent.run("Block a repeated read")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert env.commands == [
        "sed -n '10,20p' src/topics/tags.js",
        "sed -n '10,21p' src/topics/tags.js",
        "sed -n '10,21p' src/topics/tags.js # TRACE_STATUS_REPEAT_CONFIRMED",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_message = next(message["content"] for message in agent.messages if "BLOCKED:" in message["content"])
    assert "already read `src/topics/tags.js` (10-21)" in blocked_message
    assert "Treat that prior result as evidence" in blocked_message
    assert "state one root-cause hypothesis" in blocked_message
    assert "TRACE_STATUS_REPEAT_CONFIRMED" in blocked_message
    assert not any("<trace_status>" in message.get("content", "") for message in agent.messages)
    assert agent.get_traj_metadata()["trace_status_repeat_block_enabled"] is True
    assert agent.get_traj_metadata()["trace_status_repeat_block"]["reads"][0]["first_step"] == 1
    assert not agent._trace_status_ranges_overlap("10-12", "12-15")


def test_trace_status_repeat_block_allows_edit_and_inspection_of_previously_read_file(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "patched\n10 beta\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    edit_and_inspect = "sed -i 's/alpha/beta/' src/topics/tags.js && nl -ba src/topics/tags.js | sed -n '10,12p'"
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nnl -ba src/topics/tags.js | sed -n '10,12p'\n```",
                f"```bash\n{edit_and_inspect}\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
        },
    )

    exit_status, result = agent.run("Edit and inspect a previously read source file")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert env.commands == [
        "nl -ba src/topics/tags.js | sed -n '10,12p'",
        edit_and_inspect,
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    assert not any("BLOCKED:" in message.get("content", "") for message in agent.messages)


def test_trace_status_repeat_block_allows_only_one_confirmation(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "12 gamma\n", "returncode": 0},
            {"output": "12 gamma\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,20p' src/topics/tags.js\n```",
                "```bash\nsed -n '10,21p' src/topics/tags.js\n```",
                "```bash\nsed -n '10,21p' src/topics/tags.js # TRACE_STATUS_REPEAT_CONFIRMED\n```",
                "```bash\nsed -n '10,21p' src/topics/tags.js # TRACE_STATUS_REPEAT_CONFIRMED\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
        },
    )

    exit_status, result = agent.run("Allow one confirmed repeated read")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert env.commands == [
        "sed -n '10,20p' src/topics/tags.js",
        "sed -n '10,21p' src/topics/tags.js",
        "sed -n '10,21p' src/topics/tags.js # TRACE_STATUS_REPEAT_CONFIRMED",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_messages = [message["content"] for message in agent.messages if "BLOCKED:" in message["content"]]
    assert len(blocked_messages) == 1
    assert "exception since the last source edit has already been used" in blocked_messages[-1]
    assert agent.get_traj_metadata()["trace_status_repeat_block"]["repeat_confirmations_since_source_edit"] == 1


def test_trace_status_repeat_block_warns_without_blocking_post_pass_edits(default_config):
    env = StubEnvironment(
        [
            {"output": "patched\n", "returncode": 0},
            {"output": "tests passed\n", "returncode": 0},
            {"output": "patched after verification\n", "returncode": 0},
            {"output": "tests passed again\n", "returncode": 0},
            {"output": "patched after second verification\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -i 's/a/b/' src/app.py\n```",
                "```bash\npytest tests/test_app.py\n```",
                "```bash\nsed -i 's/b/c/' src/app.py\n```",
                "```bash\npytest tests/test_app.py\n```",
                "```bash\nsed -i 's/c/d/' src/app.py\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
        },
    )

    exit_status, result = agent.run("Guard edits after successful verification")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert env.commands == [
        "sed -i 's/a/b/' src/app.py",
        "pytest tests/test_app.py",
        "sed -i 's/b/c/' src/app.py",
        "pytest tests/test_app.py",
        "sed -i 's/c/d/' src/app.py",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    guard_messages = [message["content"] for message in agent.messages if "POST-PASS EDIT GUARD:" in message["content"]]
    assert len(guard_messages) == 2
    assert all("Action still executed" in message for message in guard_messages)
    assert "concrete failure output already observed" in guard_messages[-1]
    assert not any("BLOCKED:" in message.get("content", "") for message in agent.messages)
    assert agent.get_traj_metadata()["trace_status_repeat_block"]["post_pass_edit_guard_hits"] == 1


def test_trace_status_repeat_block_post_pass_guard_resets_after_failed_test(default_config):
    env = StubEnvironment(
        [
            {"output": "patched\n", "returncode": 0},
            {"output": "tests passed\n", "returncode": 0},
            {"output": "patched after verification\n", "returncode": 0},
            {"output": "test failed\n", "returncode": 1},
            {"output": "patched from failure evidence\n", "returncode": 0},
            {"output": "tests passed again\n", "returncode": 0},
            {"output": "patched after new verification\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -i 's/a/b/' src/app.py\n```",
                "```bash\npytest tests/test_app.py\n```",
                "```bash\nsed -i 's/b/c/' src/app.py\n```",
                "```bash\npytest tests/test_app.py\n```",
                "```bash\nsed -i 's/c/d/' src/app.py\n```",
                "```bash\npytest tests/test_app.py\n```",
                "```bash\nsed -i 's/d/e/' src/app.py\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
        },
    )

    exit_status, result = agent.run("Reset the post-pass guard after a failed test")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert not any("BLOCKED:" in message.get("content", "") for message in agent.messages)
    assert sum("POST-PASS EDIT GUARD:" in message.get("content", "") for message in agent.messages) == 2


def test_trace_status_large_read_block_requires_explicit_confirmation(default_config):
    env = StubEnvironment(
        [
            {"output": "large source window\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '1,2400p' src/topics/tags.js\n```",
                "```bash\nsed -n '1,2400p' src/topics/tags.js # TRACE_STATUS_LARGE_READ_BLOCK_CONFIRMED\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_large_read_block_enabled": True,
            "trace_status_large_read_block_max_lines": 1000,
        },
    )

    exit_status, result = agent.run("Block an oversized source read")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert env.commands == [
        "sed -n '1,2400p' src/topics/tags.js # TRACE_STATUS_LARGE_READ_BLOCK_CONFIRMED",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_message = next(message["content"] for message in agent.messages if "BLOCKED:" in message["content"])
    assert "configured limit of 1000 lines" in blocked_message
    assert "1-2400: 2400 lines" in blocked_message
    assert "TRACE_STATUS_LARGE_READ_BLOCK_CONFIRMED" in blocked_message


def test_trace_status_large_read_block_counts_combined_windows(default_config):
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=StubEnvironment([]),
        **{
            **default_config,
            "trace_status_large_read_block_enabled": True,
            "trace_status_large_read_block_max_lines": 1000,
        },
    )

    message = agent._trace_status_large_read_block_message(
        "sed -n '1,600p' src/one.py && sed -n '1,500p' src/two.py"
    )

    assert "configured limit of 1000 lines" in message
    assert "combined source-read windows (1100 lines)" in message
    assert "Total requested source-read length: 1100 lines" in message


def test_trace_status_long_output_truncation_keeps_command_and_head_tail(default_config):
    long_output = "A" * 2_500 + "B" * 2_500
    env = StubEnvironment(
        [
            {"output": long_output, "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nprintf source\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' && echo done\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_long_output_truncation_enabled": True,
            "trace_status_long_output_truncation_max_chars": 4_000,
            "trace_status_long_output_truncation_head_tail_chars": 300,
        },
    )

    exit_status, _ = agent.run("truncate only the observation")

    assert exit_status == "Submitted"
    assert env.commands[0] == "printf source"
    observation = agent.messages[3]["content"]
    assert "OUTPUT TRUNCATED: original output was 5,000 characters" in observation
    # The retained character budget is applied to the rendered observation, so
    # XML-like wrapper text occupies a few characters at each end.
    assert "A" * 200 in observation and "B" * 200 in observation
    assert len(observation) < 1_000


def test_trace_status_progressive_read_executes_first_chunk_without_block(default_config):
    env = StubEnvironment(
        [
            {"output": "first chunk\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '1,2400p' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' && echo done\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_progressive_read_enabled": True,
            "trace_status_progressive_read_chunk_lines": 200,
        },
    )

    exit_status, _ = agent.run("read progressively")

    assert exit_status == "Submitted"
    assert env.commands[0] == "sed -n '1,200p' src/topics/tags.js"
    observation = agent.messages[3]["content"]
    assert "PROGRESSIVE SOURCE READ" in observation
    assert "requested 1-2400, returned 1-200; remaining 201-2400" in observation
    assert "BLOCKED:" not in observation


def test_trace_status_old_observation_compaction_uses_next_agent_thought(default_config):
    source_output = "def get_prefetch_queryset():\n" + "x" * 5_000
    env = StubEnvironment(
        [
            {"output": source_output, "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    conclusion = "The reverse relation calls queryset.filter()."
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '690,825p' django/db/models/fields/related_descriptors.py\n```",
                conclusion + "\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' && echo done\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_old_observation_compaction_enabled": True,
            "trace_status_old_observation_compaction_max_chars": 4_000,
        },
    )

    exit_status, _ = agent.run("compact after use")

    assert exit_status == "Submitted"
    compacted = agent.messages[3]["content"]
    assert "<structured_source_memory>" in compacted
    assert "related_descriptors.py:690-825" in compacted
    assert conclusion in compacted
    assert "source_observation_1" in compacted
    assert agent._trace_status_raw_observation_cache["source_observation_1"] == source_output


def test_trace_status_repeat_block_allows_successful_grep_fallback_after_failed_rg(default_config):
    env = StubEnvironment(
        [
            {"output": "bash: rg: command not found\n", "returncode": 127},
            {"output": "src/topics/tags.js:10:needle\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                '```bash\nrg -n "needle" src/topics/tags.js\n```',
                "```bash\ngrep -R -n -E 'needle' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
        },
    )

    exit_status, result = agent.run("Use grep when rg is unavailable")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        'rg -n "needle" src/topics/tags.js',
        "grep -R -n -E 'needle' src/topics/tags.js",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    assert not any("BLOCKED:" in message.get("content", "") for message in agent.messages)
    searches = agent.get_traj_metadata()["trace_status_repeat_block"]["searches"]
    assert searches == [{"signature": "needle @ src/topics/tags.js", "first_step": 2, "last_step": 2, "count": 1}]


def test_trace_status_repeat_block_does_not_count_failed_inspection_toward_limit(default_config):
    env = StubEnvironment(
        [
            {"output": "missing file\n", "returncode": 1},
            {"output": "20 beta\n", "returncode": 0},
            {"output": "30 gamma\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "```bash\nsed -n '20,22p' src/topics/tags.js\n```",
                "```bash\nsed -n '30,32p' src/topics/tags.js\n```",
                "```bash\nsed -n '40,42p' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_max_consecutive_inspections": 2,
        },
    )

    exit_status, result = agent.run("Only count successful inspections")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        "sed -n '10,12p' src/topics/tags.js",
        "sed -n '20,22p' src/topics/tags.js",
        "sed -n '30,32p' src/topics/tags.js",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    assert any("2 consecutive source read/search actions" in message.get("content", "") for message in agent.messages)


def test_trace_status_repeat_block_does_not_submit_after_the_same_evidence_is_blocked_twice(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "12 gamma\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,20p' src/topics/tags.js\n```",
                "```bash\nsed -n '10,21p' src/topics/tags.js\n```",
                "```bash\nsed -n '10,21p' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
        },
    )

    exit_status, result = agent.run("Stop a repeated evidence loop")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert env.commands == [
        "sed -n '10,20p' src/topics/tags.js",
        "sed -n '10,21p' src/topics/tags.js",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]


def test_trace_status_repeat_block_limits_consecutive_inspections(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "11 beta\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "```bash\nsed -n '20,22p' src/topics/tags.js\n```",
                "```bash\nsed -n '30,32p' src/topics/tags.js\n```",
                "```bash\nsed -n '40,42p' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_max_consecutive_inspections": 2,
        },
    )

    exit_status, result = agent.run("Limit unproductive inspection")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        "sed -n '10,12p' src/topics/tags.js",
        "sed -n '20,22p' src/topics/tags.js",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_message = next(message["content"] for message in agent.messages if "consecutive source read/search" in message["content"])
    assert "2 consecutive" in blocked_message
    assert "inspect the production module that owns the failing behavior" in blocked_message


def test_trace_status_repeat_block_dynamically_allows_bounded_new_inspections(default_config):
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=StubEnvironment([]),
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_max_consecutive_inspections": 10,
            "trace_status_repeat_block_max_total_inspections_since_edit": 2,
            "trace_status_repeat_block_dynamic_inspection_enabled": True,
            "trace_status_repeat_block_dynamic_inspection_max_bonus": 1,
        },
    )
    for action in (
        "sed -n '1,2p' src/topics/tags.js",
        "sed -n '3,4p' src/topics/tags.js",
        "sed -n '5,6p' src/topics/tags.js",
    ):
        agent._update_trace_status_bar(action={"action": action}, output={"returncode": 0})

    blocked_message = agent._trace_status_total_inspection_budget_message(
        "sed -n '7,8p' src/topics/tags.js"
    )
    assert "3 total" in blocked_message
    assert "current limit: 3" in blocked_message


def test_trace_status_repeat_block_limits_test_phase_with_interleaved_inspections(default_config):
    env = StubEnvironment(
        [
            {"output": "1 failed\n", "returncode": 1},
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "1 failed\n", "returncode": 1},
            {"output": "src/topics/tags.js:10:alpha\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\npytest tests/test_tags.py::test_a\n```",
                "```bash\nsed -n '10,20p' src/topics/tags.js\n```",
                "```bash\npytest tests/test_tags.py::test_b\n```",
                "```bash\ngrep -n 'alpha' src/topics/tags.js\n```",
                "```bash\npytest tests/test_tags.py::test_c\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_max_test_phase_tests": 2,
        },
    )

    exit_status, result = agent.run("Limit test-debug loop")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        "pytest tests/test_tags.py::test_a",
        "sed -n '10,20p' src/topics/tags.js",
        "pytest tests/test_tags.py::test_b",
        "grep -n 'alpha' src/topics/tags.js",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_message = next(message["content"] for message in agent.messages if "test actions since the last edit" in message["content"])
    assert "2 test actions" in blocked_message
    assert "same test-debug phase" in blocked_message


def test_cleanup_of_python_test_files_is_not_a_test_action():
    cleanup = (
        "cd /testbed && rm -f test_multi_inherit_update.py test_multi_inherit_update2.py "
        "/tmp/fix_compiler.py /tmp/fix_subqueries.py && git diff"
    )
    assert not DefaultAgent._is_test_command(cleanup)
    assert DefaultAgent._is_test_command("cd /testbed && python test_multi_inherit_update.py")


def test_trace_status_repeat_block_warns_at_soft_limit_then_blocks_at_hard_limit(default_config):
    env = StubEnvironment([
        {"output": "1 passed\n", "returncode": 0},
        {"output": "1 passed\n", "returncode": 0},
        {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
    ])
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[
            "```bash\npytest tests/test_tags.py::test_a\n```",
            "```bash\npytest tests/test_tags.py::test_b\n```",
            "```bash\npytest tests/test_tags.py::test_c\n```",
            "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
        ]),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_soft_warning_enabled": True,
            "trace_status_repeat_block_warn_test_phase_tests": 1,
            "trace_status_repeat_block_max_test_phase_tests": 2,
        },
    )

    agent.run("Warn, then block")

    assert env.commands == [
        "pytest tests/test_tags.py::test_a",
        "pytest tests/test_tags.py::test_b",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    warnings = [m["content"] for m in agent.messages if "PACE WARNING:" in m.get("content", "")]
    assert len(warnings) == 1
    assert "tests 1/1" in warnings[0]
    assert any("2 test actions since the last edit" in m.get("content", "") for m in agent.messages)


def test_trace_status_repeat_block_resets_test_phase_after_edit(default_config):
    env = StubEnvironment(
        [
            {"output": "1 failed\n", "returncode": 1},
            {"output": "1 failed\n", "returncode": 1},
            {"output": "patched\n", "returncode": 0},
            {"output": "1 failed\n", "returncode": 1},
            {"output": "1 passed\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\npytest tests/test_tags.py::test_a\n```",
                "```bash\npytest tests/test_tags.py::test_b\n```",
                "```bash\nsed -i 's/foo/bar/' src/topics/tags.js\n```",
                "```bash\npytest tests/test_tags.py::test_c\n```",
                "```bash\npytest tests/test_tags.py::test_d\n```",
                "```bash\npytest tests/test_tags.py::test_e\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_max_test_phase_tests": 2,
        },
    )

    exit_status, result = agent.run("Reset test phase on edit")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        "pytest tests/test_tags.py::test_a",
        "pytest tests/test_tags.py::test_b",
        "sed -i 's/foo/bar/' src/topics/tags.js",
        "pytest tests/test_tags.py::test_c",
        "pytest tests/test_tags.py::test_d",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_message = next(message["content"] for message in agent.messages if "test actions since the last edit" in message["content"])
    assert "2 test actions" in blocked_message


def test_trace_status_repeat_block_scratch_edit_does_not_reset_test_phase(default_config):
    env = StubEnvironment(
        [
            {"output": "1 failed\n", "returncode": 1},
            {"output": "scratch script\n", "returncode": 0},
            {"output": "1 failed\n", "returncode": 1},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\npytest tests/test_tags.py::test_a\n```",
                "```bash\ncat > /tmp/inspect_tags.py <<'PY'\nprint('inspect')\nPY\n```",
                "```bash\npytest tests/test_tags.py::test_b\n```",
                "```bash\npytest tests/test_tags.py::test_c\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_max_test_phase_tests": 2,
        },
    )

    exit_status, result = agent.run("Do not let scratch writes reset the test phase")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        "pytest tests/test_tags.py::test_a",
        "cat > /tmp/inspect_tags.py <<'PY'\nprint('inspect')\nPY",
        "pytest tests/test_tags.py::test_b",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_message = next(message["content"] for message in agent.messages if "test actions since the last edit" in message["content"])
    assert "2 test actions" in blocked_message


def test_trace_status_repeat_block_limits_scratch_experiments(default_config):
    env = StubEnvironment(
        [
            {"output": "scratch one\n", "returncode": 0},
            {"output": "scratch run\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\ncat > /tmp/inspect_one.py <<'PY'\nprint('one')\nPY\n```",
                "```bash\npython /tmp/inspect_one.py\n```",
                "```bash\ncat > /tmp/inspect_two.py <<'PY'\nprint('two')\nPY\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_max_scratch_experiments_since_edit": 2,
        },
    )

    exit_status, result = agent.run("Limit scratch experiment loops")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        "cat > /tmp/inspect_one.py <<'PY'\nprint('one')\nPY",
        "python /tmp/inspect_one.py",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_message = next(message["content"] for message in agent.messages if "scratch experiment actions" in message["content"])
    assert "2 scratch experiment actions" in blocked_message


def test_trace_status_repeat_block_limits_total_inspections_since_source_edit(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "1 failed\n", "returncode": 1},
            {"output": "20 beta\n", "returncode": 0},
            {"output": "1 failed\n", "returncode": 1},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "```bash\npytest tests/test_tags.py::test_a\n```",
                "```bash\nsed -n '20,22p' src/topics/tags.js\n```",
                "```bash\npytest tests/test_tags.py::test_b\n```",
                "```bash\nsed -n '30,32p' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_max_consecutive_inspections": 10,
            "trace_status_repeat_block_max_total_inspections_since_edit": 2,
        },
    )

    exit_status, result = agent.run("Limit total inspection loops")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        "sed -n '10,12p' src/topics/tags.js",
        "pytest tests/test_tags.py::test_a",
        "sed -n '20,22p' src/topics/tags.js",
        "pytest tests/test_tags.py::test_b",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_message = next(message["content"] for message in agent.messages if "total source read/search actions" in message["content"])
    assert "2 total source read/search actions" in blocked_message


def test_trace_status_repeat_block_resets_total_inspections_after_source_edit(default_config):
    env = StubEnvironment(
        [
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "20 beta\n", "returncode": 0},
            {"output": "patched\n", "returncode": 0},
            {"output": "30 gamma\n", "returncode": 0},
            {"output": "40 delta\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -n '10,12p' src/topics/tags.js\n```",
                "```bash\nsed -n '20,22p' src/topics/tags.js\n```",
                "```bash\nsed -i 's/old/new/' src/topics/tags.js\n```",
                "```bash\nsed -n '30,32p' src/topics/tags.js\n```",
                "```bash\nsed -n '40,42p' src/topics/tags.js\n```",
                "```bash\nsed -n '50,52p' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_max_consecutive_inspections": 10,
            "trace_status_repeat_block_max_total_inspections_since_edit": 2,
        },
    )

    exit_status, result = agent.run("Reset total inspection budget after a source edit")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        "sed -n '10,12p' src/topics/tags.js",
        "sed -n '20,22p' src/topics/tags.js",
        "sed -i 's/old/new/' src/topics/tags.js",
        "sed -n '30,32p' src/topics/tags.js",
        "sed -n '40,42p' src/topics/tags.js",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]
    blocked_message = next(message["content"] for message in agent.messages if "total source read/search actions" in message["content"])
    assert "2 total source read/search actions" in blocked_message


def test_trace_status_repeat_block_allows_explicit_submit_after_a_verified_patch(default_config):
    env = StubEnvironment(
        [
            {"output": "patched\n", "returncode": 0},
            {"output": "1 passed\n", "returncode": 0},
            {"output": "10 alpha\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -i 's/foo/bar/' src/topics/tags.js\n```",
                "```bash\npytest tests/test_tags.py\n```",
                "```bash\nsed -n '10,20p' src/topics/tags.js\n```",
                "```bash\nsed -n '10,21p' src/topics/tags.js\n```",
                "```bash\nsed -n '10,21p' src/topics/tags.js\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
        },
    )

    exit_status, result = agent.run("Submit a verified patch instead of looping on evidence")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        "sed -i 's/foo/bar/' src/topics/tags.js",
        "pytest tests/test_tags.py",
        "sed -n '10,20p' src/topics/tags.js",
        "sed -n '10,21p' src/topics/tags.js",
    ]


def test_repeated_submit_execution_failures_stop_without_trace_status(default_config):
    env = StubEnvironment(
        [
            {"output": "Error response from daemon: No such container: abc123\n", "returncode": 1},
            {"output": "Error response from daemon: No such container: abc123\n", "returncode": 1},
            {"output": "Error response from daemon: No such container: abc123\n", "returncode": 1},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```",
                "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```",
                "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_bar_enabled": True,
            "consecutive_unproductive_failure_threshold": 3,
        },
    )

    exit_status, result = agent.run("Stop repeated failed submit attempts")

    assert exit_status == "ConsecutiveUnproductiveFailuresExceeded"
    assert "submit-command-failed" in result
    assert len(env.commands) == 3
    assert not any("<trace_status>" in msg.get("content", "") for msg in agent.messages)


def test_repeated_environment_failures_stop_without_trace_status(default_config):
    env = StubEnvironment(
        [
            {"output": "Error response from daemon: No such container: abc123\n", "returncode": 1},
            {"output": "Error response from daemon: No such container: abc123\n", "returncode": 1},
            {"output": "Error response from daemon: No such container: abc123\n", "returncode": 1},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\npython -c \"print('hello')\"\n```",
                "```bash\npython -c \"print('hello')\"\n```",
                "```bash\npython -c \"print('hello')\"\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_bar_enabled": True,
            "consecutive_unproductive_failure_threshold": 3,
        },
    )

    exit_status, result = agent.run("Stop repeated environment failures")

    assert exit_status == "ConsecutiveUnproductiveFailuresExceeded"
    assert "environment-unavailable" in result
    assert len(env.commands) == 3
    assert not any("<trace_status>" in msg.get("content", "") for msg in agent.messages)


def test_repeated_environment_failures_guard_requires_trace_status_bar(default_config):
    env = StubEnvironment(
        [
            {"output": "Error response from daemon: No such container: abc123\n", "returncode": 1},
            {"output": "Error response from daemon: No such container: abc123\n", "returncode": 1},
            {"output": "Error response from daemon: No such container: abc123\n", "returncode": 1},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\npython -c \"print('hello')\"\n```",
                "```bash\npython -c \"print('hello')\"\n```",
                "```bash\npython -c \"print('hello')\"\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "consecutive_unproductive_failure_threshold": 3,
        },
    )

    exit_status, result = agent.run("Guard is disabled unless trace status bar is enabled")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert len(env.commands) == 4


def test_trace_status_post_verify_budget_submits_after_extra_inspections(default_config):
    env = StubEnvironment(
        [
            {"output": "patched\n", "returncode": 0},
            {"output": "1 passed\n", "returncode": 0},
            {"output": "diff --git a/src/app.py b/src/app.py\n", "returncode": 0},
            {"output": "src/app.py:10:def target():\n", "returncode": 0},
            {"output": "submitted diff\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -i 's/foo/bar/' src/app.py\n```",
                "```bash\npytest tests/test_app.py\n```",
                "```bash\ngit diff\n```",
                "```bash\ngrep -rn \"target\" src/app.py\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_bar_enabled": True,
            "trace_status_post_verify_max_extra_inspections": 1,
        },
    )

    exit_status, result = agent.run("Submit after post-verify inspection budget")

    assert exit_status == "Submitted"
    assert result == "submitted diff\n"
    assert agent.model.n_calls == 4
    assert env.commands[-1] == "git add -A && git diff --cached"
    framework_messages = [msg for msg in agent.messages if "trace status post-verification budget exceeded" in msg.get("content", "")]
    assert len(framework_messages) == 1


def test_trace_status_post_verify_budget_requires_trace_status_bar(default_config):
    env = StubEnvironment(
        [
            {"output": "patched\n", "returncode": 0},
            {"output": "1 passed\n", "returncode": 0},
            {"output": "diff --git a/src/app.py b/src/app.py\n", "returncode": 0},
            {"output": "src/app.py:10:def target():\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nsed -i 's/foo/bar/' src/app.py\n```",
                "```bash\npytest tests/test_app.py\n```",
                "```bash\ngit diff\n```",
                "```bash\ngrep -rn \"target\" src/app.py\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_post_verify_max_extra_inspections": 1,
        },
    )

    exit_status, result = agent.run("No post-verify budget without trace status bar")

    assert exit_status == "Submitted"
    assert result == "done\n"
    assert agent.model.n_calls == 5
    assert env.commands[-1] == "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"


def test_trace_status_bar_extracts_sed_write_target():
    targets = DefaultAgent._extract_write_targets(
        "sed -i '33a\\\n"
        "            return value\\\n"
        "' django/utils/numberformat.py"
    )

    assert targets == {"django/utils/numberformat.py"}


def test_trace_status_bar_formats_search_and_edit_steps(default_config):
    env = StubEnvironment(
        [
            {"output": "src/topics/tags.js:10:alpha\n", "returncode": 0},
            {"output": "src/topics/tags.js:10:alpha\n", "returncode": 0},
            {"output": "patched\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\nrg 'alpha' src/topics/tags.js\n```",
                "```bash\nrg 'alpha' src/topics/tags.js\n```",
                """```bash
python - <<'PY'
from pathlib import Path
path = Path('src/topics/tags.js')
path.write_text(path.read_text())
PY
```""",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{**default_config, "instance_template": "{{task}}", "trace_status_bar_enabled": True},
    )

    exit_status, result = agent.run("Format trace status")

    assert exit_status == "Submitted"
    assert result == "done\n"
    status_messages = [msg for msg in agent.messages if "<trace_status>" in msg.get("content", "")]
    assert len(status_messages) == 1
    assert "- s1->s2 x2 `alpha @ src/topics/tags.js`" in status_messages[0]["content"]
    metadata = agent.get_traj_metadata()["trace_status_bar"]
    assert metadata["searches"] == [
        {"signature": "alpha @ src/topics/tags.js", "first_step": 1, "last_step": 2, "count": 2}
    ]
    assert metadata["edits"] == [{"target": "src/topics/tags.js", "first_step": 3, "last_step": 3, "count": 1}]


def test_trace_status_bar_treats_cat_redirect_as_write():
    action = "cd /testbed && cat > /tmp/testapp2/models.py <<'EOF'\nfrom django.db import models\nEOF"

    assert DefaultAgent._action_may_change_code(action) is True
    assert DefaultAgent._extract_write_targets(action) == {"/tmp/testapp2/models.py"}
    assert DefaultAgent._action_changes_source_code(action) is False
    assert DefaultAgent._is_scratch_experiment_action(action) is True


def test_trace_status_repeat_block_distinguishes_source_and_scratch_edits():
    source_action = "sed -i 's/foo/bar/' django/utils/numberformat.py"
    scratch_action = "python /tmp/inspect_numberformat.py"

    assert DefaultAgent._action_changes_source_code(source_action) is True
    assert DefaultAgent._action_changes_source_code(scratch_action) is False
    assert DefaultAgent._is_scratch_experiment_action(scratch_action) is True


def test_trace_status_repeat_block_uses_tracked_baseline_for_source_edits(default_config):
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=StubEnvironment([], tracked_files={"django/db/models/sql/query.py"}),
        **default_config,
    )
    agent._trace_status_tracked_files = {"django/db/models/sql/query.py"}
    scratch_action = "cat > /testbed/test_case.py <<'PY'\nprint('repro')\nPY"
    source_action = "sed -i 's/old/new/' /testbed/django/db/models/sql/query.py"
    mixed_action = """python - <<'PY'
open('/testbed/apply_fix.py', 'w').write('script')
open('/testbed/django/db/models/sql/query.py', 'w').write('patched')
PY"""

    assert agent._action_changes_tracked_source_code(scratch_action) is False
    assert agent._is_trace_status_scratch_experiment(scratch_action) is True
    assert agent._action_changes_tracked_source_code(source_action) is True
    assert agent._is_trace_status_scratch_experiment(source_action) is False
    assert agent._action_changes_tracked_source_code(mixed_action) is True
    assert agent._is_trace_status_scratch_experiment(mixed_action) is False


def test_trace_status_repeat_block_limits_untracked_workspace_scripts(default_config):
    env = StubEnvironment(
        [
            {"output": "scratch one\n", "returncode": 0},
            {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n", "returncode": 0},
        ]
    )
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "```bash\ncat > /testbed/test_case.py <<'PY'\nprint('one')\nPY\n```",
                "```bash\ncat > /testbed/test_case_2.py <<'PY'\nprint('two')\nPY\n```",
                "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=env,
        **{
            **default_config,
            "instance_template": "{{task}}",
            "trace_status_repeat_block_enabled": True,
            "trace_status_repeat_block_max_scratch_experiments_since_edit": 1,
        },
    )

    exit_status, result = agent.run("Do not let untracked workspace scripts reset budgets")

    assert exit_status == Submitted.__name__
    assert result == "done\n"
    assert env.commands == [
        "cat > /testbed/test_case.py <<'PY'\nprint('one')\nPY",
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
    ]


def test_trace_status_bar_treats_perl_in_place_edit_as_write():
    action = "perl -0pi -e 's/old/new/' django/utils/numberformat.py"

    assert DefaultAgent._action_may_change_code(action) is True


def test_trace_status_test_signature_keeps_commands_with_a_shared_prefix_distinct():
    prefix = "python -c \"" + "x" * 100
    first = prefix + " first\""
    second = prefix + " second\""

    assert DefaultAgent._shorten_command(first, 90) == DefaultAgent._shorten_command(second, 90)
    assert DefaultAgent._command_signature(first) != DefaultAgent._command_signature(second)


def test_trace_status_bar_extracts_pathlib_write_text_targets():
    action = """cd /testbed && python - <<'PY'
from pathlib import Path
path = Path('django/db/models/sql/compiler.py')
text = path.read_text()
path.write_text(text)
Path("django/db/models/query.py").write_text("content")
PY"""

    assert DefaultAgent._action_may_change_code(action) is True
    assert DefaultAgent._extract_write_targets(action) == {
        "django/db/models/query.py",
        "django/db/models/sql/compiler.py",
    }


def test_trace_status_bar_extracts_heredoc_open_write_targets():
    scratch_action = """cd /testbed && python - <<'PY'
path = "/tmp/inspect_query.py"
with open(path, "w") as handle:
    handle.write("print('inspect')")
PY"""
    source_action = """cd /testbed && python - <<'PY'
path = "django/db/models/sql/query.py"
with open(path, "w") as handle:
    handle.write("patched")
PY"""
    unresolved_action = """cd /testbed && python - <<'PY'
path = build_output_path()
with open(path, "w") as handle:
    handle.write("diagnostic")
PY"""

    assert DefaultAgent._extract_write_targets(scratch_action) == {"/tmp/inspect_query.py"}
    assert DefaultAgent._action_changes_source_code(scratch_action) is False
    assert DefaultAgent._is_scratch_experiment_action(scratch_action) is True
    assert DefaultAgent._extract_write_targets(source_action) == {"django/db/models/sql/query.py"}
    assert DefaultAgent._action_changes_source_code(source_action) is True
    assert DefaultAgent._action_changes_source_code(unresolved_action) is False


def test_trace_status_repeat_block_ignores_stderr_redirects_and_python_comparisons():
    stderr_redirect = "find / -name query.py 2>/dev/null | head -5"
    python_comparison = """python - <<'PY'
if value > 0:
    print(value)
PY"""

    assert DefaultAgent._action_may_change_code(stderr_redirect) is False
    assert DefaultAgent._action_changes_source_code(stderr_redirect) is False
    assert DefaultAgent._extract_write_targets(python_comparison) == set()
    assert DefaultAgent._action_may_change_code(python_comparison) is False
    assert DefaultAgent._action_changes_source_code(python_comparison) is False
