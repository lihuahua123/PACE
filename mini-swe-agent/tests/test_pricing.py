from minisweagent.pricing import compute_usage_cost_usd, usage_breakdown


def test_cached_prompt_tokens_are_not_billed_as_uncached_input():
    usage = {
        "prompt_tokens": 1_000,
        "input_tokens": 1_000,
        "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 800},
    }

    assert usage_breakdown(usage)["input_tokens"] == 200
    assert compute_usage_cost_usd("autodl/DeepSeek-V4-Flash", usage)[2] == 0.001106
