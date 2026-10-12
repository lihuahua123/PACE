#!/usr/bin/env python3
"""Compare SWE-bench trace token/cost/success metrics on common tasks.

Default behavior compares the three runs requested by the user and restricts
the comparison to the first 50 submitted task IDs common to all runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

# Use the exact pricing implementation used by run_my.sh, rather than keeping
# a second price table here.  The module's historical ``*_usd`` names denote
# local CNY pricing for these experiment runs.
MINISWEAGENT_SRC = Path(__file__).resolve().parents[1] / "mini-swe-agent" / "src"
_PRICING_SPEC = importlib.util.spec_from_file_location(
    "minisweagent_pricing", MINISWEAGENT_SRC / "minisweagent" / "pricing.py"
)
assert _PRICING_SPEC and _PRICING_SPEC.loader
_PRICING = importlib.util.module_from_spec(_PRICING_SPEC)
_PRICING_SPEC.loader.exec_module(_PRICING)
LOCAL_PRICING = _PRICING.LOCAL_PRICING
normalize_pricing_model_name = _PRICING.normalize_pricing_model_name
select_pricing = _PRICING.select_pricing
usage_breakdown = _PRICING.usage_breakdown

DEFAULT_RUN_DIRS = [
    Path("/workspace/pace_source/swebench/log/my-2026-08-14__08-48-42-gpt-5.6-luna_org"),
    Path("/workspace/pace_source/swebench/log/my-2026-08-14__19-21-04-DeepSeek-V4-Flash_org"),
    Path("/workspace/pace_source/swebench/log/my-2026-08-30__20-06-34-MiniMax-M2.7"),
]


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def load_json_if_exists(path_value: str | Path | None) -> Any:
    if not path_value:
        return None
    try:
        path = Path(path_value)
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return None
    return None


def extract_cache_tokens(usage: dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0

    direct = [
        usage.get("cache_tokens"),
        usage.get("cached_tokens"),
        usage.get("cache_read_input_tokens"),
        usage.get("input_cached_tokens"),
    ]
    detail_objs = [
        usage.get("prompt_tokens_details") or {},
        usage.get("input_tokens_details") or {},
    ]
    detail_vals = []
    for detail in detail_objs:
        detail_vals.extend(
            [
                detail.get("cached_tokens"),
                detail.get("cache_tokens"),
                detail.get("cache_read_input_tokens"),
            ]
        )

    candidates = [safe_int(v) for v in direct + detail_vals if v is not None]
    if not candidates:
        return 0
    return max([v for v in candidates if v >= 0] or [0])


def normalize_model_local(raw_model: str | None) -> str | None:
    normalized = normalize_pricing_model_name(str(raw_model or ""))
    return normalized if normalized in LOCAL_PRICING else None


def estimate_cost_breakdown_rmb(
    model_name: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cache_tokens: int = 0,
) -> dict[str, float | int]:
    model = normalize_model_local(model_name)
    billable_input_tokens = max(0, prompt_tokens - cache_tokens)
    if model is None or model not in LOCAL_PRICING:
        return {
            "billable_input_tokens": billable_input_tokens,
            "input_cost_rmb": 0.0,
            "cache_cost_rmb": 0.0,
            "output_cost_rmb": 0.0,
            "estimated_total_cost_rmb": 0.0,
        }

    _, price = select_pricing(model_name or "", prompt_tokens)
    if price is None:
        return {
            "billable_input_tokens": billable_input_tokens,
            "input_cost_rmb": 0.0,
            "cache_cost_rmb": 0.0,
            "output_cost_rmb": 0.0,
            "estimated_total_cost_rmb": 0.0,
        }
    input_cost_rmb = billable_input_tokens / 1_000_000 * price["input"]
    cache_cost_rmb = cache_tokens / 1_000_000 * price["cache"]
    output_cost_rmb = completion_tokens / 1_000_000 * price["output"]
    return {
        "billable_input_tokens": billable_input_tokens,
        "input_cost_rmb": input_cost_rmb,
        "cache_cost_rmb": cache_cost_rmb,
        "output_cost_rmb": output_cost_rmb,
        "estimated_total_cost_rmb": input_cost_rmb + cache_cost_rmb + output_cost_rmb,
    }


def estimate_message_cost_breakdown_rmb(
    messages: list[dict[str, Any]], default_model: str | None
) -> dict[str, float]:
    """Price each recorded response with the runner's tier selection."""
    costs = {
        "billable_input_tokens": 0.0,
        "input_cost_rmb": 0.0,
        "cache_cost_rmb": 0.0,
        "output_cost_rmb": 0.0,
    }
    for message in messages:
        response = ((message.get("extra") or {}).get("response") or {})
        usage = response.get("usage") or {}
        if not usage:
            continue
        usage_stats = usage_breakdown(usage)
        _, price = select_pricing(
            str(response.get("model") or default_model or ""), usage_stats["prompt_tokens"]
        )
        if price is None:
            continue
        costs["billable_input_tokens"] += usage_stats["input_tokens"]
        costs["input_cost_rmb"] += usage_stats["input_tokens"] * float(price["input"]) / 1_000_000
        costs["cache_cost_rmb"] += usage_stats["cached_tokens"] * float(price["cache"]) / 1_000_000
        costs["output_cost_rmb"] += usage_stats["output_tokens"] * float(price["output"]) / 1_000_000
    costs["estimated_total_cost_rmb"] = sum(
        costs[key] for key in ("input_cost_rmb", "cache_cost_rmb", "output_cost_rmb")
    )
    return costs


def build_cost_row(
    *,
    source: str,
    model: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cache_tokens: int,
    api_calls: int,
    logged_cost_usd: float | None,
    logged_instance_cost: float | None,
    success: bool | None,
    experience_retrieved: bool | None,
    cost_breakdown: dict[str, float] | None = None,
) -> dict[str, Any]:
    cost = cost_breakdown or estimate_cost_breakdown_rmb(
        model, prompt_tokens, completion_tokens, cache_tokens
    )
    return {
        "source": source,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_tokens": cache_tokens,
        "billable_input_tokens": cost["billable_input_tokens"],
        "api_calls": api_calls,
        "logged_cost_usd": logged_cost_usd,
        "logged_instance_cost": logged_instance_cost,
        "input_cost_rmb": cost["input_cost_rmb"],
        "cache_cost_rmb": cost["cache_cost_rmb"],
        "output_cost_rmb": cost["output_cost_rmb"],
        "estimated_total_cost_rmb": cost["estimated_total_cost_rmb"],
        "success": success,
        "experience_retrieved": experience_retrieved,
    }


def extract_from_regular_task_summary(task_summary_path: Path) -> dict[str, Any] | None:
    data = json.loads(task_summary_path.read_text())
    usage = data.get("usage") or {}
    if not usage:
        return None

    return build_cost_row(
        source="regular_task_summary",
        model=data.get("model"),
        prompt_tokens=safe_int(usage.get("prompt_tokens")),
        completion_tokens=safe_int(usage.get("completion_tokens")),
        cache_tokens=extract_cache_tokens(usage),
        api_calls=safe_int(usage.get("api_calls") or usage.get("request_count")),
        logged_cost_usd=float(usage.get("cost_usd") or 0.0),
        logged_instance_cost=None,
        success=bool(data.get("success_within_budget", data.get("resolved", False))),
        experience_retrieved=data.get("experience_retrieved"),
    )


def resolve_legacy_attempt_file(
    task_summary_path: Path, attempt_data: dict[str, Any], field_name: str
) -> Path | None:
    path_value = (attempt_data or {}).get(field_name)
    if path_value:
        path = Path(path_value)
        if path.exists():
            return path

    fallback_name_map = {
        "llm_rounds_json": "llm_rounds.json",
        "traj_json": "traj.json",
        "transcript_json": "transcript.json",
    }
    filename = fallback_name_map.get(field_name)
    attempt_num = safe_int((attempt_data or {}).get("attempt") or 1)
    if filename:
        fallback_path = task_summary_path.parent / f"attempt_{attempt_num}" / filename
        if fallback_path.exists():
            return fallback_path
    return None


def extract_usage_from_llm_rounds(llm_rounds_path: Path | None) -> dict[str, int] | None:
    llm_rounds = load_json_if_exists(llm_rounds_path)
    if not isinstance(llm_rounds, list):
        return None

    prompt = completion = cache = api_calls = 0
    for item in llm_rounds:
        if not isinstance(item, dict):
            continue
        usage = item.get("usage") or {}
        if not usage:
            continue
        api_calls += 1
        prompt += safe_int(usage.get("prompt_tokens"))
        completion += safe_int(usage.get("completion_tokens"))
        cache += extract_cache_tokens(usage)

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_tokens": cache,
        "api_calls": api_calls,
    }


def extract_usage_with_delta_cache_estimate_from_llm_rounds(
    llm_rounds_path: Path | None,
) -> dict[str, int] | None:
    llm_rounds = load_json_if_exists(llm_rounds_path)
    if not isinstance(llm_rounds, list):
        return None

    prompts = []
    completions = []
    for item in llm_rounds:
        if not isinstance(item, dict):
            continue
        usage = item.get("usage") or {}
        if not usage:
            continue
        prompts.append(safe_int(usage.get("prompt_tokens")))
        completions.append(safe_int(usage.get("completion_tokens")))

    if not prompts:
        return None

    incremental = [prompts[0]]
    for i in range(1, len(prompts)):
        incremental.append(max(0, prompts[i] - prompts[i - 1]))
    prompt_sum = sum(prompts)
    incremental_sum = sum(incremental)

    return {
        "prompt_tokens": prompt_sum,
        "completion_tokens": sum(completions),
        "cache_tokens": max(0, prompt_sum - incremental_sum),
        "api_calls": len(prompts),
    }


def extract_model_from_legacy_attempt(
    task_summary_data: dict[str, Any],
    attempt_data: dict[str, Any],
    task_summary_path: Path,
) -> str | None:
    if task_summary_data.get("model"):
        return task_summary_data.get("model")

    llm_rounds_path = resolve_legacy_attempt_file(task_summary_path, attempt_data, "llm_rounds_json")
    llm_rounds = load_json_if_exists(llm_rounds_path)
    if isinstance(llm_rounds, list):
        for item in llm_rounds:
            if not isinstance(item, dict):
                continue
            for key in ("model", "model_name", "model_name_or_path"):
                value = item.get(key)
                if value:
                    return value

    traj_path = resolve_legacy_attempt_file(task_summary_path, attempt_data, "traj_json")
    traj = load_json_if_exists(traj_path)
    if isinstance(traj, dict):
        info = traj.get("info") or {}
        return ((info.get("config") or {}).get("model") or {}).get("model_name")
    return None


def extract_experience_flag_from_legacy_attempt(
    task_summary_data: dict[str, Any],
    attempt_data: dict[str, Any],
    task_summary_path: Path,
) -> bool | None:
    if task_summary_data.get("experience_retrieved") is not None:
        return task_summary_data.get("experience_retrieved")

    traj_path = resolve_legacy_attempt_file(task_summary_path, attempt_data, "traj_json")
    traj = load_json_if_exists(traj_path)
    if isinstance(traj, dict) and traj.get("experience_retrieved") is not None:
        return bool(traj.get("experience_retrieved"))

    transcript_path = resolve_legacy_attempt_file(task_summary_path, attempt_data, "transcript_json")
    transcript = load_json_if_exists(transcript_path)
    if isinstance(transcript, dict) and transcript.get("experience_retrieved") is not None:
        return bool(transcript.get("experience_retrieved"))
    return None


def extract_from_legacy_task_summary(task_summary_path: Path) -> dict[str, Any] | None:
    data = json.loads(task_summary_path.read_text())
    attempts = data.get("attempts") or []
    usage = data.get("usage") or {}
    if not usage and attempts:
        usage = (attempts[0] or {}).get("usage") or {}
    if not usage:
        return None

    chosen_attempt = None
    for attempt in attempts:
        if isinstance(attempt, dict) and attempt.get("resolved"):
            chosen_attempt = attempt
            break
    if chosen_attempt is None and attempts:
        chosen_attempt = attempts[0]
    chosen_attempt = chosen_attempt or {}

    legacy_delta_cache_run_names = {
        "real-swebench-subset-100-2026-05-20__15-23-30-all_minimax",
    }
    llm_rounds_path = resolve_legacy_attempt_file(task_summary_path, chosen_attempt, "llm_rounds_json")
    if task_summary_path.parent.parent.name in legacy_delta_cache_run_names:
        llm_usage = extract_usage_with_delta_cache_estimate_from_llm_rounds(llm_rounds_path)
    else:
        llm_usage = extract_usage_from_llm_rounds(llm_rounds_path)

    prompt = safe_int((llm_usage or {}).get("prompt_tokens") or usage.get("prompt_tokens"))
    completion = safe_int((llm_usage or {}).get("completion_tokens") or usage.get("completion_tokens"))
    cache = safe_int((llm_usage or {}).get("cache_tokens") or extract_cache_tokens(usage))
    api_calls = safe_int((llm_usage or {}).get("api_calls") or usage.get("api_calls") or usage.get("request_count"))

    logged_instance_cost = None
    traj_path = resolve_legacy_attempt_file(task_summary_path, chosen_attempt, "traj_json")
    traj = load_json_if_exists(traj_path)
    if isinstance(traj, dict):
        stats = (traj.get("info") or {}).get("model_stats") or {}
        logged_instance_cost = float(stats.get("instance_cost") or 0.0)

    return build_cost_row(
        source="legacy_task_summary_attempts",
        model=extract_model_from_legacy_attempt(data, chosen_attempt, task_summary_path),
        prompt_tokens=prompt,
        completion_tokens=completion,
        cache_tokens=cache,
        api_calls=api_calls,
        logged_cost_usd=float(usage.get("cost_usd") or 0.0),
        logged_instance_cost=logged_instance_cost,
        success=bool(data.get("success_within_budget", data.get("resolved", False))),
        experience_retrieved=extract_experience_flag_from_legacy_attempt(data, chosen_attempt, task_summary_path),
    )


def extract_from_traj(traj_path: Path) -> dict[str, Any]:
    data = json.loads(traj_path.read_text())
    info = data.get("info") or {}
    stats = info.get("model_stats") or {}
    model_name = ((info.get("config") or {}).get("model") or {}).get("model_name")

    prompt = completion = cache = usage_msg_count = 0
    for msg in data.get("messages", []):
        if not isinstance(msg, dict):
            continue
        usage = (((msg.get("extra") or {}).get("response") or {}).get("usage") or {})
        if not usage:
            continue
        usage_msg_count += 1
        prompt += safe_int(usage.get("prompt_tokens"))
        completion += safe_int(usage.get("completion_tokens"))
        cache += extract_cache_tokens(usage)

    api_calls = safe_int(stats.get("api_calls")) if stats.get("api_calls") is not None else usage_msg_count
    if api_calls == 0:
        api_calls = usage_msg_count

    return build_cost_row(
        source="traj_fallback",
        model=model_name,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cache_tokens=cache,
        api_calls=api_calls,
        logged_cost_usd=None,
        logged_instance_cost=float(stats.get("instance_cost") or 0.0),
        success=None,
        experience_retrieved=bool(data.get("experience_retrieved"))
        if data.get("experience_retrieved") is not None
        else None,
        cost_breakdown=estimate_message_cost_breakdown_rmb(data.get("messages") or [], model_name),
    )


def extract_task_cost(task_dir: Path) -> dict[str, Any]:
    task_summary = task_dir / "task_summary.json"
    success_from_summary = None
    experience_from_summary = None
    ts_data = None

    if task_summary.exists():
        try:
            ts_data = json.loads(task_summary.read_text())
            success_from_summary = bool(ts_data.get("success_within_budget", ts_data.get("resolved", False)))
            experience_from_summary = ts_data.get("experience_retrieved")
        except Exception:
            ts_data = None

        if ts_data and ts_data.get("attempts"):
            legacy = extract_from_legacy_task_summary(task_summary)
            if legacy is not None:
                if legacy.get("success") is None:
                    legacy["success"] = success_from_summary
                if legacy.get("experience_retrieved") is None:
                    legacy["experience_retrieved"] = experience_from_summary
                return legacy

        regular = extract_from_regular_task_summary(task_summary)
        if regular is not None:
            if regular.get("success") is None:
                regular["success"] = success_from_summary
            if regular.get("experience_retrieved") is None:
                regular["experience_retrieved"] = experience_from_summary
            return regular

    traj_candidates = sorted(task_dir.glob("*/*.traj.json"))
    if traj_candidates:
        out = extract_from_traj(traj_candidates[0])
        if out.get("success") is None:
            out["success"] = success_from_summary
        if out.get("experience_retrieved") is None:
            out["experience_retrieved"] = experience_from_summary
        return out

    return {
        "source": "missing",
        "model": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_tokens": 0,
        "billable_input_tokens": 0,
        "api_calls": 0,
        "logged_cost_usd": None,
        "logged_instance_cost": None,
        "input_cost_rmb": 0.0,
        "cache_cost_rmb": 0.0,
        "output_cost_rmb": 0.0,
        "estimated_total_cost_rmb": 0.0,
        "success": success_from_summary,
        "experience_retrieved": experience_from_summary,
    }


def is_submitted_task(task_dir: Path) -> bool:
    task_summary = task_dir / "task_summary.json"
    traj_candidates = sorted(task_dir.glob("*/*.traj.json"))
    traj_status = ""
    if traj_candidates:
        try:
            traj_data = json.loads(traj_candidates[0].read_text())
            traj_status = str((traj_data.get("info") or {}).get("exit_status") or "").strip().lower()
        except Exception:
            traj_status = ""

    if (task_dir / "preds.json").exists():
        return True
    if not task_summary.exists():
        return traj_status == "limitsexceeded"

    data = load_json_if_exists(task_summary) or {}
    status = str(data.get("agent_exit_status") or "").strip().lower()
    if status == "submitted":
        return True
    if safe_int(data.get("attempt_count")) > 0:
        return True
    if data.get("attempts"):
        return True
    if traj_status in {"submitted", "limitsexceeded"}:
        return True
    return False


def list_submitted_task_ids(run_dir: Path) -> set[str]:
    return {p.name for p in run_dir.iterdir() if p.is_dir() and is_submitted_task(p)}


def common_task_ids(run_dirs: list[Path], limit: int | None) -> list[str]:
    submitted_sets = [list_submitted_task_ids(d) for d in run_dirs]
    common_ids = sorted(set.intersection(*submitted_sets)) if submitted_sets else []
    if limit is not None:
        common_ids = common_ids[:limit]
    return common_ids


def summarize_run(run_dir: Path, allowed_task_ids: set[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    task_dirs = sorted(
        p
        for p in run_dir.iterdir()
        if p.is_dir() and p.name in allowed_task_ids and is_submitted_task(p)
    )

    rows = []
    for task_dir in task_dirs:
        row = extract_task_cost(task_dir)
        row["task_id"] = task_dir.name
        row["subset_dir"] = run_dir.name
        rows.append(row)

    task_df = pd.DataFrame(rows)
    if task_df.empty:
        summary = {
            "subset_dir": run_dir.name,
            "task_count": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_tokens": 0,
            "billable_input_tokens": 0,
            "input_cost_rmb": 0.0,
            "cache_cost_rmb": 0.0,
            "output_cost_rmb": 0.0,
            "estimated_total_cost_rmb": 0.0,
        }
        return summary, task_df

    success_count = int(task_df["success"].fillna(False).astype(bool).sum())
    task_count = int(len(task_df))
    summary = {
        "subset_dir": run_dir.name,
        "task_count": task_count,
        "success_count": success_count,
        "success_rate": success_count / task_count if task_count else 0.0,
        "api_calls": int(task_df["api_calls"].fillna(0).sum()),
        "prompt_tokens": int(task_df["prompt_tokens"].sum()),
        "completion_tokens": int(task_df["completion_tokens"].sum()),
        "cache_tokens": int(task_df["cache_tokens"].sum()),
        "billable_input_tokens": int(task_df["billable_input_tokens"].sum()),
        "input_cost_rmb": float(task_df["input_cost_rmb"].sum()),
        "cache_cost_rmb": float(task_df["cache_cost_rmb"].sum()),
        "output_cost_rmb": float(task_df["output_cost_rmb"].sum()),
        "estimated_total_cost_rmb": float(task_df["estimated_total_cost_rmb"].sum()),
    }
    return summary, task_df


def build_comparison(run_dirs: list[Path], limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    task_ids = common_task_ids(run_dirs, limit)
    allowed = set(task_ids)

    summaries = []
    task_rows = []
    for run_dir in run_dirs:
        summary, task_df = summarize_run(run_dir, allowed)
        summaries.append(summary)
        if not task_df.empty:
            task_rows.extend(task_df.to_dict("records"))

    summary_df = pd.DataFrame(summaries)
    task_df = pd.DataFrame(task_rows)

    if not summary_df.empty:
        summary_df["success_rate_pct"] = summary_df["success_rate"] * 100
        summary_df = summary_df[
            [
                "subset_dir",
                "task_count",
                "success_count",
                "success_rate_pct",
                "prompt_tokens",
                "completion_tokens",
                "cache_tokens",
                "billable_input_tokens",
                "input_cost_rmb",
                "cache_cost_rmb",
                "output_cost_rmb",
                "estimated_total_cost_rmb",
                "api_calls",
            ]
        ]

    return summary_df, task_df, task_ids


def display_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    renamed = summary_df.rename(
        columns={
            "subset_dir": "目录",
            "prompt_tokens": "输入 token",
            "completion_tokens": "输出 token",
            "cache_tokens": "缓存 token",
            "input_cost_rmb": "输入成本 RMB",
            "cache_cost_rmb": "缓存成本 RMB",
            "output_cost_rmb": "输出成本 RMB",
            "estimated_total_cost_rmb": "总成本 RMB",
            "api_calls": "api_calls",
            "success_rate_pct": "成功率%",
        }
    )
    return renamed[
        [
            "目录",
            "输入 token",
            "输出 token",
            "缓存 token",
            "输入成本 RMB",
            "缓存成本 RMB",
            "输出成本 RMB",
            "总成本 RMB",
            "api_calls",
            "成功率%",
        ]
    ]


def write_outputs(
    summary_df: pd.DataFrame,
    task_df: pd.DataFrame,
    task_ids: list[str],
    output_dir: Path,
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    table_df = display_table(summary_df).copy()
    for col in ["输入成本 RMB", "缓存成本 RMB", "输出成本 RMB", "总成本 RMB"]:
        table_df[col] = table_df[col].round(6)
    table_df["成功率%"] = table_df["成功率%"].round(2)

    table_df.to_csv(output_dir / "common50_summary.csv", index=False)
    summary_df.to_csv(output_dir / "common50_summary_raw.csv", index=False)
    task_df.to_csv(output_dir / "common50_task_details.csv", index=False)
    (output_dir / "common50_task_ids.txt").write_text("\n".join(task_ids) + "\n")
    try:
        summary_markdown = table_df.to_markdown(index=False)
    except ImportError:
        # Keep plot generation independent of the optional ``tabulate`` package.
        summary_markdown = table_df.to_csv(index=False)
    (output_dir / "common50_summary.md").write_text(summary_markdown + "\n")

    print(summary_markdown)
    print(f"\ncommon task count: {len(task_ids)}")
    print(f"wrote: {output_dir / 'common50_summary.csv'}")
    print(f"wrote: {output_dir / 'common50_task_details.csv'}")

    write_svg_boxplots(task_df, output_dir / "common50_boxplots.svg")

    if make_plots:
        plot_summary(task_df, output_dir)
        plot_cost_tail(task_df, output_dir)


def model_display_name(raw_model: str | None, subset_dir: str | None = None) -> str:
    normalized = normalize_model_local(raw_model)
    if normalized:
        return normalized
    if subset_dir:
        for candidate in LOCAL_PRICING:
            if candidate.lower() in subset_dir.lower():
                return candidate
    # Provider-qualified model IDs (for example, ``openai/gpt-5.6-luna``)
    # should be shown as their model component in chart labels.
    if raw_model:
        return str(raw_model).rstrip("/").split("/")[-1]
    return str(subset_dir or "unknown")


def prepare_boxplot_groups(task_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if task_df.empty:
        return task_df.copy(), []

    df = task_df.copy()
    df["success_bool"] = df["success"].fillna(False).astype(bool)
    df["result_label"] = df["success_bool"].map({True: "success", False: "failure"})
    df["model_label"] = [
        model_display_name(model, subset_dir)
        for model, subset_dir in zip(df.get("model", []), df.get("subset_dir", []))
    ]
    df["total_tokens"] = df["prompt_tokens"].fillna(0).astype(float) + df["completion_tokens"].fillna(0).astype(float)
    df["api_calls"] = df["api_calls"].fillna(0).astype(float)

    run_order = []
    for subset_dir in df["subset_dir"].tolist():
        label = model_display_name(
            df.loc[df["subset_dir"] == subset_dir, "model"].dropna().iloc[0]
            if not df.loc[df["subset_dir"] == subset_dir, "model"].dropna().empty
            else None,
            subset_dir,
        )
        if label not in run_order:
            run_order.append(label)

    labels = []
    for model in run_order:
        for result in ["success", "failure"]:
            labels.append(f"{model}\n{result}")
    return df, labels


def box_stats(values: list[float]) -> dict[str, Any] | None:
    clean = pd.Series([v for v in values if pd.notna(v)])
    if clean.empty:
        return None

    q1 = float(clean.quantile(0.25))
    median = float(clean.quantile(0.5))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    low_bound = q1 - 1.5 * iqr
    high_bound = q3 + 1.5 * iqr
    inliers = clean[(clean >= low_bound) & (clean <= high_bound)]
    whisker_low = float(inliers.min()) if not inliers.empty else float(clean.min())
    whisker_high = float(inliers.max()) if not inliers.empty else float(clean.max())
    outliers = [float(v) for v in clean[(clean < whisker_low) | (clean > whisker_high)].tolist()]
    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "outliers": outliers,
        "n": int(clean.size),
    }


def write_svg_boxplots(task_df: pd.DataFrame, output_path: Path) -> None:
    df, labels = prepare_boxplot_groups(task_df)
    width = 1700
    height = 620
    panel_w = 760
    panel_h = 430
    left1 = 72
    left2 = 918
    top = 52
    bottom = top + panel_h
    box_w = 64
    colors = {
        "token_box": "#D9E6F2",
        "step_box": "#F6D8BD",
        "median": "#FF7F0E",
        "axis": "#222222",
        "grid": "#DCDCDC",
        "text": "#111111",
        "outlier": "#111111",
    }

    def esc(text: object) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def group_values(metric: str) -> list[list[float]]:
        values = []
        for label in labels:
            model, result = label.split("\n", 1)
            one = df[(df["model_label"] == model) & (df["result_label"] == result)][metric]
            values.append([float(v) for v in one.tolist() if pd.notna(v) and float(v) > 0])
        return values

    def log_ticks(min_value: float, max_value: float) -> list[float]:
        import math

        lo = int(math.floor(math.log10(max(1.0, min_value))))
        hi = int(math.ceil(math.log10(max(1.0, max_value))))
        return [10.0**i for i in range(lo, hi + 1)]

    def draw_panel(
        x0: int,
        title: str,
        ylabel: str,
        values_by_group: list[list[float]],
        fill: str,
        *,
        log_scale: bool = False,
    ) -> list[str]:
        flat = [v for values in values_by_group for v in values]
        if not flat:
            flat = [1.0]
        if log_scale:
            ticks = log_ticks(min(flat), max(flat))
            lo = min(ticks)
            hi = max(ticks)

            def y_map(value: float) -> float:
                import math

                value = max(value, lo)
                return bottom - (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo) or 1) * panel_h

            tick_pairs = [(tick, y_map(tick), f"10^{int(__import__('math').log10(tick))}") for tick in ticks]
        else:
            hi = max(flat) * 1.08
            lo = 0.0
            if hi <= 0:
                hi = 1.0

            def y_map(value: float) -> float:
                return bottom - (value - lo) / (hi - lo) * panel_h

            tick_count = 5
            tick_pairs = [(hi * i / tick_count, y_map(hi * i / tick_count), f"{hi * i / tick_count:.0f}") for i in range(tick_count + 1)]

        slot = panel_w / max(1, len(labels))
        out = [
            f'<text x="{x0 + panel_w / 2}" y="27" text-anchor="middle" font-size="20" fill="{colors["text"]}">{esc(title)}</text>',
            f'<line x1="{x0}" y1="{bottom}" x2="{x0 + panel_w}" y2="{bottom}" stroke="{colors["axis"]}" />',
            f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="{colors["axis"]}" />',
            f'<text x="{x0 - 48}" y="{top + panel_h / 2}" transform="rotate(-90 {x0 - 48} {top + panel_h / 2})" '
            f'text-anchor="middle" font-size="14" fill="{colors["text"]}">{esc(ylabel)}</text>',
        ]
        for value, y, text in tick_pairs:
            out.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" '
                f'stroke="{colors["grid"]}" stroke-opacity="0.65" />'
            )
            out.append(
                f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="{colors["text"]}">{esc(text)}</text>'
            )

        for i, (label, values) in enumerate(zip(labels, values_by_group)):
            stats = box_stats(values)
            cx = x0 + slot * i + slot / 2
            if stats is not None:
                y_q1 = y_map(stats["q1"])
                y_q3 = y_map(stats["q3"])
                y_med = y_map(stats["median"])
                y_low = y_map(stats["whisker_low"])
                y_high = y_map(stats["whisker_high"])
                out.extend(
                    [
                        f'<line x1="{cx:.1f}" y1="{y_high:.1f}" x2="{cx:.1f}" y2="{y_q3:.1f}" stroke="{colors["axis"]}" />',
                        f'<line x1="{cx:.1f}" y1="{y_q1:.1f}" x2="{cx:.1f}" y2="{y_low:.1f}" stroke="{colors["axis"]}" />',
                        f'<line x1="{cx - box_w / 3:.1f}" y1="{y_high:.1f}" x2="{cx + box_w / 3:.1f}" y2="{y_high:.1f}" stroke="{colors["axis"]}" />',
                        f'<line x1="{cx - box_w / 3:.1f}" y1="{y_low:.1f}" x2="{cx + box_w / 3:.1f}" y2="{y_low:.1f}" stroke="{colors["axis"]}" />',
                        f'<rect x="{cx - box_w / 2:.1f}" y="{min(y_q1, y_q3):.1f}" width="{box_w}" '
                        f'height="{abs(y_q1 - y_q3):.1f}" fill="{fill}" stroke="{colors["axis"]}" stroke-width="1.5" />',
                        f'<line x1="{cx - box_w / 2:.1f}" y1="{y_med:.1f}" x2="{cx + box_w / 2:.1f}" '
                        f'y2="{y_med:.1f}" stroke="{colors["median"]}" stroke-width="2" />',
                    ]
                )
                for outlier in stats["outliers"]:
                    y = y_map(outlier)
                    out.append(
                        f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="4.5" fill="none" stroke="{colors["outlier"]}" stroke-width="1.4" />'
                    )

            first, second = label.split("\n", 1)
            out.append(
                f'<text x="{cx:.1f}" y="{bottom + 24}" text-anchor="middle" font-size="14" fill="{colors["text"]}">{esc(first)}</text>'
            )
            out.append(
                f'<text x="{cx:.1f}" y="{bottom + 43}" text-anchor="middle" font-size="14" fill="{colors["text"]}">{esc(second)}</text>'
            )
        return out

    token_values = group_values("total_tokens")
    step_values = group_values("api_calls")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF" />',
        *draw_panel(
            left1,
            "RQ1: Total tokens by success/failure",
            "Total tokens (log scale)",
            token_values,
            colors["token_box"],
            log_scale=True,
        ),
        *draw_panel(
            left2,
            "RQ1: Steps by success/failure",
            "Assistant steps / API calls",
            step_values,
            colors["step_box"],
            log_scale=False,
        ),
        "</svg>",
    ]
    output_path.write_text("\n".join(parts) + "\n")
    print(f"wrote: {output_path}")


def plot_summary(task_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("\nmatplotlib is not installed; skipped PNG plots.")
        print("Install it with: python3 -m pip install matplotlib")
        return

    df, labels = prepare_boxplot_groups(task_df)
    if df.empty:
        print("\nNo task details available; skipped PNG plots.")
        return

    def values(metric: str) -> list[list[float]]:
        out = []
        for label in labels:
            model, result = label.split("\n", 1)
            series = df[(df["model_label"] == model) & (df["result_label"] == result)][metric]
            out.append([float(v) for v in series.tolist() if pd.notna(v) and float(v) > 0])
        return out

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
    token_plot = axes[0].boxplot(values("total_tokens"), tick_labels=labels, patch_artist=True)
    axes[0].set_title("RQ1: Total tokens by success/failure")
    axes[0].set_ylabel("Total tokens (log scale)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    for patch in token_plot["boxes"]:
        patch.set_facecolor("#D9E6F2")

    step_plot = axes[1].boxplot(values("api_calls"), tick_labels=labels, patch_artist=True)
    axes[1].set_title("RQ1: Steps by success/failure")
    axes[1].set_ylabel("Assistant steps / API calls")
    axes[1].grid(True, alpha=0.25)
    for patch in step_plot["boxes"]:
        patch.set_facecolor("#F6D8BD")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=0)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(10)

    fig.tight_layout()
    out = output_dir / "common50_boxplots.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote: {out}")


def plot_cost_tail(task_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot common-task costs from the runner's local CNY price schedule."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    df = task_df.copy()
    df["cost_cny"] = pd.to_numeric(df["estimated_total_cost_rmb"], errors="coerce")
    df = df[df["cost_cny"].gt(0)].copy()
    if df.empty:
        print("\nNo per-instance CNY costs available; skipped cost-tail plot.")
        return

    df["model_label"] = [
        model_display_name(model, subset_dir)
        for model, subset_dir in zip(df["model"], df["subset_dir"])
    ]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    for color, (label, group) in zip(colors, df.groupby("model_label", sort=False)):
        costs = sorted(group["cost_cny"])
        count = len(costs)
        ecdf_y = [(index + 1) / count for index in range(count)]
        pareto_y = []
        running_cost = 0.0
        total_cost = sum(costs)
        for cost in reversed(costs):
            running_cost += cost
            pareto_y.append(running_cost / total_cost)
        legend = (
            f"{label}  (P50 {pd.Series(costs).quantile(.5):.3f}, "
            f"P95 {pd.Series(costs).quantile(.95):.3f}, max {costs[-1]:.3f} CNY)"
        )
        axes[0].step(costs, ecdf_y, where="post", color=color, linewidth=2.3, label=legend)
        axes[1].plot(
            [0, *[(index + 1) / count for index in range(count)]],
            [0, *pareto_y],
            color=color,
            linewidth=2.3,
            label=label,
        )

    axes[0].set_title("Per-task cost ECDF (common 50)")
    axes[0].set_xlabel("Per-task cost (CNY, log scale)")
    axes[0].set_ylabel("Cumulative share of tasks")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].set_title("Cost concentration / Pareto curve (common 50)")
    axes[1].set_xlabel("Most expensive tasks included")
    axes[1].set_ylabel("Cumulative share of total cost")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xticks([0, 0.1, 0.2, 0.5, 1])
    axes[1].set_xticklabels(["0%", "10%", "20%", "50%", "100%"])
    axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axes[1].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    for suffix in ("png", "svg"):
        output_path = output_dir / f"common50_cost_tail.{suffix}"
        fig.savefig(output_path, dpi=180 if suffix == "png" else None)
        print(f"wrote: {output_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SWE-bench trace token/cost/success metrics on common submitted tasks."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        default=DEFAULT_RUN_DIRS,
        help="Run directories. Defaults to the three requested trace directories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Use the first N sorted common submitted task IDs. Use --limit 0 for all common tasks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/pace_source/scripts/model_cost_compare_common50"),
        help="Directory for CSV/Markdown/plot outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(d) for d in args.run_dirs]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)

    limit = None if args.limit == 0 else args.limit
    summary_df, task_df, task_ids = build_comparison(run_dirs, limit=limit)
    write_outputs(summary_df, task_df, task_ids, args.output_dir, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
