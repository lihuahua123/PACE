#!/usr/bin/env python3
"""Count per-task Mini SWE Agent step categories and draw a boxplot.

The category rules intentionally reuse scripts/classify_intent.py, where:
- grep/rg/ag/find become search intents
- cat/nl/head/tail/awk/sed -n become read intents
- sed -i/perl -i/heredoc/tee/write helpers become edit intents
- pytest/unittest/jest/go test/etc. become verify intents

This script maps classify_intent's high-level "verify" bucket to "test" to
match the analysis vocabulary requested here.

For streak metrics, source read/search uses consecutive successful repository
inspection actions. Test phase streak counts test attempts since the last edit,
including failed tests, and allows read/search/other actions to be interleaved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from classify_intent import (
    INTENT_TO_HIGH_LEVEL,
    classify_command,
    extract_observation_text,
    extract_steps,
    iter_traces,
    trace_id_for,
)


DEFAULT_TRACE_ROOT = Path(
    "/workspace/pace_source/swebench/log/"
    "my-2026-08-14__19-21-04-DeepSeek-V4-Flash_org"
)
DEFAULT_OUTPUT_ROOT = Path("/workspace/pace_source/scripts/outputs/task_step_categories")
DEFAULT_TRAJ_GLOB = "**/*.traj.json"
THRESHOLD_SCAN_PERCENTILES = (0.50, 0.75, 0.90, 0.95, 0.99)
CATEGORY_ORDER = ["search", "read", "edit", "test", "other"]
CATEGORY_COLORS = {
    "search": "#7c9b6f",
    "read": "#5a7d9a",
    "edit": "#4a8a5a",
    "test": "#c77f4f",
    "other": "#9a9a9a",
}
EDIT_PATTERNS = (
    r"\bsed\s+-i\b",
    r"\bperl\s+-\S*i\b",
    r"\bapply_patch\b",
    r"\bpython3?\b.*\bwrite_text\b",
    r"\bpython3?\b.*\bopen\([^)]*,\s*['\"](?:w|a)",
    r"\bcat\s+<<[\s\S]*>",
    r"\bcat\s+>{1,2}\b",
    r"\btee\s+",
)
SCRATCH_PATH_PATTERNS = (
    r"(?:^|/)(?:tmp|\.tmp)(?:/|$)",
    r"(?:^|/)(?:var/tmp)(?:/|$)",
)
SCRATCH_WRITE_PATTERNS = (
    r"\bsed\s+-i\b",
    r"\bperl\s+-\S*i\b",
    r"\bapply_patch\b",
    r"\bpython3?\b.*\b(?:write_text|open\([^)]*,\s*['\"](?:w|a))",
    r"\bcat\s+<<[\s\S]*>",
    r"\bcat\s+>{1,2}\b",
    r"\btee\s+",
    r">\s*[^&\s]+",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument(
        "--threshold-scan-roots",
        type=Path,
        nargs="+",
        help="Run offline TRACE_STATUS_REPEAT_BLOCK threshold scanning for multiple trace roots.",
    )
    parser.add_argument("--traj-glob", default=DEFAULT_TRAJ_GLOB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def looks_like_edit_command(raw_command: str) -> bool:
    return any(re.search(pattern, raw_command, re.S) for pattern in EDIT_PATTERNS)


def looks_like_scratch_experiment(raw_command: str) -> bool:
    """Mirror the runtime's temporary-script experiment classification."""
    normalized = raw_command.replace("\\\n", " ")
    if any(re.search(pattern, normalized) for pattern in SCRATCH_WRITE_PATTERNS):
        if any(re.search(pattern, normalized) for pattern in SCRATCH_PATH_PATTERNS):
            return True
    if re.search(r"\bpython3?\s+([^\s|;&]+)", normalized):
        python_script = re.search(r"\bpython3?\s+([^\s|;&]+)", normalized).group(1)
        if python_script.endswith(".py") and any(
            re.search(pattern, python_script) for pattern in SCRATCH_PATH_PATTERNS
        ):
            return True
    return False


def is_source_edit(category: str, raw_command: str) -> bool:
    return category == "edit" and not looks_like_scratch_experiment(raw_command)


def command_category(raw_command: str, intent: str) -> str:
    high_level = INTENT_TO_HIGH_LEVEL.get(intent, "other")
    if high_level == "verify":
        return "test"
    if high_level == "edit" or looks_like_edit_command(raw_command):
        return "edit"
    if high_level in {"search", "read"}:
        return high_level
    return "other"


def classify_trace(traj_path: Path, trace_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(traj_path)
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError(f"Unsupported trajectory format: {traj_path}")

    trace_id = trace_id_for(traj_path, trace_root)
    counts: Counter[str] = Counter({category: 0 for category in CATEGORY_ORDER})
    intent_counts: Counter[str] = Counter()
    step_rows: list[dict[str, Any]] = []
    max_success_streaks: Counter[str] = Counter({category: 0 for category in CATEGORY_ORDER})
    current_success_category = ""
    current_success_streak = 0
    current_success_source_inspection_streak = 0
    max_success_source_inspection_streak = 0
    current_total_inspections_since_edit = 0
    max_total_inspections_since_edit = 0
    current_scratch_experiments_since_edit = 0
    max_scratch_experiments_since_edit = 0
    current_test_phase_streak = 0
    max_test_phase_streak = 0

    for step in extract_steps(trace_id, traj_path, messages):
        observation_text = extract_observation_text(step.observation_raw)
        result = classify_command(step.raw_command, observation_text)
        category = command_category(step.raw_command, result.intent)
        succeeded = step.returncode == 0
        counts[category] += 1
        intent_counts[result.intent] += 1
        if succeeded:
            if category == current_success_category:
                current_success_streak += 1
            else:
                current_success_category = category
                current_success_streak = 1
            max_success_streaks[category] = max(max_success_streaks[category], current_success_streak)
        else:
            current_success_category = ""
            current_success_streak = 0
        if succeeded and category in {"search", "read"}:
            current_success_source_inspection_streak += 1
            max_success_source_inspection_streak = max(
                max_success_source_inspection_streak,
                current_success_source_inspection_streak,
            )
        else:
            current_success_source_inspection_streak = 0
        source_edit = is_source_edit(category, step.raw_command)
        if source_edit:
            current_total_inspections_since_edit = 0
            current_scratch_experiments_since_edit = 0
            current_test_phase_streak = 0
        else:
            if succeeded and category in {"search", "read"}:
                current_total_inspections_since_edit += 1
                max_total_inspections_since_edit = max(
                    max_total_inspections_since_edit,
                    current_total_inspections_since_edit,
                )
            if succeeded and looks_like_scratch_experiment(step.raw_command):
                current_scratch_experiments_since_edit += 1
                max_scratch_experiments_since_edit = max(
                    max_scratch_experiments_since_edit,
                    current_scratch_experiments_since_edit,
                )
        # Preserve the existing test-phase metric's category-level semantics.
        if category == "edit":
            current_test_phase_streak = 0
        elif category == "test":
            current_test_phase_streak += 1
            max_test_phase_streak = max(max_test_phase_streak, current_test_phase_streak)
        step_rows.append(
            {
                "trace_id": trace_id,
                "traj_path": str(traj_path),
                "step_index": step.step_index,
                "message_index": step.message_index,
                "category": category,
                "intent": result.intent,
                "rule": result.rule,
                "returncode": step.returncode,
                "command_head": result.command_head,
                "raw_command": step.raw_command,
            }
        )

    summary = {
        "trace_id": trace_id,
        "traj_path": str(traj_path),
        "total_steps": sum(counts.values()),
        **{f"{category}_steps": counts[category] for category in CATEGORY_ORDER},
        **{f"{category}_max_success_streak": max_success_streaks[category] for category in CATEGORY_ORDER},
        "source_read_search_max_success_streak": max_success_source_inspection_streak,
        "max_total_inspections_since_edit": max_total_inspections_since_edit,
        "max_scratch_experiments_since_edit": max_scratch_experiments_since_edit,
        "test_phase_streak": max_test_phase_streak,
        "intent_counts": json.dumps(dict(intent_counts), ensure_ascii=False, sort_keys=True),
    }
    return summary, step_rows


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = pos - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def box_stats(values: list[int]) -> dict[str, float]:
    sorted_values = sorted(values)
    q1 = percentile(sorted_values, 0.25)
    median = percentile(sorted_values, 0.50)
    q3 = percentile(sorted_values, 0.75)
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    non_outliers = [value for value in sorted_values if low_fence <= value <= high_fence]
    return {
        "min": float(non_outliers[0] if non_outliers else sorted_values[0]),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(non_outliers[-1] if non_outliers else sorted_values[-1]),
        "mean": sum(sorted_values) / len(sorted_values) if sorted_values else 0.0,
    }


REPEAT_BLOCK_METRICS = {
    "consecutive_inspections": "source_read_search_max_success_streak",
    "total_inspections": "max_total_inspections_since_edit",
    "scratch_experiments": "max_scratch_experiments_since_edit",
    "test_phase_tests": "test_phase_streak",
}


def threshold_label(q: float) -> str:
    return f"P{int(q * 100):02d}"


def threshold_config_from_rows(task_rows: list[dict[str, Any]], q: float) -> dict[str, int]:
    return {
        metric: max(1, math.ceil(percentile(sorted(int(row[field]) for row in task_rows), q)))
        for metric, field in REPEAT_BLOCK_METRICS.items()
    }


def simulate_budget_blocks(
    task_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    thresholds: dict[str, int],
) -> dict[str, Any]:
    """Counterfactually replay only the four repeat-block budgets.

    This intentionally excludes repeated-evidence blocks. A blocked action does
    not update simulated state, matching DefaultAgent.execute_action(). The
    result therefore answers: how many actions in the archived trace would have
    been stopped by these budget thresholds?
    """
    del task_rows  # Kept in the signature to make the relationship explicit.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in step_rows:
        grouped.setdefault(str(row["trace_id"]), []).append(row)

    blocked_actions = 0
    blocked_tasks: set[str] = set()
    block_counts = Counter()
    for trace_id, rows in grouped.items():
        consecutive = 0
        total = 0
        scratch = 0
        tests = 0
        for row in rows:
            category = str(row["category"])
            raw_command = str(row["raw_command"])
            succeeded = row["returncode"] == 0
            inspection = succeeded and category in {"search", "read"}
            source_edit = is_source_edit(category, raw_command)
            scratch_action = looks_like_scratch_experiment(raw_command)
            is_test = category == "test"

            block_kind = ""
            if inspection and consecutive >= thresholds["consecutive_inspections"]:
                block_kind = "consecutive_inspections"
            elif inspection and total >= thresholds["total_inspections"]:
                block_kind = "total_inspections"
            elif is_test and tests >= thresholds["test_phase_tests"]:
                block_kind = "test_phase_tests"
            elif scratch_action and scratch >= thresholds["scratch_experiments"]:
                block_kind = "scratch_experiments"

            if block_kind:
                blocked_actions += 1
                blocked_tasks.add(trace_id)
                block_counts[block_kind] += 1
                continue

            # This mirrors the runtime state transitions after an action runs.
            if inspection:
                consecutive += 1
                total += 1
            else:
                consecutive = 0
            if source_edit:
                consecutive = 0
                total = 0
                scratch = 0
                tests = 0
            elif succeeded and scratch_action:
                scratch += 1
            if is_test and not source_edit:
                tests += 1

    return {
        "blocked_tasks": len(blocked_tasks),
        "blocked_actions": blocked_actions,
        "blocked_consecutive_inspections": block_counts["consecutive_inspections"],
        "blocked_total_inspections": block_counts["total_inspections"],
        "blocked_scratch_experiments": block_counts["scratch_experiments"],
        "blocked_test_phase_tests": block_counts["test_phase_tests"],
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def short_model_name(model: str) -> str:
    replacements = {
        "my-2026-08-16__08-43-07-MiniMax-M2.7_org": "MiniMax-M2.7",
        "my-2026-08-14__19-21-04-DeepSeek-V4-Flash_org": "DeepSeek-V4-Flash",
        "my-2026-08-14__08-48-42-gpt-5.6-luna_org": "GPT-5.6-luna",
    }
    return replacements.get(model, model)


def draw_threshold_scan_svg_fallback(
    output_dir: Path,
    metric_values_by_model: dict[str, dict[str, list[int]]],
    scan_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Create simple SVG versions when matplotlib is unavailable."""
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    models = list(metric_values_by_model)
    metrics = [
        ("consecutive_inspections", "Consecutive inspections"),
        ("total_inspections", "Total inspections since edit"),
        ("scratch_experiments", "Scratch experiments since edit"),
        ("test_phase_tests", "Test actions since edit"),
    ]
    paths: dict[str, str] = {}

    def esc(value: Any) -> str:
        return (str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))

    def line_plot(
        path: Path,
        title: str,
        x_labels: list[str],
        series: list[tuple[str, list[float], str]],
        y_label: str,
        width: int = 960,
        height: int = 560,
    ) -> None:
        left, right, top, bottom = 90, 35, 58, 80
        plot_w, plot_h = width - left - right, height - top - bottom
        ymax = max(max(values, default=1) for _, values, _ in series)
        ymax = max(1.0, ymax * 1.08)

        def x_at(index: int) -> float:
            return left + (index / max(1, len(x_labels) - 1)) * plot_w

        def y_at(value: float) -> float:
            return top + plot_h - value / ymax * plot_h

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{left}" y="30" font-family="Arial" font-size="22" font-weight="bold">{esc(title)}</text>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#444"/>',
        ]
        for tick in range(6):
            value = ymax * tick / 5
            y = y_at(value)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            parts.append(f'<text x="{left - 8}" y="{y + 6:.1f}" text-anchor="end" font-family="Arial" font-size="18">{value:.1f}</text>')
        for index, label in enumerate(x_labels):
            parts.append(f'<text x="{x_at(index):.1f}" y="{top + plot_h + 30}" text-anchor="middle" font-family="Arial" font-size="20">{esc(label)}</text>')
        for label, values, color in series:
            points = " ".join(f"{x_at(i):.1f},{y_at(value):.1f}" for i, value in enumerate(values))
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for i, value in enumerate(values):
                parts.append(f'<circle cx="{x_at(i):.1f}" cy="{y_at(value):.1f}" r="3.5" fill="{color}"/>')
        for index, (label, _values, color) in enumerate(series):
            x = left + 18 + (index % 3) * 230
            y = height - 28 - (index // 3) * 18
            parts.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 18}" y2="{y - 4}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text x="{x + 24}" y="{y}" font-family="Arial" font-size="18">{esc(label)}</text>')
        parts.append(f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="20">{esc(y_label)}</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")

    # ECDF as four small panels in one SVG.
    width, height = 1100, 760
    panel_w, panel_h = 500, 300
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="35" y="30" font-family="Arial" font-size="22" font-weight="bold">Empirical distributions of repeat-block metrics</text>']
    for panel_index, (metric, label) in enumerate(metrics):
        origin_x = 55 + (panel_index % 2) * 530
        origin_y = 55 + (panel_index // 2) * 350
        left, top, plot_w, plot_h = origin_x + 55, origin_y + 25, panel_w - 85, panel_h - 75
        max_value = max((max(metric_values_by_model[model][metric], default=1) for model in models), default=1)
        parts.append(f'<text x="{origin_x + 8}" y="{origin_y + 15}" font-family="Arial" font-size="16" font-weight="bold">{esc(label)}</text>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444"/><line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#444"/>')
        for color, model in zip(colors, models):
            values = sorted(metric_values_by_model[model][metric])
            points = " ".join(f"{left + value / max_value * plot_w:.1f},{top + plot_h - (i + 1) / len(values) * plot_h:.1f}" for i, value in enumerate(values)) if values else ""
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 34}" text-anchor="middle" font-family="Arial" font-size="18">Maximum count</text>')
        parts.append(f'<text x="{origin_x + 12}" y="{top + plot_h / 2:.1f}" transform="rotate(-90 {origin_x + 12} {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="18">Cumulative proportion</text>')
    for index, model in enumerate(models):
        x = 70 + index * 270
        parts.append(f'<line x1="{x}" y1="738" x2="{x + 18}" y2="738" stroke="{colors[index]}" stroke-width="3"/><text x="{x + 24}" y="744" font-family="Arial" font-size="18">{esc(short_model_name(model))}</text>')
    parts.append("</svg>")
    ecdf_path = output_dir / "repeat_block_metric_ecdf.svg"
    ecdf_path.write_text("\n".join(parts), encoding="utf-8")
    paths["metric_ecdf_svg"] = str(ecdf_path)

    percentiles = [threshold_label(q) for q in THRESHOLD_SCAN_PERCENTILES]
    joint_rows = [row for row in scan_rows if row["scan_type"].startswith("joint_")]
    curve_series = []
    for color, model in zip(colors, models):
        rows = {row["scan_type"].removeprefix("joint_"): row for row in joint_rows if row["model"] == model}
        curve_series.append((short_model_name(model), [rows[p]["blocked_task_rate_pct"] for p in percentiles], color))
    curve_path = output_dir / "repeat_block_joint_task_rate_curve.svg"
    line_plot(curve_path, "Blocked task rate under joint configurations", percentiles, curve_series, "Blocked tasks (%)")
    paths["joint_task_curve_svg"] = str(curve_path)

    heatmap_path = output_dir / "repeat_block_joint_task_rate_heatmap.svg"
    cell_w, cell_h, left, top = 105, 48, 210, 80
    max_rate = max((row["blocked_task_rate_pct"] for row in joint_rows), default=1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="850" height="330"><rect width="100%" height="100%" fill="white"/><text x="35" y="35" font-family="Arial" font-size="20" font-weight="bold">Blocked task rate (%) under joint configurations</text>']
    for j, p in enumerate(percentiles):
        parts.append(f'<text x="{left + j * cell_w + cell_w / 2:.1f}" y="{top - 12}" text-anchor="middle" font-family="Arial" font-size="13">{p}</text>')
    for i, model in enumerate(models):
        parts.append(f'<text x="{left - 12}" y="{top + i * cell_h + cell_h / 2 + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{esc(short_model_name(model))}</text>')
        rows = {row["scan_type"].removeprefix("joint_"): row for row in joint_rows if row["model"] == model}
        for j, p in enumerate(percentiles):
            value = rows[p]["blocked_task_rate_pct"]
            intensity = value / max_rate if max_rate else 0
            color = f"rgb(255,{int(235 - 150 * intensity)},{int(210 - 170 * intensity)})"
            x, y = left + j * cell_w, top + i * cell_h
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="white"/><text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 5}" text-anchor="middle" font-family="Arial" font-size="13">{value:.1f}</text>')
    parts.append("</svg>")
    heatmap_path.write_text("\n".join(parts), encoding="utf-8")
    paths["joint_heatmap_svg"] = str(heatmap_path)
    return paths


def draw_threshold_scan_plots(
    output_dir: Path,
    metric_values_by_model: dict[str, dict[str, list[int]]],
    scan_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Draw paper-oriented plots for the offline threshold scan."""
    if plt is None:
        return draw_threshold_scan_svg_fallback(output_dir, metric_values_by_model, scan_rows)

    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    models = list(metric_values_by_model)
    metric_labels = {
        "consecutive_inspections": "Consecutive inspections",
        "total_inspections": "Total inspections since edit",
        "scratch_experiments": "Scratch experiments since edit",
        "test_phase_tests": "Test actions since edit",
    }
    output_paths: dict[str, str] = {}

    # Figure 1: empirical distributions of the four threshold metrics.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), dpi=180)
    for ax, (metric, label) in zip(axes.flat, metric_labels.items()):
        for color, model in zip(colors, models):
            values = sorted(metric_values_by_model[model][metric])
            if not values:
                continue
            y_values = [(index + 1) / len(values) for index in range(len(values))]
            ax.step(values, y_values, where="post", label=short_model_name(model), color=color)
        ax.set_title(label)
        ax.set_xlabel("Maximum count per trace")
        ax.set_ylabel("Cumulative proportion")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="both", color="#e5e5e5", linewidth=0.7)
    axes.flat[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Empirical distributions of repeat-block metrics", fontsize=14)
    fig.tight_layout()
    base = output_dir / "repeat_block_metric_ecdf"
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    output_paths["metric_ecdf_png"] = str(base.with_suffix(".png"))
    output_paths["metric_ecdf_svg"] = str(base.with_suffix(".svg"))

    # Figure 2: blocked task/action rate under joint percentile configurations.
    percentiles = [threshold_label(q) for q in THRESHOLD_SCAN_PERCENTILES]
    joint_rows = [row for row in scan_rows if row["scan_type"].startswith("joint_")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=180)
    for color, model in zip(colors, models):
        rows = [row for row in joint_rows if row["model"] == model]
        rows.sort(key=lambda row: percentiles.index(row["scan_type"].removeprefix("joint_")))
        x = list(range(len(rows)))
        axes[0].plot(x, [row["blocked_task_rate_pct"] for row in rows], marker="o", label=short_model_name(model), color=color)
        axes[1].plot(x, [row["blocked_action_rate_pct"] for row in rows], marker="o", label=short_model_name(model), color=color)
    for ax, ylabel, title in (
        (axes[0], "Blocked tasks (%)", "Task blocking rate"),
        (axes[1], "Blocked actions (%)", "Action blocking rate"),
    ):
        ax.set_xticks(range(len(percentiles)), percentiles)
        ax.set_xlabel("Joint percentile threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="both", color="#e5e5e5", linewidth=0.7)
    axes[0].legend(fontsize=8)
    fig.suptitle("Counterfactual blocking under joint threshold configurations", fontsize=14)
    fig.tight_layout()
    base = output_dir / "repeat_block_joint_threshold_curves"
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    output_paths["joint_curves_png"] = str(base.with_suffix(".png"))
    output_paths["joint_curves_svg"] = str(base.with_suffix(".svg"))

    # Figure 3: compact heatmap for the main paper table replacement.
    matrix = []
    for model in models:
        rows = {row["scan_type"].removeprefix("joint_"): row for row in joint_rows if row["model"] == model}
        matrix.append([rows[percentile]["blocked_task_rate_pct"] for percentile in percentiles])
    fig, ax = plt.subplots(figsize=(7.4, 3.7), dpi=180)
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(max(row) for row in matrix) if matrix else 1, aspect="auto")
    ax.set_xticks(range(len(percentiles)), percentiles)
    ax.set_yticks(range(len(models)), [short_model_name(model) for model in models])
    ax.set_xlabel("Joint percentile threshold")
    ax.set_title("Blocked task rate (%) under joint configurations")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(column_index, row_index, f"{value:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Blocked tasks (%)")
    fig.tight_layout()
    base = output_dir / "repeat_block_joint_task_rate_heatmap"
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    output_paths["joint_heatmap_png"] = str(base.with_suffix(".png"))
    output_paths["joint_heatmap_svg"] = str(base.with_suffix(".svg"))
    return output_paths


def run_threshold_scan(args: argparse.Namespace) -> None:
    output_dir = args.output_root / "trace_status_repeat_block_threshold_scan"
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    metric_values_by_model: dict[str, dict[str, list[int]]] = {}

    for trace_root in args.threshold_scan_roots:
        traces = iter_traces(trace_root, args.traj_glob)
        if not traces:
            raise SystemExit(f"No trajectory files found under {trace_root} with glob {args.traj_glob!r}")
        task_rows: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        for traj_path in traces:
            task_summary, trace_steps = classify_trace(traj_path, trace_root)
            task_rows.append(task_summary)
            step_rows.extend(trace_steps)

        model = trace_root.name
        metric_values_by_model[model] = {
            metric: [int(row[field]) for row in task_rows]
            for metric, field in REPEAT_BLOCK_METRICS.items()
        }
        for metric, field in REPEAT_BLOCK_METRICS.items():
            values = sorted(int(row[field]) for row in task_rows)
            for q in THRESHOLD_SCAN_PERCENTILES:
                threshold = max(1, math.ceil(percentile(values, q)))
                distribution_rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "percentile": threshold_label(q),
                        "percentile_value": round(percentile(values, q), 3),
                        "threshold": threshold,
                        "task_count": len(task_rows),
                        "tasks_at_or_above_threshold": sum(value >= threshold for value in values),
                    }
                )

        configs: list[tuple[str, dict[str, int]]] = []
        for q in THRESHOLD_SCAN_PERCENTILES:
            configs.append((f"joint_{threshold_label(q)}", threshold_config_from_rows(task_rows, q)))
            for metric in REPEAT_BLOCK_METRICS:
                config = {name: 10**9 for name in REPEAT_BLOCK_METRICS}
                config[metric] = threshold_config_from_rows(task_rows, q)[metric]
                configs.append((f"single_{metric}_{threshold_label(q)}", config))

        for scan_type, config in configs:
            result = simulate_budget_blocks(task_rows, step_rows, config)
            task_count = len(task_rows)
            action_count = len(step_rows)
            scan_rows.append(
                {
                    "model": model,
                    "scan_type": scan_type,
                    "task_count": task_count,
                    "action_count": action_count,
                    "threshold_consecutive_inspections": config["consecutive_inspections"],
                    "threshold_total_inspections": config["total_inspections"],
                    "threshold_scratch_experiments": config["scratch_experiments"],
                    "threshold_test_phase_tests": config["test_phase_tests"],
                    **result,
                    "blocked_task_rate_pct": round(100 * result["blocked_tasks"] / task_count, 2),
                    "blocked_action_rate_pct": round(100 * result["blocked_actions"] / action_count, 2),
                }
            )

    plot_paths = draw_threshold_scan_plots(output_dir, metric_values_by_model, scan_rows)
    scan_csv = output_dir / "trace_status_repeat_block_threshold_scan.csv"
    distribution_csv = output_dir / "trace_status_repeat_block_threshold_distributions.csv"
    scan_json = output_dir / "trace_status_repeat_block_threshold_scan.json"
    scan_md = output_dir / "trace_status_repeat_block_threshold_scan.md"
    scan_fields = list(scan_rows[0]) if scan_rows else []
    distribution_fields = list(distribution_rows[0]) if distribution_rows else []
    write_csv(scan_csv, scan_rows, scan_fields)
    write_csv(distribution_csv, distribution_rows, distribution_fields)
    write_json(
        scan_json,
        {
            "trace_roots": [str(root) for root in args.threshold_scan_roots],
            "percentiles": [threshold_label(q) for q in THRESHOLD_SCAN_PERCENTILES],
            "metric_fields": REPEAT_BLOCK_METRICS,
            "scan_semantics": (
                "Counterfactual replay of archived traces using only the four budget thresholds. "
                "Repeated-evidence blocks are excluded; blocked actions do not update simulated state."
            ),
            "rows": scan_rows,
            "distributions": distribution_rows,
            "plots": plot_paths,
        },
    )
    joint_rows = [row for row in scan_rows if row["scan_type"].startswith("joint_")]
    columns = [
        "model",
        "scan_type",
        "threshold_consecutive_inspections",
        "threshold_total_inspections",
        "threshold_scratch_experiments",
        "threshold_test_phase_tests",
        "blocked_tasks",
        "blocked_task_rate_pct",
        "blocked_actions",
        "blocked_action_rate_pct",
        "blocked_consecutive_inspections",
        "blocked_total_inspections",
        "blocked_scratch_experiments",
        "blocked_test_phase_tests",
    ]
    scan_md.write_text(
        "# TRACE_STATUS_REPEAT_BLOCK offline threshold scan\n\n"
        "Joint scans set all four thresholds to the same empirical percentile-derived configuration. "
        "Single-metric scans vary one threshold while setting the other three to a very large value.\n\n"
        + "## Joint percentile configurations\n\n"
        + markdown_table(joint_rows, columns)
        + "\n## Single-metric percentile configurations\n\n"
        + markdown_table(
            [
                {
                    "model": row["model"],
                    "metric_percentile": row["scan_type"].removeprefix("single_").replace("_", " "),
                    "threshold": (
                        row["threshold_consecutive_inspections"]
                        if row["scan_type"].startswith("single_consecutive_inspections")
                        else row["threshold_total_inspections"]
                        if row["scan_type"].startswith("single_total_inspections")
                        else row["threshold_scratch_experiments"]
                        if row["scan_type"].startswith("single_scratch_experiments")
                        else row["threshold_test_phase_tests"]
                    ),
                    "blocked_tasks": row["blocked_tasks"],
                    "blocked_task_rate_pct": row["blocked_task_rate_pct"],
                    "blocked_actions": row["blocked_actions"],
                    "blocked_action_rate_pct": row["blocked_action_rate_pct"],
                }
                for row in scan_rows
                if row["scan_type"].startswith("single_")
            ],
            [
                "model",
                "metric_percentile",
                "threshold",
                "blocked_tasks",
                "blocked_task_rate_pct",
                "blocked_actions",
                "blocked_action_rate_pct",
            ],
        ),
        encoding="utf-8",
    )
    print(f"Wrote threshold scan tables under {output_dir}")
    print(scan_md.read_text(encoding="utf-8"))


def draw_svg_boxplot(task_rows: list[dict[str, Any]], output_base: Path, value_suffix: str, title: str, ylabel: str) -> str:
    series = {
        category: [int(row[f"{category}_{value_suffix}"]) for row in task_rows]
        for category in CATEGORY_ORDER
    }
    stats = {category: box_stats(values) for category, values in series.items()}
    ymax = max((max(values) for values in series.values() if values), default=1)
    ymax = max(1, int(ymax * 1.12) + 1)

    width, height = 920, 560
    left, right, top, bottom = 86, 34, 52, 82
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_at(index: int) -> float:
        return left + (index + 0.5) * plot_w / len(CATEGORY_ORDER)

    def y_at(value: float) -> float:
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#222">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#444"/>',
    ]
    tick_count = 6
    for i in range(tick_count + 1):
        value = ymax * i / tick_count
        y = y_at(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#555">{value:.0f}</text>'
        )

    box_w = min(82, plot_w / len(CATEGORY_ORDER) * 0.46)
    for index, category in enumerate(CATEGORY_ORDER):
        x = x_at(index)
        stat = stats[category]
        y_q1, y_med, y_q3 = y_at(stat["q1"]), y_at(stat["median"]), y_at(stat["q3"])
        y_min, y_max, y_mean = y_at(stat["min"]), y_at(stat["max"]), y_at(stat["mean"])
        color = CATEGORY_COLORS[category]
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{y_max:.2f}" x2="{x:.2f}" y2="{y_min:.2f}" stroke="#555"/>',
                f'<line x1="{x - box_w / 3:.2f}" y1="{y_max:.2f}" x2="{x + box_w / 3:.2f}" y2="{y_max:.2f}" stroke="#555"/>',
                f'<line x1="{x - box_w / 3:.2f}" y1="{y_min:.2f}" x2="{x + box_w / 3:.2f}" y2="{y_min:.2f}" stroke="#555"/>',
                f'<rect x="{x - box_w / 2:.2f}" y="{y_q3:.2f}" width="{box_w:.2f}" height="{max(1, y_q1 - y_q3):.2f}" fill="{color}" fill-opacity="0.72" stroke="#444"/>',
                f'<line x1="{x - box_w / 2:.2f}" y1="{y_med:.2f}" x2="{x + box_w / 2:.2f}" y2="{y_med:.2f}" stroke="#222" stroke-width="2"/>',
                f'<line x1="{x - box_w * 0.62:.2f}" y1="{y_q3:.2f}" x2="{x + box_w * 0.62:.2f}" y2="{y_q3:.2f}" stroke="#b33a2f" stroke-width="2.2"/>',
                f'<text x="{x + box_w * 0.66:.2f}" y="{y_q3 + 4:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#b33a2f">P75={stat["q3"]:.1f}</text>',
                f'<path d="M {x:.2f} {y_mean - 5:.2f} L {x + 5:.2f} {y_mean:.2f} L {x:.2f} {y_mean + 5:.2f} L {x - 5:.2f} {y_mean:.2f} Z" fill="#222"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333">{category}</text>',
            ]
        )

    parts.append(
        f'<text x="22" y="{top + plot_h / 2:.2f}" transform="rotate(-90 22 {top + plot_h / 2:.2f})" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333">{ylabel}</text>'
    )
    parts.append("</svg>")

    svg_path = output_base.with_suffix(".svg")
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return str(svg_path)


def draw_boxplot(
    task_rows: list[dict[str, Any]],
    output_base: Path,
    value_suffix: str = "steps",
    title: str = "Per-task step counts by category",
    ylabel: str = "Steps per task",
) -> dict[str, str]:
    values = [[int(row[f"{category}_{value_suffix}"]) for row in task_rows] for category in CATEGORY_ORDER]

    if plt is None:
        return {"boxplot_svg": draw_svg_boxplot(task_rows, output_base, value_suffix, title, ylabel)}

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    box = ax.boxplot(
        values,
        tick_labels=CATEGORY_ORDER,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#222222",
            "markeredgecolor": "#222222",
            "markersize": 4,
        },
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch, category in zip(box["boxes"], CATEGORY_ORDER):
        patch.set_facecolor(CATEGORY_COLORS[category])
        patch.set_alpha(0.72)
        patch.set_edgecolor("#444444")

    for index, category_values in enumerate(values, start=1):
        p75 = percentile(sorted(category_values), 0.75)
        ax.hlines(p75, index - 0.34, index + 0.34, colors="#b33a2f", linewidth=1.6, zorder=3)
        ax.annotate(
            f"P75={p75:.1f}",
            xy=(index + 0.36, p75),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#b33a2f",
        )

    ax.set_xlabel("Step category")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"boxplot_png": str(png_path), "boxplot_svg": str(svg_path)}


def draw_single_svg_boxplot(
    values: list[int],
    output_base: Path,
    title: str,
    ylabel: str,
    label: str,
) -> str:
    stat = box_stats(values)
    ymax = max(max(values, default=1), 1)
    ymax = max(1, int(ymax * 1.12) + 1)
    width, height = 520, 560
    left, right, top, bottom = 90, 42, 52, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    x = left + plot_w / 2
    box_w = 92

    def y_at(value: float) -> float:
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="28" y="34" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#222">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#444"/>',
    ]
    for i in range(7):
        value = ymax * i / 6
        y = y_at(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#555">{value:.0f}</text>'
        )

    y_q1, y_med, y_q3 = y_at(stat["q1"]), y_at(stat["median"]), y_at(stat["q3"])
    y_min, y_max, y_mean = y_at(stat["min"]), y_at(stat["max"]), y_at(stat["mean"])
    parts.extend(
        [
            f'<line x1="{x:.2f}" y1="{y_max:.2f}" x2="{x:.2f}" y2="{y_min:.2f}" stroke="#555"/>',
            f'<line x1="{x - box_w / 3:.2f}" y1="{y_max:.2f}" x2="{x + box_w / 3:.2f}" y2="{y_max:.2f}" stroke="#555"/>',
            f'<line x1="{x - box_w / 3:.2f}" y1="{y_min:.2f}" x2="{x + box_w / 3:.2f}" y2="{y_min:.2f}" stroke="#555"/>',
            f'<rect x="{x - box_w / 2:.2f}" y="{y_q3:.2f}" width="{box_w:.2f}" height="{max(1, y_q1 - y_q3):.2f}" fill="#6d8f91" fill-opacity="0.76" stroke="#444"/>',
            f'<line x1="{x - box_w / 2:.2f}" y1="{y_med:.2f}" x2="{x + box_w / 2:.2f}" y2="{y_med:.2f}" stroke="#222" stroke-width="2"/>',
            f'<line x1="{x - box_w * 0.62:.2f}" y1="{y_q3:.2f}" x2="{x + box_w * 0.62:.2f}" y2="{y_q3:.2f}" stroke="#b33a2f" stroke-width="2.2"/>',
            f'<text x="{x + box_w * 0.68:.2f}" y="{y_q3 + 4:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#b33a2f">P75={stat["q3"]:.1f}</text>',
            f'<path d="M {x:.2f} {y_mean - 5:.2f} L {x + 5:.2f} {y_mean:.2f} L {x:.2f} {y_mean + 5:.2f} L {x - 5:.2f} {y_mean:.2f} Z" fill="#222"/>',
            f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333">{label}</text>',
        ]
    )
    parts.append(
        f'<text x="22" y="{top + plot_h / 2:.2f}" transform="rotate(-90 22 {top + plot_h / 2:.2f})" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333">{ylabel}</text>'
    )
    parts.append("</svg>")

    svg_path = output_base.with_suffix(".svg")
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return str(svg_path)


def draw_single_boxplot(
    values: list[int],
    output_base: Path,
    title: str,
    ylabel: str,
    label: str,
) -> dict[str, str]:
    if plt is None:
        return {"boxplot_svg": draw_single_svg_boxplot(values, output_base, title, ylabel, label)}

    p75 = percentile(sorted(values), 0.75)
    fig, ax = plt.subplots(figsize=(4.8, 5.2), dpi=160)
    box = ax.boxplot(
        [values],
        tick_labels=[label],
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#222222",
            "markeredgecolor": "#222222",
            "markersize": 4,
        },
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    box["boxes"][0].set_facecolor("#6d8f91")
    box["boxes"][0].set_alpha(0.76)
    box["boxes"][0].set_edgecolor("#444444")
    ax.hlines(p75, 0.66, 1.34, colors="#b33a2f", linewidth=1.6, zorder=3)
    ax.annotate(f"P75={p75:.1f}", xy=(1.36, p75), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color="#b33a2f")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"boxplot_png": str(png_path), "boxplot_svg": str(svg_path)}


def draw_named_series_svg_boxplot(
    series: list[tuple[str, list[int], str]],
    output_base: Path,
    title: str,
    ylabel: str,
    ylim: tuple[int, int] | None = None,
) -> str:
    stats = [
        (label, values, color, box_stats([value for value in values if ylim is None or value <= ylim[1]]))
        for label, values, color in series
    ]
    ymax = ylim[1] if ylim else max((max(values) for _, values, _ in series if values), default=1)
    ymax = max(1, int(ymax * 1.12) + 1) if ylim is None else ymax

    width, height = 760, 560
    left, right, top, bottom = 90, 46, 52, 96
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_at(index: int) -> float:
        return left + (index + 0.5) * plot_w / len(series)

    def y_at(value: float) -> float:
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="28" y="34" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#222">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#444"/>',
    ]
    for i in range(7):
        value = ymax * i / 6
        y = y_at(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#555">{value:.0f}</text>'
        )

    box_w = min(98, plot_w / len(series) * 0.38)
    for index, (label, _values, color, stat) in enumerate(stats):
        x = x_at(index)
        y_q1, y_med, y_q3 = y_at(stat["q1"]), y_at(stat["median"]), y_at(stat["q3"])
        y_min, y_max, y_mean = y_at(stat["min"]), y_at(stat["max"]), y_at(stat["mean"])
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{y_max:.2f}" x2="{x:.2f}" y2="{y_min:.2f}" stroke="#555"/>',
                f'<line x1="{x - box_w / 3:.2f}" y1="{y_max:.2f}" x2="{x + box_w / 3:.2f}" y2="{y_max:.2f}" stroke="#555"/>',
                f'<line x1="{x - box_w / 3:.2f}" y1="{y_min:.2f}" x2="{x + box_w / 3:.2f}" y2="{y_min:.2f}" stroke="#555"/>',
                f'<rect x="{x - box_w / 2:.2f}" y="{y_q3:.2f}" width="{box_w:.2f}" height="{max(1, y_q1 - y_q3):.2f}" fill="{color}" fill-opacity="0.76" stroke="#444"/>',
                f'<line x1="{x - box_w / 2:.2f}" y1="{y_med:.2f}" x2="{x + box_w / 2:.2f}" y2="{y_med:.2f}" stroke="#222" stroke-width="2"/>',
                f'<line x1="{x - box_w * 0.62:.2f}" y1="{y_q3:.2f}" x2="{x + box_w * 0.62:.2f}" y2="{y_q3:.2f}" stroke="#b33a2f" stroke-width="2.2"/>',
                f'<text x="{x + box_w * 0.68:.2f}" y="{y_q3 + 4:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#b33a2f">P75={stat["q3"]:.1f}</text>',
                f'<path d="M {x:.2f} {y_mean - 5:.2f} L {x + 5:.2f} {y_mean:.2f} L {x:.2f} {y_mean + 5:.2f} L {x - 5:.2f} {y_mean:.2f} Z" fill="#222"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333">{label}</text>',
            ]
        )
    parts.append(
        f'<text x="22" y="{top + plot_h / 2:.2f}" transform="rotate(-90 22 {top + plot_h / 2:.2f})" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333">{ylabel}</text>'
    )
    parts.append("</svg>")

    svg_path = output_base.with_suffix(".svg")
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return str(svg_path)


def draw_named_series_boxplot(
    series: list[tuple[str, list[int], str]],
    output_base: Path,
    title: str,
    ylabel: str,
    ylim: tuple[int, int] | None = None,
) -> dict[str, str]:
    if plt is None:
        return {"boxplot_svg": draw_named_series_svg_boxplot(series, output_base, title, ylabel, ylim)}

    labels = [label for label, _values, _color in series]
    values = [values for _label, values, _color in series]
    colors = [color for _label, _values, color in series]
    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=160)
    box = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#222222",
            "markeredgecolor": "#222222",
            "markersize": 4,
        },
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.76)
        patch.set_edgecolor("#444444")
    for index, series_values in enumerate(values, start=1):
        p75 = percentile(sorted(series_values), 0.75)
        ax.hlines(p75, index - 0.34, index + 0.34, colors="#b33a2f", linewidth=1.6, zorder=3)
        ax.annotate(
            f"P75={p75:.1f}",
            xy=(index + 0.36, p75),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#b33a2f",
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"boxplot_png": str(png_path), "boxplot_svg": str(svg_path)}


def main() -> None:
    args = parse_args()
    if args.threshold_scan_roots:
        run_threshold_scan(args)
        return
    traces = iter_traces(args.trace_root, args.traj_glob)
    if not traces:
        raise SystemExit(f"No trajectory files found under {args.trace_root} with glob {args.traj_glob!r}")

    output_dir = args.output_root / args.trace_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    task_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for traj_path in traces:
        task_summary, trace_steps = classify_trace(traj_path, args.trace_root)
        task_rows.append(task_summary)
        step_rows.extend(trace_steps)

    task_csv = output_dir / "task_step_category_counts.csv"
    step_csv = output_dir / "step_category_rows.csv"
    summary_json = output_dir / "summary.json"

    write_csv(
        task_csv,
        task_rows,
        [
            "trace_id",
            "traj_path",
            "total_steps",
            "search_steps",
            "read_steps",
            "edit_steps",
            "test_steps",
            "other_steps",
            "search_max_success_streak",
            "read_max_success_streak",
            "edit_max_success_streak",
            "test_max_success_streak",
            "other_max_success_streak",
            "source_read_search_max_success_streak",
            "max_total_inspections_since_edit",
            "max_scratch_experiments_since_edit",
            "test_phase_streak",
            "intent_counts",
        ],
    )
    write_csv(
        step_csv,
        step_rows,
        [
            "trace_id",
            "traj_path",
            "step_index",
            "message_index",
            "category",
            "intent",
            "rule",
            "returncode",
            "command_head",
            "raw_command",
        ],
    )
    plot_paths = draw_boxplot(task_rows, output_dir / "task_step_category_boxplot")
    success_streak_plot_paths = draw_boxplot(
        task_rows,
        output_dir / "task_step_category_success_streak_boxplot",
        value_suffix="max_success_streak",
        title="Per-task longest successful streak by category",
        ylabel="Longest consecutive successful steps per task",
    )
    source_inspection_streaks = [int(row["source_read_search_max_success_streak"]) for row in task_rows]
    test_phase_streaks = [int(row["test_phase_streak"]) for row in task_rows]
    phase_streak_plot_paths = draw_named_series_boxplot(
        [
            ("source read/search", source_inspection_streaks, "#6d8f91"),
            ("test phase", test_phase_streaks, "#c77f4f"),
        ],
        output_dir / "source_inspection_and_test_phase_streak_boxplot",
        title="Per-task source inspection and test-phase attempts",
        ylabel="Longest streak / attempt count per task",
        ylim=(0, 40),
    )
    repeat_block_series = [
        (
            "consecutive inspections",
            source_inspection_streaks,
            "#6d8f91",
        ),
        (
            "total inspections",
            [int(row["max_total_inspections_since_edit"]) for row in task_rows],
            "#78956b",
        ),
        (
            "scratch experiments",
            [int(row["max_scratch_experiments_since_edit"]) for row in task_rows],
            "#b58b52",
        ),
        (
            "test phase tests",
            test_phase_streaks,
            "#c77f4f",
        ),
    ]
    repeat_block_plot_paths = draw_named_series_boxplot(
        repeat_block_series,
        output_dir / "trace_status_repeat_block_thresholds_boxplot",
        title="TRACE_STATUS_REPEAT_BLOCK metrics",
        ylabel="Per-task maximum",
    )

    totals = {
        category: sum(int(row[f"{category}_steps"]) for row in task_rows)
        for category in CATEGORY_ORDER
    }
    p75_by_category = {
        category: percentile(sorted(int(row[f"{category}_steps"]) for row in task_rows), 0.75)
        for category in CATEGORY_ORDER
    }
    success_streak_p75_by_category = {
        category: percentile(sorted(int(row[f"{category}_max_success_streak"]) for row in task_rows), 0.75)
        for category in CATEGORY_ORDER
    }
    sorted_source_inspection_streaks = sorted(source_inspection_streaks)
    sorted_test_phase_streaks = sorted(test_phase_streaks)
    total_inspection_values = [int(row["max_total_inspections_since_edit"]) for row in task_rows]
    scratch_experiment_values = [int(row["max_scratch_experiments_since_edit"]) for row in task_rows]
    repeat_block_p75 = {
        "TRACE_STATUS_REPEAT_BLOCK_MAX_CONSECUTIVE_INSPECTIONS": percentile(
            sorted_source_inspection_streaks, 0.75
        ),
        "TRACE_STATUS_REPEAT_BLOCK_MAX_TOTAL_INSPECTIONS": percentile(
            sorted(total_inspection_values), 0.75
        ),
        "TRACE_STATUS_REPEAT_BLOCK_MAX_SCRATCH_EXPERIMENTS": percentile(
            sorted(scratch_experiment_values), 0.75
        ),
        "TRACE_STATUS_REPEAT_BLOCK_MAX_TEST_PHASE_TESTS": percentile(
            sorted_test_phase_streaks, 0.75
        ),
    }
    repeat_block_config = {
        key: math.ceil(value) for key, value in repeat_block_p75.items()
    }
    summary = {
        "trace_root": str(args.trace_root),
        "traj_glob": args.traj_glob,
        "task_count": len(task_rows),
        "step_count": len(step_rows),
        "category_order": CATEGORY_ORDER,
        "category_totals": totals,
        "category_p75": p75_by_category,
        "category_max_success_streak_p75": success_streak_p75_by_category,
        "source_read_search_max_success_streak_p75": percentile(sorted_source_inspection_streaks, 0.75),
        "source_read_search_max_success_streak_ge_12": sum(value >= 12 for value in source_inspection_streaks),
        "test_phase_streak_p75": percentile(sorted_test_phase_streaks, 0.75),
        "trace_status_repeat_block_p75": repeat_block_p75,
        "trace_status_repeat_block_recommended_config": repeat_block_config,
        "test_phase_streak_ge_12": sum(value >= 12 for value in test_phase_streaks),
        "metric_definitions": {
            "source_read_search_max_success_streak": (
                "Longest consecutive successful source read/search inspection run. "
                "Failures and edit/test/other actions break the streak."
            ),
            "test_phase_streak": (
                "Largest number of test attempts since the previous edit. Failed tests count; "
                "read/search/other actions may be interleaved and do not break the phase."
            ),
            "max_total_inspections_since_edit": (
                "Maximum number of successful source read/search inspections since the previous source edit."
            ),
            "max_scratch_experiments_since_edit": (
                "Maximum number of successful scratch experiment actions since the previous source edit."
            ),
        },
        "classification_source": "scripts/classify_intent.py",
        "outputs": {
            "task_counts_csv": str(task_csv),
            "step_rows_csv": str(step_csv),
            **plot_paths,
            **{
                key.replace("boxplot", "success_streak_boxplot"): value
                for key, value in success_streak_plot_paths.items()
            },
            **{key.replace("boxplot", "phase_streak_boxplot"): value for key, value in phase_streak_plot_paths.items()},
            **{key.replace("boxplot", "repeat_block_boxplot"): value for key, value in repeat_block_plot_paths.items()},
        },
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nCopyable TRACE_STATUS_REPEAT_BLOCK configuration:")
    for key, value in repeat_block_config.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
