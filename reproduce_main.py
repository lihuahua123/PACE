#!/usr/bin/env python3
"""Recompute the archived no-control and PACE rows from bundled task traces."""
from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / 'scripts/compare_trace_costs_common50_important.py'
spec = importlib.util.spec_from_file_location('pace_costs', MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

GROUPS = {
    'MiniMax no control': ['minimax_nocontrol_1', 'minimax_nocontrol_2', 'minimax_nocontrol_3'],
    'MiniMax PACE': ['minimax_pace_1', 'minimax_pace_2', 'minimax_pace_3'],
    'luna no control': ['luna_nocontrol_1', 'luna_nocontrol_2', 'luna_nocontrol_3'],
    'luna PACE': ['luna_pace_2', 'luna_pace_3'],
    'DeepSeek no control': ['deepseek_nocontrol'],
    'DeepSeek PACE': ['deepseek_pace'],
}


def summarize_run(label: str, cohort: str = 'traces') -> tuple[int, float, float, int]:
    run_dir = ROOT / cohort / label
    tasks = sorted(p for p in run_dir.iterdir() if p.is_dir())
    assert len(tasks) == 50, f'{label}: expected 50 tasks, found {len(tasks)}'
    rows = [module.extract_task_cost(p) for p in tasks]
    assert all(row.get('success') is not None for row in rows), label
    return (
        sum(bool(row['success']) for row in rows),
        sum(row['estimated_total_cost_rmb'] for row in rows),
        sum(row['prompt_tokens'] + row['completion_tokens'] for row in rows) / 1_000_000,
        sum(row['api_calls'] for row in rows),
    )


def print_group(group: str, runs: list[tuple[int, float, float, int]]) -> None:
    print(group)
    print('  successes:', ', '.join(f'{r[0]}/50' for r in runs))
    for name, values, unit, digits in [
        ('SR', [r[0] * 2 for r in runs], '%', 1),
        ('cost', [r[1] for r in runs], ' CNY', 3),
        ('tokens', [r[2] for r in runs], 'M', 3),
        ('API calls', [r[3] for r in runs], '', 1),
    ]:
        mean = statistics.mean(values)
        spread = f' ± {statistics.stdev(values):.{digits}f}' if len(values) > 1 else ''
        print(f'  {name}: {mean:.{digits}f}{spread}{unit}')


def main() -> None:
    for group, labels in GROUPS.items():
        print_group(group, [summarize_run(label) for label in labels])
    labels = GROUPS['MiniMax PACE']
    print_group('MiniMax PACE original independent runs',
                [summarize_run(label, 'traces_original') for label in labels])


if __name__ == '__main__':
    main()
