#!/usr/bin/env python3

"""Collect per-instance .pred files into evaluation-friendly aggregate outputs."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from minisweagent.utils.log import logger

from .common import prediction_record, write_json_atomic

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def gather_predictions(predictions_dir: str | Path, model_name: str) -> dict[str, dict[str, str]]:
    root = Path(predictions_dir)
    gathered: dict[str, dict[str, str]] = {}
    for pred_path in sorted(root.glob("*/*.pred")):
        instance_id = pred_path.stem
        gathered[instance_id] = prediction_record(instance_id, model_name, pred_path.read_text(encoding="utf-8"))
    return gathered


def predictions_as_list(predictions: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    return [predictions[instance_id] for instance_id in sorted(predictions)]


# fmt: off
@app.command()
def main(
    predictions_dir: Path = typer.Option(..., "--predictions-dir", exists=True, file_okay=False, dir_okay=True, readable=True, help="Directory that contains per-instance subdirectories with .pred files"),
    model_name: str = typer.Option("mini-swe-agent", "--model-name", help="Model name to stamp into aggregated predictions"),
    output: Path = typer.Option(..., "-o", "--output", help="Output JSON file"),
    jsonl_output: Path | None = typer.Option(None, "--jsonl-output", help="Optional JSONL output path"),
) -> None:
    # fmt: on
    predictions = gather_predictions(predictions_dir, model_name)
    write_json_atomic(output, predictions)
    if jsonl_output is not None:
        jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(json.dumps(record) for record in predictions_as_list(predictions)) + "\n"
        jsonl_output.write_text(payload, encoding="utf-8")
    logger.info(f"Gathered {len(predictions)} predictions into {output}")


if __name__ == "__main__":
    app()

