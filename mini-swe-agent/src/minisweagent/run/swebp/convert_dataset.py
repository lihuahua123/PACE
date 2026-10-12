#!/usr/bin/env python3

"""Convert SWE-bench Pro style dataset rows into generic batch instances."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import typer

from minisweagent.utils.log import logger

from .batch_instances import BatchInstance, dump_batch_instances

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def convert_record_to_batch_instance(
    record: dict[str, Any],
    *,
    image_repo: str = "jefzda/sweap-images",
    repo_name: str = "app",
    image_tag_field: str = "dockerhub_tag",
    image_name_field: str = "image_name",
) -> BatchInstance:
    image_name = record.get(image_name_field)
    if not image_name:
        dockerhub_tag = record.get(image_tag_field)
        if not dockerhub_tag:
            raise KeyError(
                f"Record {record.get('instance_id', '<unknown>')} is missing '{image_name_field}' and '{image_tag_field}'"
            )
        image_name = f"{image_repo}:{dockerhub_tag}"

    known_fields = {
        "instance_id",
        "problem_statement",
        "base_commit",
        image_tag_field,
        image_name_field,
        "repo_name",
    }
    extra_fields = {key: value for key, value in record.items() if key not in known_fields}
    if image_tag_field in record:
        extra_fields[image_tag_field] = record[image_tag_field]

    return BatchInstance(
        instance_id=str(record["instance_id"]),
        problem_statement=str(record["problem_statement"]),
        base_commit=str(record["base_commit"]),
        image_name=str(image_name),
        repo_name=str(record.get("repo_name") or repo_name),
        extra_fields=extra_fields,
    )


def filter_records(
    records: list[dict[str, Any]], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict[str, Any]]:
    if shuffle:
        records = sorted(records.copy(), key=lambda record: str(record["instance_id"]))
        random.seed(42)
        random.shuffle(records)
    if filter_spec:
        records = [record for record in records if re.match(filter_spec, str(record["instance_id"]))]
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        records = records[slice(*values)]
    return records


def load_dataset_records(dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for swebp-convert. Install the dev dependencies first.") from exc
    return [dict(record) for record in load_dataset(dataset_name, split=split)]


# fmt: off
@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", help="Hugging Face dataset name or local dataset path"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    output: Path = typer.Option(..., "-o", "--output", help="Output YAML path"),
    image_repo: str = typer.Option("jefzda/sweap-images", "--image-repo", help="Docker repository used to build image_name"),
    repo_name: str = typer.Option("app", "--repo-name", help="Fallback repo_name when the dataset does not provide one"),
    image_tag_field: str = typer.Option("dockerhub_tag", "--image-tag-field", help="Field containing the docker tag"),
    image_name_field: str = typer.Option("image_name", "--image-name-field", help="Field containing a fully-qualified image name"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g. '0:5')"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle records before filtering"),
) -> None:
    # fmt: on
    records = load_dataset_records(dataset, split)
    records = filter_records(records, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    instances = [
        convert_record_to_batch_instance(
            record,
            image_repo=image_repo,
            repo_name=repo_name,
            image_tag_field=image_tag_field,
            image_name_field=image_name_field,
        )
        for record in records
    ]
    dump_batch_instances(instances, output)
    logger.info(f"Wrote {len(instances)} batch instances to {output}")


if __name__ == "__main__":
    app()

