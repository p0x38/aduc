"""Tests for dataset deduplication."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chord_dataset.dedupe import dedupe_file, find_duplicate_prompts, write_deduplicated


def write_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def test_dedupe_finds_exact_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "dataset.jsonl"

    record = {
        "prompt": "Create a C major progression.",
        "output": {
            "metadata": {
                "bpm": 100,
                "signature": [4, 4],
                "scale": "C major",
            },
            "shared": {},
            "progression_data": [
                {"name": "C"},
                {"name": "G"},
            ],
        },
    }

    write_jsonl(source, [record, record])

    result = dedupe_file(source)

    assert result.records == 2
    assert result.unique_records == 1
    assert result.duplicate_records == 1
    assert len(result.groups) == 1
    assert result.groups[0].lines == (1, 2)


def test_dedupe_keeps_unique_records(tmp_path: Path) -> None:
    source = tmp_path / "dataset.jsonl"

    records = [
        {
            "prompt": "Create A.",
            "output": {},
        },
        {
            "prompt": "Create B.",
            "output": {},
        },
    ]

    write_jsonl(source, records)

    result = dedupe_file(source)

    assert result.records == 2
    assert result.unique_records == 2
    assert result.duplicate_records == 0
    assert result.groups == []


def test_dedupe_is_order_independent(tmp_path: Path) -> None:
    source = tmp_path / "dataset.jsonl"

    record = {
        "b": 2,
        "a": 1,
    }

    equivalent = {
        "a": 1,
        "b": 2,
    }

    write_jsonl(source, [record, equivalent])

    result = dedupe_file(source)

    assert result.duplicate_records == 1


def test_dedupe_handles_empty_dataset(tmp_path: Path) -> None:
    source = tmp_path / "dataset.jsonl"
    source.write_text("", encoding="utf-8")

    result = dedupe_file(source)

    assert result.records == 0
    assert result.unique_records == 0
    assert result.duplicate_records == 0
    assert result.groups == []


def test_write_deduplicated_removes_exact_duplicates(
    tmp_path: Path,
) -> None:
    """Writing a deduplicated dataset should remove exact duplicates."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "deduped.jsonl"

    record = {
        "prompt": "Create a progression.",
        "output": {
            "metadata": {
                "bpm": 100,
                "signature": [4, 4],
                "scale": "C major",
            },
            "shared": {},
            "progression_data": [
                {"name": "C"},
                {"name": "G"},
            ],
        },
    }

    write_jsonl(source, [record, record])

    written = write_deduplicated(source, destination)

    assert written == 1

    lines = destination.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0]) == record


def test_find_duplicate_prompts(tmp_path: Path) -> None:
    """Duplicate prompts should be grouped after normalization."""
    source = tmp_path / "dataset.jsonl"

    records = [
        {"prompt": "Create a C major progression.", "output": {}},
        {"prompt": "  CREATE   A C MAJOR PROGRESSION.  ", "output": {}},
        {"prompt": "Create a G major progression.", "output": {}},
    ]

    write_jsonl(source, records)

    groups = find_duplicate_prompts(source)

    assert len(groups) == 1
    assert groups[0].lines == (1, 2)
    assert groups[0].prompt == "Create a C major progression."


def test_find_duplicate_prompts_ignores_empty_prompts(
    tmp_path: Path,
) -> None:
    """Empty prompts should not produce duplicate groups."""
    source = tmp_path / "dataset.jsonl"

    write_jsonl(
        source,
        [
            {"prompt": " ", "output": {}},
            {"prompt": "  ", "output": {}},
        ],
    )

    assert find_duplicate_prompts(source) == []


def test_find_duplicate_prompts_allows_different_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.jsonl"

    write_jsonl(
        source,
        [
            {
                "prompt": "Create a C major progression.",
                "output": {"progression": ["C", "G", "Am", "F"]},
            },
            {
                "prompt": " create   a c major progression. ",
                "output": {"progression": ["C", "Am", "F", "G"]},
            },
        ],
    )

    groups = find_duplicate_prompts(source)

    assert len(groups) == 1
    assert groups[0].lines == (1, 2)
