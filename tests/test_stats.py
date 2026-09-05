"""Tests for dataset statistics."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from chord_dataset.operation import PromptOperation
from chord_dataset.stats import DatasetStats, collect_stats
from chord_dataset.validator import ValidationError


def write_jsonl(
    path: Path,
    records: list[object],
) -> None:
    """Write records to a JSONL file."""
    path.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records),
        encoding="utf-8",
    )


def make_record(
    *,
    bpm: int = 100,
    scale: str = "C major",
    signature: list[int] | None = None,
    prompt: str = "Generate a progression.",
    progression_data: list[Any] | None = None,
    shared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a valid dataset record."""
    if signature is None:
        signature = [4, 4]

    if progression_data is None:
        progression_data = [
            {"name": "C"},
            {"name": "G"},
            {"name": "Am"},
            {"name": "F"},
        ]

    if shared is None:
        shared = {}

    return {
        "prompt": prompt,
        "output": {
            "metadata": {
                "bpm": bpm,
                "signature": signature,
                "scale": scale,
            },
            "shared": shared,
            "progression_data": progression_data,
        },
    }


def test_collect_stats_empty_file(tmp_path: Path) -> None:
    """An empty dataset should return zero statistics."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    result = collect_stats(path)

    assert result == DatasetStats(
        records=0,
        total_chords=0,
        chord_groups=0,
        shared_references=0,
        shared_structures=0,
        average_bpm=0.0,
        min_bpm=0,
        max_bpm=0,
        scales=Counter[str](),
        signatures=Counter[tuple[int, int]](),
        prompt_starters=Counter[str](),
        operations=Counter[PromptOperation](),
    )


def test_collect_stats_counts_basic_dataset(tmp_path: Path) -> None:
    """Basic records should contribute to the expected statistics."""
    path = tmp_path / "dataset.jsonl"

    write_jsonl(
        path,
        [
            make_record(
                bpm=90,
                scale="C major",
                prompt="Generate something.",
            ),
            make_record(
                bpm=110,
                scale="A minor",
                prompt="Create something.",
            ),
        ],
    )

    result = collect_stats(path)

    assert result.records == 2
    assert result.total_chords == 8
    assert result.chord_groups == 0
    assert result.shared_references == 0
    assert result.shared_structures == 0
    assert result.average_bpm == 100
    assert result.min_bpm == 90
    assert result.max_bpm == 110
    assert result.scales["C major"] == 1
    assert result.scales["A minor"] == 1
    assert result.signatures[(4, 4)] == 2
    assert result.prompt_starters["Generate"] == 1
    assert result.prompt_starters["Create"] == 1


def test_collect_stats_counts_chord_groups(tmp_path: Path) -> None:
    """Chord groups should count as groups and contain their chords."""
    path = tmp_path / "dataset.jsonl"

    write_jsonl(
        path,
        [
            make_record(
                progression_data=[
                    {"name": "Cmaj7"},
                    {
                        "chords": [
                            {"name": "Dm7"},
                            {"name": "G7"},
                        ]
                    },
                ],
            )
        ],
    )

    result = collect_stats(path)

    assert result.records == 1
    assert result.chord_groups == 1
    assert result.total_chords == 3


def test_collect_stats_counts_shared_data(tmp_path: Path) -> None:
    """Shared definitions and references should be counted."""
    path = tmp_path / "dataset.jsonl"

    write_jsonl(
        path,
        [
            make_record(
                shared={
                    "main": {
                        "chords": [
                            {"name": "Cmaj7"},
                            {"name": "Am7"},
                        ]
                    },
                    "turnaround": {
                        "chords": [
                            {"name": "Dm7"},
                            {"name": "G7"},
                        ]
                    },
                },
                progression_data=[
                    {"shared": "$main"},
                    {"shared": "$turnaround"},
                ],
            )
        ],
    )

    result = collect_stats(path)

    assert result.shared_structures == 2
    assert result.shared_references == 2
    assert result.total_chords == 0


def test_collect_stats_counts_mixed_progression(tmp_path: Path) -> None:
    """Normal chords, groups, and references should all be handled."""
    path = tmp_path / "dataset.jsonl"

    write_jsonl(
        path,
        [
            make_record(
                shared={
                    "turnaround": {
                        "chords": [
                            {"name": "Dm7"},
                            {"name": "G7"},
                        ]
                    }
                },
                progression_data=[
                    {"name": "Cmaj7"},
                    {
                        "chords": [
                            {"name": "Am7"},
                            {"name": "Dm7"},
                        ]
                    },
                    {"shared": "$turnaround"},
                ],
            )
        ],
    )

    result = collect_stats(path)

    assert result.total_chords == 3
    assert result.chord_groups == 1
    assert result.shared_references == 1
    assert result.shared_structures == 1


def test_collect_stats_counts_signatures(tmp_path: Path) -> None:
    """Different time signatures should be tracked separately."""
    path = tmp_path / "dataset.jsonl"

    write_jsonl(
        path,
        [
            make_record(signature=[4, 4]),
            make_record(signature=[4, 4]),
            make_record(signature=[3, 4]),
            make_record(signature=[6, 8]),
        ],
    )

    result = collect_stats(path)

    assert result.signatures[(4, 4)] == 2
    assert result.signatures[(3, 4)] == 1
    assert result.signatures[(6, 8)] == 1


def test_collect_stats_tracks_prompt_starters(tmp_path: Path) -> None:
    """The first word of each prompt should be counted."""
    path = tmp_path / "dataset.jsonl"

    write_jsonl(
        path,
        [
            make_record(prompt="Generate a progression."),
            make_record(prompt="Generate another progression."),
            make_record(prompt="Create a progression."),
            make_record(prompt="Reharmonize this progression."),
        ],
    )

    result = collect_stats(path)

    assert result.prompt_starters["Generate"] == 2
    assert result.prompt_starters["Create"] == 1
    assert result.prompt_starters["Reharmonize"] == 1


def test_collect_stats_rejects_missing_shared_reference(
    tmp_path: Path,
) -> None:
    """Invalid shared references should fail statistics collection."""
    path = tmp_path / "dataset.jsonl"

    write_jsonl(
        path,
        [
            make_record(
                progression_data=[
                    {"shared": "$missing"},
                ],
            )
        ],
    )

    with pytest.raises(
        ValidationError,
        match=r"missing shared structure '\$missing'",
    ):
        collect_stats(path)


def test_collect_stats_rejects_invalid_record(
    tmp_path: Path,
) -> None:
    """Invalid records should fail statistics collection."""
    path = tmp_path / "dataset.jsonl"

    record = make_record()
    del record["output"]["metadata"]["scale"]

    write_jsonl(path, [record])

    with pytest.raises(
        ValidationError,
        match=r"line 1: invalid dataset record",
    ):
        collect_stats(path)


def test_collect_stats_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON should fail statistics collection."""
    path = tmp_path / "dataset.jsonl"

    write_jsonl(path, [make_record()])
    with path.open("a", encoding="utf-8") as file:
        file.write('{"prompt":"broken"\n')

    with pytest.raises(
        ValidationError,
        match=r"line 2: invalid JSON",
    ):
        collect_stats(path)
