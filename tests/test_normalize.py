"""Tests for dataset normalization."""

import json
from pathlib import Path
from typing import Any

import pytest

from chord_dataset.models import Chord, ChordGroup, DatasetRecord
from chord_dataset.normalize import (
    NormalizationResult,
    normalize_file,
    normalize_record,
)
from chord_dataset.validator import ValidationError


def make_record(
    *,
    scale: str = "C major",
    progression_data: list[dict[str, Any]] | None = None,
    shared: dict[str, Any] | None = None,
) -> DatasetRecord:
    """Create a test record."""
    if progression_data is None:
        progression_data = [
            {"name": "Cmaj7"},
            {"name": "G7"},
        ]

    if shared is None:
        shared = {}

    return DatasetRecord.model_validate(
        {
            "prompt": "Generate a progression.",
            "output": {
                "metadata": {
                    "bpm": 100,
                    "signature": [4, 4],
                    "scale": scale,
                },
                "shared": shared,
                "progression_data": progression_data,
            },
        }
    )


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write records to JSONL."""
    path.write_text(
        "".join(
            f"{json.dumps(record, ensure_ascii=False)}\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_normalize_record_preserves_valid_data() -> None:
    """Already-normalized records should remain unchanged."""
    record = make_record()

    normalized = normalize_record(record)

    assert normalized == record


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("A minor", "A natural minor"),
        ("C minor", "C natural minor"),
        ("F# minor", "F# natural minor"),
        ("  C major  ", "C major"),
    ],
)
def test_normalize_record_normalizes_scale(
    source: str,
    expected: str,
) -> None:
    """Common scale aliases should be normalized."""
    record = make_record(scale=source)

    normalized = normalize_record(record)

    assert normalized.output.metadata.scale == expected


def test_normalize_record_normalizes_chord_whitespace() -> None:
    """Chord names should have normalized whitespace."""
    record = make_record(
        progression_data=[
            {"name": " Cmaj7 "},
            {"name": "Db  sus4"},
        ]
    )

    normalized = normalize_record(record)

    first = normalized.output.progression_data[0]
    second = normalized.output.progression_data[1]

    assert isinstance(first, Chord)
    assert isinstance(second, Chord)

    assert first.name == "Cmaj7"
    assert second.name == "Db sus4"


def test_normalize_record_normalizes_group_chords() -> None:
    """Chord groups should have normalized chord names."""
    record = make_record(
        progression_data=[
            {
                "chords": [
                    {"name": " Gm7 "},
                    {"name": " C7  "},
                ]
            }
        ]
    )

    normalized = normalize_record(record)

    group = normalized.output.progression_data[0]

    assert isinstance(group, ChordGroup)
    assert [chord.name for chord in group.chords] == ["Gm7", "C7"]


def test_normalize_record_normalizes_shared_data() -> None:
    """Shared chord structures should also be normalized."""
    record = make_record(
        shared={
            "main": {
                "chords": [
                    {"name": " Am7 "},
                    {"name": " F  maj7 "},
                ]
            }
        },
        progression_data=[
            {"shared": "$main"},
        ],
    )

    normalized = normalize_record(record)

    assert normalized.output.shared["main"].chords[0].name == "Am7"
    assert normalized.output.shared["main"].chords[1].name == "F maj7"


def test_normalize_file_writes_output(tmp_path: Path) -> None:
    """normalize_file should write a normalized JSONL file."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "normalized.jsonl"

    record = make_record(scale="A minor")

    write_jsonl(
        source,
        [record.model_dump(mode="json")],
    )

    result = normalize_file(source, destination)

    assert result == NormalizationResult(
        records=1,
        changed_records=1,
    )
    assert destination.is_file()


def test_normalize_file_preserves_unchanged_records(
    tmp_path: Path,
) -> None:
    """Unchanged records should not be counted as changed."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "normalized.jsonl"

    record = make_record()

    write_jsonl(
        source,
        [record.model_dump(mode="json")],
    )

    result = normalize_file(source, destination)

    assert result.records == 1
    assert result.changed_records == 0


def test_normalize_file_creates_parent_directory(
    tmp_path: Path,
) -> None:
    """Missing destination directories should be created."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "nested" / "output.jsonl"

    record = make_record()

    write_jsonl(
        source,
        [record.model_dump(mode="json")],
    )

    normalize_file(source, destination)

    assert destination.is_file()


def test_normalize_file_preserves_unicode(
    tmp_path: Path,
) -> None:
    """Unicode prompts should survive normalization."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "normalized.jsonl"

    record = make_record()
    data = record.model_dump(mode="json")
    data["prompt"] = "王道のコード進行を作ってください。"

    write_jsonl(source, [data])

    normalize_file(source, destination)

    exported = json.loads(
        destination.read_text(encoding="utf-8").splitlines()[0]
    )

    assert exported["prompt"] == "王道のコード進行を作ってください。"


def test_normalize_file_rejects_invalid_record(
    tmp_path: Path,
) -> None:
    """Invalid records should fail normalization."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "normalized.jsonl"

    record = make_record().model_dump(mode="json")
    del record["output"]["metadata"]["scale"]

    write_jsonl(source, [record])

    with pytest.raises(
        ValidationError,
        match=r"line 1: invalid dataset record",
    ):
        normalize_file(source, destination)


def test_normalize_file_rejects_missing_shared_reference(
    tmp_path: Path,
) -> None:
    """Broken shared references should fail normalization."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "normalized.jsonl"

    record = make_record(
        progression_data=[
            {"shared": "$missing"},
        ]
    )

    write_jsonl(
        source,
        [record.model_dump(mode="json")],
    )

    with pytest.raises(
        ValidationError,
        match=r"missing shared structure '\$missing'",
    ):
        normalize_file(source, destination)


def test_normalize_file_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON should fail normalization."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "normalized.jsonl"

    record = make_record()

    write_jsonl(
        source,
        [record.model_dump(mode="json")],
    )

    with source.open("a", encoding="utf-8") as file:
        file.write('{"prompt":"broken"\n')

    with pytest.raises(
        ValidationError,
        match=r"line 2: invalid JSON",
    ):
        normalize_file(source, destination)
