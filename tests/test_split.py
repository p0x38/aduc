"""Tests for dataset splitting."""

from pathlib import Path
from typing import Any

import pytest

from chord_dataset.split import (
    DatasetSplit,
    load_and_split,
    split_records,
    write_split,
)


def make_records(count: int) -> list[dict[str, Any]]:
    """Create uniquely identifiable test records."""
    return [
        {
            "prompt": f"Generate progression {index}.",
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
        for index in range(count)
    ]


def record_ids(records: list[dict[str, Any]]) -> set[str]:
    """Return the prompt identifiers for a set of records."""
    return {str(record["prompt"]) for record in records}


def test_split_records_returns_dataset_split() -> None:
    """Splitting records should return a DatasetSplit."""
    records = make_records(10)

    result = split_records(records)

    assert isinstance(result, DatasetSplit)
    assert result.total == 10


def test_split_records_default_counts() -> None:
    """The default split should use an 80/10/10 distribution."""
    records = make_records(100)

    result = split_records(records, seed=42)

    assert len(result.train) == 80
    assert len(result.validation) == 10
    assert len(result.test) == 10
    assert result.total == 100


def test_split_records_preserves_all_records() -> None:
    """Splitting must not lose or duplicate records."""
    records = make_records(217)

    result = split_records(records, seed=42)

    original = record_ids(records)
    combined = (
        record_ids(result.train)
        | record_ids(result.validation)
        | record_ids(result.test)
    )

    assert combined == original
    assert (
        len(result.train)
        + len(result.validation)
        + len(result.test)
        == len(records)
    )


def test_split_records_has_no_overlap() -> None:
    """Train, validation, and test sets must not overlap."""
    records = make_records(100)

    result = split_records(records, seed=42)

    train = record_ids(result.train)
    validation = record_ids(result.validation)
    test = record_ids(result.test)

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_split_records_is_deterministic() -> None:
    """The same seed must produce the same split."""
    records = make_records(100)

    first = split_records(records, seed=123)
    second = split_records(records, seed=123)

    assert first == second


def test_split_records_changes_with_seed() -> None:
    """Different seeds should normally produce different splits."""
    records = make_records(100)

    first = split_records(records, seed=123)
    second = split_records(records, seed=456)

    assert first != second


def test_split_records_does_not_modify_input() -> None:
    """Splitting should not reorder or modify the input list."""
    records = make_records(20)
    original = list(records)

    split_records(records, seed=42)

    assert records == original


def test_split_records_accepts_custom_ratios() -> None:
    """Custom train and validation ratios should be respected."""
    records = make_records(100)

    result = split_records(
        records,
        train_ratio=0.7,
        validation_ratio=0.2,
        seed=42,
    )

    assert len(result.train) == 70
    assert len(result.validation) == 20
    assert len(result.test) == 10


@pytest.mark.parametrize(
    ("train_ratio", "validation_ratio"),
    [
        (-0.1, 0.1),
        (0.8, -0.1),
        (0.8, 0.3),
        (1.1, 0.0),
        (0.0, 1.1),
    ],
)
def test_split_records_rejects_invalid_ratios(
    train_ratio: float,
    validation_ratio: float,
) -> None:
    """Invalid split ratios should raise ValueError."""
    with pytest.raises(ValueError):
        split_records(
            make_records(10),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )


def test_split_records_accepts_zero_test_ratio() -> None:
    """A zero test ratio should be allowed."""
    result = split_records(
        make_records(10),
        train_ratio=0.8,
        validation_ratio=0.2,
    )

    assert len(result.train) == 8
    assert len(result.validation) == 2
    assert len(result.test) == 0


def test_split_records_accepts_zero_validation_ratio() -> None:
    """A zero validation ratio should be allowed."""
    result = split_records(
        make_records(10),
        train_ratio=0.8,
        validation_ratio=0.0,
    )

    assert len(result.train) == 8
    assert len(result.validation) == 0
    assert len(result.test) == 2


def test_split_records_handles_empty_input() -> None:
    """An empty input should produce three empty splits."""
    result = split_records([])

    assert result.train == []
    assert result.validation == []
    assert result.test == []
    assert result.total == 0


def test_split_records_handles_single_record() -> None:
    """A single record should still be assigned to one split."""
    result = split_records(make_records(1), seed=42)

    assert result.total == 1
    assert len(result.train) + len(result.validation) + len(result.test) == 1


def test_load_and_split_reads_jsonl(tmp_path: Path) -> None:
    """load_and_split should load and split a JSONL file."""
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "\n".join(
            [
                (
                    '{"prompt":"one","output":{"metadata":'
                    '{"bpm":100,"signature":[4,4],"scale":"C major"},'
                    '"shared":{},"progression_data":[{"name":"C"}]}}'
                ),
                (
                    '{"prompt":"two","output":{"metadata":'
                    '{"bpm":100,"signature":[4,4],"scale":"C major"},'
                    '"shared":{},"progression_data":[{"name":"G"}]}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_and_split(path, seed=42)

    assert result.total == 2


def test_write_split_creates_output_files(tmp_path: Path) -> None:
    """write_split should create all three JSONL files."""
    split = DatasetSplit(
        train=make_records(2),
        validation=make_records(1),
        test=make_records(1),
    )

    output_dir = tmp_path / "split"

    write_split(split, output_dir)

    assert (output_dir / "train.jsonl").is_file()
    assert (output_dir / "validation.jsonl").is_file()
    assert (output_dir / "test.jsonl").is_file()


def test_write_split_creates_output_directory(tmp_path: Path) -> None:
    """write_split should create a missing output directory."""
    split = DatasetSplit(
        train=make_records(1),
        validation=[],
        test=[],
    )

    output_dir = tmp_path / "nested" / "split"

    write_split(split, output_dir)

    assert output_dir.is_dir()


def test_write_split_writes_valid_jsonl(tmp_path: Path) -> None:
    """Written split files should contain one JSON object per line."""
    split = DatasetSplit(
        train=make_records(2),
        validation=make_records(1),
        test=make_records(1),
    )

    output_dir = tmp_path / "split"
    write_split(split, output_dir)

    train_lines = (
        output_dir / "train.jsonl"
    ).read_text(encoding="utf-8").splitlines()

    assert len(train_lines) == 2
    assert all(line.startswith("{") for line in train_lines)
    assert all(line.endswith("}") for line in train_lines)
