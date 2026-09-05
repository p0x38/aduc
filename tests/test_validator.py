"""Tests for dataset validation."""

import json
from pathlib import Path

import pytest

from chord_dataset.validator import (
    ValidationError,
    ValidationResult,
    validate_file,
    validate_shared_references,
)

VALID_RECORD = {
    "prompt": "Generate a progression in C major.",
    "output": {
        "metadata": {
            "bpm": 97,
            "signature": [4, 4],
            "scale": "C major",
        },
        "shared": {},
        "progression_data": [
            {"name": "C"},
            {"name": "G"},
            {"name": "Am"},
            {"name": "F"},
        ],
    },
}


def write_jsonl(
    path: Path,
    records: list[object],
) -> None:
    """Write records to a JSONL file."""
    path.write_text(
        "".join(
            f"{json.dumps(record, ensure_ascii=False)}\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_validate_file_accepts_valid_dataset(tmp_path: Path) -> None:
    """A valid JSONL dataset should pass validation."""
    path = tmp_path / "dataset.jsonl"

    write_jsonl(
        path,
        [
            VALID_RECORD,
            VALID_RECORD,
        ],
    )

    result = validate_file(path)

    assert result == ValidationResult(
        total_lines=2,
        valid_records=2,
    )


def test_validate_file_accepts_blank_lines(tmp_path: Path) -> None:
    """Blank lines should be ignored by validation."""
    path = tmp_path / "dataset.jsonl"

    path.write_text(
        json.dumps(VALID_RECORD) + "\n"
        "\n"
        "   \n"
        + json.dumps(VALID_RECORD) + "\n",
        encoding="utf-8",
    )

    result = validate_file(path)

    assert result.total_lines == 2
    assert result.valid_records == 2


def test_validate_file_rejects_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON should fail validation."""
    path = tmp_path / "dataset.jsonl"

    path.write_text(
        json.dumps(VALID_RECORD) + "\n"
        '{"prompt":"broken"\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match=r"line 2: invalid JSON",
    ):
        validate_file(path)


def test_validate_file_rejects_missing_scale(tmp_path: Path) -> None:
    """Records without scale metadata should fail validation."""
    path = tmp_path / "dataset.jsonl"

    record = json.loads(json.dumps(VALID_RECORD))
    del record["output"]["metadata"]["scale"]

    write_jsonl(path, [record])

    with pytest.raises(
        ValidationError,
        match=r"line 1: output\.metadata\.scale",
    ):
        validate_file(path)


def test_validate_file_rejects_invalid_bpm(tmp_path: Path) -> None:
    """Records with an invalid BPM should fail validation."""
    path = tmp_path / "dataset.jsonl"

    record = json.loads(json.dumps(VALID_RECORD))
    record["output"]["metadata"]["bpm"] = 0

    write_jsonl(path, [record])

    with pytest.raises(
        ValidationError,
        match=r"line 1: output\.metadata\.bpm",
    ):
        validate_file(path)


def test_validate_file_rejects_invalid_shared_reference(
    tmp_path: Path,
) -> None:
    """A reference to an unknown shared structure should fail."""
    path = tmp_path / "dataset.jsonl"

    record = json.loads(json.dumps(VALID_RECORD))
    record["output"]["progression_data"] = [
        {"shared": "$missing"},
    ]

    write_jsonl(path, [record])

    with pytest.raises(
        ValidationError,
        match=r"missing shared structure '\$missing'",
    ):
        validate_file(path)


def test_validate_file_accepts_existing_shared_reference(
    tmp_path: Path,
) -> None:
    """A reference to an existing shared structure should pass."""
    path = tmp_path / "dataset.jsonl"

    record = json.loads(json.dumps(VALID_RECORD))
    record["output"]["shared"] = {
        "turnaround": {
            "chords": [
                {"name": "Dm7"},
                {"name": "G7"},
            ],
        },
    }
    record["output"]["progression_data"] = [
        {"name": "Cmaj7"},
        {"shared": "$turnaround"},
    ]

    write_jsonl(path, [record])

    result = validate_file(path)

    assert result.valid_records == 1


def test_validate_shared_references_ignores_normal_chords() -> None:
    """Normal chords should not be treated as references."""
    from chord_dataset.models import DatasetRecord

    record = DatasetRecord.model_validate(VALID_RECORD)

    validate_shared_references(1, record)


def test_validate_shared_references_reports_index() -> None:
    """Missing references should identify the progression index."""
    from chord_dataset.models import DatasetRecord

    record_data = json.loads(json.dumps(VALID_RECORD))
    record_data["output"]["progression_data"] = [
        {"name": "C"},
        {"shared": "$missing"},
    ]

    record = DatasetRecord.model_validate(record_data)

    with pytest.raises(
        ValidationError,
        match=r"progression_data\[1\]",
    ):
        validate_shared_references(7, record)


def test_validate_file_result_is_frozen(tmp_path: Path) -> None:
    """ValidationResult should be immutable."""
    path = tmp_path / "dataset.jsonl"

    write_jsonl(path, [VALID_RECORD])

    result = validate_file(path)

    with pytest.raises(AttributeError):
        result.valid_records = 999  # type: ignore[misc]