"""Tests for dataset conversion."""

import json
from pathlib import Path
from typing import Any

import pytest

from chord_dataset.convert import (
    ConversionResult,
    convert_jsonl,
    convert_record,
)
from chord_dataset.formats import OutputFormat
from chord_dataset.models import DatasetRecord
from chord_dataset.validator import ValidationError


def make_record() -> DatasetRecord:
    """Create a valid test record."""
    return DatasetRecord.model_validate({
        "prompt": "Create a progression in C major.",
        "output": {
            "metadata": {
                "bpm": 97,
                "signature": [4, 4],
                "scale": "C major",
            },
            "shared": {
                "turnaround": {
                    "chords": [
                        {"name": "Dm7"},
                        {"name": "G7"},
                    ],
                },
            },
            "progression_data": [
                {"name": "Cmaj7"},
                {"shared": "$turnaround"},
            ],
        },
    })


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write records to JSONL."""
    path.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records),
        encoding="utf-8",
    )


def test_convert_record_preserves_prompt() -> None:
    """The original prompt should be preserved."""
    record = make_record()

    converted = convert_record(record)

    assert converted["prompt"] == record.prompt


def test_convert_record_uses_response_field() -> None:
    """The structured output should become the response."""
    record = make_record()

    converted = convert_record(record)

    assert "response" in converted
    assert "output" not in converted


def test_convert_record_preserves_metadata() -> None:
    """Metadata should survive conversion."""
    record = make_record()

    converted = convert_record(record)

    response = converted["response"]

    assert response["metadata"]["bpm"] == 97
    assert response["metadata"]["signature"] == [4, 4]
    assert response["metadata"]["scale"] == "C major"


def test_convert_record_preserves_shared_data() -> None:
    """Shared structures should survive conversion."""
    record = make_record()

    converted = convert_record(record)

    shared = converted["response"]["shared"]

    assert shared["turnaround"]["chords"] == [
        {"name": "Dm7"},
        {"name": "G7"},
    ]


def test_convert_record_preserves_shared_reference() -> None:
    """Shared references should survive conversion."""
    record = make_record()

    converted = convert_record(record)

    progression = converted["response"]["progression_data"]

    assert progression[1] == {"shared": "$turnaround"}


def test_convert_jsonl_creates_destination(
    tmp_path: Path,
) -> None:
    """Conversion should create the destination file."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "training.jsonl"

    record = make_record()

    write_jsonl(
        source,
        [record.model_dump(mode="json")],
    )

    result = convert_jsonl(source, destination)

    assert result == ConversionResult(
        records=1,
        output_format=OutputFormat.STRUCTURED,
    )
    assert destination.is_file()


def test_convert_jsonl_supports_completion_format(
    tmp_path: Path,
) -> None:
    """Conversion should support the completion format."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "completion.jsonl"

    write_jsonl(
        source,
        [make_record().model_dump(mode="json")],
    )

    result = convert_jsonl(
        source,
        destination,
        output_format=OutputFormat.COMPLETION,
    )

    assert result.output_format is OutputFormat.COMPLETION

    converted = json.loads(
        destination.read_text(encoding="utf-8").splitlines()[0]
    )

    assert set(converted) == {"prompt", "completion"}
    assert isinstance(converted["completion"], str)


def test_convert_jsonl_supports_chat_format(
    tmp_path: Path,
) -> None:
    """Conversion should support the chat format."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "chat.jsonl"

    write_jsonl(
        source,
        [make_record().model_dump(mode="json")],
    )

    result = convert_jsonl(
        source,
        destination,
        output_format=OutputFormat.CHAT,
    )

    assert result.output_format is OutputFormat.CHAT

    converted = json.loads(
        destination.read_text(encoding="utf-8").splitlines()[0]
    )

    assert list(converted) == ["messages"]
    assert len(converted["messages"]) == 3


def test_convert_jsonl_creates_parent_directory(
    tmp_path: Path,
) -> None:
    """Conversion should create missing parent directories."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "nested" / "training.jsonl"

    record = make_record()

    write_jsonl(
        source,
        [record.model_dump(mode="json")],
    )

    result = convert_jsonl(source, destination)

    assert result.records == 1
    assert destination.is_file()


def test_convert_jsonl_preserves_all_records(
    tmp_path: Path,
) -> None:
    """Every input record should produce one output record."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "training.jsonl"

    records = [
        make_record().model_dump(mode="json"),
        make_record().model_dump(mode="json"),
        make_record().model_dump(mode="json"),
    ]

    write_jsonl(source, records)

    result = convert_jsonl(source, destination)

    assert result.records == 3

    lines = destination.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3


def test_convert_jsonl_is_valid_jsonl(
    tmp_path: Path,
) -> None:
    """Every converted line should be valid JSON."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "training.jsonl"

    write_jsonl(
        source,
        [make_record().model_dump(mode="json")],
    )

    convert_jsonl(source, destination)

    lines = destination.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    value = json.loads(lines[0])

    assert isinstance(value, dict)
    assert "prompt" in value
    assert "response" in value


def test_convert_jsonl_preserves_unicode(
    tmp_path: Path,
) -> None:
    """Unicode prompts should survive conversion."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "training.jsonl"

    record = make_record().model_dump(mode="json")
    record["prompt"] = "王道のコード進行を作ってください。"

    write_jsonl(source, [record])

    convert_jsonl(source, destination)

    converted = json.loads(destination.read_text(encoding="utf-8").splitlines()[0])

    assert converted["prompt"] == "王道のコード進行を作ってください。"


def test_convert_jsonl_rejects_invalid_record(
    tmp_path: Path,
) -> None:
    """Invalid records should fail conversion."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "training.jsonl"

    record = make_record().model_dump(mode="json")
    del record["output"]["metadata"]["scale"]

    write_jsonl(source, [record])

    with pytest.raises(
        ValidationError,
        match=r"line 1: invalid dataset record",
    ):
        convert_jsonl(source, destination)


def test_convert_jsonl_rejects_missing_shared_reference(
    tmp_path: Path,
) -> None:
    """Missing shared references should fail conversion."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "training.jsonl"

    record = make_record().model_dump(mode="json")
    record["output"]["progression_data"] = [
        {"shared": "$missing"},
    ]

    write_jsonl(source, [record])

    with pytest.raises(
        ValidationError,
        match=r"missing shared structure '\$missing'",
    ):
        convert_jsonl(source, destination)


def test_convert_jsonl_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON should fail conversion."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "training.jsonl"

    write_jsonl(
        source,
        [make_record().model_dump(mode="json")],
    )

    with source.open("a", encoding="utf-8") as file:
        file.write('{"prompt":"broken"\n')

    with pytest.raises(
        ValidationError,
        match=r"line 2: invalid JSON",
    ):
        convert_jsonl(source, destination)


def test_convert_jsonl_is_deterministic(
    tmp_path: Path,
) -> None:
    """Converting the same source twice should produce identical output."""
    source = tmp_path / "source.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_jsonl(
        source,
        [
            make_record().model_dump(mode="json"),
            make_record().model_dump(mode="json"),
        ],
    )

    convert_jsonl(source, first)
    convert_jsonl(source, second)

    assert first.read_bytes() == second.read_bytes()


def test_convert_empty_dataset(
    tmp_path: Path,
) -> None:
    """An empty dataset should convert successfully."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "training.jsonl"

    source.write_text("", encoding="utf-8")

    result = convert_jsonl(source, destination)

    assert result.records == 0
    assert destination.is_file()
    assert destination.read_text(encoding="utf-8") == ""
