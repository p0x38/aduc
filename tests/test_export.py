"""Tests for dataset export."""

import json
from pathlib import Path
from typing import Any

import pytest

from chord_dataset.export import export_jsonl
from chord_dataset.validator import ValidationError


def make_record(index: int = 0) -> dict[str, Any]:
    """Create a valid dataset record."""
    return {
        "prompt": f"Generate progression {index}.",
        "output": {
            "metadata": {
                "bpm": 100 + index,
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into Python objects."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_export_jsonl_creates_destination(tmp_path: Path) -> None:
    """Export should create the destination file."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "export.jsonl"

    write_jsonl(source, [make_record(0)])

    exported = export_jsonl(source, destination)

    assert exported == 1
    assert destination.is_file()


def test_export_jsonl_creates_parent_directory(
    tmp_path: Path,
) -> None:
    """Export should create missing parent directories."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "nested" / "data" / "export.jsonl"

    write_jsonl(source, [make_record(0)])

    exported = export_jsonl(source, destination)

    assert exported == 1
    assert destination.is_file()


def test_export_jsonl_preserves_data(tmp_path: Path) -> None:
    """Export should preserve the dataset semantics."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "export.jsonl"

    records = [
        make_record(0),
        make_record(1),
        make_record(2),
    ]

    write_jsonl(source, records)

    exported = export_jsonl(source, destination)

    assert exported == 3
    assert read_jsonl(destination) == records


def test_export_jsonl_handles_shared_references(
    tmp_path: Path,
) -> None:
    """Valid shared references should be exported."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "export.jsonl"

    record = make_record()

    output = record["output"]
    output["shared"] = {
        "turnaround": {
            "chords": [
                {"name": "Dm7"},
                {"name": "G7"},
            ],
        },
    }
    output["progression_data"] = [
        {"name": "Cmaj7"},
        {"shared": "$turnaround"},
    ]

    write_jsonl(source, [record])

    exported = export_jsonl(source, destination)

    assert exported == 1
    assert read_jsonl(destination) == [record]


def test_export_jsonl_rejects_invalid_record(
    tmp_path: Path,
) -> None:
    """Invalid dataset records should fail export."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "export.jsonl"

    record = make_record()
    del record["output"]["metadata"]["scale"]

    write_jsonl(source, [record])

    with pytest.raises(
        ValidationError,
        match=r"line 1: invalid dataset record",
    ):
        export_jsonl(source, destination)


def test_export_jsonl_rejects_invalid_shared_reference(
    tmp_path: Path,
) -> None:
    """Missing shared references should fail export."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "export.jsonl"

    record = make_record()
    record["output"]["progression_data"] = [
        {"shared": "$missing"},
    ]

    write_jsonl(source, [record])

    with pytest.raises(
        ValidationError,
        match=r"missing shared structure '\$missing'",
    ):
        export_jsonl(source, destination)


def test_export_jsonl_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON should fail export."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "export.jsonl"

    write_jsonl(source, [make_record(0)])

    with source.open("a", encoding="utf-8") as file:
        file.write('{"prompt":"broken"\n')

    with pytest.raises(
        ValidationError,
        match=r"line 2: invalid JSON",
    ):
        export_jsonl(source, destination)


def test_export_jsonl_preserves_unicode(tmp_path: Path) -> None:
    """Unicode prompt text should survive export."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "export.jsonl"

    record = make_record()
    record["prompt"] = "王道のコード進行を作ってください。"

    write_jsonl(source, [record])

    export_jsonl(source, destination)

    exported = read_jsonl(destination)

    assert exported[0]["prompt"] == "王道のコード進行を作ってください。"


def test_export_jsonl_uses_one_record_per_line(
    tmp_path: Path,
) -> None:
    """Every exported record should occupy exactly one line."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "export.jsonl"

    write_jsonl(
        source,
        [
            make_record(0),
            make_record(1),
            make_record(2),
        ],
    )

    export_jsonl(source, destination)

    lines = destination.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3
    assert all(line.startswith("{") for line in lines)
    assert all(line.endswith("}") for line in lines)


def test_export_jsonl_is_deterministic(tmp_path: Path) -> None:
    """Exporting the same data twice should produce identical output."""
    source = tmp_path / "source.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_jsonl(
        source,
        [
            make_record(0),
            make_record(1),
        ],
    )

    export_jsonl(source, first)
    export_jsonl(source, second)

    assert first.read_bytes() == second.read_bytes()


def test_export_jsonl_empty_dataset(tmp_path: Path) -> None:
    """An empty dataset should export successfully."""
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "export.jsonl"

    source.write_text("", encoding="utf-8")

    exported = export_jsonl(source, destination)

    assert exported == 0
    assert destination.is_file()
    assert destination.read_text(encoding="utf-8") == ""
