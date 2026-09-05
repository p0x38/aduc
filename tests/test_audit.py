"""Tests for dataset auditing."""

import json
from pathlib import Path

from chord_dataset.audit import audit_file


VALID_RECORD = {
    "prompt": "Generate a progression in C major.",
    "output": {
        "metadata": {"bpm": 100, "signature": [4, 4], "scale": "C major"},
        "shared": {},
        "progression_data": [{"name": "C"}, {"name": "G"}],
    },
}


def write_jsonl(path: Path, records: list[object]) -> None:
    """Write records to a JSONL file."""
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_audit_file_passes_clean_dataset(tmp_path: Path) -> None:
    """A clean dataset should pass the audit."""
    path = tmp_path / "dataset.jsonl"
    write_jsonl(path, [VALID_RECORD])

    result = audit_file(path)

    assert result.passed
    assert result.records == 1
    assert result.issues == []
    assert result.duplicates.duplicate_records == 0
    assert result.duplicate_prompts == []


def test_audit_file_detects_prompt_scale_mismatch(tmp_path: Path) -> None:
    """An explicit prompt scale should match metadata."""
    path = tmp_path / "dataset.jsonl"
    record = json.loads(json.dumps(VALID_RECORD))
    record["prompt"] = "Generate a progression in C major."
    record["output"]["metadata"]["scale"] = "F major"
    write_jsonl(path, [record])

    result = audit_file(path)

    assert not result.passed
    assert len(result.issues) == 1
    assert "prompt/metadata scale mismatch" in result.issues[0].message
    assert "C major" in result.issues[0].message
    assert "F major" in result.issues[0].message


def test_audit_file_reports_exact_duplicates(tmp_path: Path) -> None:
    """The audit should report exact duplicate records."""
    path = tmp_path / "dataset.jsonl"
    write_jsonl(path, [VALID_RECORD, VALID_RECORD])

    result = audit_file(path)

    assert result.passed
    assert result.duplicates.duplicate_records == 1
    assert result.duplicates.groups[0].lines == (1, 2)


def test_audit_file_reports_duplicate_prompts(tmp_path: Path) -> None:
    """The audit should report normalized duplicate prompts."""
    path = tmp_path / "dataset.jsonl"
    first = json.loads(json.dumps(VALID_RECORD))
    second = json.loads(json.dumps(VALID_RECORD))
    second["prompt"] = "  GENERATE   A PROGRESSION IN C MAJOR.  "
    second["output"]["metadata"]["bpm"] = 120
    write_jsonl(path, [first, second])

    result = audit_file(path)

    assert result.passed
    assert len(result.duplicate_prompts) == 1
    assert result.duplicate_prompts[0].lines == (1, 2)
