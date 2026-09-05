"""Tests for the dataset audit CLI command."""

import json
from pathlib import Path

from click.testing import CliRunner

from chord_dataset.cli import main


def test_audit_command_reports_scale_mismatch(tmp_path: Path) -> None:
    """The audit command should report prompt/metadata mismatches."""
    path = tmp_path / "dataset.jsonl"
    record = {
        "prompt": "Generate a progression in C major.",
        "output": {
            "metadata": {
                "bpm": 100,
                "signature": [4, 4],
                "scale": "F major",
            },
            "shared": {},
            "progression_data": [{"name": "C"}],
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["audit", str(path)])

    assert result.exit_code == 1
    assert "Dataset Audit" in result.output
    assert "Issues: 1" in result.output
    assert "prompt/metadata scale mismatch" in result.output


def test_audit_command_passes_clean_dataset(tmp_path: Path) -> None:
    """The audit command should succeed for a clean dataset."""
    path = tmp_path / "dataset.jsonl"
    record = {
        "prompt": "Generate a progression in C major.",
        "output": {
            "metadata": {
                "bpm": 100,
                "signature": [4, 4],
                "scale": "C major",
            },
            "shared": {},
            "progression_data": [{"name": "C"}],
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["audit", str(path)])

    assert result.exit_code == 0
    assert "Issues: 0" in result.output
    assert "Dataset audit passed" in result.output
