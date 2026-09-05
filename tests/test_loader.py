"""Tests for JSONL loading."""

import json
from pathlib import Path

import pytest

from chord_dataset.loader import JSONLLoadError, load_jsonl


def test_load_jsonl_reads_objects(tmp_path: Path) -> None:
    """Valid JSON objects should be loaded with their line numbers."""
    path = tmp_path / "dataset.jsonl"

    path.write_text(
        '{"prompt":"first"}\n'
        '{"prompt":"second"}\n',
        encoding="utf-8",
    )

    records = list(load_jsonl(path))

    assert records == [
        (1, {"prompt": "first"}),
        (2, {"prompt": "second"}),
    ]


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines should not produce records."""
    path = tmp_path / "dataset.jsonl"

    path.write_text(
        '{"prompt":"first"}\n'
        "\n"
        "   \n"
        '{"prompt":"second"}\n',
        encoding="utf-8",
    )

    records = list(load_jsonl(path))

    assert records == [
        (1, {"prompt": "first"}),
        (4, {"prompt": "second"}),
    ]


def test_load_jsonl_rejects_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON should raise JSONLLoadError."""
    path = tmp_path / "dataset.jsonl"

    path.write_text(
        '{"prompt":"valid"}\n'
        '{"prompt":"broken"\n',
        encoding="utf-8",
    )

    with pytest.raises(JSONLLoadError, match=r"line 2: invalid JSON"):
        list(load_jsonl(path))


def test_load_jsonl_rejects_non_object(tmp_path: Path) -> None:
    """Top-level arrays and primitives should be rejected."""
    path = tmp_path / "dataset.jsonl"

    path.write_text(
        json.dumps(["not", "an", "object"]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        JSONLLoadError,
        match="top-level JSON value must be an object",
    ):
        list(load_jsonl(path))


def test_load_jsonl_rejects_missing_file(tmp_path: Path) -> None:
    """Missing files should produce a JSONLLoadError."""
    path = tmp_path / "missing.jsonl"

    with pytest.raises(JSONLLoadError, match="unable to read"):
        list(load_jsonl(path))


def test_load_jsonl_preserves_complex_values(tmp_path: Path) -> None:
    """Nested JSON objects and arrays should be preserved."""
    path = tmp_path / "dataset.jsonl"

    value = {
        "prompt": "Generate a progression.",
        "output": {
            "metadata": {
                "bpm": 97,
                "signature": [4, 4],
            },
            "progression_data": [
                {"name": "Cmaj7"},
                {
                    "chords": [
                        {"name": "Dm7"},
                        {"name": "G7"},
                    ]
                },
            ],
        },
    }

    path.write_text(
        json.dumps(value) + "\n",
        encoding="utf-8",
    )

    records = list(load_jsonl(path))

    assert records == [(1, value)]