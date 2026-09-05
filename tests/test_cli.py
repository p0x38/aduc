"""Tests for the command-line interface."""

import json
from pathlib import Path

from click.testing import CliRunner

from chord_dataset.cli import main


def make_record(index: int) -> dict[str, object]:
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


def write_jsonl(path: Path, count: int) -> None:
    """Write test records to a JSONL file."""
    path.write_text(
        "".join(
            f"{json.dumps(make_record(index))}\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def test_validate_command_succeeds(tmp_path: Path) -> None:
    """The validate command should accept a valid dataset."""
    path = tmp_path / "dataset.jsonl"
    write_jsonl(path, 3)

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(path)])

    assert result.exit_code == 0
    assert "Dataset is valid" in result.output
    assert "Records: 3" in result.output
    assert "Lines checked: 3" in result.output


def test_validate_command_fails_on_invalid_dataset(
    tmp_path: Path,
) -> None:
    """The validate command should report invalid datasets."""
    path = tmp_path / "dataset.jsonl"

    path.write_text(
        '{"prompt":"invalid"}\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(path)])

    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "output" in result.output


def test_validate_command_fails_on_invalid_json(
    tmp_path: Path,
) -> None:
    """The validate command should report malformed JSON."""
    path = tmp_path / "dataset.jsonl"

    path.write_text(
        '{"prompt":"broken"\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(path)])

    assert result.exit_code != 0
    assert "invalid JSON" in result.output


def test_stats_command_succeeds(tmp_path: Path) -> None:
    """The stats command should display dataset statistics."""
    path = tmp_path / "dataset.jsonl"
    write_jsonl(path, 5)

    runner = CliRunner()
    result = runner.invoke(main, ["stats", str(path)])

    assert result.exit_code == 0
    assert "Dataset Statistics" in result.output
    assert "Records" in result.output
    assert "5" in result.output
    assert "C major" in result.output
    assert "4/4" in result.output
    assert "generation" in result.output


def test_stats_command_fails_on_invalid_dataset(
    tmp_path: Path,
) -> None:
    """The stats command should reject invalid datasets."""
    path = tmp_path / "dataset.jsonl"

    path.write_text(
        '{"prompt":"invalid"}\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["stats", str(path)])

    assert result.exit_code != 0
    assert "Error:" in result.output


def test_split_command_creates_files(tmp_path: Path) -> None:
    """The split command should create all output files."""
    input_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "split"

    write_jsonl(input_path, 100)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["split", str(input_path), str(output_dir)],
    )

    assert result.exit_code == 0
    assert "Dataset split successfully" in result.output

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"

    assert train_path.is_file()
    assert validation_path.is_file()
    assert test_path.is_file()


def test_split_command_reports_counts(tmp_path: Path) -> None:
    """The split command should report the resulting counts."""
    input_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "split"

    write_jsonl(input_path, 100)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["split", str(input_path), str(output_dir)],
    )

    assert result.exit_code == 0
    assert "Train: 80" in result.output
    assert "Validation: 10" in result.output
    assert "Test: 10" in result.output
    assert "Total: 100" in result.output


def test_split_command_accepts_custom_ratios(
    tmp_path: Path,
) -> None:
    """The split command should accept custom ratios."""
    input_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "split"

    write_jsonl(input_path, 100)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "split",
            str(input_path),
            str(output_dir),
            "--train-ratio",
            "0.7",
            "--validation-ratio",
            "0.2",
        ],
    )

    assert result.exit_code == 0
    assert "Train: 70" in result.output
    assert "Validation: 20" in result.output
    assert "Test: 10" in result.output


def test_split_command_accepts_seed(
    tmp_path: Path,
) -> None:
    """The split command should accept a deterministic seed."""
    input_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "split"

    write_jsonl(input_path, 20)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "split",
            str(input_path),
            str(output_dir),
            "--seed",
            "123",
        ],
    )

    assert result.exit_code == 0


def test_split_command_rejects_invalid_ratios(
    tmp_path: Path,
) -> None:
    """The split command should reject invalid ratios."""
    input_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "split"

    write_jsonl(input_path, 10)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "split",
            str(input_path),
            str(output_dir),
            "--train-ratio",
            "0.8",
            "--validation-ratio",
            "0.3",
        ],
    )

    assert result.exit_code != 0
    assert "must not exceed 1" in result.output


def test_split_command_preserves_record_count(
    tmp_path: Path,
) -> None:
    """The split command must not lose records."""
    input_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "split"

    write_jsonl(input_path, 217)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["split", str(input_path), str(output_dir)],
    )

    assert result.exit_code == 0

    total = sum(
        len((output_dir / filename).read_text(encoding="utf-8").splitlines())
        for filename in (
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
        )
    )

    assert total == 217


def test_help_command_succeeds() -> None:
    """The top-level help command should work."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "Manage and validate chord progression datasets." in result.output
    assert "validate" in result.output
    assert "stats" in result.output
    assert "split" in result.output


def test_dedupe_command(tmp_path: Path) -> None:
    source = tmp_path / "dataset.jsonl"

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
            ],
        },
    }

    source.write_text(
        json.dumps(record, ensure_ascii=False) + "\n"
        + json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["dedupe", str(source)])

    assert result.exit_code == 0
    assert "Exact duplicates found" in result.output
    assert "Duplicate records: 1" in result.output


def test_dedupe_command_writes_output(tmp_path: Path) -> None:
    """The dedupe command should write a deduplicated dataset."""
    source = tmp_path / "dataset.jsonl"
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
            ],
        },
    }

    source.write_text(
        json.dumps(record, ensure_ascii=False)
        + "\n"
        + json.dumps(record, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "dedupe",
            str(source),
            "--write",
            str(destination),
        ],
    )

    assert result.exit_code == 0
    assert "Deduplicated dataset written" in result.output
    assert "Records written: 1" in result.output

    lines = destination.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
