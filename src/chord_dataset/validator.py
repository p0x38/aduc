"""Dataset validation logic."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from .loader import JSONLLoadError, load_jsonl
from .models import DatasetRecord, SharedReference


class ValidationError(ValueError):
    """Raised when a dataset fails validation."""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Summary of a dataset validation run."""

    total_lines: int
    valid_records: int


def validate_file(path: Path) -> ValidationResult:
    """Validate every record in a JSONL file."""

    total_lines = 0
    valid_records = 0

    try:
        records = load_jsonl(path)

        for line_number, data in records:
            total_lines += 1

            try:
                record = DatasetRecord.model_validate(data)
            except PydanticValidationError as exc:
                raise ValidationError(
                    format_validation_error(line_number, exc)
                ) from exc

            validate_shared_references(line_number, record)
            valid_records += 1

    except JSONLLoadError as exc:
        raise ValidationError(str(exc)) from exc

    return ValidationResult(
        total_lines=total_lines,
        valid_records=valid_records,
    )


def validate_shared_references(
    line_number: int,
    record: DatasetRecord,
) -> None:
    """Ensure every shared reference resolves to an existing key."""

    shared = record.output.shared

    for index, item in enumerate(record.output.progression_data):
        if not isinstance(item, SharedReference):
            continue

        reference = item.shared
        key = reference.removeprefix("$")

        if key not in shared:
            raise ValidationError(
                f"line {line_number}: progression_data[{index}] "
                f"references missing shared structure {reference!r}"
            )


def format_validation_error(
    line_number: int,
    error: PydanticValidationError,
) -> str:
    """Format a Pydantic validation error for CLI output."""

    first_error = error.errors()[0]

    location = ".".join(str(part) for part in first_error["loc"])
    message = first_error["msg"]

    return f"line {line_number}: {location}: {message}"
