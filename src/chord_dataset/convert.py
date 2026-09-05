"""Dataset conversion utilities."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from .formats import OutputFormat, format_record
from .loader import JSONLLoadError, load_jsonl
from .models import DatasetRecord
from .validator import ValidationError, validate_shared_references


def convert_record(record: DatasetRecord) -> dict[str, Any]:
    """Convert a record using the default structured format."""
    return format_record(record, OutputFormat.STRUCTURED)


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Summary of a dataset conversion run."""

    records: int
    output_format: OutputFormat


def convert_jsonl(
    source: Path,
    destination: Path,
    *,
    output_format: OutputFormat = OutputFormat.STRUCTURED,
) -> ConversionResult:
    """Convert a validated dataset into a training format."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    records = 0

    try:
        loaded_records = load_jsonl(source)

        with destination.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            for line_number, data in loaded_records:
                try:
                    record = DatasetRecord.model_validate(data)
                except PydanticValidationError as exc:
                    raise ValidationError(
                        f"line {line_number}: invalid dataset record: {exc}"
                    ) from exc

                validate_shared_references(line_number, record)

                converted = format_record(record, output_format)

                file.write(
                    json.dumps(
                        converted,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                file.write("\n")

                records += 1

    except JSONLLoadError as exc:
        raise ValidationError(str(exc)) from exc

    return ConversionResult(
        records=records,
        output_format=output_format,
    )
