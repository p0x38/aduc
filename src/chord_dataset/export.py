"""Dataset export utilities."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from .loader import JSONLLoadError, load_jsonl
from .models import DatasetRecord
from .validator import ValidationError, validate_shared_references


def export_jsonl(
    source: Path,
    destination: Path,
) -> int:
    """Validate and export a dataset as deterministic JSONL.

    The source data is not modified semantically. Records are parsed through
    the dataset models to ensure that only valid records are exported.

    Args:
        source: Input JSONL file.
        destination: Output JSONL file.

    Returns:
        Number of exported records.

    Raises:
        ValidationError: If the input file is invalid.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    exported = 0

    try:
        records = load_jsonl(source)

        with destination.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            for line_number, data in records:
                try:
                    record = DatasetRecord.model_validate(data)
                except PydanticValidationError as exc:
                    raise ValidationError(
                        f"line {line_number}: invalid dataset record: {exc}"
                    ) from exc

                validate_shared_references(line_number, record)

                file.write(_serialize_record(record))
                file.write("\n")

                exported += 1

    except JSONLLoadError as exc:
        raise ValidationError(str(exc)) from exc

    return exported


def _serialize_record(record: DatasetRecord) -> str:
    """Serialize a dataset record to compact deterministic JSON."""
    data: dict[str, Any] = record.model_dump(mode="json")

    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )