"""Dataset normalization utilities."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from .loader import JSONLLoadError, load_jsonl
from .models import DatasetRecord
from .validator import ValidationError, validate_shared_references


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Summary of a dataset normalization run."""

    records: int
    changed_records: int


def normalize_record(record: DatasetRecord) -> DatasetRecord:
    """Normalize non-semantic formatting in a dataset record."""

    data = record.model_dump(mode="python")

    metadata = data["output"]["metadata"]
    metadata["scale"] = _normalize_scale(metadata["scale"])

    for item in data["output"]["progression_data"]:
        _normalize_progression_item(item)

    for shared_data in data["output"]["shared"].values():
        for chord in shared_data["chords"]:
            chord["name"] = _normalize_chord_name(chord["name"])

    return DatasetRecord.model_validate(data)


def normalize_file(
    source: Path,
    destination: Path,
) -> NormalizationResult:
    """Normalize a JSONL dataset and write the result."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    records = 0
    changed_records = 0

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

                normalized = normalize_record(record)

                if normalized.model_dump(mode="json") != record.model_dump(
                    mode="json"
                ):
                    changed_records += 1

                file.write(
                    json.dumps(
                        normalized.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                file.write("\n")

                records += 1

    except JSONLLoadError as exc:
        raise ValidationError(str(exc)) from exc

    return NormalizationResult(
        records=records,
        changed_records=changed_records,
    )


def _normalize_progression_item(item: dict[str, Any]) -> None:
    """Normalize a progression item in place."""

    if "name" in item:
        item["name"] = _normalize_chord_name(item["name"])
        return

    if "chords" in item:
        for chord in item["chords"]:
            chord["name"] = _normalize_chord_name(chord["name"])
        return

    if "shared" in item:
        item["shared"] = _normalize_shared_reference(item["shared"])


def _normalize_scale(scale: str) -> str:
    """Normalize common scale-name aliases."""
    normalized = " ".join(scale.strip().split())

    aliases = {
        "A minor": "A natural minor",
        "Bb minor": "Bb natural minor",
        "C minor": "C natural minor",
        "D minor": "D natural minor",
        "E minor": "E natural minor",
        "F minor": "F natural minor",
        "G minor": "G natural minor",
        "C# minor": "C# natural minor",
        "F# minor": "F# natural minor",
    }

    return aliases.get(normalized, normalized)


def _normalize_chord_name(name: str) -> str:
    """Normalize whitespace in a chord name."""
    return " ".join(name.strip().split())


def _normalize_shared_reference(reference: str) -> str:
    """Normalize a shared reference."""
    return reference.strip()
