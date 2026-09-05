"""Dataset statistics."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from .loader import JSONLLoadError, load_jsonl
from .models import ChordGroup, DatasetRecord, SharedReference
from .operation import PromptOperation, classify_operation
from .validator import ValidationError, validate_shared_references


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Statistics collected from a chord progression dataset."""

    records: int
    total_chords: int
    chord_groups: int
    shared_references: int
    shared_structures: int
    average_bpm: float
    min_bpm: int
    max_bpm: int
    scales: Counter[str]
    signatures: Counter[tuple[int, int]]
    prompt_starters: Counter[str]
    operations: Counter[PromptOperation]


def collect_stats(path: Path) -> DatasetStats:
    """Collect statistics from a validated JSONL dataset."""

    records = 0
    total_chords = 0
    chord_groups = 0
    shared_references = 0
    shared_structures = 0

    bpm_values: list[int] = []
    scales: Counter[str] = Counter()
    signatures: Counter[tuple[int, int]] = Counter()
    prompt_starters: Counter[str] = Counter()
    operations: Counter[PromptOperation] = Counter()

    try:
        loaded_records = load_jsonl(path)

        for line_number, data in loaded_records:
            try:
                record = DatasetRecord.model_validate(data)
            except PydanticValidationError as exc:
                raise ValidationError(
                    f"line {line_number}: invalid dataset record: {exc}"
                ) from exc

            validate_shared_references(line_number, record)

            records += 1
            shared_structures += len(record.output.shared)

            metadata = record.output.metadata

            bpm_values.append(metadata.bpm)
            scales[metadata.scale] += 1
            signatures[metadata.signature] += 1

            prompt_starter = record.prompt.split(maxsplit=1)[0]
            prompt_starters[prompt_starter] += 1

            operations[classify_operation(record.prompt)] += 1

            for item in record.output.progression_data:
                if isinstance(item, ChordGroup):
                    chord_groups += 1
                    total_chords += len(item.chords)
                elif isinstance(item, SharedReference):
                    shared_references += 1
                else:
                    total_chords += 1

    except JSONLLoadError as exc:
        raise ValidationError(str(exc)) from exc

    if not bpm_values:
        return DatasetStats(
            records=0,
            total_chords=0,
            chord_groups=0,
            shared_references=0,
            shared_structures=0,
            average_bpm=0.0,
            min_bpm=0,
            max_bpm=0,
            scales=Counter(),
            signatures=Counter(),
            prompt_starters=Counter(),
            operations=Counter(),
        )

    return DatasetStats(
        records=records,
        total_chords=total_chords,
        chord_groups=chord_groups,
        shared_references=shared_references,
        shared_structures=shared_structures,
        average_bpm=sum(bpm_values) / len(bpm_values),
        min_bpm=min(bpm_values),
        max_bpm=max(bpm_values),
        scales=scales,
        signatures=signatures,
        prompt_starters=prompt_starters,
        operations=operations,
    )
