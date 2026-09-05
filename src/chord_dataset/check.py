"""Dataset quality checks."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from .loader import JSONLLoadError, load_jsonl
from .models import ChordGroup, DatasetRecord, SharedReference
from .validator import ValidationError, validate_shared_references


@dataclass(frozen=True, slots=True)
class CheckIssue:
    """A single dataset quality issue."""

    line_number: int
    message: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of a dataset quality check."""

    records: int
    issues: list[CheckIssue]

    @property
    def passed(self) -> bool:
        """Return whether the dataset passed all checks."""
        return not self.issues


def check_file(path: Path) -> CheckResult:
    """Run quality checks against a dataset file."""
    issues: list[CheckIssue] = []
    records = 0

    try:
        loaded_records = load_jsonl(path)

        for line_number, data in loaded_records:
            try:
                record = DatasetRecord.model_validate(data)
            except PydanticValidationError as exc:
                issues.append(
                    CheckIssue(
                        line_number,
                        f"invalid dataset record: {exc}",
                    )
                )
                continue

            try:
                validate_shared_references(line_number, record)
            except ValidationError as exc:
                issues.append(
                    CheckIssue(
                        line_number,
                        str(exc),
                    )
                )
                continue

            records += 1
            issues.extend(_check_record(line_number, record))

    except JSONLLoadError as exc:
        raise ValidationError(str(exc)) from exc

    return CheckResult(
        records=records,
        issues=issues,
    )


def _check_record(
    line_number: int,
    record: DatasetRecord,
) -> list[CheckIssue]:
    """Check semantic dataset quality."""
    issues: list[CheckIssue] = []

    if not record.output.progression_data:
        issues.append(
            CheckIssue(
                line_number,
                "progression_data is empty",
            )
        )
        return issues

    concrete_chords = _count_concrete_chords(record)

    if concrete_chords == 0:
        issues.append(
            CheckIssue(
                line_number,
                "progression contains no concrete chords",
            )
        )

    for index, item in enumerate(record.output.progression_data):
        if not isinstance(item, ChordGroup):
            continue

        if len(item.chords) == 0:
            issues.append(
                CheckIssue(
                    line_number,
                    f"progression_data[{index}] contains an empty chord group",
                )
            )

    return issues


def _count_concrete_chords(record: DatasetRecord) -> int:
    """Count concrete chords, including chords inside shared references."""
    count = 0

    for item in record.output.progression_data:
        if isinstance(item, ChordGroup):
            count += len(item.chords)
        elif isinstance(item, SharedReference):
            key = item.shared.removeprefix("$")
            shared_data = record.output.shared.get(key)

            if shared_data is not None:
                count += len(shared_data.chords)
        else:
            count += 1

    return count
