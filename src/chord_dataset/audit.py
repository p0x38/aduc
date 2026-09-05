"""Dataset audit utilities."""

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from .check import CheckIssue, check_file
from .dedupe import DedupeResult, PromptDuplicateGroup, dedupe_file, find_duplicate_prompts
from .loader import JSONLLoadError, load_jsonl
from .models import DatasetRecord
from .validator import ValidationError


_SCALE_PATTERN = re.compile(
    r"\b([A-Ga-g](?:#|b)?\s+(?:natural\s+minor|harmonic\s+minor|"
    r"melodic\s+minor|major|minor|mixolydian|dorian|phrygian|"
    r"lydian|locrian))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Result of a dataset audit."""

    records: int
    issues: list[CheckIssue]
    duplicates: DedupeResult
    duplicate_prompts: list[PromptDuplicateGroup]

    @property
    def passed(self) -> bool:
        """Return whether the dataset passed semantic audit checks."""
        return not self.issues


def audit_file(path: Path) -> AuditResult:
    """Audit dataset structure, semantics, and duplicate records."""
    check = check_file(path)
    duplicates = dedupe_file(path)
    duplicate_prompts = find_duplicate_prompts(path)

    issues = list(check.issues)
    issues.extend(_check_prompt_scales(path))

    return AuditResult(
        records=check.records,
        issues=issues,
        duplicates=duplicates,
        duplicate_prompts=duplicate_prompts,
    )


def _check_prompt_scales(path: Path) -> list[CheckIssue]:
    """Check explicit prompt scales against metadata scales."""
    issues: list[CheckIssue] = []

    try:
        for line_number, data in load_jsonl(path):
            try:
                record = DatasetRecord.model_validate(data)
            except PydanticValidationError:
                continue

            scales = {
                _normalize_scale(match.group(1))
                for match in _SCALE_PATTERN.finditer(record.prompt)
            }

            if len(scales) != 1:
                continue

            prompt_scale = next(iter(scales))
            metadata_scale = _normalize_scale(record.output.metadata.scale)

            if prompt_scale != metadata_scale:
                issues.append(
                    CheckIssue(
                        line_number,
                        "prompt/metadata scale mismatch: "
                        f"prompt says {prompt_scale!r}, "
                        f"metadata says {metadata_scale!r}",
                    )
                )
    except JSONLLoadError as exc:
        raise ValidationError(str(exc)) from exc

    return issues


def _normalize_scale(value: str) -> str:
    """Normalize a scale name for comparison."""
    return " ".join(value.lower().split())
