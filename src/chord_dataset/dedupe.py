"""Dataset deduplication utilities."""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import JSONLLoadError, load_jsonl
from .validator import ValidationError


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """A group of records with identical content."""

    lines: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DedupeResult:
    """Result of a deduplication pass."""

    records: int
    unique_records: int
    duplicate_records: int
    groups: list[DuplicateGroup]


@dataclass(frozen=True, slots=True)
class PromptDuplicateGroup:
    """A group of records with equivalent normalized prompts."""

    lines: tuple[int, ...]
    prompt: str


def dedupe_file(path: Path) -> DedupeResult:
    """Find exact duplicate records in a JSONL dataset."""
    hashes: dict[str, list[int]] = {}
    records = 0

    try:
        for line_number, record in load_jsonl(path):
            records += 1
            digest = _record_hash(record)
            hashes.setdefault(digest, []).append(line_number)
    except JSONLLoadError as exc:
        raise ValidationError(str(exc)) from exc

    groups = [
        DuplicateGroup(lines=tuple(lines))
        for lines in hashes.values()
        if len(lines) > 1
    ]

    duplicate_records = sum(
        len(group.lines) - 1
        for group in groups
    )

    return DedupeResult(
        records=records,
        unique_records=records - duplicate_records,
        duplicate_records=duplicate_records,
        groups=groups,
    )


def find_duplicate_prompts(
    path: Path,
) -> list[PromptDuplicateGroup]:
    """Find records with identical normalized prompts."""
    prompts: dict[str, list[int]] = defaultdict(list)
    display_prompts: dict[str, str] = {}

    try:
        for line_number, record in load_jsonl(path):
            prompt = record.get("prompt")

            if not isinstance(prompt, str):
                continue

            normalized = _normalize_prompt(prompt)

            if not normalized:
                continue

            prompts[normalized].append(line_number)
            display_prompts.setdefault(normalized, prompt)

    except JSONLLoadError as exc:
        raise ValidationError(str(exc)) from exc

    return [
        PromptDuplicateGroup(
            lines=tuple(lines),
            prompt=display_prompts[normalized],
        )
        for normalized, lines in prompts.items()
        if len(lines) > 1
    ]


def write_deduplicated(
    source: Path,
    destination: Path,
) -> int:
    """Write a dataset with exact duplicate records removed."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0

    try:
        loaded_records = load_jsonl(source)

        with destination.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            for _, record in loaded_records:
                serialized = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )

                if serialized in seen:
                    continue

                seen.add(serialized)
                file.write(serialized)
                file.write("\n")
                written += 1

    except JSONLLoadError as exc:
        raise ValidationError(str(exc)) from exc

    return written


def _normalize_prompt(prompt: str) -> str:
    """Normalize a prompt for duplicate comparison."""
    normalized = prompt.strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _record_hash(record: dict[str, Any]) -> str:
    """Create a deterministic hash for a record."""
    serialized = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8"),
    ).hexdigest()
