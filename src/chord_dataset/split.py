"""Dataset splitting utilities."""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import JSONLLoadError, load_jsonl


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A dataset split containing raw JSON records."""

    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]

    @property
    def total(self) -> int:
        """Return the total number of records."""
        return len(self.train) + len(self.validation) + len(self.test)


def split_records(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    """Split records into train, validation, and test sets.

    Records are shuffled deterministically using ``seed``.
    """
    if train_ratio < 0:
        raise ValueError("train_ratio must be non-negative")

    if validation_ratio < 0:
        raise ValueError("validation_ratio must be non-negative")

    ratio_sum = train_ratio + validation_ratio

    if ratio_sum > 1.0 and not math.isclose(ratio_sum, 1.0):
        raise ValueError("train_ratio + validation_ratio must not exceed 1")

    test_ratio = max(0.0, 1.0 - ratio_sum)

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)

    train_end = round(total * train_ratio)
    validation_end = train_end + round(total * validation_ratio)

    # Avoid accidental overflow from floating-point edge cases.
    train_end = min(train_end, total)
    validation_end = min(validation_end, total)

    train = shuffled[:train_end]
    validation = shuffled[train_end:validation_end]
    test = shuffled[validation_end:]

    # Keep this variable meaningful and make the intended ratio explicit.
    _ = test_ratio

    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
    )


def load_and_split(
    path: Path,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    """Load a JSONL file and split its records."""
    records: list[dict[str, Any]] = []

    try:
        for _, record in load_jsonl(path):
            records.append(record)
    except JSONLLoadError:
        raise

    return split_records(
        records,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )


def write_split(
    split: DatasetSplit,
    output_dir: Path,
) -> None:
    """Write a dataset split as three JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_dir / "train.jsonl", split.train)
    _write_jsonl(output_dir / "validation.jsonl", split.validation)
    _write_jsonl(output_dir / "test.jsonl", split.test)


def _write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write raw records to a JSONL file."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")
