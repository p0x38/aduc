"""JSONL dataset loading utilities."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class JSONLLoadError(ValueError):
    """Raised when a JSONL file cannot be read or parsed."""


def load_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield parsed JSON objects together with their line numbers."""

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise JSONLLoadError(
                        f"line {line_number}: invalid JSON: {exc.msg}"
                    ) from exc

                if not isinstance(value, dict):
                    raise JSONLLoadError(
                        f"line {line_number}: top-level JSON value "
                        "must be an object"
                    )

                yield line_number, value

    except OSError as exc:
        raise JSONLLoadError(f"unable to read {path}: {exc}") from exc