"""Training dataset output formats."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Final

from .models import DatasetRecord


class OutputFormat(StrEnum):
    """Supported training output formats."""

    STRUCTURED = "structured"
    COMPLETION = "completion"
    CHAT = "chat"


class RecordFormatter(ABC):
    """Base class for training record formatters."""

    @abstractmethod
    def format(self, record: DatasetRecord) -> dict[str, Any]:
        """Format a dataset record."""


class StructuredFormatter(RecordFormatter):
    """Format records as prompt/response objects."""

    def format(self, record: DatasetRecord) -> dict[str, Any]:
        """Return a structured prompt/response record."""
        return {
            "prompt": record.prompt,
            "response": record.output.model_dump(mode="json"),
        }


class CompletionFormatter(RecordFormatter):
    """Format records as plain prompt/completion text."""

    def format(self, record: DatasetRecord) -> dict[str, Any]:
        """Return a prompt/completion record."""
        response = json.dumps(
            record.output.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return {
            "prompt": record.prompt,
            "completion": response,
        }


class ChatFormatter(RecordFormatter):
    """Format records as chat messages."""

    SYSTEM_MESSAGE: Final = (
        "You generate and transform chord progressions using the "
        "provided musical instructions."
    )

    def format(self, record: DatasetRecord) -> dict[str, Any]:
        """Return a chat-style training record."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": record.prompt,
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        record.output.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
        }


_FORMATTERS: Final[dict[OutputFormat, RecordFormatter]] = {
    OutputFormat.STRUCTURED: StructuredFormatter(),
    OutputFormat.COMPLETION: CompletionFormatter(),
    OutputFormat.CHAT: ChatFormatter(),
}


def get_formatter(output_format: OutputFormat) -> RecordFormatter:
    """Return the formatter for an output format."""
    try:
        return _FORMATTERS[output_format]
    except KeyError as exc:
        raise ValueError(
            f"unsupported output format: {output_format!r}"
        ) from exc


def format_record(
    record: DatasetRecord,
    output_format: OutputFormat,
) -> dict[str, Any]:
    """Format a record using the requested output format."""
    return get_formatter(output_format).format(record)
