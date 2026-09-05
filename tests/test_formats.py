"""Tests for training output formats."""

import json

from chord_dataset.formats import (
    ChatFormatter,
    CompletionFormatter,
    OutputFormat,
    StructuredFormatter,
    format_record,
    get_formatter,
)
from chord_dataset.models import DatasetRecord


def make_record() -> DatasetRecord:
    """Create a valid test record."""
    return DatasetRecord.model_validate(
        {
            "prompt": "Create a progression in C major.",
            "output": {
                "metadata": {
                    "bpm": 97,
                    "signature": [4, 4],
                    "scale": "C major",
                },
                "shared": {
                    "turnaround": {
                        "chords": [
                            {"name": "Dm7"},
                            {"name": "G7"},
                        ]
                    }
                },
                "progression_data": [
                    {"name": "Cmaj7"},
                    {"shared": "$turnaround"},
                ],
            },
        }
    )


def test_get_formatter_returns_structured_formatter() -> None:
    """Structured format should return its formatter."""
    assert isinstance(
        get_formatter(OutputFormat.STRUCTURED),
        StructuredFormatter,
    )


def test_get_formatter_returns_completion_formatter() -> None:
    """Completion format should return its formatter."""
    assert isinstance(
        get_formatter(OutputFormat.COMPLETION),
        CompletionFormatter,
    )


def test_get_formatter_returns_chat_formatter() -> None:
    """Chat format should return its formatter."""
    assert isinstance(
        get_formatter(OutputFormat.CHAT),
        ChatFormatter,
    )


def test_structured_formatter() -> None:
    """Structured output should preserve the full response object."""
    result = StructuredFormatter().format(make_record())

    assert result["prompt"] == "Create a progression in C major."
    assert "response" in result
    assert "metadata" in result["response"]
    assert result["response"]["metadata"]["bpm"] == 97


def test_completion_formatter() -> None:
    """Completion output should contain serialized structured data."""
    result = CompletionFormatter().format(make_record())

    assert result["prompt"] == "Create a progression in C major."
    assert "completion" in result
    assert isinstance(result["completion"], str)

    completion = json.loads(result["completion"])

    assert completion["metadata"]["scale"] == "C major"


def test_chat_formatter() -> None:
    """Chat output should contain system, user, and assistant messages."""
    result = ChatFormatter().format(make_record())

    messages = result["messages"]

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"

    assert messages[1]["content"] == "Create a progression in C major."

    assistant = json.loads(messages[2]["content"])

    assert assistant["metadata"]["bpm"] == 97


def test_format_record_dispatches() -> None:
    """format_record should use the requested formatter."""
    record = make_record()

    structured = format_record(
        record,
        OutputFormat.STRUCTURED,
    )
    completion = format_record(
        record,
        OutputFormat.COMPLETION,
    )
    chat = format_record(
        record,
        OutputFormat.CHAT,
    )

    assert set(structured) == {"prompt", "response"}
    assert set(completion) == {"prompt", "completion"}
    assert set(chat) == {"messages"}


def test_structured_format_preserves_shared_reference() -> None:
    """Structured output should preserve shared references."""
    result = format_record(
        make_record(),
        OutputFormat.STRUCTURED,
    )

    progression = result["response"]["progression_data"]

    assert progression[1] == {"shared": "$turnaround"}


def test_completion_format_is_valid_json() -> None:
    """The completion field should itself contain valid JSON."""
    result = format_record(
        make_record(),
        OutputFormat.COMPLETION,
    )

    json.loads(result["completion"])


def test_chat_assistant_content_is_valid_json() -> None:
    """Chat assistant content should contain valid JSON."""
    result = format_record(
        make_record(),
        OutputFormat.CHAT,
    )

    assistant_content = result["messages"][2]["content"]

    json.loads(assistant_content)
