"""Prompt operation classification."""

import re
from enum import StrEnum


class PromptOperation(StrEnum):
    """Supported dataset prompt operations."""

    GENERATE = "generation"
    CONTINUE = "continuation"
    COMPLETE = "completion"
    EXTEND = "extension"
    REPEAT = "repetition"
    REHARMONIZE = "reharmonization"
    SIMPLIFY = "simplification"
    TRANSPOSE = "transposition"
    ADD_PASSING = "add_passing_chords"
    ADD_TURNAROUND = "add_turnaround"
    ADD_SECONDARY_DOMINANT = "add_secondary_dominant"
    DARKEN = "darkening"
    BRIGHTEN = "brightening"
    MODIFY = "modification"
    STYLE = "style_change"
    OTHER = "other"


def classify_operation(prompt: str) -> PromptOperation:
    """Classify a prompt by its requested musical operation."""

    text = prompt.strip().lower()

    if not text:
        return PromptOperation.OTHER

    # Explicit transformation operations take priority.
    if _contains_phrase(
        text,
        "reharmonize",
        "re-harmonize",
        "reharmonise",
        "re-harmonise",
    ):
        return PromptOperation.REHARMONIZE

    if _contains_phrase(text, "transpose", "transposition"):
        return PromptOperation.TRANSPOSE

    if _contains_phrase(text, "simplify"):
        return PromptOperation.SIMPLIFY

    if _contains_phrase(text, "reduce"):
        return PromptOperation.SIMPLIFY

    if _contains_any(
        text,
        "add a turnaround",
        "add turnaround",
        "with a turnaround",
        "shared turnaround",
        "turnaround",
    ):
        return PromptOperation.ADD_TURNAROUND

    if _contains_any(
        text,
        "add passing",
        "passing chord",
        "passing chords",
        "passing-diminished",
    ):
        return PromptOperation.ADD_PASSING

    if _contains_phrase(
        text,
        "secondary dominant",
        "secondary dominants",
    ):
        return PromptOperation.ADD_SECONDARY_DOMINANT

    if _contains_any(
        text,
        "darker",
        "dark atmosphere",
        "make it dark",
        "make this dark",
        "more emotional",
        "moody",
    ):
        return PromptOperation.DARKEN

    if _contains_any(
        text,
        "brighter",
        "bright atmosphere",
        "make it bright",
        "make this bright",
        "cheerful",
        "hopeful",
        "uplifting",
    ):
        return PromptOperation.BRIGHTEN

    if _contains_phrase(text, "continue", "continuation"):
        return PromptOperation.CONTINUE

    if _contains_phrase(text, "complete", "completion"):
        return PromptOperation.COMPLETE

    if _contains_phrase(text, "extend", "extension"):
        return PromptOperation.EXTEND

    if _contains_phrase(text, "repeat", "repeated", "repetition"):
        return PromptOperation.REPEAT

    if _contains_phrase(
        text,
        "modify",
        "rework",
        "alter",
        "change",
        "dress it up",
    ):
        return PromptOperation.MODIFY

    # A style name by itself does not imply that the operation is
    # "style_change". For example, "Build a jazz progression" is still
    # a generation request.
    if _looks_like_generation(text):
        return PromptOperation.GENERATE

    if _contains_any(
        text,
        "style",
        "styled",
        "style change",
    ):
        return PromptOperation.STYLE

    return PromptOperation.OTHER


def _contains_phrase(text: str, *phrases: str) -> bool:
    """Return whether any phrase occurs as a complete word sequence."""
    return any(
        re.search(rf"\b{re.escape(phrase)}\b", text) is not None
        for phrase in phrases
    )


def _contains_any(text: str, *terms: str) -> bool:
    """Return whether any term occurs in the text."""
    return any(term in text for term in terms)


def _looks_like_generation(text: str) -> bool:
    """Determine whether a prompt primarily requests generation."""
    generation_terms = (
        "generate",
        "create",
        "build",
        "craft",
        "compose",
        "produce",
        "design",
        "write",
        "come up with",
        "put together",
        "give me",
        "try",
        "set up",
        "shape",
        "center",
        "hold",
        "use",
        "represent",
        "make a",
    )

    return _contains_any(text, *generation_terms)
