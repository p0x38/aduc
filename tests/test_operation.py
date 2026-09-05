"""Tests for prompt operation classification."""

import pytest

from chord_dataset.operation import (
    PromptOperation,
    classify_operation,
)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "Generate a chord progression in C major.",
            PromptOperation.GENERATE,
        ),
        (
            "Create a bright progression in D major.",
            PromptOperation.GENERATE,
        ),
        (
            "Build a jazz progression in Bb major.",
            PromptOperation.GENERATE,
        ),
        (
            "Write a progression in A minor.",
            PromptOperation.GENERATE,
        ),
        (
            "Continue this progression in Eb major: Ebmaj7 | Gm7.",
            PromptOperation.CONTINUE,
        ),
        (
            "Complete this progression in E major: Emaj7 | C#m7.",
            PromptOperation.COMPLETE,
        ),
        (
            "Extend this four-bar progression into eight bars.",
            PromptOperation.EXTEND,
        ),
        (
            "Repeat the main progression twice.",
            PromptOperation.REPEAT,
        ),
        (
            "Repeat the hook twice and finish with a turnaround.",
            PromptOperation.ADD_TURNAROUND,
        ),
        (
            "Reharmonize C-Am-F-G with richer jazz chords.",
            PromptOperation.REHARMONIZE,
        ),
        (
            "Simplify this jazz progression into basic triads.",
            PromptOperation.SIMPLIFY,
        ),
        (
            "Transpose C-Am-F-G from C major to E major.",
            PromptOperation.TRANSPOSE,
        ),
        (
            "Add passing chords to C-F-G-C.",
            PromptOperation.ADD_PASSING,
        ),
        (
            "Add a secondary dominant before D minor.",
            PromptOperation.ADD_SECONDARY_DOMINANT,
        ),
        (
            "Make this progression darker while preserving the root motion.",
            PromptOperation.DARKEN,
        ),
        (
            "Make this progression brighter with major ninths.",
            PromptOperation.BRIGHTEN,
        ),
        (
            "Modify the progression by replacing the final chord.",
            PromptOperation.MODIFY,
        ),
        (
            "Rework this progression into a city-pop style.",
            PromptOperation.MODIFY,
        ),
    ],
)
def test_classify_operation(
    prompt: str,
    expected: PromptOperation,
) -> None:
    """Prompts should be classified according to their main operation."""
    assert classify_operation(prompt) is expected


def test_generation_verbs_are_not_required_to_match() -> None:
    """Different generation phrasings should map to the same operation."""
    prompts = [
        "Give me a progression in C major.",
        "Put together a progression in G major.",
        "Come up with a progression in A minor.",
        "Try a progression in F major.",
    ]

    assert all(
        classify_operation(prompt) is PromptOperation.GENERATE
        for prompt in prompts
    )


def test_operation_enum_values_are_stable() -> None:
    """Serialized operation values should remain stable."""
    assert PromptOperation.GENERATE.value == "generation"
    assert PromptOperation.CONTINUE.value == "continuation"
    assert PromptOperation.REHARMONIZE.value == "reharmonization"
    assert PromptOperation.TRANSPOSE.value == "transposition"


def test_unknown_prompt_returns_other() -> None:
    """Unknown prompt forms should fall back to other."""
    assert (
        classify_operation("Something completely unspecified.")
        is PromptOperation.OTHER
    )
