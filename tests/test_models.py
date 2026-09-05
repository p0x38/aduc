"""Tests for dataset Pydantic models."""

import pytest
from pydantic import ValidationError

from chord_dataset.models import (
    Chord,
    ChordGroup,
    DatasetRecord,
    Metadata,
    Output,
    SharedData,
    SharedReference,
)


def make_output(**overrides: object) -> dict[str, object]:
    """Create a minimal valid output object."""
    output: dict[str, object] = {
        "metadata": {
            "bpm": 97,
            "signature": [4, 4],
            "scale": "C major",
        },
        "shared": {},
        "progression_data": [{"name": "C"}],
    }
    output.update(overrides)
    return output


def make_record(**overrides: object) -> dict[str, object]:
    """Create a minimal valid dataset record."""
    record: dict[str, object] = {
        "prompt": "Generate a progression in C major.",
        "output": make_output(),
    }
    record.update(overrides)
    return record


def test_chord_accepts_valid_name() -> None:
    """A chord with a valid name should be accepted."""
    chord = Chord(name="Cmaj7")

    assert chord.name == "Cmaj7"


@pytest.mark.parametrize("name", ["", "   "])
def test_chord_rejects_empty_name(name: str) -> None:
    """An empty chord name should be rejected."""
    with pytest.raises(ValidationError):
        Chord(name=name)


def test_chord_group_accepts_chords() -> None:
    """A chord group should accept one or more chords."""
    group = ChordGroup(
        chords=[
            Chord(name="Bm7"),
            Chord(name="Cm9"),
        ],
    )

    assert [chord.name for chord in group.chords] == ["Bm7", "Cm9"]


def test_chord_group_rejects_empty_list() -> None:
    """A chord group must contain at least one chord."""
    with pytest.raises(ValidationError):
        ChordGroup(chords=[])


def test_shared_reference_accepts_dollar_prefix() -> None:
    """A shared reference must use the $name format."""
    reference = SharedReference(shared="$turnaround")

    assert reference.shared == "$turnaround"


@pytest.mark.parametrize("reference", ["turnaround", "", "$"])
def test_shared_reference_rejects_invalid_value(reference: str) -> None:
    """Invalid shared references should be rejected."""
    with pytest.raises(ValidationError):
        SharedReference(shared=reference)


def test_shared_data_accepts_chords() -> None:
    """Shared data should contain at least one chord."""
    shared = SharedData(
        chords=[
            Chord(name="Dm7"),
            Chord(name="G7"),
        ],
    )

    assert len(shared.chords) == 2


def test_metadata_accepts_valid_values() -> None:
    """Valid metadata should be accepted."""
    metadata = Metadata(
        bpm=120,
        signature=(4, 4),
        scale="C major",
    )

    assert metadata.bpm == 120
    assert metadata.signature == (4, 4)
    assert metadata.scale == "C major"


@pytest.mark.parametrize("bpm", [0, -1])
def test_metadata_rejects_invalid_bpm(bpm: int) -> None:
    """BPM must be positive."""
    with pytest.raises(ValidationError):
        Metadata(
            bpm=bpm,
            signature=(4, 4),
            scale="C major",
        )


@pytest.mark.parametrize(
    "signature",
    [
        (0, 4),
        (4, 0),
        (-1, 4),
        (4, -1),
    ],
)
def test_metadata_rejects_invalid_signature(
    signature: tuple[int, int],
) -> None:
    """Time-signature components must be positive."""
    with pytest.raises(ValidationError):
        Metadata(
            bpm=120,
            signature=signature,
            scale="C major",
        )


def test_metadata_rejects_empty_scale() -> None:
    """An empty scale name should be rejected."""
    with pytest.raises(ValidationError):
        Metadata(
            bpm=120,
            signature=(4, 4),
            scale="   ",
        )


def test_output_accepts_chord() -> None:
    """Output should accept a normal chord."""
    output = Output.model_validate(make_output())

    item = output.progression_data[0]

    assert isinstance(item, Chord)
    assert item.name == "C"


def test_output_accepts_chord_group() -> None:
    """Output should accept a chord group."""
    output = Output.model_validate(
        make_output(
            progression_data=[
                {
                    "chords": [
                        {"name": "Gm7"},
                        {"name": "C7"},
                    ]
                }
            ]
        )
    )

    group = output.progression_data[0]

    assert isinstance(group, ChordGroup)
    assert [chord.name for chord in group.chords] == ["Gm7", "C7"]


def test_output_accepts_shared_reference() -> None:
    """Output should accept a shared reference."""
    output = Output.model_validate(
        make_output(
            shared={
                "turnaround": {
                    "chords": [
                        {"name": "Dm7"},
                        {"name": "G7"},
                    ]
                }
            },
            progression_data=[
                {"shared": "$turnaround"},
            ],
        )
    )

    reference = output.progression_data[0]

    assert isinstance(reference, SharedReference)
    assert reference.shared == "$turnaround"


def test_output_rejects_empty_progression() -> None:
    """Progression data must contain at least one item."""
    with pytest.raises(ValidationError):
        Output.model_validate(
            make_output(
                progression_data=[],
            )
        )


def test_record_accepts_valid_data() -> None:
    """A complete valid dataset record should be accepted."""
    record = DatasetRecord.model_validate(make_record())

    assert record.prompt == "Generate a progression in C major."
    assert record.output.metadata.scale == "C major"


def test_record_rejects_empty_prompt() -> None:
    """An empty prompt should be rejected."""
    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(
            make_record(prompt="   "),
        )


def test_record_requires_metadata() -> None:
    """Metadata is required."""
    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(
            {
                "prompt": "Generate a progression.",
                "output": {
                    "shared": {},
                    "progression_data": [{"name": "C"}],
                },
            }
        )


def test_record_requires_scale() -> None:
    """Scale is required by the current dataset schema."""
    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(
            {
                "prompt": "Generate a progression.",
                "output": {
                    "metadata": {
                        "bpm": 97,
                        "signature": [4, 4],
                    },
                    "shared": {},
                    "progression_data": [{"name": "C"}],
                },
            }
        )


def test_record_rejects_nested_array_item() -> None:
    """Progression items must not be accidentally wrapped in an array."""
    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(
            make_record(
                output=make_output(
                    progression_data=[
                        [
                            {
                                "chords": [
                                    {"name": "Gm7"},
                                    {"name": "C7"},
                                ]
                            }
                        ]
                    ]
                )
            )
        )