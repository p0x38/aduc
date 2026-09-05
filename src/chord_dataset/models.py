"""Pydantic models for chord progression datasets."""

from typing import Annotated

from pydantic import BaseModel, Field, RootModel, field_validator


class Metadata(BaseModel):
    """Musical metadata associated with a progression."""

    bpm: Annotated[int, Field(ge=1)]
    signature: tuple[
        Annotated[int, Field(gt=0)],
        Annotated[int, Field(gt=0)],
    ]
    scale: str

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, value: str) -> str:
        """Ensure the scale name is not empty."""
        value = value.strip()

        if not value:
            raise ValueError("scale must not be empty")

        return value


class Chord(BaseModel):
    """A single chord."""

    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Ensure the chord name is not empty."""
        value = value.strip()

        if not value:
            raise ValueError("chord name must not be empty")

        return value


class ChordGroup(BaseModel):
    """A group of chords played within a subdivided musical unit."""

    chords: list[Chord] = Field(min_length=1)


class SharedReference(BaseModel):
    """A reference to a shared progression fragment."""

    shared: str

    @field_validator("shared")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        """Ensure the shared reference uses the $name format."""
        if not value.startswith("$"):
            raise ValueError("shared reference must start with '$'")

        if len(value) == 1:
            raise ValueError("shared reference must contain a name")

        return value


ChordItem = Chord | ChordGroup | SharedReference


class SharedData(BaseModel):
    """Reusable chord structure."""

    chords: list[Chord] = Field(min_length=1)


class Output(BaseModel):
    """Generated chord progression output."""

    metadata: Metadata
    shared: dict[str, SharedData]
    progression_data: list[ChordItem] = Field(min_length=1)


class DatasetRecord(BaseModel):
    """A single JSONL dataset record."""

    prompt: str
    output: Output

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Ensure the prompt is not empty."""
        value = value.strip()

        if not value:
            raise ValueError("prompt must not be empty")

        return value


class Dataset(RootModel[list[DatasetRecord]]):
    """A collection of dataset records."""