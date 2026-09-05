"""Command-line interface for chord-dataset."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .audit import audit_file, format_diagnostic
from .convert import convert_jsonl
from .dedupe import (
    dedupe_file,
    find_duplicate_prompts,
    write_deduplicated,
)
from .export import export_jsonl
from .formats import OutputFormat
from .normalize import normalize_file
from .split import load_and_split, write_split
from .stats import collect_stats
from .validator import ValidationError, validate_file

console = Console()


@click.group()
def main() -> None:
    """Manage and validate chord progression datasets."""


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(path: Path) -> None:
    """Validate a JSONL dataset."""
    try:
        result = validate_file(path)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc

    console.print("[green]✓ Dataset is valid[/green]")
    console.print()
    console.print(f"Records: {result.valid_records}")
    console.print(f"Lines checked: {result.total_lines}")


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def audit(path: Path) -> None:
    """Audit dataset quality and report inconsistencies."""
    try:
        result = audit_file(path)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc

    console.print("[bold]Dataset Audit[/bold]")
    console.print()
    console.print(f"Records: {result.records}")
    console.print(f"Issues: {len(result.issues)}")
    console.print(f"Exact duplicate groups: {len(result.duplicates.groups)}")
    console.print(f"Duplicate records: {result.duplicates.duplicate_records}")
    console.print(f"Duplicate prompt groups: {len(result.duplicate_prompts)}")

    if result.issues:
        console.print()
        console.print("[yellow]Issues[/yellow]")
        for issue in result.issues:
            console.print(
                format_diagnostic(
                    path,
                    line=issue.line_number,
                    column=1,
                    message=issue.message,
                )
            )

    if result.duplicates.groups:
        console.print()
        console.print("[yellow]Exact duplicate groups[/yellow]")
        for group in result.duplicates.groups:
            console.print(f"  lines {', '.join(map(str, group.lines))}")

    if result.duplicate_prompts:
        console.print()
        console.print("[yellow]Duplicate prompt groups[/yellow]")
        for group in result.duplicate_prompts:
            console.print(f"  lines {', '.join(map(str, group.lines))}")
            console.print(f"    {group.prompt}")

    if result.passed:
        console.print()
        console.print("[green]✓ Dataset audit passed[/green]")
    else:
        raise click.exceptions.Exit(1)


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def stats(path: Path) -> None:
    """Display statistics for a JSONL dataset."""
    try:
        result = collect_stats(path)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc

    console.print("[bold]Dataset Statistics[/bold]")
    console.print()
    summary = Table(show_header=False, box=None)
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Records", str(result.records))
    summary.add_row("Total chords", str(result.total_chords))
    summary.add_row("Chord groups", str(result.chord_groups))
    summary.add_row("Shared references", str(result.shared_references))
    summary.add_row("Shared structures", str(result.shared_structures))
    summary.add_row("Average BPM", f"{result.average_bpm:.1f}")
    summary.add_row("BPM range", f"{result.min_bpm}–{result.max_bpm}")
    console.print(summary)
    console.print()
    scale_table = Table(title="Scales")
    scale_table.add_column("Scale")
    scale_table.add_column("Records", justify="right")
    for scale, count in result.scales.most_common():
        scale_table.add_row(scale, str(count))
    console.print(scale_table)
    console.print()
    signature_table = Table(title="Time Signatures")
    signature_table.add_column("Signature")
    signature_table.add_column("Records", justify="right")
    for (numerator, denominator), count in result.signatures.most_common():
        signature_table.add_row(f"{numerator}/{denominator}", str(count))
    console.print(signature_table)
    console.print()
    prompt_table = Table(title="Prompt Starters")
    prompt_table.add_column("Starter")
    prompt_table.add_column("Records", justify="right")
    for starter, count in result.prompt_starters.most_common():
        prompt_table.add_row(starter, str(count))
    console.print(prompt_table)
    console.print()
    operation_table = Table(title="Prompt Operations")
    operation_table.add_column("Operation")
    operation_table.add_column("Records", justify="right")
    for operation, count in result.operations.most_common():
        operation_table.add_row(operation.value, str(count))
    console.print(operation_table)


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--train-ratio",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.8,
    show_default=True,
)
@click.option(
    "--validation-ratio",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.1,
    show_default=True,
)
@click.option("--seed", type=int, default=42, show_default=True)
def split(
    path: Path, output_dir: Path, train_ratio: float, validation_ratio: float, seed: int
) -> None:
    """Split a JSONL dataset into train, validation, and test sets."""
    try:
        dataset = load_and_split(
            path, train_ratio=train_ratio, validation_ratio=validation_ratio, seed=seed
        )
        write_split(dataset, output_dir)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print("[green]✓ Dataset split successfully[/green]")
    console.print()
    console.print(f"Train: {len(dataset.train)}")
    console.print(f"Validation: {len(dataset.validation)}")
    console.print(f"Test: {len(dataset.test)}")
    console.print(f"Total: {dataset.total}")


@main.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("destination", type=click.Path(dir_okay=False, path_type=Path))
def export(source: Path, destination: Path) -> None:
    """Validate and export a JSONL dataset."""
    try:
        exported = export_jsonl(source, destination)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print("[green]✓ Dataset exported successfully[/green]")
    console.print(f"Records: {exported}")
    console.print(f"Output: {destination}")


@main.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("destination", type=click.Path(dir_okay=False, path_type=Path))
def normalize(source: Path, destination: Path) -> None:
    """Normalize a JSONL dataset."""
    try:
        result = normalize_file(source, destination)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print("[green]✓ Dataset normalized successfully[/green]")
    console.print(f"Records: {result.records}")
    console.print(f"Changed: {result.changed_records}")
    console.print(f"Output: {destination}")


@main.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("destination", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(
        [output_format.value for output_format in OutputFormat], case_sensitive=False
    ),
    default=OutputFormat.STRUCTURED.value,
    show_default=True,
)
def convert(source: Path, destination: Path, output_format: str) -> None:
    """Convert a dataset into a training-oriented JSONL format."""
    try:
        result = convert_jsonl(
            source, destination, output_format=OutputFormat(output_format.lower())
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print("[green]✓ Dataset converted successfully[/green]")
    console.print(f"Records: {result.records}")
    console.print(f"Format: {result.output_format.value}")
    console.print(f"Output: {destination}")


@main.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--write",
    "destination",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the deduplicated dataset to this file.",
)
def dedupe(source: Path, destination: Path | None) -> None:
    """Find and optionally remove exact duplicate records."""
    result = dedupe_file(source)
    click.echo("Exact duplicates found")
    click.echo(f"Records: {result.records}")
    click.echo(f"Unique records: {result.unique_records}")
    click.echo(f"Duplicate records: {result.duplicate_records}")
    if result.groups:
        click.echo("Duplicate groups:")
        for group in result.groups:
            click.echo(f"  lines {', '.join(map(str, group.lines))}")
    if destination is not None:
        write_deduplicated(source, destination)
        click.echo(f"Deduplicated dataset written to {destination}")
        click.echo(f"Records written: {result.unique_records}")


@main.command("dedupe-prompts")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def dedupe_prompts(source: Path) -> None:
    """Find records with duplicate normalized prompts."""
    try:
        groups = find_duplicate_prompts(source)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    if not groups:
        console.print("[green]✓ No duplicate prompts found[/green]")
        return
    console.print("[yellow]Duplicate prompts found[/yellow]")
    console.print(f"Groups: {len(groups)}")
    for index, group in enumerate(groups, start=1):
        lines = ", ".join(str(line) for line in group.lines)
        console.print(f"Group {index}: lines {lines}")
        console.print(f"  {group.prompt}")


if __name__ == "__main__":
    main()
