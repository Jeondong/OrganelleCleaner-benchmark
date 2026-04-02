#!/usr/bin/env python3
"""Combine OrganelleCleaner report.tsv files into a master validation table."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT_DIR = Path("/jdh/SRA/hifiasm_runs")
OUTPUT_DIR = ROOT_DIR / "validation_integrated"
MASTER_OUTPUT = OUTPUT_DIR / "all_reports_combined.tsv"
SUMMARY_OUTPUT = OUTPUT_DIR / "all_reports_combined_summary.tsv"
REPORT_GLOB = "*/organelle_cleaner_runs/*/report.tsv"


def find_report_files(root_dir: Path) -> list[Path]:
    """Return sorted report.tsv paths matching the expected OrganelleCleaner layout."""
    return sorted(path for path in root_dir.glob(REPORT_GLOB) if path.is_file())


def extract_species_and_mode(report_path: Path, root_dir: Path) -> tuple[str, str]:
    """Extract species and mode from a report.tsv path."""
    relative_parts = report_path.relative_to(root_dir).parts
    if len(relative_parts) < 4:
        raise ValueError(f"Path does not match expected structure: {report_path}")

    species = relative_parts[0]
    if relative_parts[1] != "organelle_cleaner_runs":
        raise ValueError(f"Path does not contain organelle_cleaner_runs at expected position: {report_path}")
    mode = relative_parts[2]
    return species, mode


def format_column_diff(reference_columns: Iterable[str], other_columns: Iterable[str]) -> str:
    """Return a short summary of header differences."""
    reference_set = set(reference_columns)
    other_set = set(other_columns)
    missing = sorted(reference_set - other_set)
    extra = sorted(other_set - reference_set)
    parts: list[str] = []
    if missing:
        parts.append(f"missing={missing}")
    if extra:
        parts.append(f"extra={extra}")
    return "; ".join(parts) if parts else "no differences"


def detect_classification_column(columns: list[str]) -> str | None:
    """Find the best classification-like column name without changing source headers."""
    normalized = {column: "".join(ch for ch in column.lower() if ch.isalnum()) for column in columns}
    priorities = ["classification", "class", "contigclass"]

    for wanted in priorities:
        for column in columns:
            if normalized[column] == wanted:
                return column

    for wanted in priorities:
        for column in columns:
            if wanted in normalized[column]:
                return column
    return None


def build_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize rows and class counts per species and mode."""
    summary_rows: list[dict[str, object]] = []
    classification_col = detect_classification_column(combined_df.columns.tolist())

    for (species, mode), group_df in combined_df.groupby(["species", "mode"], sort=True, dropna=False):
        row: dict[str, object] = {
            "species": species,
            "mode": mode,
            "n_rows": int(len(group_df)),
        }
        if classification_col is None:
            row["n_organelle"] = pd.NA
            row["n_nuclear"] = pd.NA
        else:
            classification = (
                group_df[classification_col]
                .astype("string")
                .str.strip()
                .str.lower()
            )
            row["n_organelle"] = int((classification == "organelle").sum())
            row["n_nuclear"] = int((classification == "nuclear").sum())
        summary_rows.append(row)

    return pd.DataFrame(summary_rows, columns=["species", "mode", "n_rows", "n_organelle", "n_nuclear"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_paths = find_report_files(ROOT_DIR)
    if not report_paths:
        raise FileNotFoundError(f"No report.tsv files found under {ROOT_DIR} using pattern {REPORT_GLOB!r}")

    print(f"Found {len(report_paths)} report.tsv files:")
    for path in report_paths:
        print(f"  {path}")

    combined_frames: list[pd.DataFrame] = []
    headers_by_file: dict[Path, list[str]] = {}
    warnings: list[str] = []
    detected_modes: set[str] = set()
    detected_species: set[str] = set()

    for report_path in report_paths:
        try:
            species, mode = extract_species_and_mode(report_path, ROOT_DIR)
        except Exception as exc:
            warnings.append(f"WARNING: skipping path with unexpected structure: {report_path} ({exc})")
            continue

        detected_species.add(species)
        detected_modes.add(mode)

        try:
            header_df = pd.read_csv(report_path, sep="\t", nrows=0)
            headers_by_file[report_path] = header_df.columns.tolist()
        except Exception as exc:
            warnings.append(f"WARNING: failed to inspect header for {report_path}: {exc}")
            continue

        try:
            report_df = pd.read_csv(report_path, sep="\t", dtype="string")
        except Exception as exc:
            warnings.append(f"WARNING: failed to read {report_path}: {exc}")
            continue

        path_species = pd.Series([species] * len(report_df), dtype="string")
        path_mode = pd.Series([mode] * len(report_df), dtype="string")

        if "species" in report_df.columns:
            report_df["species"] = path_species
        else:
            report_df.insert(0, "species", path_species)

        if "mode" in report_df.columns:
            report_df["mode"] = path_mode
        else:
            report_df.insert(1, "mode", path_mode)

        combined_frames.append(report_df)

    if not combined_frames:
        raise RuntimeError("No readable report.tsv files were combined.")

    reference_path = next(iter(headers_by_file))
    reference_columns = headers_by_file[reference_path]
    mismatches = {
        path: format_column_diff(reference_columns, columns)
        for path, columns in headers_by_file.items()
        if columns != reference_columns
    }

    if mismatches:
        print("\nHeader differences detected relative to:")
        print(f"  {reference_path}")
        for path, diff in mismatches.items():
            print(f"  {path}: {diff}")
    else:
        print("\nNo header differences detected across readable files.")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(warning)

    combined_df = pd.concat(combined_frames, ignore_index=True, sort=False)
    summary_df = build_summary(combined_df)

    combined_df.to_csv(MASTER_OUTPUT, sep="\t", index=False, na_rep="NA")
    summary_df.to_csv(SUMMARY_OUTPUT, sep="\t", index=False, na_rep="NA")

    print("\nOutputs written:")
    print(f"  {MASTER_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")
    print("\nRun summary:")
    print(f"  number of report.tsv files combined: {len(combined_frames)}")
    print(f"  total number of rows: {len(combined_df)}")
    print(f"  detected modes: {sorted(detected_modes)}")
    print(f"  detected species: {sorted(detected_species)}")


if __name__ == "__main__":
    main()
