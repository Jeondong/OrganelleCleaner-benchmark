#!/usr/bin/env python3
"""Plot publication-style contig length distributions for organelle-cleaning validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path("/jdh/SRA/hifiasm_runs/validation_integrated")
ROOT_DIR = Path("/jdh/SRA/hifiasm_runs")
INTEGRATED_LONG_PATH = BASE_DIR / "validation_integrated_long.tsv"
INTEGRATED_SPECIES_SUMMARY_PATH = BASE_DIR / "validation_integrated_species_summary.tsv"
OUTPUT_PATH = BASE_DIR / "figure6_length_distribution_hybrid.svg"

TARGET_MODE = "hybrid"
MIN_LENGTH_BP = 1000
N_BINS = 50

MODE_DIR_MAP = {
    "graph": "graph-only",
    "blast": "blast-only",
    "hybrid": "hybrid",
}

COLORS = {
    "removed": "#ba3f4b",
    "retained": "#2f6c8f",
    "before": "#7b7b7b",
    "after": "#2f6c8f",
}


@dataclass
class ColumnSelection:
    length: Optional[str]
    mode: Optional[str]
    label: Optional[str]


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def detect_column(columns: Iterable[str], priorities: list[str], tokens: list[str]) -> Optional[str]:
    columns = list(columns)
    normalized = {column: normalize_name(column) for column in columns}

    for wanted in priorities:
        wanted_norm = normalize_name(wanted)
        for column in columns:
            if normalized[column] == wanted_norm:
                return column

    scored: list[tuple[int, str]] = []
    for column in columns:
        score = 0
        norm = normalized[column]
        for token in tokens:
            if token in norm:
                score += 1
        if score > 0:
            scored.append((score, column))

    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], item[1]))[1]


def detect_columns(df: pd.DataFrame) -> ColumnSelection:
    columns = list(df.columns)
    length_col = detect_column(
        columns,
        priorities=["contig_length", "length_bp", "contig_len", "length", "size", "seq_length"],
        tokens=["length", "size", "len", "bp"],
    )
    mode_col = detect_column(
        columns,
        priorities=["mode", "run_mode", "cleaning_mode", "method"],
        tokens=["mode", "method"],
    )
    label_col = detect_column(
        columns,
        priorities=["set_class", "classification", "class", "contig_class", "status"],
        tokens=["setclass", "classification", "class", "status"],
    )
    return ColumnSelection(length=length_col, mode=mode_col, label=label_col)


def choose_mode_base(species_dir: Path) -> Optional[Path]:
    candidate = species_dir / "organelle_cleaner_runs"
    if candidate.is_dir():
        return candidate
    if any((species_dir / mode_dir).is_dir() for mode_dir in MODE_DIR_MAP.values()):
        return species_dir
    return None


def infer_set_labels(series: pd.Series, source_column: str) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    unique_values = {value for value in normalized.unique() if value}

    if unique_values.issubset({"removed", "retained"}) and unique_values:
        return normalized

    if unique_values & {"organelle", "nuclear"}:
        # The integrated report stores biological classification, not the plotting label.
        # Organelle-classified contigs are the contigs removed by OrganelleCleaner,
        # while nuclear-classified contigs are retained after cleaning.
        mapped = normalized.map({"organelle": "removed", "nuclear": "retained"})
        return mapped

    raise ValueError(
        f"Unable to infer removed/retained labels from column '{source_column}' "
        f"with values: {sorted(unique_values)[:8]}"
    )


def filter_lengths(df: pd.DataFrame, length_col: str, set_col: str) -> pd.DataFrame:
    filtered = df.copy()
    filtered[length_col] = pd.to_numeric(filtered[length_col], errors="coerce")
    filtered = filtered.dropna(subset=[length_col, set_col]).copy()
    filtered = filtered[filtered[length_col] >= MIN_LENGTH_BP].copy()
    filtered = filtered[np.isfinite(filtered[length_col])].copy()
    filtered = filtered[filtered[length_col] > 0].copy()
    return filtered


def load_integrated_long() -> tuple[Optional[pd.DataFrame], ColumnSelection, str]:
    long_df = pd.read_csv(INTEGRATED_LONG_PATH, sep="\t", dtype=str)
    detected = detect_columns(long_df)

    if detected.mode is None or detected.label is None:
        reason = "integrated long table is missing a detectable mode or label column"
        return None, detected, reason

    if detected.length is None:
        reason = "integrated long table has no detectable contig length column"
        return None, detected, reason

    working = long_df.copy()
    working[detected.mode] = working[detected.mode].fillna("").astype(str).str.strip().str.lower()
    working = working[working[detected.mode] == TARGET_MODE].copy()
    if working.empty:
        reason = f"integrated long table contains no rows for mode '{TARGET_MODE}'"
        return None, detected, reason

    working["set_label"] = infer_set_labels(working[detected.label], detected.label)
    working = filter_lengths(working, detected.length, "set_label")
    working = working[working["set_label"].isin({"removed", "retained"})].copy()
    if working.empty:
        reason = "integrated long table has no valid hybrid rows after length and label filtering"
        return None, detected, reason

    return working, detected, "integrated long table"


def discover_species() -> list[str]:
    species_summary = pd.read_csv(INTEGRATED_SPECIES_SUMMARY_PATH, sep="\t", dtype=str)
    if "species" not in species_summary.columns:
        raise ValueError("validation_integrated_species_summary.tsv is missing the 'species' column.")
    return sorted(species_summary["species"].dropna().astype(str).unique().tolist())


def load_species_reports() -> tuple[pd.DataFrame, ColumnSelection]:
    species_names = discover_species()
    collected: list[pd.DataFrame] = []
    used_columns: Optional[ColumnSelection] = None

    for species in species_names:
        species_dir = ROOT_DIR / species
        mode_base = choose_mode_base(species_dir)
        if mode_base is None:
            continue
        report_path = mode_base / MODE_DIR_MAP[TARGET_MODE] / "report.tsv"
        if not report_path.is_file():
            continue

        report_df = pd.read_csv(report_path, sep="\t", dtype=str)
        detected = detect_columns(report_df)
        if detected.length is None or detected.label is None:
            continue

        working = report_df.copy()
        if detected.mode is not None:
            working[detected.mode] = working[detected.mode].fillna("").astype(str).str.strip().str.lower()
            working = working[working[detected.mode] == TARGET_MODE].copy()
        else:
            working["_mode"] = TARGET_MODE
            detected = ColumnSelection(length=detected.length, mode="_mode", label=detected.label)

        if working.empty:
            continue

        working["species"] = species
        working["set_label"] = infer_set_labels(working[detected.label], detected.label)
        working = filter_lengths(working, detected.length, "set_label")
        working = working[working["set_label"].isin({"removed", "retained"})].copy()
        if working.empty:
            continue

        collected.append(working[["species", detected.mode, detected.length, "set_label"]].rename(columns={
            detected.mode: "mode",
            detected.length: "length_bp",
        }))
        used_columns = detected

    if not collected or used_columns is None:
        raise RuntimeError("No usable species-level report.tsv files were found for hybrid mode.")

    combined = pd.concat(collected, ignore_index=True)
    return combined, used_columns


def gaussian_smooth(values: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def compute_density(log_lengths: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density, edges = np.histogram(log_lengths, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, gaussian_smooth(density)


def format_bp(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} Mb"
    if value >= 1_000:
        return f"{value / 1_000:.1f} kb"
    return f"{value:.0f} bp"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.color": "#d7dde3",
            "grid.linewidth": 0.7,
        }
    )


def add_distribution(ax: plt.Axes, values_bp: np.ndarray, bins: np.ndarray, color: str, label: str) -> None:
    log_values = np.log10(values_bp)
    centers, density = compute_density(log_values, bins)
    ax.fill_between(centers, density, color=color, alpha=0.18, linewidth=0)
    ax.plot(centers, density, color=color, linewidth=2.2, label=label)

    median_log = np.log10(np.median(values_bp))
    ax.axvline(median_log, color=color, linestyle="--", linewidth=1.2, alpha=0.9)


def apply_axis_format(ax: plt.Axes) -> None:
    tick_values = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8], dtype=float)
    tick_positions = np.log10(tick_values)
    tick_labels = ["1 kb", "10 kb", "100 kb", "1 Mb", "10 Mb", "100 Mb"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, axis="y", alpha=0.85)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlabel("Contig length (log10 bp)")
    ax.set_ylabel("Density")


def build_figure(df: pd.DataFrame) -> None:
    removed = df.loc[df["set_label"] == "removed", "length_bp"].to_numpy(dtype=float)
    retained = df.loc[df["set_label"] == "retained", "length_bp"].to_numpy(dtype=float)
    before = df["length_bp"].to_numpy(dtype=float)
    after = retained.copy()

    if len(removed) == 0 or len(retained) == 0:
        raise RuntimeError("Both removed and retained sets must contain at least one contig.")

    configure_style()

    combined_log = np.log10(before)
    xmin = np.floor(combined_log.min() * 10) / 10
    xmax = np.ceil(combined_log.max() * 10) / 10
    bins = np.linspace(xmin, xmax, N_BINS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)

    add_distribution(axes[0], removed, bins, COLORS["removed"], f"Removed (n={len(removed):,})")
    add_distribution(axes[0], retained, bins, COLORS["retained"], f"Retained (n={len(retained):,})")
    axes[0].set_title("A. Removed vs retained contigs", loc="left", fontweight="bold")
    apply_axis_format(axes[0])
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].text(
        0.02,
        0.98,
        (
            f"Median removed: {format_bp(np.median(removed))}\n"
            f"Median retained: {format_bp(np.median(retained))}"
        ),
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        color="#333333",
    )

    add_distribution(axes[1], before, bins, COLORS["before"], f"Before cleaning (n={len(before):,})")
    add_distribution(axes[1], after, bins, COLORS["after"], f"After cleaning (n={len(after):,})")
    axes[1].set_title("B. Before vs after cleaning", loc="left", fontweight="bold")
    apply_axis_format(axes[1])
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].text(
        0.02,
        0.98,
        (
            f"Median before: {format_bp(np.median(before))}\n"
            f"Median after: {format_bp(np.median(after))}"
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        color="#333333",
    )

    fig.savefig(OUTPUT_PATH, format="svg", dpi=300)
    plt.close(fig)


def main() -> None:
    integrated_df, integrated_detected, integrated_reason = load_integrated_long()

    print(f"Integrated long length column: {integrated_detected.length}")
    print(f"Integrated long mode column: {integrated_detected.mode}")
    print(f"Integrated long removed/retained label column: {integrated_detected.label}")

    if integrated_df is None:
        print(f"Integrated long table unsuitable: {integrated_reason}")
        plot_df, fallback_detected = load_species_reports()
        print("Using fallback source: species-level report.tsv files")
        print(f"Fallback length column: {fallback_detected.length}")
        print(f"Fallback mode column: {fallback_detected.mode}")
        print(f"Fallback removed/retained label column: {fallback_detected.label}")
    else:
        plot_df = integrated_df.rename(columns={integrated_detected.length: "length_bp", integrated_detected.mode: "mode"})
        print("Using source: validation_integrated_long.tsv")
        print(f"Length column used: {integrated_detected.length}")
        print(f"Mode column used: {integrated_detected.mode}")
        print(f"Removed/retained label column used: {integrated_detected.label}")

    removed_n = int((plot_df["set_label"] == "removed").sum())
    retained_n = int((plot_df["set_label"] == "retained").sum())
    before_n = len(plot_df)
    after_n = retained_n

    print(f"Mode plotted: {TARGET_MODE}")
    print(f"Minimum length filter: {MIN_LENGTH_BP} bp")
    print(f"Removed contigs used: {removed_n}")
    print(f"Retained contigs used: {retained_n}")
    print(f"Before cleaning contigs used: {before_n}")
    print(f"After cleaning contigs used: {after_n}")

    build_figure(plot_df)
    print(f"Wrote SVG: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
