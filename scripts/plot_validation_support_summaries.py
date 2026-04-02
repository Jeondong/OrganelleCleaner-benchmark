#!/usr/bin/env python3
"""Create publication-quality SVG figures from integrated validation TSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("/jdh/SRA/hifiasm_runs/validation_integrated")
SUMMARY_PATH = BASE_DIR / "validation_integrated_summary.tsv"
MODE_SUMMARY_PATH = BASE_DIR / "validation_integrated_mode_summary.tsv"
SPECIES_SUMMARY_PATH = BASE_DIR / "validation_integrated_species_summary.tsv"
LONG_PATH = BASE_DIR / "validation_integrated_long.tsv"

MODE_ORDER = ["graph", "blast", "hybrid"]
MODE_LABELS = {"graph": "Graph-only", "blast": "BLAST-only", "hybrid": "Hybrid"}

COLORS = {
    "dual": "#1f3b5d",
    "retained_dual": "#9c2f2f",
    "sequence": "#4c78a8",
    "topology": "#59a14f",
    "strong_topology": "#2f6b3f",
}


def configure_style() -> None:
    """Apply a minimal scientific visual style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def require_columns(dataframe: pd.DataFrame, required: Iterable[str], label: str) -> None:
    """Raise a helpful error if required columns are missing."""
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {', '.join(missing)}. "
            f"Available columns: {', '.join(dataframe.columns)}"
        )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all integrated validation tables."""
    for path in [SUMMARY_PATH, MODE_SUMMARY_PATH, SPECIES_SUMMARY_PATH, LONG_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    summary_df = pd.read_csv(SUMMARY_PATH, sep="\t")
    mode_summary_df = pd.read_csv(MODE_SUMMARY_PATH, sep="\t")
    species_summary_df = pd.read_csv(SPECIES_SUMMARY_PATH, sep="\t")
    long_df = pd.read_csv(LONG_PATH, sep="\t")

    require_columns(
        summary_df,
        [
            "species",
            "mode",
            "removed_dual_supported_pct",
            "removed_sequence_supported_pct",
            "removed_topology_supported_pct",
            "removed_strong_topology_supported_pct",
            "retained_dual_supported_pct",
            "retained_sequence_supported_pct",
            "retained_topology_supported_pct",
            "retained_strong_topology_supported_pct",
        ],
        SUMMARY_PATH.name,
    )
    require_columns(
        mode_summary_df,
        [
            "mode",
            "removed_dual_supported_pct",
            "retained_dual_supported_pct",
            "removed_sequence_supported_pct",
            "removed_topology_supported_pct",
            "removed_strong_topology_supported_pct",
            "retained_sequence_supported_pct",
            "retained_topology_supported_pct",
            "retained_strong_topology_supported_pct",
        ],
        MODE_SUMMARY_PATH.name,
    )
    require_columns(species_summary_df, ["species"], SPECIES_SUMMARY_PATH.name)
    require_columns(long_df, ["species", "mode", "contig_id"], LONG_PATH.name)

    return summary_df, mode_summary_df, species_summary_df, long_df


def standardize_mode_order(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted in the desired mode order."""
    ordered = dataframe.copy()
    ordered["mode"] = pd.Categorical(ordered["mode"], categories=MODE_ORDER, ordered=True)
    ordered = ordered.sort_values("mode").reset_index(drop=True)
    return ordered


def format_mode_labels(modes: Sequence[str]) -> list[str]:
    """Return display labels for mode names."""
    return [MODE_LABELS.get(mode, str(mode)) for mode in modes]


def style_axis(ax: plt.Axes, ylabel: str, title: str, ylim: tuple[float, float] | None = None) -> None:
    """Apply consistent axis formatting."""
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14, weight="bold")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Write a figure as SVG with tight layout."""
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def grouped_bar_chart(
    dataframe: pd.DataFrame,
    x_column: str,
    series_columns: Sequence[str],
    series_labels: Sequence[str],
    colors: Sequence[str],
    ylabel: str,
    title: str,
    output_path: Path,
    rotate_xticks: bool = False,
    xtick_labels: Sequence[str] | None = None,
    y_max: float | None = None,
) -> None:
    """Create a grouped bar chart with publication-friendly sizing."""
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x_values = list(range(len(dataframe)))
    n_series = len(series_columns)
    bar_width = 0.72 / n_series
    offsets = [((index - (n_series - 1) / 2) * bar_width) for index in range(n_series)]

    for offset, column, label, color in zip(offsets, series_columns, series_labels, colors):
        ax.bar(
            [x + offset for x in x_values],
            dataframe[column],
            width=bar_width,
            label=label,
            color=color,
            edgecolor="none",
        )

    ax.set_xticks(x_values)
    labels = list(xtick_labels) if xtick_labels is not None else dataframe[x_column].astype(str).tolist()
    ax.set_xticklabels(labels, rotation=35 if rotate_xticks else 0, ha="right" if rotate_xticks else "center")
    upper = y_max if y_max is not None else max(100.0, float(dataframe[list(series_columns)].max().max()) * 1.08)
    style_axis(ax, ylabel=ylabel, title=title, ylim=(0, upper))
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_path)


def single_bar_chart(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    color: str,
    ylabel: str,
    title: str,
    output_path: Path,
    rotate_xticks: bool = True,
) -> None:
    """Create a single-series bar chart."""
    fig_width = max(10.5, len(dataframe) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))
    x_values = list(range(len(dataframe)))
    ax.bar(x_values, dataframe[y_column], width=0.72, color=color, edgecolor="none")
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        dataframe[x_column].astype(str).tolist(),
        rotation=40 if rotate_xticks else 0,
        ha="right" if rotate_xticks else "center",
    )
    upper = max(100.0, float(dataframe[y_column].max()) * 1.08 if len(dataframe) else 100.0)
    style_axis(ax, ylabel=ylabel, title=title, ylim=(0, upper))
    save_figure(fig, output_path)


def build_figures(summary_df: pd.DataFrame, mode_summary_df: pd.DataFrame) -> list[Path]:
    """Create all requested SVG figures."""
    written: list[Path] = []
    mode_summary_df = standardize_mode_order(mode_summary_df)
    mode_labels = format_mode_labels(mode_summary_df["mode"].astype(str).tolist())

    figure_specs = [
        {
            "filename": "mode_dual_support_summary.svg",
            "columns": ["removed_dual_supported_pct", "retained_dual_supported_pct"],
            "labels": ["Removed: dual-supported", "Retained: dual-supported"],
            "colors": [COLORS["dual"], COLORS["retained_dual"]],
            "title": "Dual-Supported Contigs by OrganelleCleaner Mode",
        },
        {
            "filename": "mode_support_composition_removed.svg",
            "columns": [
                "removed_sequence_supported_pct",
                "removed_topology_supported_pct",
                "removed_strong_topology_supported_pct",
                "removed_dual_supported_pct",
            ],
            "labels": [
                "Sequence-supported",
                "Topology-supported",
                "Strong topology-supported",
                "Dual-supported",
            ],
            "colors": [
                COLORS["sequence"],
                COLORS["topology"],
                COLORS["strong_topology"],
                COLORS["dual"],
            ],
            "title": "Support Composition Among Removed Contigs",
        },
        {
            "filename": "mode_support_composition_retained.svg",
            "columns": [
                "retained_sequence_supported_pct",
                "retained_topology_supported_pct",
                "retained_strong_topology_supported_pct",
                "retained_dual_supported_pct",
            ],
            "labels": [
                "Sequence-supported",
                "Topology-supported",
                "Strong topology-supported",
                "Dual-supported",
            ],
            "colors": [
                COLORS["sequence"],
                COLORS["topology"],
                COLORS["strong_topology"],
                COLORS["retained_dual"],
            ],
            "title": "Support Composition Among Retained Contigs",
        },
        {
            "filename": "mode_sequence_vs_topology_removed.svg",
            "columns": [
                "removed_sequence_supported_pct",
                "removed_topology_supported_pct",
            ],
            "labels": ["Sequence-supported", "Topology-supported"],
            "colors": [COLORS["sequence"], COLORS["topology"]],
            "title": "Sequence vs Topology Support Among Removed Contigs",
        },
    ]

    for spec in figure_specs:
        output_path = BASE_DIR / spec["filename"]
        grouped_bar_chart(
            dataframe=mode_summary_df,
            x_column="mode",
            series_columns=spec["columns"],
            series_labels=spec["labels"],
            colors=spec["colors"],
            ylabel="Contigs (%)",
            title=spec["title"],
            output_path=output_path,
            xtick_labels=mode_labels,
            y_max=100.0,
        )
        written.append(output_path)

    hybrid_df = summary_df[summary_df["mode"].astype(str) == "hybrid"].copy()
    if hybrid_df.empty:
        raise ValueError("validation_integrated_summary.tsv contains no rows for mode == 'hybrid'.")

    retained_sorted = hybrid_df.sort_values(
        "retained_dual_supported_pct", ascending=False
    ).reset_index(drop=True)
    removed_sorted = hybrid_df.sort_values(
        "removed_dual_supported_pct", ascending=False
    ).reset_index(drop=True)

    retained_output = BASE_DIR / "species_retained_dual_supported_pct.svg"
    single_bar_chart(
        dataframe=retained_sorted,
        x_column="species",
        y_column="retained_dual_supported_pct",
        color=COLORS["retained_dual"],
        ylabel="Hybrid retained dual-supported contigs (%)",
        title="Species-Specific Dual-Supported Retained Contigs in Hybrid Mode",
        output_path=retained_output,
        rotate_xticks=True,
    )
    written.append(retained_output)

    removed_output = BASE_DIR / "species_removed_dual_supported_pct.svg"
    single_bar_chart(
        dataframe=removed_sorted,
        x_column="species",
        y_column="removed_dual_supported_pct",
        color=COLORS["dual"],
        ylabel="Hybrid removed dual-supported contigs (%)",
        title="Species-Specific Dual-Supported Removed Contigs in Hybrid Mode",
        output_path=removed_output,
        rotate_xticks=True,
    )
    written.append(removed_output)

    return written


def main() -> int:
    """Run the plotting workflow and print written SVG paths."""
    configure_style()
    summary_df, mode_summary_df, species_summary_df, long_df = load_inputs()
    _ = species_summary_df, long_df
    written = build_figures(summary_df, mode_summary_df)

    print("Wrote SVG figures:")
    for path in written:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
