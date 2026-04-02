#!/usr/bin/env python3
"""Generate Figure 7: GC content and coverage-related features for hybrid mode."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_PATH = Path("/jdh/SRA/hifiasm_runs/validation_integrated/all_reports_combined.tsv")
OUTPUT_PATH = Path("/jdh/SRA/hifiasm_runs/validation_integrated/figure7_gc_coverage.svg")

TARGET_MODE = "hybrid"
VALID_CLASSIFICATIONS = {"organelle": "removed", "nuclear": "retained"}
COVERAGE_PRIORITY = [
    "coverage",
    "blast_merged_coverage_fraction",
    "plastid_merged_coverage_fraction",
    "mit_merged_coverage_fraction",
]
COLORS = {
    "removed": "#c44e52",
    "retained": "#4c72b0",
}
MAX_SCATTER_PER_GROUP = 10_000
HIST_BINS = 40
RANDOM_SEED = 42


def configure_style() -> None:
    """Apply publication-oriented plotting defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )


def load_filtered_data(path: Path) -> pd.DataFrame:
    """Load the combined TSV and retain only hybrid removed/retained rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Input TSV not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype="string")
    if "mode" not in df.columns:
        raise ValueError("Input TSV is missing required column: mode")
    if "classification" not in df.columns:
        raise ValueError("Input TSV is missing required column: classification")
    if "gc_content" not in df.columns:
        raise ValueError("Input TSV is missing required column: gc_content")

    filtered = df[df["mode"].astype(str).str.strip().str.lower() == TARGET_MODE].copy()
    filtered["classification"] = filtered["classification"].astype(str).str.strip().str.lower()
    filtered = filtered[filtered["classification"].isin(VALID_CLASSIFICATIONS)].copy()
    filtered["group"] = filtered["classification"].map(VALID_CLASSIFICATIONS)
    filtered["gc_content"] = pd.to_numeric(filtered["gc_content"], errors="coerce")

    if filtered.empty:
        raise ValueError(
            "No rows remain after filtering for mode == 'hybrid' and "
            "classification in {'organelle', 'nuclear'}."
        )

    return filtered


def select_coverage_column(df: pd.DataFrame) -> tuple[str, pd.Series]:
    """Choose the first usable coverage-like column by priority."""
    diagnostics: list[str] = []

    for column in COVERAGE_PRIORITY:
        if column not in df.columns:
            diagnostics.append(f"{column}: missing")
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        non_null = int(numeric.notna().sum())
        fraction = non_null / len(df) if len(df) else 0.0
        diagnostics.append(f"{column}: non-null numeric={non_null}/{len(df)} ({fraction:.1%})")

        if non_null == 0:
            continue
        if column == "coverage" and fraction < 0.5:
            continue
        return column, numeric

    diagnostic_text = "; ".join(diagnostics) if diagnostics else "no candidate columns inspected"
    raise ValueError(
        "No usable coverage-like column was found. "
        f"Inspection summary: {diagnostic_text}"
    )


def coverage_axis_label(column: str) -> str:
    """Return a human-readable axis label for the selected coverage feature."""
    label_map = {
        "coverage": "Coverage",
        "blast_merged_coverage_fraction": "BLAST merged coverage fraction",
        "plastid_merged_coverage_fraction": "Plastid merged coverage fraction",
        "mit_merged_coverage_fraction": "Mitochondrial merged coverage fraction",
    }
    return label_map.get(column, column.replace("_", " "))


def add_histogram_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_column: str,
    title: str,
    xlabel: str,
) -> None:
    """Plot overlaid density histograms and median lines for removed vs retained."""
    panel_df = df[["group", value_column]].dropna().copy()
    if panel_df.empty:
        raise ValueError(f"No usable data available for panel '{title}' from column '{value_column}'.")

    values = panel_df[value_column].astype(float)
    bins = np.histogram_bin_edges(values.to_numpy(), bins=HIST_BINS)

    ymax = 0.0
    medians: dict[str, float] = {}
    for group in ("removed", "retained"):
        group_values = panel_df.loc[panel_df["group"] == group, value_column].astype(float).to_numpy()
        if len(group_values) == 0:
            continue

        ax.hist(
            group_values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.22,
            linewidth=1.4,
            color=COLORS[group],
            edgecolor=COLORS[group],
            label=f"{group.capitalize()} (n={len(group_values):,})",
        )
        counts, _ = np.histogram(group_values, bins=bins, density=True)
        if len(counts):
            ymax = max(ymax, float(np.nanmax(counts)))
        medians[group] = float(np.nanmedian(group_values))

    if not medians:
        raise ValueError(f"No removed/retained values remained for panel '{title}'.")

    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    for group, median in medians.items():
        ax.axvline(median, color=COLORS[group], linestyle="--", linewidth=1.3, alpha=0.95)

    text_lines = [f"{group.capitalize()} median: {median:.3g}" for group, median in medians.items()]
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
    )
    ax.legend(frameon=False, loc="upper right")
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.margins(x=0.03)


def sample_scatter(group_df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Downsample scatter points per group when necessary."""
    if len(group_df) <= max_points:
        return group_df
    return group_df.sample(n=max_points, random_state=RANDOM_SEED)


def add_scatter_panel(ax: plt.Axes, df: pd.DataFrame, coverage_column: str, ylabel: str) -> None:
    """Plot GC content vs selected coverage-like feature for removed vs retained."""
    panel_df = df[["group", "gc_content", coverage_column]].dropna().copy()
    if panel_df.empty:
        raise ValueError(
            f"No usable data available for scatter panel with columns 'gc_content' and '{coverage_column}'."
        )

    panel_df["gc_content"] = panel_df["gc_content"].astype(float)
    panel_df[coverage_column] = panel_df[coverage_column].astype(float)

    scatter_counts: dict[str, int] = {}
    for group in ("removed", "retained"):
        group_df = panel_df[panel_df["group"] == group].copy()
        if group_df.empty:
            continue
        sampled = sample_scatter(group_df, MAX_SCATTER_PER_GROUP)
        scatter_counts[group] = len(sampled)
        ax.scatter(
            sampled["gc_content"],
            sampled[coverage_column],
            s=14,
            alpha=0.2,
            color=COLORS[group],
            edgecolors="none",
            label=f"{group.capitalize()} (n={len(sampled):,})",
        )

    if not scatter_counts:
        raise ValueError("No removed/retained values remained for the scatter panel.")

    ax.set_title("C. GC content vs coverage-like feature", loc="left", fontweight="bold")
    ax.set_xlabel("GC content (%)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.margins(x=0.04, y=0.05)


def build_figure(df: pd.DataFrame, coverage_column: str) -> None:
    """Create the full three-panel figure."""
    configure_style()
    ylabel = coverage_axis_label(coverage_column)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)

    add_histogram_panel(
        axes[0],
        df=df,
        value_column="gc_content",
        title="A. GC content distribution",
        xlabel="GC content (%)",
    )
    add_histogram_panel(
        axes[1],
        df=df,
        value_column=coverage_column,
        title="B. Coverage-related distribution",
        xlabel=ylabel,
    )
    add_scatter_panel(
        axes[2],
        df=df,
        coverage_column=coverage_column,
        ylabel=ylabel,
    )

    fig.suptitle(
        "Figure 7. GC content and coverage-related features (hybrid mode)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_PATH, format="svg", dpi=300)
    plt.close(fig)


def main() -> None:
    df = load_filtered_data(INPUT_PATH)
    coverage_column, numeric_coverage = select_coverage_column(df)
    df[coverage_column] = numeric_coverage

    removed_n = int((df["group"] == "removed").sum())
    retained_n = int((df["group"] == "retained").sum())

    print(f"Removed contigs: {removed_n}")
    print(f"Retained contigs: {retained_n}")
    print(f"Selected coverage column: {coverage_column}")

    build_figure(df, coverage_column)
    print(f"Wrote SVG: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
