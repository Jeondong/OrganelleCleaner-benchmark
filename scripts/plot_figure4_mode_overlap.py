#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


CATEGORY_ORDER = [
    "graph_only",
    "blast_only",
    "graph_blast_only",
    "graph_hybrid_only",
    "blast_hybrid_only",
    "all_three",
    "hybrid_only",
]

CATEGORY_COLORS = {
    "graph_only": "#4C78A8",
    "blast_only": "#E45756",
    "graph_blast_only": "#B279A2",
    "graph_hybrid_only": "#72B7B2",
    "blast_hybrid_only": "#F58518",
    "all_three": "#7F7F7F",
    "hybrid_only": "#54A24B",
}

CATEGORY_LABELS = {
    "graph_only": "Graph only",
    "blast_only": "BLAST only",
    "graph_blast_only": "Graph + BLAST",
    "graph_hybrid_only": "Graph + Hybrid",
    "blast_hybrid_only": "BLAST + Hybrid",
    "all_three": "All three",
    "hybrid_only": "Hybrid only",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Figure 4 stacked overlap summaries as SVG."
    )
    parser.add_argument(
        "--plot_table",
        required=True,
        help="Path to figure4_plot_table.tsv",
    )
    parser.add_argument(
        "--species_summary",
        required=True,
        help="Path to species_method_overlap_summary.tsv",
    )
    parser.add_argument(
        "--overall_summary",
        required=True,
        help="Path to overall_method_overlap_summary.tsv",
    )
    parser.add_argument(
        "--out",
        default="figure4_method_overlap.svg",
        help="Output SVG filename",
    )
    return parser.parse_args()


def italic_species(name: str) -> str:
    parts = name.replace("_", " ").split()
    return r"$\it{" + r"\ ".join(parts) + "}$"


def load_species_order(species_summary_path: str):
    df = pd.read_csv(species_summary_path, sep="\t")
    required = {"species", "hybrid_only_n", "union_all_n"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"species summary missing required columns: {sorted(missing)}")

    # Sort by hybrid_only descending, then union descending
    df = df.sort_values(
        ["hybrid_only_n", "union_all_n"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return df["species"].tolist(), df


def load_plot_table(plot_table_path: str, species_order):
    df = pd.read_csv(plot_table_path, sep="\t")
    required = {"species", "category", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"plot table missing required columns: {sorted(missing)}")

    df = df[df["category"].isin(CATEGORY_ORDER)].copy()
    df["species"] = pd.Categorical(df["species"], categories=species_order, ordered=True)
    df = df.sort_values(["species"])

    pivot = df.pivot_table(
        index="species",
        columns="category",
        values="count",
        fill_value=0
    )

    for cat in CATEGORY_ORDER:
        if cat not in pivot.columns:
            pivot[cat] = 0

    pivot = pivot[CATEGORY_ORDER]
    pivot = pivot.loc[species_order]

    return pivot


def load_overall_counts(overall_summary_path: str):
    df = pd.read_csv(overall_summary_path, sep="\t")
    if df.shape[0] != 1:
        raise ValueError("overall summary should have exactly one row")

    row = df.iloc[0]

    overall_counts = {
        "graph_only": int(row["graph_only_n"]),
        "blast_only": int(row["blast_only_n"]),
        "graph_blast_only": int(row["graph_blast_overlap_n"] - row["all_three_overlap_n"]),
        "graph_hybrid_only": int(row["graph_hybrid_overlap_n"] - row["all_three_overlap_n"]),
        "blast_hybrid_only": int(row["blast_hybrid_overlap_n"] - row["all_three_overlap_n"]),
        "all_three": int(row["all_three_overlap_n"]),
        "hybrid_only": int(row["hybrid_only_n"]),
    }

    return overall_counts


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_axisbelow(True)


def plot_species_stacked(ax, pivot_df):
    y = np.arange(len(pivot_df.index))
    left = np.zeros(len(pivot_df.index))

    for cat in CATEGORY_ORDER:
        vals = pivot_df[cat].to_numpy()
        ax.barh(
            y,
            vals,
            left=left,
            color=CATEGORY_COLORS[cat],
            edgecolor="white",
            linewidth=0.5,
            label=CATEGORY_LABELS[cat],
        )
        left += vals

    labels = [italic_species(s) for s in pivot_df.index.tolist()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of candidate organelle contigs", fontsize=12)
    ax.set_title("(a) Species-wise overlap composition", fontsize=13, loc="left")
    style_axis(ax)


def plot_overall_stacked(ax, overall_counts):
    left = 0
    total = sum(overall_counts.values())

    for cat in CATEGORY_ORDER:
        val = overall_counts[cat]
        ax.barh(
            [0],
            [val],
            left=[left],
            color=CATEGORY_COLORS[cat],
            edgecolor="white",
            linewidth=0.6,
        )
        if val > 0:
            ax.text(
                left + val / 2,
                0,
                str(val),
                ha="center",
                va="center",
                fontsize=9,
            )
        left += val

    ax.set_yticks([0])
    ax.set_yticklabels(["Overall"], fontsize=11)
    ax.set_xlabel("Number of candidate organelle contigs", fontsize=12)
    ax.set_title("(b) Overall overlap composition", fontsize=13, loc="left")
    ax.set_xlim(0, total * 1.02)
    style_axis(ax)


def main():
    args = parse_args()

    species_order, species_summary_df = load_species_order(args.species_summary)
    pivot_df = load_plot_table(args.plot_table, species_order)
    overall_counts = load_overall_counts(args.overall_summary)

    fig = plt.figure(figsize=(10.5, 8.2))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.8, 1.3], hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    plot_species_stacked(ax1, pivot_df)
    plot_overall_stacked(ax2, overall_counts)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[cat])
        for cat in CATEGORY_ORDER
    ]
    labels = [CATEGORY_LABELS[cat] for cat in CATEGORY_ORDER]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()
