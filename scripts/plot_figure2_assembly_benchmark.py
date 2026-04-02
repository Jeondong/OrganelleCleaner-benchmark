#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_TSV = """species\tinitial_busco\tgraph_busco\tblast_busco\thybrid_busco\tinitial_contigs\tgraph_contigs\tblast_contigs\thybrid_contigs\tinitial_size\tgraph_size\tblast_size\thybrid_size
Amborella_trichopoda\t99.10%\t99.10%\t99.10%\t99.10%\t545\t512\t349\t150\t747343385\t744918533\t732164603\t724030252
Arabidopsis_thaliana\t99.40%\t99.40%\t99.40%\t99.40%\t1558\t1533\t1451\t791\t192963338\t191050972\t186224673\t163834661
Carica_papaya\t99.30%\t99.30%\t99.30%\t99.30%\t210\t194\t180\t86\t340694735\t339295472\t338605927\t334877314
Citrullus_lanatus\t99.30%\t99.30%\t99.30%\t99.30%\t1373\t1305\t1202\t658\t451111037\t445392545\t438139146\t419581028
Daucus_carota\t99.70%\t99.70%\t99.70%\t99.70%\t2174\t2056\t1740\t1501\t577898321\t568925427\t547084339\t536963666
Glycine_max\t99.90%\t99.90%\t99.90%\t99.90%\t237\t214\t200\t120\t1023434414\t1021621364\t1021076269\t1017962643
Malus_domestica\t99.60%\t99.60%\t99.60%\t99.60%\t1623\t1609\t1622\t1524\t753739991\t752553887\t753635688\t750709477
Oryza_sativa\t99.90%\t99.90%\t99.90%\t99.90%\t595\t569\t421\t125\t431485722\t429489695\t419448837\t407413746
Salix_wilsonii\t99.30%\t99.30%\t99.30%\t99.30%\t118\t110\t114\t99\t360498294\t359854945\t360264315\t359920752
Solanum_lycopersicum\t97.90%\t97.90%\t97.90%\t97.90%\t1305\t1241\t1256\t551\t830577855\t825666957\t827352659\t807262817
Vitis_vinifera\t99.90%\t99.90%\t99.90%\t99.90%\t334\t320\t310\t82\t512940333\t511676517\t511512547\t503245655
"""


METHODS = ["initial", "graph", "blast", "hybrid"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot OrganelleCleaner benchmark figures and save as SVG."
    )
    parser.add_argument(
        "-i", "--input",
        help="Input CSV/TSV file. If omitted, built-in example data will be used.",
        default=None
    )
    parser.add_argument(
        "-o", "--outdir",
        help="Output directory for SVG files.",
        default="benchmark_svgs"
    )
    return parser.parse_args()


def load_data(input_file=None):
    if input_file:
        # Try normal CSV first; if everything collapses into one column, fall back to TSV.
        df = pd.read_csv(input_file)
        if len(df.columns) == 1:
            df = pd.read_csv(input_file, sep="\t")
    else:
        df = pd.read_csv(StringIO(DEFAULT_TSV), sep="\t")

    df.columns = df.columns.str.strip()

    # percent columns -> float
    busco_cols = [f"{m}_busco" for m in METHODS]
    for col in busco_cols:
        df[col] = df[col].astype(str).str.replace("%", "", regex=False).astype(float)

    # numeric columns
    numeric_cols = []
    for m in METHODS:
        numeric_cols.append(f"{m}_contigs")
        numeric_cols.append(f"{m}_size")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # species label: replace _ with space
    df["species_label"] = df["species"].str.replace("_", " ", regex=False)

    # sort by hybrid contigs reduction for better visualization
    df["hybrid_contig_reduction_pct"] = (
        (df["initial_contigs"] - df["hybrid_contigs"]) / df["initial_contigs"] * 100
    )
    df = df.sort_values("hybrid_contig_reduction_pct", ascending=True).reset_index(drop=True)

    return df


def italic_species_labels(ax, labels):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([rf"$\it{{{lab.replace(' ', r'\ ')}}}$" for lab in labels], fontsize=10)


def save_close(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_contigs_absolute(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))

    y = np.arange(len(df))
    labels = df["species_label"].tolist()

    for i, row in df.iterrows():
        xvals = [row[f"{m}_contigs"] for m in METHODS]
        ax.plot(xvals, [i] * 4, marker="o", linewidth=1.8)

    ax.set_xlabel("Number of contigs", fontsize=12)
    ax.set_title("Contig counts across methods", fontsize=13)
    ax.set_xticks([])  # methods are not on x-axis here; values are x-axis
    italic_species_labels(ax, labels)

    # method annotations at top-right as legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='-', label='initial'),
        Line2D([0], [0], marker='o', linestyle='-', label='graph'),
        Line2D([0], [0], marker='o', linestyle='-', label='blast'),
        Line2D([0], [0], marker='o', linestyle='-', label='hybrid'),
    ]
    # A simple text legend is better since colors are auto-cycled per species, not per method
    ax.text(
        0.99, 0.02,
        "Each horizontal line = one species\nPoints ordered as: initial → graph → blast → hybrid",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9
    )

    save_close(fig, os.path.join(outdir, "contigs_absolute.svg"))


def plot_contigs_reduction(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = df["species_label"].tolist()
    y = np.arange(len(df))

    graph_red = (df["initial_contigs"] - df["graph_contigs"]) / df["initial_contigs"] * 100
    blast_red = (df["initial_contigs"] - df["blast_contigs"]) / df["initial_contigs"] * 100
    hybrid_red = (df["initial_contigs"] - df["hybrid_contigs"]) / df["initial_contigs"] * 100

    h = 0.22
    ax.barh(y - h, graph_red, height=h, label="graph")
    ax.barh(y,     blast_red, height=h, label="blast")
    ax.barh(y + h, hybrid_red, height=h, label="hybrid")

    ax.set_xlabel("Contig reduction (%) vs. initial", fontsize=12)
    ax.set_title("Contig reduction by method", fontsize=13)
    italic_species_labels(ax, labels)
    ax.legend(frameon=False)

    save_close(fig, os.path.join(outdir, "contigs_reduction_percent.svg"))


def plot_size_absolute(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))

    y = np.arange(len(df))
    labels = df["species_label"].tolist()

    for i, row in df.iterrows():
        xvals = [row[f"{m}_size"] / 1e6 for m in METHODS]  # Mb
        ax.plot(xvals, [i] * 4, marker="o", linewidth=1.8)

    ax.set_xlabel("Assembly size (Mb)", fontsize=12)
    ax.set_title("Assembly size across methods", fontsize=13)
    ax.set_xticks([])
    italic_species_labels(ax, labels)

    ax.text(
        0.99, 0.02,
        "Each horizontal line = one species\nPoints ordered as: initial → graph → blast → hybrid",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9
    )

    save_close(fig, os.path.join(outdir, "assembly_size_absolute.svg"))


def plot_size_reduction(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = df["species_label"].tolist()
    y = np.arange(len(df))

    graph_red = (df["initial_size"] - df["graph_size"]) / df["initial_size"] * 100
    blast_red = (df["initial_size"] - df["blast_size"]) / df["initial_size"] * 100
    hybrid_red = (df["initial_size"] - df["hybrid_size"]) / df["initial_size"] * 100

    h = 0.22
    ax.barh(y - h, graph_red, height=h, label="graph")
    ax.barh(y,     blast_red, height=h, label="blast")
    ax.barh(y + h, hybrid_red, height=h, label="hybrid")

    ax.set_xlabel("Assembly size reduction (%) vs. initial", fontsize=12)
    ax.set_title("Assembly size reduction by method", fontsize=13)
    italic_species_labels(ax, labels)
    ax.legend(frameon=False)

    save_close(fig, os.path.join(outdir, "assembly_size_reduction_percent.svg"))


def plot_busco_heatmap(df, outdir):
    fig, ax = plt.subplots(figsize=(6, 6))

    heat = df[[f"{m}_busco" for m in METHODS]].to_numpy()
    im = ax.imshow(heat, aspect="auto")

    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels(METHODS, fontsize=10)
    italic_species_labels(ax, df["species_label"].tolist())
    ax.set_title("BUSCO completeness (%)", fontsize=13)

    # annotate values
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("BUSCO (%)", fontsize=10)

    save_close(fig, os.path.join(outdir, "busco_heatmap.svg"))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.input)

    plot_contigs_absolute(df, args.outdir)
    plot_contigs_reduction(df, args.outdir)
    plot_size_absolute(df, args.outdir)
    plot_size_reduction(df, args.outdir)
    plot_busco_heatmap(df, args.outdir)

    print(f"[OK] SVG files saved in: {args.outdir}")


if __name__ == "__main__":
    main()
