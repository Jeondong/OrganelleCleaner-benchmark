#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_TSV = """species\tgenome size (Mbp)\tgraph-only (s)\tblast-only (s)\thybrid (s)\ttotal three modes (s)
Arabidopsis_thaliana\t192.96\t11\t22\t24\t57
Carica_papaya\t340.69\t13\t25\t27\t65
Salix_wilsonii\t360.5\t13\t26\t28\t67
Oryza_sativa\t431.49\t14\t28\t30\t72
Citrullus_lanatus\t451.11\t15\t29\t31\t75
Vitis_vinifera\t512.94\t16\t31\t33\t80
Daucus_carota\t577.9\t17\t33\t35\t85
Amborella_trichopoda\t747.34\t18\t36\t38\t92
Malus_domestica\t753.74\t18\t36\t39\t93
Solanum_lycopersicum\t830.58\t19\t38\t40\t97
Glycine_max\t1023.43\t20\t40\t40\t100
"""

METHODS = ["graph-only (s)", "blast-only (s)", "hybrid (s)"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot OrganelleCleaner runtime benchmark figures from xlsx/csv/tsv."
    )
    parser.add_argument(
        "-i", "--input",
        help="Input XLSX/CSV/TSV file. If omitted, built-in example data will be used.",
        default=None
    )
    parser.add_argument(
        "-o", "--outdir",
        help="Output directory for figure files.",
        default="runtime_svgs"
    )
    return parser.parse_args()


def _standardize_columns(df):
    rename_map = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in ["species"]:
            rename_map[c] = "species"
        elif key in ["genome size (mbp)", "genome_size_mbp", "genome size", "genome_size"]:
            rename_map[c] = "genome size (Mbp)"
        elif key in ["graph-only (s)", "graph_only_s", "graph-only", "graph only (s)", "graph only"]:
            rename_map[c] = "graph-only (s)"
        elif key in ["blast-only (s)", "blast_only_s", "blast-only", "blast only (s)", "blast only"]:
            rename_map[c] = "blast-only (s)"
        elif key in ["hybrid (s)", "hybrid_s", "hybrid"]:
            rename_map[c] = "hybrid (s)"
        elif key in ["total three modes (s)", "total_three_modes_s", "total (s)", "total"]:
            rename_map[c] = "total three modes (s)"
    df = df.rename(columns=rename_map)
    return df


def load_data(input_file=None):
    if input_file:
        lower = input_file.lower()
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            df = pd.read_excel(input_file)
        else:
            df = pd.read_csv(input_file)
            if len(df.columns) == 1:
                df = pd.read_csv(input_file, sep="\t")
    else:
        df = pd.read_csv(StringIO(DEFAULT_TSV), sep="\t")

    df.columns = df.columns.str.strip()
    df = _standardize_columns(df)

    required = ["species", "genome size (Mbp)", "graph-only (s)", "blast-only (s)", "hybrid (s)", "total three modes (s)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nDetected columns: {list(df.columns)}")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col])

    df["species_label"] = df["species"].astype(str).str.replace("_", " ", regex=False)
    df = df.sort_values("genome size (Mbp)", ascending=True).reset_index(drop=True)
    return df


def italic_species_labels(ax, labels):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([rf"$\it{{{lab.replace(' ', r'\ ')}}}$" for lab in labels], fontsize=10)


def save_close(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_stacked_runtime(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(df))
    labels = df["species_label"].tolist()

    graph = df["graph-only (s)"].to_numpy()
    blast = df["blast-only (s)"].to_numpy()
    hybrid = df["hybrid (s)"].to_numpy()

    ax.barh(y, graph, label="graph-only")
    ax.barh(y, blast, left=graph, label="blast-only")
    ax.barh(y, hybrid, left=graph + blast, label="hybrid")

    italic_species_labels(ax, labels)
    ax.set_xlabel("Runtime (s)", fontsize=12)
    ax.set_title("Runtime across three modes", fontsize=13)
    ax.legend(frameon=False)

    save_close(fig, os.path.join(outdir, "runtime_stacked_by_species.svg"))


def plot_total_vs_genome(df, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))

    x = df["genome size (Mbp)"].to_numpy()
    y = df["total three modes (s)"].to_numpy()

    ax.scatter(x, y)
    for _, row in df.iterrows():
        ax.annotate(
            row["species_label"].replace(" ", "\n"),
            (row["genome size (Mbp)"], row["total three modes (s)"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8
        )

    coef = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 100)
    yfit = coef[0] * xfit + coef[1]
    ax.plot(xfit, yfit, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Genome size (Mbp)", fontsize=12)
    ax.set_ylabel("Total runtime for all three modes (s)", fontsize=12)
    ax.set_title("Total runtime scales with genome size", fontsize=13)

    save_close(fig, os.path.join(outdir, "total_runtime_vs_genome_size.svg"))


def plot_mode_vs_genome(df, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))

    x = df["genome size (Mbp)"].to_numpy()
    for method in METHODS:
        ax.plot(x, df[method].to_numpy(), marker="o", linewidth=1.8, label=method.replace(" (s)", ""))

    ax.set_xlabel("Genome size (Mbp)", fontsize=12)
    ax.set_ylabel("Runtime (s)", fontsize=12)
    ax.set_title("Per-mode runtime vs. genome size", fontsize=13)
    ax.legend(frameon=False)

    save_close(fig, os.path.join(outdir, "mode_runtime_vs_genome_size.svg"))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.input)

    plot_stacked_runtime(df, args.outdir)
    plot_total_vs_genome(df, args.outdir)
    plot_mode_vs_genome(df, args.outdir)

    print(f"[OK] Figures saved in: {args.outdir}")


if __name__ == "__main__":
    main()
