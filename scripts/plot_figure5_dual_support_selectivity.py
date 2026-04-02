#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

BASE_DIR = Path("/jdh/SRA/hifiasm_runs/validation_integrated")
MODE_SUMMARY_PATH = BASE_DIR / "validation_integrated_mode_summary.tsv"
OUTPUT_PATH = BASE_DIR / "figure2_dual_supported_removed_retained.svg"

MODE_ORDER = ["blast", "hybrid"]
MODE_LABELS = {
    "blast": "BLAST-only",
    "hybrid": "Hybrid",
}

COLORS = {
    "blast": "#5b84b1",
    "hybrid": "#1f3b5d",
}

def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })

def add_value_labels(ax, bars, offset=1.2, fontsize=10):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

def main():
    configure_style()

    df = pd.read_csv(MODE_SUMMARY_PATH, sep="\t")

    required = [
        "mode",
        "removed_dual_supported_pct",
        "retained_dual_supported_pct",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Graph-only 제외: dual support는 sequence evidence가 있어야 하므로 비교 부적절
    df = df[df["mode"].isin(MODE_ORDER)].copy()
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    df = df.sort_values("mode").reset_index(drop=True)

    if df.empty or len(df) != 2:
        raise ValueError("Expected BLAST-only and Hybrid rows in validation_integrated_mode_summary.tsv")

    x = range(len(df))
    colors = [COLORS[m] for m in df["mode"].astype(str)]
    xticklabels = [MODE_LABELS[m] for m in df["mode"].astype(str)]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))

    # Panel A: Removed
    ax = axes[0]
    removed_vals = df["removed_dual_supported_pct"].tolist()
    bars = ax.bar(x, removed_vals, color=colors, width=0.62, edgecolor="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Dual-supported contigs within set (%)")
    ax.set_title("Removed set", weight="bold", pad=10)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    add_value_labels(ax, bars, offset=1.2, fontsize=10)

    # Panel B: Retained
    ax = axes[1]
    retained_vals = df["retained_dual_supported_pct"].tolist()
    bars = ax.bar(x, retained_vals, color=colors, width=0.62, edgecolor="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(0, 40)
    ax.set_ylabel("Dual-supported contigs within set (%)")
    ax.set_title("Retained set", weight="bold", pad=10)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    add_value_labels(ax, bars, offset=0.8, fontsize=10)

    # inset for low retained values
    axins = inset_axes(ax, width="42%", height="42%", loc="upper right", borderpad=1.6)
    inset_bars = axins.bar(x, retained_vals, color=colors, width=0.62, edgecolor="none")
    axins.set_ylim(0, 5)
    axins.set_xticks(list(x))
    axins.set_xticklabels(["BLAST", "Hybrid"], fontsize=8)
    axins.tick_params(axis="y", labelsize=8)
    axins.grid(axis="y", color="#e3e3e3", linewidth=0.6, alpha=0.8)
    axins.set_title("Zoom: 0–5%", fontsize=9, pad=4)

    # overall title
    fig.suptitle(
        "Hybrid mode selectively removes dual-supported contigs",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
