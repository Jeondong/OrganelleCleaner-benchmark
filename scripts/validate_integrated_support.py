#!/usr/bin/env python3
"""Integrated validation of removed and retained contigs.

This script validates OrganelleCleaner outputs using both:
1) sequence evidence (BLAST-derived support)
2) topology evidence (graph/topology-derived support)

It scans species directories under a root directory, locates
OrganelleCleaner outputs for graph-only, blast-only, and hybrid modes,
and summarizes support among removed and retained contigs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


MODE_DIR_MAP = {
    "graph-only": "graph",
    "blast-only": "blast",
    "hybrid": "hybrid",
}

REQUIRED_FILES = (
    "organelle_contigs.txt",
    "nuclear_contigs.txt",
    "report.tsv",
)

SUMMARY_COLUMNS = [
    "species",
    "mode",

    "removed_n",
    "removed_sequence_supported_n",
    "removed_sequence_supported_pct",
    "removed_topology_supported_n",
    "removed_topology_supported_pct",
    "removed_strong_topology_supported_n",
    "removed_strong_topology_supported_pct",
    "removed_dual_supported_n",
    "removed_dual_supported_pct",

    "retained_n",
    "retained_sequence_supported_n",
    "retained_sequence_supported_pct",
    "retained_topology_supported_n",
    "retained_topology_supported_pct",
    "retained_strong_topology_supported_n",
    "retained_strong_topology_supported_pct",
    "retained_dual_supported_n",
    "retained_dual_supported_pct",
]

LONG_COLUMNS = [
    "species",
    "mode",
    "contig_id",
    "set_class",
    "classification",
    "blast_support_level",
    "sequence_supported",
    "topology_supported",
    "strong_topology_supported",
    "dual_supported",
    "topology_signal_count",
    "topology_strong_signal_count",
    "is_circular",
    "is_isolated",
    "is_compact_component",
    "graph_score",
    "blast_score",
    "final_score",
]

AGG_BASE_COLUMNS = [
    "removed_n",
    "removed_sequence_supported_n",
    "removed_topology_supported_n",
    "removed_strong_topology_supported_n",
    "removed_dual_supported_n",
    "retained_n",
    "retained_sequence_supported_n",
    "retained_topology_supported_n",
    "retained_strong_topology_supported_n",
    "retained_dual_supported_n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrated sequence/topology validation for OrganelleCleaner outputs."
    )
    parser.add_argument("--root", required=True, help="Root directory containing species folders.")
    parser.add_argument("--outdir", required=True, help="Output directory name or path.")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def is_hidden_or_noise(path: Path) -> bool:
    name = path.name
    if name.startswith(".") or name.startswith(".nfs"):
        return True
    if name in {"result", "__pycache__", "validation_summary"}:
        return True
    return False


def resolve_outdir(root: Path, outdir_arg: str) -> Path:
    outdir = Path(outdir_arg)
    if not outdir.is_absolute():
        outdir = root / outdir
    return outdir


def choose_mode_base(species_dir: Path) -> Optional[Path]:
    pattern_a = species_dir / "organelle_cleaner_runs"
    if pattern_a.is_dir():
        return pattern_a
    if any((species_dir / subdir).is_dir() for subdir in MODE_DIR_MAP):
        return species_dir
    return None


def read_contig_list(path: Path) -> List[str]:
    contigs: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            contig = line.split()[0]
            if contig:
                contigs.append(contig)
    return contigs


def format_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def to_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def detect_columns(report_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    columns = list(report_df.columns)
    normalized = {col: normalize_name(col) for col in columns}

    def find_by_priority(priorities: Sequence[str]) -> Optional[str]:
        priority_map = {normalize_name(name): name for name in priorities}
        for col in columns:
            if normalized[col] in priority_map:
                return col
        for wanted in priorities:
            wanted_norm = normalize_name(wanted)
            for col in columns:
                if wanted_norm and wanted_norm in normalized[col]:
                    return col
        return None

    return {
        "contig": find_by_priority(["contig_id", "contig", "seq_name", "sequence", "id"]),
        "classification": find_by_priority(["classification", "class", "contig_class"]),
        "support_level": find_by_priority(["blast_support_level", "support_level", "blast_level"]),
        "topology_signal_count": find_by_priority(["topology_signal_count"]),
        "topology_strong_signal_count": find_by_priority(["topology_strong_signal_count"]),
        "is_circular": find_by_priority(["is_circular"]),
        "is_isolated": find_by_priority(["is_isolated"]),
        "is_compact_component": find_by_priority(["is_compact_component"]),
        "graph_score": find_by_priority(["graph_score"]),
        "blast_score": find_by_priority(["blast_score"]),
        "final_score": find_by_priority(["final_score"]),
    }


def discover_species_modes(root: Path, outdir_path: Path) -> Tuple[List[Dict[str, Path]], List[str]]:
    discovered: List[Dict[str, Path]] = []
    warnings: List[str] = []
    outdir_name = outdir_path.name

    for species_dir in sorted(root.iterdir(), key=lambda path: path.name):
        if not species_dir.is_dir():
            continue
        if is_hidden_or_noise(species_dir):
            continue
        if species_dir.name == outdir_name and outdir_path.parent == root:
            continue

        mode_base = choose_mode_base(species_dir)
        if mode_base is None:
            continue

        species_found = False
        for mode_folder, mode_name in MODE_DIR_MAP.items():
            mode_dir = mode_base / mode_folder
            if not mode_dir.is_dir():
                warnings.append(
                    f"SKIP_MODE\t{species_dir.name}\t{mode_name}\tmissing mode directory: {mode_dir}"
                )
                continue

            missing_files = [
                filename for filename in REQUIRED_FILES if not (mode_dir / filename).is_file()
            ]
            if missing_files:
                warnings.append(
                    f"SKIP_MODE\t{species_dir.name}\t{mode_name}\tmissing files: {', '.join(missing_files)}"
                )
                continue

            discovered.append(
                {
                    "species": species_dir.name,
                    "mode": mode_name,
                    "mode_dir": mode_dir,
                    "organelle_contigs": mode_dir / "organelle_contigs.txt",
                    "nuclear_contigs": mode_dir / "nuclear_contigs.txt",
                    "report": mode_dir / "report.tsv",
                }
            )
            species_found = True

        if not species_found:
            warnings.append(
                f"SKIP_SPECIES\t{species_dir.name}\tno valid modes with all required files"
            )

    return discovered, warnings


def aggregate_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col, as_index=False)[AGG_BASE_COLUMNS].sum().copy()

    grouped["removed_sequence_supported_pct"] = grouped.apply(
        lambda row: format_pct(int(row["removed_sequence_supported_n"]), int(row["removed_n"])),
        axis=1,
    )
    grouped["removed_topology_supported_pct"] = grouped.apply(
        lambda row: format_pct(int(row["removed_topology_supported_n"]), int(row["removed_n"])),
        axis=1,
    )
    grouped["removed_strong_topology_supported_pct"] = grouped.apply(
        lambda row: format_pct(int(row["removed_strong_topology_supported_n"]), int(row["removed_n"])),
        axis=1,
    )
    grouped["removed_dual_supported_pct"] = grouped.apply(
        lambda row: format_pct(int(row["removed_dual_supported_n"]), int(row["removed_n"])),
        axis=1,
    )

    grouped["retained_sequence_supported_pct"] = grouped.apply(
        lambda row: format_pct(int(row["retained_sequence_supported_n"]), int(row["retained_n"])),
        axis=1,
    )
    grouped["retained_topology_supported_pct"] = grouped.apply(
        lambda row: format_pct(int(row["retained_topology_supported_n"]), int(row["retained_n"])),
        axis=1,
    )
    grouped["retained_strong_topology_supported_pct"] = grouped.apply(
        lambda row: format_pct(int(row["retained_strong_topology_supported_n"]), int(row["retained_n"])),
        axis=1,
    )
    grouped["retained_dual_supported_pct"] = grouped.apply(
        lambda row: format_pct(int(row["retained_dual_supported_n"]), int(row["retained_n"])),
        axis=1,
    )

    ordered = [group_col] + [c for c in SUMMARY_COLUMNS if c not in {"species", "mode"}]
    return grouped[[c for c in ordered if c in grouped.columns]]


def build_validation_tables(
    discovered: Sequence[Dict[str, Path]], warnings: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []

    for item in discovered:
        species = str(item["species"])
        mode = str(item["mode"])
        report_path = Path(item["report"])

        try:
            report_df = pd.read_csv(report_path, sep="\t", dtype=str)
        except Exception as exc:
            warnings.append(f"SKIP_MODE\t{species}\t{mode}\tfailed to read report.tsv: {exc}")
            continue

        columns = detect_columns(report_df)
        missing_critical = [
            name for name in ("contig", "classification", "support_level")
            if columns[name] is None
        ]
        if missing_critical:
            warnings.append(
                f"SKIP_MODE\t{species}\t{mode}\tmissing detectable columns: {', '.join(missing_critical)}"
            )
            continue

        contig_col = columns["contig"]
        classification_col = columns["classification"]
        support_level_col = columns["support_level"]

        assert contig_col is not None
        assert classification_col is not None
        assert support_level_col is not None

        working_df = report_df.copy()
        working_df[contig_col] = working_df[contig_col].astype(str).str.strip()
        working_df = working_df[working_df[contig_col] != ""].copy()
        working_df = working_df.drop_duplicates(subset=[contig_col], keep="first")

        working_df[classification_col] = (
            working_df[classification_col].fillna("").astype(str).str.strip().str.lower()
        )
        working_df[support_level_col] = (
            working_df[support_level_col].fillna("").astype(str).str.strip().str.lower()
        )

        # sequence support
        working_df["sequence_supported"] = (
            (working_df[support_level_col] != "") &
            (working_df[support_level_col] != "no_blast_support")
        )

        # topology support
        if columns["topology_signal_count"] is not None:
            working_df["topology_signal_count"] = to_numeric_series(
                working_df[columns["topology_signal_count"]]
            )
        else:
            working_df["topology_signal_count"] = 0

        if columns["topology_strong_signal_count"] is not None:
            working_df["topology_strong_signal_count"] = to_numeric_series(
                working_df[columns["topology_strong_signal_count"]]
            )
        else:
            working_df["topology_strong_signal_count"] = 0

        if columns["is_circular"] is not None:
            working_df["is_circular"] = to_bool_series(working_df[columns["is_circular"]])
        else:
            working_df["is_circular"] = False

        if columns["is_isolated"] is not None:
            working_df["is_isolated"] = to_bool_series(working_df[columns["is_isolated"]])
        else:
            working_df["is_isolated"] = False

        if columns["is_compact_component"] is not None:
            working_df["is_compact_component"] = to_bool_series(
                working_df[columns["is_compact_component"]]
            )
        else:
            working_df["is_compact_component"] = False

        if columns["graph_score"] is not None:
            working_df["graph_score"] = to_numeric_series(working_df[columns["graph_score"]])
        else:
            working_df["graph_score"] = 0

        if columns["blast_score"] is not None:
            working_df["blast_score"] = to_numeric_series(working_df[columns["blast_score"]])
        else:
            working_df["blast_score"] = 0

        if columns["final_score"] is not None:
            working_df["final_score"] = to_numeric_series(working_df[columns["final_score"]])
        else:
            working_df["final_score"] = 0

        # Conservative topology support:
        # any topology signal OR circular OR compact component
        working_df["topology_supported"] = (
            (working_df["topology_signal_count"] >= 1) |
            working_df["is_circular"] |
            working_df["is_compact_component"]
        )

        # Strong topology support:
        # strong topology signal OR (circular + compact)
        working_df["strong_topology_supported"] = (
            (working_df["topology_strong_signal_count"] >= 1) |
            (working_df["is_circular"] & working_df["is_compact_component"])
        )

        working_df["dual_supported"] = (
            working_df["sequence_supported"] & working_df["topology_supported"]
        )

        working_df = working_df[working_df[classification_col].isin({"organelle", "nuclear"})].copy()

        # Optional cross-check with txt lists
        try:
            removed_contigs = set(read_contig_list(Path(item["organelle_contigs"])))
            retained_contigs = set(read_contig_list(Path(item["nuclear_contigs"])))
            report_removed = set(
                working_df.loc[working_df[classification_col] == "organelle", contig_col].astype(str)
            )
            report_retained = set(
                working_df.loc[working_df[classification_col] == "nuclear", contig_col].astype(str)
            )
            if report_removed != removed_contigs:
                warnings.append(
                    f"MISMATCH\t{species}\t{mode}\torganelle_contigs.txt differs from report.tsv classification "
                    f"(report={len(report_removed)}, file={len(removed_contigs)})"
                )
            if report_retained != retained_contigs:
                warnings.append(
                    f"MISMATCH\t{species}\t{mode}\tnuclear_contigs.txt differs from report.tsv classification "
                    f"(report={len(report_retained)}, file={len(retained_contigs)})"
                )
        except Exception as exc:
            warnings.append(f"WARNING\t{species}\t{mode}\tfailed contig-list cross-check: {exc}")

        working_df["species"] = species
        working_df["mode"] = mode
        working_df["contig_id"] = working_df[contig_col]
        working_df["classification"] = working_df[classification_col]
        working_df["blast_support_level"] = working_df[support_level_col]
        working_df["set_class"] = working_df["classification"].map(
            {"organelle": "removed", "nuclear": "retained"}
        )

        subset_df = working_df[LONG_COLUMNS].copy()

        removed_df = subset_df[subset_df["set_class"] == "removed"].copy()
        retained_df = subset_df[subset_df["set_class"] == "retained"].copy()

        def count_true(df: pd.DataFrame, col: str) -> int:
            return int(df[col].sum())

        summary_rows.append(
            {
                "species": species,
                "mode": mode,

                "removed_n": len(removed_df),
                "removed_sequence_supported_n": count_true(removed_df, "sequence_supported"),
                "removed_sequence_supported_pct": format_pct(count_true(removed_df, "sequence_supported"), len(removed_df)),
                "removed_topology_supported_n": count_true(removed_df, "topology_supported"),
                "removed_topology_supported_pct": format_pct(count_true(removed_df, "topology_supported"), len(removed_df)),
                "removed_strong_topology_supported_n": count_true(removed_df, "strong_topology_supported"),
                "removed_strong_topology_supported_pct": format_pct(count_true(removed_df, "strong_topology_supported"), len(removed_df)),
                "removed_dual_supported_n": count_true(removed_df, "dual_supported"),
                "removed_dual_supported_pct": format_pct(count_true(removed_df, "dual_supported"), len(removed_df)),

                "retained_n": len(retained_df),
                "retained_sequence_supported_n": count_true(retained_df, "sequence_supported"),
                "retained_sequence_supported_pct": format_pct(count_true(retained_df, "sequence_supported"), len(retained_df)),
                "retained_topology_supported_n": count_true(retained_df, "topology_supported"),
                "retained_topology_supported_pct": format_pct(count_true(retained_df, "topology_supported"), len(retained_df)),
                "retained_strong_topology_supported_n": count_true(retained_df, "strong_topology_supported"),
                "retained_strong_topology_supported_pct": format_pct(count_true(retained_df, "strong_topology_supported"), len(retained_df)),
                "retained_dual_supported_n": count_true(retained_df, "dual_supported"),
                "retained_dual_supported_pct": format_pct(count_true(retained_df, "dual_supported"), len(retained_df)),
            }
        )

        long_rows.extend(subset_df.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    long_df = pd.DataFrame(long_rows, columns=LONG_COLUMNS)
    return summary_df, long_df


def write_outputs(
    outdir: Path,
    summary_df: pd.DataFrame,
    long_df: pd.DataFrame,
    warnings: Sequence[str],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df = summary_df.sort_values(["species", "mode"]).reset_index(drop=True)
    long_df = long_df.sort_values(["species", "mode", "set_class", "contig_id"]).reset_index(drop=True)

    if summary_df.empty:
        mode_summary_df = pd.DataFrame(columns=["mode"] + [c for c in SUMMARY_COLUMNS if c not in {"species", "mode"}])
        species_summary_df = pd.DataFrame(columns=["species"] + [c for c in SUMMARY_COLUMNS if c not in {"species", "mode"}])
    else:
        mode_summary_df = aggregate_summary(summary_df, "mode").sort_values("mode").reset_index(drop=True)
        species_summary_df = aggregate_summary(summary_df, "species").sort_values("species").reset_index(drop=True)

    summary_df.to_csv(outdir / "validation_integrated_summary.tsv", sep="\t", index=False)
    long_df.to_csv(outdir / "validation_integrated_long.tsv", sep="\t", index=False)
    mode_summary_df.to_csv(outdir / "validation_integrated_mode_summary.tsv", sep="\t", index=False)
    species_summary_df.to_csv(outdir / "validation_integrated_species_summary.tsv", sep="\t", index=False)

    warning_path = outdir / "processing_warnings.log"
    with warning_path.open("w", encoding="utf-8") as handle:
        if warnings:
            for line in warnings:
                handle.write(f"{line}\n")
        else:
            handle.write("No warnings.\n")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    outdir = resolve_outdir(root, args.outdir)

    discovered, warnings = discover_species_modes(root, outdir)
    summary_df, long_df = build_validation_tables(discovered, warnings)
    write_outputs(outdir, summary_df, long_df, warnings)

    print(
        f"Processed species-mode combinations: {len(summary_df)} "
        f"across {summary_df['species'].nunique() if not summary_df.empty else 0} species."
    )
    print(f"Warnings written to: {outdir / 'processing_warnings.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
