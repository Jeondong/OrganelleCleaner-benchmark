#!/usr/bin/env python3
"""Summarize method overlap for OrganelleCleaner benchmark species folders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd


MODE_NAMES = ("graph", "blast", "hybrid")
MODE_DIRS = {
    "graph": "graph-only",
    "blast": "blast-only",
    "hybrid": "hybrid",
}

SPECIES_SUMMARY_COLUMNS = [
    "species",
    "graph_n",
    "blast_n",
    "hybrid_n",
    "graph_only_n",
    "blast_only_n",
    "hybrid_only_n",
    "graph_blast_overlap_n",
    "graph_hybrid_overlap_n",
    "blast_hybrid_overlap_n",
    "all_three_overlap_n",
    "union_all_n",
    "hybrid_vs_union_recall_pct",
    "hybrid_vs_graph_pct",
    "hybrid_vs_blast_pct",
]

MEMBERSHIP_COLUMNS = [
    "species",
    "contig_id",
    "in_graph",
    "in_blast",
    "in_hybrid",
    "membership_class",
]

OVERALL_SUMMARY_COLUMNS = [
    "graph_n",
    "blast_n",
    "hybrid_n",
    "graph_only_n",
    "blast_only_n",
    "hybrid_only_n",
    "graph_blast_overlap_n",
    "graph_hybrid_overlap_n",
    "blast_hybrid_overlap_n",
    "all_three_overlap_n",
    "union_all_n",
    "hybrid_vs_union_recall_pct",
]

PLOT_COLUMNS = ["species", "category", "count"]

PLOT_CATEGORIES = [
    "graph_only",
    "blast_only",
    "hybrid_only",
    "graph_blast_only",
    "graph_hybrid_only",
    "blast_hybrid_only",
    "all_three",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize overlap between graph-only, blast-only, and hybrid "
            "organelle contig detection results across species folders."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing species folders.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for summary tables and warning log.",
    )
    return parser.parse_args()


def safe_pct(numerator: int, denominator: int) -> float:
    """Return a percentage, avoiding division-by-zero failures."""
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def is_ignored_name(name: str) -> bool:
    """Return True if a top-level entry should be ignored during discovery."""
    return (
        not name
        or name.startswith(".")
        or name.startswith(".nfs")
        or name == "result"
        or name == "__pycache__"
    )


def read_contig_file(path: Path) -> Tuple[Set[str], List[str]]:
    """Read a contig list file into a set and return any parsing warnings."""
    contigs: Set[str] = set()
    warnings: List[str] = []

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                contig_id = raw_line.strip()
                if not contig_id:
                    continue
                contigs.add(contig_id)
    except OSError as exc:
        warnings.append(f"Failed to read file: {path} ({exc})")

    return contigs, warnings


def resolve_species_mode_files(
    species_dir: Path,
) -> Tuple[Optional[Dict[str, Path]], List[str]]:
    """Resolve required contig files for a species using supported path patterns."""
    species = species_dir.name
    warnings: List[str] = []

    pattern_a = {
        mode: species_dir / "organelle_cleaner_runs" / MODE_DIRS[mode] / "organelle_contigs.txt"
        for mode in MODE_NAMES
    }
    pattern_b = {
        mode: species_dir / MODE_DIRS[mode] / "organelle_contigs.txt"
        for mode in MODE_NAMES
    }

    if all(path.is_file() for path in pattern_a.values()):
        return pattern_a, warnings

    if all(path.is_file() for path in pattern_b.values()):
        warnings.append(
            f"Using alternate direct mode-path layout for species '{species}' under {species_dir}"
        )
        return pattern_b, warnings

    missing_a = [str(path) for path in pattern_a.values() if not path.is_file()]
    missing_b = [str(path) for path in pattern_b.values() if not path.is_file()]
    warnings.append(
        f"Skipping species '{species}': required files not found under supported layouts. "
        f"Missing Pattern A files: {', '.join(missing_a) if missing_a else 'none'}; "
        f"Missing Pattern B files: {', '.join(missing_b) if missing_b else 'none'}"
    )
    return None, warnings


def looks_like_species_dir(entry: Path) -> bool:
    """Return True if a directory looks like a species result folder."""
    if (entry / "organelle_cleaner_runs").is_dir():
        return True
    return any((entry / MODE_DIRS[mode]).is_dir() for mode in MODE_NAMES)


def discover_species_candidates(root: Path, outdir: Path) -> List[Path]:
    """Discover species candidate directories under the root."""
    species_dirs: List[Path] = []
    try:
        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            if is_ignored_name(entry.name):
                continue
            if not entry.is_dir():
                continue
            if entry.resolve() == outdir:
                continue
            if looks_like_species_dir(entry):
                species_dirs.append(entry)
    except OSError:
        return []
    return species_dirs


def collect_species_contigs(
    species_dir: Path,
) -> Tuple[bool, Dict[str, Set[str]], List[str]]:
    """Load contig sets for all required modes for one species."""
    warnings: List[str] = []
    contig_sets: Dict[str, Set[str]] = {}
    mode_files, path_warnings = resolve_species_mode_files(species_dir)
    warnings.extend(path_warnings)
    if mode_files is None:
        return False, {}, warnings

    for mode in MODE_NAMES:
        contig_path = mode_files[mode]
        contigs, read_warnings = read_contig_file(contig_path)
        contig_sets[mode] = contigs
        warnings.extend(read_warnings)

    return True, contig_sets, warnings


def membership_class(in_graph: bool, in_blast: bool, in_hybrid: bool) -> str:
    """Map method membership booleans to the requested class label."""
    if in_graph and in_blast and in_hybrid:
        return "all_three"
    if in_graph and in_blast:
        return "graph_blast"
    if in_graph and in_hybrid:
        return "graph_hybrid"
    if in_blast and in_hybrid:
        return "blast_hybrid"
    if in_graph:
        return "graph_only"
    if in_blast:
        return "blast_only"
    return "hybrid_only"


def compute_overlap_counts(
    graph: Set[str],
    blast: Set[str],
    hybrid: Set[str],
) -> Dict[str, int]:
    """Compute overlap counts for three sets."""
    all_three = graph & blast & hybrid
    graph_blast = graph & blast
    graph_hybrid = graph & hybrid
    blast_hybrid = blast & hybrid
    union_all = graph | blast | hybrid

    return {
        "graph_n": len(graph),
        "blast_n": len(blast),
        "hybrid_n": len(hybrid),
        "graph_only_n": len(graph - blast - hybrid),
        "blast_only_n": len(blast - graph - hybrid),
        "hybrid_only_n": len(hybrid - graph - blast),
        "graph_blast_overlap_n": len(graph_blast),
        "graph_hybrid_overlap_n": len(graph_hybrid),
        "blast_hybrid_overlap_n": len(blast_hybrid),
        "all_three_overlap_n": len(all_three),
        "union_all_n": len(union_all),
    }


def build_species_outputs(
    species: str,
    graph: Set[str],
    blast: Set[str],
    hybrid: Set[str],
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    """Build per-species summary, long membership rows, and plot rows."""
    counts = compute_overlap_counts(graph, blast, hybrid)
    union_all = graph | blast | hybrid

    summary_row: Dict[str, object] = {
        "species": species,
        **counts,
        "hybrid_vs_union_recall_pct": safe_pct(counts["hybrid_n"], counts["union_all_n"]),
        "hybrid_vs_graph_pct": safe_pct(counts["hybrid_n"], counts["graph_n"]),
        "hybrid_vs_blast_pct": safe_pct(counts["hybrid_n"], counts["blast_n"]),
    }

    membership_rows: List[Dict[str, object]] = []
    for contig_id in sorted(union_all):
        in_graph = contig_id in graph
        in_blast = contig_id in blast
        in_hybrid = contig_id in hybrid
        membership_rows.append(
            {
                "species": species,
                "contig_id": contig_id,
                "in_graph": in_graph,
                "in_blast": in_blast,
                "in_hybrid": in_hybrid,
                "membership_class": membership_class(in_graph, in_blast, in_hybrid),
            }
        )

    all_three = graph & blast & hybrid
    plot_counts = {
        "graph_only": len(graph - blast - hybrid),
        "blast_only": len(blast - graph - hybrid),
        "hybrid_only": len(hybrid - graph - blast),
        "graph_blast_only": len((graph & blast) - hybrid),
        "graph_hybrid_only": len((graph & hybrid) - blast),
        "blast_hybrid_only": len((blast & hybrid) - graph),
        "all_three": len(all_three),
    }
    plot_rows = [
        {"species": species, "category": category, "count": plot_counts[category]}
        for category in PLOT_CATEGORIES
    ]

    return summary_row, membership_rows, plot_rows


def build_overall_summary(prefixed_sets: Dict[str, Set[str]]) -> Dict[str, object]:
    """Build dataset-level summary across all species using species-prefixed IDs."""
    graph = prefixed_sets["graph"]
    blast = prefixed_sets["blast"]
    hybrid = prefixed_sets["hybrid"]
    counts = compute_overlap_counts(graph, blast, hybrid)
    return {
        **counts,
        "hybrid_vs_union_recall_pct": safe_pct(counts["hybrid_n"], counts["union_all_n"]),
    }


def ensure_output_dir(outdir: Path) -> None:
    """Create the output directory if needed."""
    outdir.mkdir(parents=True, exist_ok=True)


def write_tsv(rows: Sequence[Dict[str, object]], columns: Sequence[str], path: Path) -> None:
    """Write rows to a TSV with a stable column order."""
    dataframe = pd.DataFrame(list(rows), columns=list(columns))
    dataframe.to_csv(path, sep="\t", index=False)


def write_warning_log(warnings: Sequence[str], path: Path) -> None:
    """Write warnings to the processing log."""
    with path.open("w", encoding="utf-8") as handle:
        if warnings:
            for message in warnings:
                handle.write(f"{message}\n")
        else:
            handle.write("No warnings.\n")


def main() -> int:
    """Run the summary workflow."""
    args = parse_args()
    root = Path(args.root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir = outdir.resolve()

    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    warning_messages: List[str] = []
    processed_species = 0
    skipped_species = 0

    species_summary_rows: List[Dict[str, object]] = []
    membership_rows: List[Dict[str, object]] = []
    plot_rows: List[Dict[str, object]] = []

    overall_prefixed_sets: Dict[str, Set[str]] = {mode: set() for mode in MODE_NAMES}

    species_dirs = discover_species_candidates(root, outdir)
    if not species_dirs:
        warning_messages.append(f"No candidate species directories found under {root}")

    ensure_output_dir(outdir)

    for species_dir in species_dirs:
        species = species_dir.name
        valid, contig_sets, species_warnings = collect_species_contigs(species_dir)
        warning_messages.extend(species_warnings)
        if not valid:
            skipped_species += 1
            continue

        graph = contig_sets["graph"]
        blast = contig_sets["blast"]
        hybrid = contig_sets["hybrid"]

        summary_row, species_membership_rows, species_plot_rows = build_species_outputs(
            species=species,
            graph=graph,
            blast=blast,
            hybrid=hybrid,
        )
        species_summary_rows.append(summary_row)
        membership_rows.extend(species_membership_rows)
        plot_rows.extend(species_plot_rows)

        for mode, contigs in contig_sets.items():
            overall_prefixed_sets[mode].update(f"{species}|{contig_id}" for contig_id in contigs)

        processed_species += 1

    overall_summary_row = build_overall_summary(overall_prefixed_sets)

    write_tsv(
        species_summary_rows,
        SPECIES_SUMMARY_COLUMNS,
        outdir / "species_method_overlap_summary.tsv",
    )
    write_tsv(
        membership_rows,
        MEMBERSHIP_COLUMNS,
        outdir / "species_method_membership_long.tsv",
    )
    write_tsv(
        [overall_summary_row],
        OVERALL_SUMMARY_COLUMNS,
        outdir / "overall_method_overlap_summary.tsv",
    )
    write_tsv(
        plot_rows,
        PLOT_COLUMNS,
        outdir / "figure4_plot_table.tsv",
    )
    write_warning_log(warning_messages, outdir / "processing_warnings.log")

    print(f"Processed {processed_species} species")
    print(f"Skipped {skipped_species} species")
    print(f"Output written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
