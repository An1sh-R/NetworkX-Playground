"""
NetworkX Playground — main exploration script.

Loads SNAP graphs, computes metrics, generates matched synthetic graphs,
compares them, and writes tables, plots, and a summary report.
"""

from __future__ import annotations

import sys
from typing import Any, cast

# Windows consoles often default to cp1252; use UTF-8 for metric symbols.
if callable(getattr(sys.stdout, "reconfigure", None)):
    stdout = cast(Any, sys.stdout)
    stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import pandas as pd

from metrics import compute_all_metrics, load_snap_graph, print_basic_info
from plots import (
    plot_degree_distribution,
    plot_log_log_degree_distribution,
    plot_metric_comparison,
    plot_multi_metric_comparison,
    plot_small_graph_example,
    save_metrics_table,
)
from synthetic import SYNTHETIC_DESCRIPTIONS, generate_all_synthetic

# --- Paths ---
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_PATH = OUTPUT_DIR / "summary_report.txt"

# SNAP edge-list files (downloaded from https://snap.stanford.edu/data/)
SNAP_FILES = {
    "email-Eu-core": DATA_DIR / "email-Eu-core.txt.gz",
    "ca-GrQc": DATA_DIR / "ca-GrQc.txt.gz",
    "facebook-combined": DATA_DIR / "facebook_combined.txt.gz",
}

# Metrics shown in comparison tables
COMPARISON_COLUMNS = [
    "name",
    "nodes",
    "edges",
    "average_degree",
    "density",
    "clustering_coefficient",
    "transitivity",
    "assortativity",
    "modularity",
    "degree_heterogeneity",
    "avg_shortest_path_lcc",
    "diameter_lcc",
]


def metrics_to_row(metrics: dict) -> dict:
    """Flatten metrics dict for pandas (exclude non-tabular fields)."""
    return {k: metrics[k] for k in COMPARISON_COLUMNS if k in metrics}


def interpret_comparison(snap_name: str, comparison_df: pd.DataFrame) -> str:
    """
    Short interpretation: which synthetic model fits best and what differs.
    Uses Euclidean distance on normalized structural metrics.
    """
    structural = [
        "clustering_coefficient",
        "transitivity",
        "assortativity",
        "modularity",
        "degree_heterogeneity",
        "avg_shortest_path_lcc",
        "diameter_lcc",
    ]

    snap_row = comparison_df[comparison_df["name"] == f"{snap_name} (SNAP)"]
    synth_rows = comparison_df[comparison_df["name"] != f"{snap_name} (SNAP)"]

    if snap_row.empty or synth_rows.empty:
        return "Insufficient data for interpretation."

    snap_vals = snap_row.iloc[0]
    distances = {}

    for _, row in synth_rows.iterrows():
        diffs = []
        for col in structural:
            s_val = snap_vals[col]
            r_val = row[col]
            if pd.isna(s_val) or pd.isna(r_val):
                continue
            # Normalize by snap value to balance scales; use |snap| for zero.
            denom = abs(s_val) if abs(s_val) > 1e-9 else 1.0
            diffs.append(((r_val - s_val) / denom) ** 2)
        distances[row["name"]] = sum(diffs) ** 0.5 if diffs else float("inf")

    best_model = min(distances.items(), key=lambda item: item[1])[0]
    worst_metric_diff = None
    max_diff = -1.0

    best_row = synth_rows[synth_rows["name"] == best_model].iloc[0]
    for col in structural:
        s_val = snap_vals[col]
        r_val = best_row[col]
        if pd.isna(s_val) or pd.isna(r_val):
            continue
        diff = abs(r_val - s_val)
        if diff > max_diff:
            max_diff = diff
            worst_metric_diff = col

    lines = [
        f"Best overall match: {best_model}",
        f"Largest remaining gap (vs SNAP): {worst_metric_diff}",
        "",
        "Interpretation:",
        f"- {best_model} captures the most structural features of {snap_name}.",
        "- Real social networks often combine high clustering with heterogeneous degrees;",
        "  no single simple model reproduces every metric.",
        "- Large gaps in clustering/transitivity suggest triadic closure beyond random wiring.",
        "- Gaps in degree heterogeneity/assortativity point to hub structure and",
        "  preferential attachment not present in lattice or ER models.",
    ]
    return "\n".join(lines)


def print_synthetic_explanations() -> None:
    """Print how each synthetic model is built and what it represents."""
    print("\n" + "=" * 60)
    print("SYNTHETIC GRAPH MODELS")
    print("=" * 60)
    for model, info in SYNTHETIC_DESCRIPTIONS.items():
        print(f"\n--- {model} ---")
        print(f"  Generation:   {info['generation']}")
        print(f"  Parameters:   {info['parameters']}")
        print(f"  Real-world:   {info['real_world']}")
        print(f"  Strengths:    {info['strengths']}")
        print(f"  Limitations:  {info['limitations']}")


def analyze_snap_graph(filepath: Path, snap_name: str, seed: int = 42) -> tuple:
    """Load one SNAP graph, analyze it, build synthetic counterparts, compare."""
    print(f"\nLoading SNAP dataset: {filepath.name}", flush=True)
    G = load_snap_graph(str(filepath))

    snap_metrics = compute_all_metrics(G, f"{snap_name} (SNAP)")
    print_basic_info(snap_metrics)

    print("\n--- Advanced Metrics ---")
    print(f"  Degree heterogeneity:     {snap_metrics['degree_heterogeneity']:.4f}")
    print(f"  Clustering coefficient:   {snap_metrics['clustering_coefficient']:.4f}")
    print(f"  Transitivity:             {snap_metrics['transitivity']:.4f}")
    print(f"  Assortativity:            {snap_metrics['assortativity']:.4f}")
    print(f"  Communities (Louvain):    {snap_metrics['num_communities']}")
    print(f"  Modularity:               {snap_metrics['modularity']:.4f}")
    print(f"  Avg shortest path (LCC):  {snap_metrics['avg_shortest_path_lcc']}")
    print(f"  Diameter (LCC):           {snap_metrics['diameter_lcc']}")

    n = snap_metrics["nodes"]
    avg_degree = snap_metrics["average_degree"]

    print(f"\nGenerating synthetic graphs matched to n={n}, avg_degree={avg_degree:.2f}", flush=True)
    synthetic_graphs = generate_all_synthetic(n, avg_degree, seed=seed)

    all_metrics = [snap_metrics]
    for model_name, synth_g in synthetic_graphs.items():
        m = compute_all_metrics(synth_g, model_name)
        all_metrics.append(m)

    comparison_df = pd.DataFrame([metrics_to_row(m) for m in all_metrics])

    # Console table
    print(f"\n--- Comparison Table: {snap_name} ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(comparison_df.to_string(index=False))

    # Save CSV
    csv_path = TABLES_DIR / f"comparison_{snap_name}.csv"
    save_metrics_table(comparison_df, csv_path)
    print(f"\nSaved table: {csv_path}")

    # Plots for SNAP graph
    plot_degree_distribution(
        snap_metrics["degrees"],
        snap_name,
        PLOTS_DIR / snap_name / "degree_dist.png",
    )
    plot_log_log_degree_distribution(
        snap_metrics["degrees"],
        snap_name,
        PLOTS_DIR / snap_name / "degree_dist_loglog.png",
    )

    # Plots for each graph in the comparison
    for m in all_metrics:
        safe_name = m["name"].replace(" ", "_").replace("(", "").replace(")", "")
        subdir = PLOTS_DIR / snap_name / safe_name
        plot_degree_distribution(m["degrees"], m["name"], subdir / "degree_dist.png")
        plot_log_log_degree_distribution(
            m["degrees"], m["name"], subdir / "degree_dist_loglog.png"
        )
        plot_small_graph_example(
            m["graph"], m["name"], subdir / "graph_example.png"
        )

    # Metric comparison bar charts
    for metric, label in [
        ("clustering_coefficient", "Clustering Coefficient"),
        ("modularity", "Modularity (Louvain)"),
        ("assortativity", "Degree Assortativity"),
        ("degree_heterogeneity", "Degree Heterogeneity"),
        ("avg_shortest_path_lcc", "Avg Shortest Path (LCC)"),
        ("diameter_lcc", "Diameter (LCC)"),
    ]:
        plot_metric_comparison(
            comparison_df,
            metric,
            f"{label} — {snap_name}",
            PLOTS_DIR / snap_name / f"compare_{metric}.png",
            ylabel=label,
        )

    plot_multi_metric_comparison(
        comparison_df,
        [
            "clustering_coefficient",
            "transitivity",
            "assortativity",
            "modularity",
            "degree_heterogeneity",
        ],
        f"Normalized Structural Metrics — {snap_name}",
        PLOTS_DIR / snap_name / "compare_multi_normalized.png",
    )

    interpretation = interpret_comparison(snap_name, comparison_df)
    print(f"\n--- Interpretation: {snap_name} ---")
    print(interpretation)

    return snap_name, comparison_df, interpretation, all_metrics


def write_summary_report(results: list) -> None:
    """Final summary across all SNAP datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "NetworkX Playground — Summary Report",
        "=" * 50,
        "",
        "This report summarizes comparisons between SNAP real-world graphs",
        "and four synthetic models: Erdős–Rényi, Watts–Strogatz,",
        "Barabási–Albert, and 2D Grid/Lattice.",
        "",
    ]

    structural = [
        "clustering_coefficient",
        "transitivity",
        "assortativity",
        "modularity",
        "degree_heterogeneity",
        "avg_shortest_path_lcc",
        "diameter_lcc",
    ]

    for snap_name, comparison_df, interpretation, _ in results:
        lines.append(f"\n{'=' * 50}")
        lines.append(f"Dataset: {snap_name}")
        lines.append("=" * 50)
        lines.append(interpretation)

        snap_row = comparison_df[comparison_df["name"] == f"{snap_name} (SNAP)"].iloc[0]
        synth_rows = comparison_df[comparison_df["name"] != f"{snap_name} (SNAP)"]

        lines.append("\nMetrics differing most from best-matching synthetic:")
        best_name = interpretation.split("\n")[0].replace("Best overall match: ", "")
        best_row = synth_rows[synth_rows["name"] == best_name].iloc[0]

        diffs = []
        for col in structural:
            s_val, b_val = snap_row[col], best_row[col]
            if pd.notna(s_val) and pd.notna(b_val):
                diffs.append((col, abs(s_val - b_val)))
        diffs.sort(key=lambda x: x[1], reverse=True)
        for col, diff in diffs[:3]:
            lines.append(f"  - {col}: gap = {diff:.4f}")

    lines.extend(
        [
            "\n" + "=" * 50,
            "KEY INSIGHTS",
            "=" * 50,
            "",
            "1. Social and collaboration networks usually show higher clustering than",
            "   Erdős–Rényi random graphs at the same density — friends of friends connect.",
            "",
            "2. Watts–Strogatz often matches clustering and short paths well but misses",
            "   degree heterogeneity seen in real networks.",
            "",
            "3. Barabási–Albert captures hub-heavy degree distributions but typically",
            "   under-estimates clustering and community modularity.",
            "",
            "4. Grid lattices have high local clustering but large diameter — poor fit",
            "   for most social networks despite spatial intuition.",
            "",
            "5. No single generative model replaces empirical data; each highlights",
            "   different mechanisms (randomness, small-world, preferential attachment, space).",
            "",
            "Outputs written to:",
            f"  Tables: {TABLES_DIR}",
            f"  Plots:  {PLOTS_DIR}",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary report saved: {REPORT_PATH}")


def main() -> None:
    print("NetworkX Playground — Graph Exploration")
    print("=" * 60)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify datasets exist
    missing = [name for name, path in SNAP_FILES.items() if not path.exists()]
    if missing:
        print("Missing SNAP files in data/:")
        for name in missing:
            print(f"  - {name}")
        print("\nDownload from https://snap.stanford.edu/data/ and place in data/")
        return

    print_synthetic_explanations()

    results = []
    for snap_name, filepath in SNAP_FILES.items():
        result = analyze_snap_graph(filepath, snap_name)
        results.append(result)

    write_summary_report(results)

    print("\n" + "=" * 60)
    print("Done! Check outputs/ for CSV files, plots, and summary_report.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
