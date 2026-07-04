"""
Visualization helpers for graph exploration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files without a display
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_degree_distribution(
    degrees: list[int], title: str, output_path: Path
) -> None:
    """
    Linear-scale degree distribution histogram.

    Uses a histogram rather than one bar per exact degree because real networks
    often have many unique degree values and a bar-per-degree view can look too
    sparse or visually empty for skewed distributions.
    """
    _ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(8, 5))
    if not degrees:
        ax.text(
            0.5,
            0.5,
            "No degree data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        values = np.asarray(degrees, dtype=float)
        bins = min(50, max(10, int(np.sqrt(len(values)))))
        ax.hist(values, bins=bins, color="steelblue", edgecolor="white")

    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.set_title(f"Degree Distribution — {title}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_log_log_degree_distribution(
    degrees: list[int], title: str, output_path: Path
) -> None:
    """
    Log-log degree distribution plot.

    A straight-line tail on log-log axes suggests power-law (scale-free) behavior.
    """
    _ensure_dir(output_path.parent)
    dist = pd.Series(degrees).value_counts().sort_index()
    x = np.asarray(dist.index.to_numpy(dtype=float), dtype=float)
    y = np.asarray(dist.to_numpy(dtype=float), dtype=float)
    mask = (x > 0) & (y > 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x[mask], y[mask], color="darkorange", s=30, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (log)")
    ax.set_ylabel("Count (log)")
    ax.set_title(f"Log-Log Degree Distribution — {title}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    ylabel: str | None = None,
) -> None:
    """Bar chart comparing one metric across SNAP and synthetic graphs."""
    _ensure_dir(output_path.parent)
    plot_df = df.dropna(subset=[metric]).copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [
        "crimson" if "SNAP" in name or name.endswith("(SNAP)") else "steelblue"
        for name in plot_df["name"]
    ]
    ax.bar(plot_df["name"], plot_df[metric], color=colors, edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel(ylabel or metric)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_multi_metric_comparison(
    df: pd.DataFrame, metrics: list[str], title: str, output_path: Path
) -> None:
    """Grouped bar chart for several normalized metrics."""
    _ensure_dir(output_path.parent)
    plot_df = df.set_index("name")[metrics].copy()

    # Normalize each metric to [0, 1] for visual comparison on same axes.
    normalized = plot_df.apply(
        lambda col: (col - col.min()) / (col.max() - col.min())
        if col.max() != col.min()
        else col * 0
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    normalized.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title(title)
    ax.set_ylabel("Normalized value (0 = min, 1 = max per metric)")
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_small_graph_example(
    G: nx.Graph, title: str, output_path: Path, max_nodes: int = 80
) -> None:
    """
    Visualize a small subgraph so you can see local structure.

    nx.spring_layout(G, seed=seed):
      - Force-directed layout using Fruchterman-Reingold.
      - Returns dict node -> (x, y) positions.
      - Good for intuition; not unique or geographically meaningful.

    For large graphs we sample the LCC to keep the drawing readable.
    """
    _ensure_dir(output_path.parent)

    if G.number_of_nodes() > max_nodes:
        # nx.connected_components + subgraph to sample a readable piece.
        lcc_nodes = max(nx.connected_components(G), key=len)
        lcc = G.subgraph(lcc_nodes).copy()
        sample_nodes = list(lcc.nodes())[:max_nodes]
        G = G.subgraph(sample_nodes).copy()

    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # nx.draw_networkx(): renders nodes and edges using matplotlib axes.
    degrees = dict(G.degree())
    node_sizes = [80 + 20 * degrees[n] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color="steelblue", alpha=0.85, ax=ax
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics_table(df: pd.DataFrame, output_path: Path) -> None:
    """Export comparison table to CSV."""
    _ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
