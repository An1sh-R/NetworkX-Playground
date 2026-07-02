"""
Synthetic graph generators matched to empirical SNAP graphs.

Each generator includes comments on parameters, real-world analogies,
strengths, and limitations.
"""

from __future__ import annotations

import math

import networkx as nx


def _even_k(target_k: int, n: int) -> int:
    """Watts-Strogatz needs even k; clamp to valid range."""
    k = max(2, int(round(target_k)))
    if k % 2 == 1:
        k += 1
    k = min(k, n - 1 if (n - 1) % 2 == 0 else n - 2)
    return max(2, k)


def generate_erdos_renyi(n: int, avg_degree: float, seed: int = 42) -> nx.Graph:
    """
    Erdős-Rényi G(n, p) random graph.

    nx.erdos_renyi_graph(n, p, seed=seed):
      - Adds each possible edge independently with probability p.
      - Returns a Graph with ~p * n*(n-1)/2 edges on average.

    Parameters:
      - n: number of nodes
      - p ≈ avg_degree / (n - 1) chosen so expected degree ≈ avg_degree

    Models: simple null model — who connects to whom at random.
    Strengths: mathematically tractable baseline; easy to generate.
    Limitations: Poisson degree distribution; low clustering; no hubs.
    """
    p = avg_degree / (n - 1) if n > 1 else 0.0
    p = min(max(p, 0.0), 1.0)
    return nx.erdos_renyi_graph(n, p, seed=seed)


def generate_watts_strogatz(n: int, avg_degree: float, seed: int = 42) -> nx.Graph:
    """
    Watts–Strogatz small-world graph.

    nx.watts_strogatz_graph(n, k, p, seed=seed):
      - Starts with a ring lattice where each node connects to k/2 neighbors
        on each side, then rewires each edge with probability p.
      - Returns a connected-ish graph with high clustering and short paths.

    Parameters:
      - n: number of nodes
      - k: each node's degree in the initial ring (set from avg_degree)
      - p=0.1: rewiring probability (classic small-world value)

    Models: social networks, neural networks — clustered yet small diameter.
    Strengths: captures high clustering + short paths.
    Limitations: degree distribution is narrow; not scale-free.
    """
    k = _even_k(int(round(avg_degree)), n)
    return nx.watts_strogatz_graph(n, k, p=0.1, seed=seed)


def generate_barabasi_albert(n: int, avg_degree: float, seed: int = 42) -> nx.Graph:
    """
    Barabási–Albert preferential attachment (scale-free) graph.

    nx.barabasi_albert_graph(n, m, seed=seed):
      - Starts with m nodes, then adds n-m nodes one at a time.
      - Each new node attaches m edges to existing nodes with probability
        proportional to their current degree (rich get richer).
      - Returns a Graph with power-law-like degree tail.

    Parameters:
      - n: final number of nodes
      - m: edges added per new node (m ≈ avg_degree / 2 for undirected BA)

    Models: web links, citations, social influence networks.
    Strengths: heavy-tailed degrees; hub structure.
    Limitations: low clustering vs real social nets; no community structure
    unless extended variants are used.
    """
    m = max(1, int(round(avg_degree / 2)))
    m = min(m, n - 1)
    return nx.barabasi_albert_graph(n, m, seed=seed)


def generate_grid(n: int, seed: int = 42) -> nx.Graph:
    """
    2D grid / lattice graph.

    nx.grid_2d_graph(m, n):
      - Nodes are (row, col) pairs; edges connect orthogonal neighbors.
      - Returns a Graph with m*n nodes and ~2*m*n - m - n edges.

    Parameters:
      - Dimensions chosen so m * n ~ target node count n_nodes.

    Note: A plain 2D grid has average degree ~4 (max 4). It cannot match
    high average degrees of social networks — it serves as a spatial baseline.

    Models: road grids, chip layouts, spatial contact networks.
    Strengths: very high clustering locally; predictable regular structure.
    Limitations: high diameter; uniform low degree; poor model for social hubs.
    """
    side = max(2, int(math.sqrt(n)))
    rows = side
    cols = max(2, int(math.ceil(n / rows)))

    G = nx.grid_2d_graph(rows, cols)
    # Relabel to integers 0..N-1 for consistency with SNAP graphs.
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Trim extra nodes if grid is slightly larger than n.
    if G.number_of_nodes() > n:
        nodes_to_remove = list(range(n, G.number_of_nodes()))
        G.remove_nodes_from(nodes_to_remove)

    return G


def generate_all_synthetic(
    n: int, avg_degree: float, seed: int = 42
) -> dict[str, nx.Graph]:
    """Generate all four synthetic models matched to n and avg_degree."""
    return {
        "Erdos-Renyi": generate_erdos_renyi(n, avg_degree, seed),
        "Watts-Strogatz": generate_watts_strogatz(n, avg_degree, seed),
        "Barabasi-Albert": generate_barabasi_albert(n, avg_degree, seed),
        "Grid-Lattice": generate_grid(n, seed),
    }


SYNTHETIC_DESCRIPTIONS = {
    "Erdos-Renyi": {
        "generation": "Each edge exists independently with probability p ~ avg_degree/(n-1).",
        "parameters": "n (nodes), p (edge probability derived from target average degree).",
        "real_world": "Baseline random wiring; rarely a perfect social model but useful for null hypotheses.",
        "strengths": "Simple, fast, well-studied theory.",
        "limitations": "No hubs, low clustering, narrow degree distribution.",
    },
    "Watts-Strogatz": {
        "generation": "Ring lattice with local neighbors, then random rewiring.",
        "parameters": "n, k (ring degree from avg_degree), p=0.1 rewiring probability.",
        "real_world": "Networks that are highly clustered yet have short paths.",
        "strengths": "Captures small-world property.",
        "limitations": "Not scale-free; degree distribution remains peaked.",
    },
    "Barabasi-Albert": {
        "generation": "Preferential attachment: new nodes favor high-degree targets.",
        "parameters": "n, m ~ avg_degree/2 edges per new node.",
        "real_world": "Web graphs, citations, influence-driven growth.",
        "strengths": "Produces hub-heavy scale-free-like tails.",
        "limitations": "Lower clustering than many social networks; grows sequentially.",
    },
    "Grid-Lattice": {
        "generation": "Nodes on a 2D grid; edges between adjacent cells.",
        "parameters": "Grid dimensions chosen to approximate n nodes.",
        "real_world": "Spatial networks, physical adjacency, infrastructure grids.",
        "strengths": "Regular structure, high local clustering.",
        "limitations": "Large diameter, uniform degree, no preferential attachment.",
    },
}
