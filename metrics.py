"""
Graph metrics for NetworkX exploration.

Each function computes one metric and includes comments explaining
what it measures, why it is useful, and how to interpret high/low values.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np


def load_snap_graph(filepath: str) -> nx.Graph:
    """
    Load a SNAP edge-list file into an undirected NetworkX graph.

    nx.read_edgelist():
      - Reads lines of 'node1 node2' pairs from a text file.
      - Returns a Graph (or DiGraph if create_using is set).
      - Assumes nodes are strings unless nodetype is given.

    We use comments='#' because SNAP files often start with header lines.
    nodetype=int converts node labels to integers.
    We treat SNAP social networks as undirected for comparison with
    classic random-graph models (ER, WS, BA, grid).
    """
    G = nx.read_edgelist(
        filepath,
        comments="#",
        nodetype=int,
        create_using=nx.Graph(),
    )
    # nx.Graph() removes duplicate edges and self-loops automatically.
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def basic_graph_info(G: nx.Graph) -> dict[str, Any]:
    """Print-friendly basic statistics for any graph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # nx.degree() returns a DegreeView: node -> degree mapping.
    # list(G.degree()) gives [(node, degree), ...] pairs.
    degrees = [d for _, d in G.degree()]

    # nx.number_connected_components() counts disconnected pieces.
    components = list(nx.connected_components(G))
    num_components = len(components)

    # Largest connected component (LCC) is used for path-based metrics.
    lcc_nodes = max(components, key=len) if components else set()
    lcc = G.subgraph(lcc_nodes).copy()

    # Density = actual edges / possible edges.
    # nx.density(G): returns m / (n*(n-1)/2) for undirected graphs.
    # High density -> nearly complete; low density -> sparse network.
    density = nx.density(G)

    return {
        "nodes": n,
        "edges": m,
        "average_degree": (2 * m / n) if n > 0 else 0.0,
        "degree_min": int(min(degrees)) if degrees else 0,
        "degree_max": int(max(degrees)) if degrees else 0,
        "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
        "degree_median": float(np.median(degrees)) if degrees else 0.0,
        "density": density,
        "connected_components": num_components,
        "lcc_nodes": lcc.number_of_nodes(),
        "lcc_edges": lcc.number_of_edges(),
        "lcc_fraction": (lcc.number_of_nodes() / n) if n > 0 else 0.0,
        "degrees": degrees,
        "lcc": lcc,
    }


def degree_distribution(degrees: list[int]) -> dict[int, int]:
    """
    Degree distribution: how many nodes have each degree k.

    Useful for spotting scale-free tails (few hubs, many low-degree nodes)
    vs Poisson-like random graphs (degrees cluster near the mean).
    """
    dist: dict[int, int] = {}
    for d in degrees:
        dist[d] = dist.get(d, 0) + 1
    return dist


def degree_heterogeneity(degrees: list[int]) -> float:
    """
    Degree heterogeneity (coefficient of variation of degree).

    Measures: std(degree) / mean(degree).
    Useful: summarizes how unequal degrees are across nodes.
    High values: hub-dominated networks (scale-free-like).
    Low values: more uniform degrees (lattice/random-like).
    """
    if not degrees:
        return 0.0
    mean_d = np.mean(degrees)
    if mean_d == 0:
        return 0.0
    return float(np.std(degrees) / mean_d)


def clustering_coefficient(G: nx.Graph) -> float:
    """
    Average clustering coefficient.

    nx.average_clustering(G):
      - For each node, measures how connected its neighbors are to each other.
      - Returns the mean of local clustering coefficients.

    Useful: detects "friends of friends are also friends" (triadic closure).
    High values: strong local clustering (social groups, grids).
    Low values: few triangles (tree-like or random sparse graphs).
    """
    if G.number_of_nodes() == 0:
        return 0.0
    return float(nx.average_clustering(G))


def transitivity(G: nx.Graph) -> float:
    """
    Global transitivity (clustering coefficient of the whole graph).

    nx.transitivity(G):
      - Ratio of closed triangles to connected triples.
      - Returns a single graph-level value in [0, 1].

    Useful: complements average clustering; less sensitive to low-degree nodes.
    High values: many triangles globally.
    Low values: open, tree-like structure.
    """
    if G.number_of_nodes() == 0:
        return 0.0
    return float(nx.transitivity(G))


def assortativity(G: nx.Graph) -> float:
    """
    Degree assortativity coefficient.

    nx.degree_assortativity_coefficient(G):
      - Pearson correlation of degrees at both ends of edges.
      - Returns value in [-1, 1].

    Useful: reveals whether hubs connect to hubs or to low-degree nodes.
    High positive: assortative mixing (similar degrees connect).
    Negative: disassortative (hubs link to periphery) common in biological/tech nets.
    Near zero: degree-independent wiring (random-like).
    """
    if G.number_of_edges() == 0:
        return 0.0
    return float(nx.degree_assortativity_coefficient(G))


def louvain_communities(G: nx.Graph) -> tuple[dict[Any, int], int, float]:
    """
    Community detection with the Louvain method.

    nx.community.louvain_communities(G, seed=42):
      - Greedy modularity optimization (built into NetworkX).
      - Returns a list of sets; each set is one community.

    nx.community.modularity(G, communities):
      - Measures quality of community split (density within vs between groups).
      - Higher modularity -> stronger community structure.

    Assumption: works on undirected graphs; disconnected graphs still get
    partitions but interpretation is easier on a single large component.

    Returns: (partition, num_communities, modularity)
    """
    if G.number_of_nodes() == 0:
        return {}, 0, 0.0

    # Louvain on empty-edge graphs can fail; handle gracefully.
    if G.number_of_edges() == 0:
        partition = {node: 0 for node in G.nodes()}
        return partition, 1, 0.0

    communities = list(nx.community.louvain_communities(G, seed=42))
    partition = {node: i for i, comm in enumerate(communities) for node in comm}
    num_communities = len(communities)
    modularity = nx.community.modularity(G, communities)
    return partition, num_communities, float(modularity)


def path_metrics_on_lcc(lcc: nx.Graph) -> tuple[float | None, int | None]:
    """
    Average shortest path length and diameter on the largest connected component.

    nx.average_shortest_path_length(G):
      - Mean of shortest paths between all node pairs.
      - Requires connected G.

    nx.diameter(G):
      - Longest shortest path between any two nodes.
      - Requires connected G.

    Useful: "small-world" networks have short paths despite local clustering.
    Low avg path / diameter: small world, easy information spread.
    High values: large or lattice-like distances.

    Returns (None, None) if LCC has fewer than 2 nodes.
    """
    if lcc.number_of_nodes() < 2:
        return None, None

    avg_path = float(nx.average_shortest_path_length(lcc))
    diameter = int(nx.diameter(lcc))
    return avg_path, diameter


def compute_all_metrics(G: nx.Graph, name: str) -> dict[str, Any]:
    """Compute every metric used in SNAP vs synthetic comparisons."""
    info = basic_graph_info(G)
    lcc = info["lcc"]
    degrees = info["degrees"]

    partition, num_communities, modularity = louvain_communities(G)
    avg_path, diameter = path_metrics_on_lcc(lcc)

    return {
        "name": name,
        "nodes": info["nodes"],
        "edges": info["edges"],
        "average_degree": info["average_degree"],
        "degree_min": info["degree_min"],
        "degree_max": info["degree_max"],
        "degree_mean": info["degree_mean"],
        "degree_median": info["degree_median"],
        "density": info["density"],
        "connected_components": info["connected_components"],
        "lcc_nodes": info["lcc_nodes"],
        "lcc_fraction": info["lcc_fraction"],
        "degree_heterogeneity": degree_heterogeneity(degrees),
        "clustering_coefficient": clustering_coefficient(G),
        "transitivity": transitivity(G),
        "assortativity": assortativity(G),
        "num_communities": num_communities,
        "modularity": modularity,
        "avg_shortest_path_lcc": avg_path,
        "diameter_lcc": diameter,
        "degrees": degrees,
        "degree_distribution": degree_distribution(degrees),
        "partition": partition,
        "graph": G,
        "lcc": lcc,
    }


def print_basic_info(metrics: dict[str, Any]) -> None:
    """Nicely formatted console output for basic graph properties."""
    print(f"\n{'=' * 60}")
    print(f"Graph: {metrics['name']}")
    print(f"{'=' * 60}")
    print(f"  Nodes:                  {metrics['nodes']:,}")
    print(f"  Edges:                  {metrics['edges']:,}")
    print(f"  Average degree:         {metrics['average_degree']:.4f}")
    print(f"  Degree (min/mean/med/max): {metrics['degree_min']} / "
          f"{metrics['degree_mean']:.2f} / {metrics['degree_median']:.2f} / "
          f"{metrics['degree_max']}")
    print(f"  Density:                {metrics['density']:.6f}")
    print(f"  Connected components:   {metrics['connected_components']}")
    print(f"  Largest component:      {metrics['lcc_nodes']:,} nodes "
          f"({metrics['lcc_fraction']:.1%} of graph)")
