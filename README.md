# NetworkX Playground

A standalone exploratory project for learning **NetworkX**, graph theory concepts, graph metrics, and graph generation. Real SNAP datasets are compared against four classic synthetic models.

## What This Project Does

1. Loads SNAP edge-list graphs from `data/`
2. Prints basic graph statistics and computes structural metrics
3. Generates matched synthetic graphs (same node count, similar average degree)
4. Compares real vs synthetic networks with tables and plots
5. Writes CSV exports, visualizations, and a summary report

## Project Structure

```
NetworkX Playground/
├── main.py           # Run this — orchestrates the full exploration
├── metrics.py        # Graph loading, metrics, NetworkX function notes
├── synthetic.py      # Erdős–Rényi, Watts–Strogatz, BA, Grid generators
├── plots.py          # Degree distributions, comparisons, graph drawings
├── requirements.txt
├── data/             # SNAP edge-list files (.txt.gz)
└── outputs/
    ├── tables/       # Comparison CSV files (one per SNAP graph)
    ├── plots/        # Degree plots, metric bar charts, graph drawings
    └── summary_report.txt
```

## Setup

```bash
pip install -r requirements.txt
```

### SNAP Datasets

Three small datasets are included (or download from [SNAP](https://snap.stanford.edu/data/)):

| File | Description | Nodes | Edges |
|------|-------------|-------|-------|
| `email-Eu-core.txt.gz` | Email network (EU research institution) | ~986 | ~25k |
| `ca-GrQc.txt.gz` | General Relativity collaboration (Arxiv) | ~5.2k | ~14.5k |
| `facebook_combined.txt.gz` | Facebook social circles (combined ego networks) | ~4k | ~88k |

Place any additional SNAP edge-list files in `data/` and add them to `SNAP_FILES` in `main.py`.

## Run

```bash
python main.py
```

## Expected Outputs

### Console
- Basic info for each SNAP graph (nodes, edges, degree stats, density, components)
- Advanced metrics (clustering, transitivity, assortativity, Louvain communities, paths)
- Pandas comparison tables (SNAP vs four synthetic models)
- Short interpretation after each dataset

### Files
- `outputs/tables/comparison_<dataset>.csv` — metric comparison tables
- `outputs/plots/<dataset>/` — degree distributions, log-log plots, metric bar charts, small graph drawings
- `outputs/summary_report.txt` — which model fits best per dataset and key insights

---

## Graph Metrics Explained

### Basic Properties

| Metric | What it measures | High value | Low value |
|--------|------------------|------------|-----------|
| **Nodes / Edges** | Size of the network | Larger network | Smaller network |
| **Average degree** | Mean number of connections per node | Denser local connectivity | Sparse network |
| **Density** | Fraction of possible edges that exist | Nearly complete graph | Very sparse |
| **Connected components** | Number of disconnected pieces | Fragmented network | Single piece |
| **Largest connected component (LCC)** | Biggest reachable subgraph | Most nodes in one piece | Many small islands |

### Structural Metrics

| Metric | What it measures | High value | Low value |
|--------|------------------|------------|-----------|
| **Degree distribution** | How many nodes have degree *k* | Reveals hub structure | — |
| **Degree heterogeneity** | std(degree) / mean(degree) | Hub-dominated, unequal | Uniform degrees |
| **Clustering coefficient** | How often neighbors of a node connect | Tight local groups | Tree-like, open |
| **Transitivity** | Global ratio of triangles to triples | Many closed triangles | Few triangles |
| **Assortativity** | Do similar degrees connect? | Assortative (like-with-like) | Disassortative (hubs to periphery) |
| **Louvain communities** | Groups with dense internal links | Clear community structure | Weak grouping |
| **Modularity** | Quality of community partition | Strong communities | Random-like mixing |
| **Avg shortest path (LCC)** | Typical distance between nodes | Larger world | Small-world |
| **Diameter (LCC)** | Longest shortest path | Stretched network | Compact network |

---

## Synthetic Graph Models

### Erdős–Rényi (G(n, p))
- **Idea:** Each edge appears independently with probability *p*.
- **Parameters:** *n* nodes, *p* ≈ avg_degree / (n−1).
- **Models:** Random null hypothesis; rarely perfect for social nets.
- **Strengths:** Simple, well-understood theory.
- **Limitations:** No hubs, low clustering.

### Watts–Strogatz
- **Idea:** Ring lattice + random rewiring.
- **Parameters:** *n*, *k* (ring degree), *p* (rewire probability, default 0.1).
- **Models:** Small-world social and neural networks.
- **Strengths:** High clustering + short paths.
- **Limitations:** Narrow degree distribution.

### Barabási–Albert
- **Idea:** Preferential attachment — new nodes favor high-degree targets.
- **Parameters:** *n*, *m* edges per new node ≈ avg_degree/2.
- **Models:** Web, citations, influence networks.
- **Strengths:** Scale-free-like hubs.
- **Limitations:** Lower clustering than many social graphs.

### 2D Grid / Lattice
- **Idea:** Nodes on a grid; edges between neighbors.
- **Parameters:** Grid dimensions ≈ √n × √n.
- **Models:** Roads, spatial contact, chip layout.
- **Strengths:** Regular, high local clustering.
- **Limitations:** Large diameter, uniform degree.

---

## Important NetworkX Functions Used

| Function | What it does | Returns | Notes |
|----------|--------------|---------|-------|
| `nx.read_edgelist()` | Load edges from text file | `Graph` | Use `comments='#'` for SNAP headers |
| `nx.Graph()` | Undirected graph container | Empty graph | Ignores duplicate edges |
| `G.number_of_nodes()` / `number_of_edges()` | Count nodes and edges | `int` | — |
| `G.degree()` | Degree per node | `DegreeView` | Iterate as `(node, degree)` |
| `nx.density(G)` | Edge fraction of complete graph | `float` in [0,1] | — |
| `nx.connected_components(G)` | Connected pieces | Iterator of node sets | — |
| `G.subgraph(nodes)` | Extract induced subgraph | `Graph` | Copy for safe mutation |
| `nx.average_clustering(G)` | Mean local clustering | `float` | Social cohesion |
| `nx.transitivity(G)` | Global clustering | `float` | Triangle density |
| `nx.degree_assortativity_coefficient(G)` | Degree correlation | `float` in [-1,1] | Needs edges |
| `nx.average_shortest_path_length(G)` | Mean path length | `float` | **Requires connected G** |
| `nx.diameter(G)` | Longest shortest path | `int` | **Requires connected G** |
| `nx.erdos_renyi_graph(n, p)` | Random graph | `Graph` | — |
| `nx.watts_strogatz_graph(n, k, p)` | Small-world graph | `Graph` | *k* must be even |
| `nx.barabasi_albert_graph(n, m)` | Scale-free graph | `Graph` | Sequential growth |
| `nx.grid_2d_graph(rows, cols)` | 2D lattice | `Graph` | Tuple node labels |
| `nx.spring_layout(G)` | Force-directed layout | `dict` node→(x,y) | For visualization |
| `nx.draw_networkx()` | Render graph | — | Uses matplotlib |

| `nx.community.louvain_communities(G)` | Greedy modularity optimization | `list` of node sets |
| `nx.community.modularity(G, communities)` | Partition quality score | `float` |

---

## Learning Goals

After running this project you should understand:

- How to load and inspect graphs in NetworkX
- What common metrics tell you about real networks
- How synthetic models differ and what each is good for
- Why no single model captures all properties of social networks

## License

Educational / exploratory use. SNAP datasets have their own citation requirements — see [snap.stanford.edu/data](https://snap.stanford.edu/data/).
