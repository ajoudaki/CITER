#!/usr/bin/env python3
"""
Analyze and visualize the extracted Mathlib 4 DAG.

This script provides utilities for:
1. Loading extracted DAGs
2. Computing graph metrics (centrality, clustering, etc.)
3. Finding important theorems/files
4. Visualizing subgraphs
5. Exporting to various formats

Usage:
    python analyze_dag.py --input-dir ./output --analysis all
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from collections import Counter


def load_graph_json(path: Path) -> nx.DiGraph:
    """Load a graph from JSON format."""
    with open(path, 'r') as f:
        data = json.load(f)

    G = nx.DiGraph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])
    return G


def load_theorems(path: Path) -> Dict:
    """Load theorem metadata."""
    with open(path, 'r') as f:
        return json.load(f)


class DAGAnalyzer:
    """Analyze Mathlib DAG structure."""

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
        self.file_graph: Optional[nx.DiGraph] = None
        self.theorem_graph: Optional[nx.DiGraph] = None
        self.theorems: Dict = {}

    def load(self):
        """Load all extracted data."""
        file_path = self.input_dir / "file_dependencies.json"
        if file_path.exists():
            self.file_graph = load_graph_json(file_path)
            print(f"Loaded file graph: {self.file_graph.number_of_nodes()} nodes, "
                  f"{self.file_graph.number_of_edges()} edges")

        thm_path = self.input_dir / "theorem_dependencies.json"
        if thm_path.exists():
            self.theorem_graph = load_graph_json(thm_path)
            print(f"Loaded theorem graph: {self.theorem_graph.number_of_nodes()} nodes, "
                  f"{self.theorem_graph.number_of_edges()} edges")

        theorems_path = self.input_dir / "theorems.json"
        if theorems_path.exists():
            self.theorems = load_theorems(theorems_path)
            print(f"Loaded {len(self.theorems)} theorem metadata entries")

    def compute_centrality(self, graph: nx.DiGraph, top_k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Compute various centrality measures."""
        results = {}

        print("Computing PageRank...")
        pagerank = nx.pagerank(graph)
        results["pagerank"] = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]

        print("Computing in-degree centrality...")
        in_degree = dict(graph.in_degree())
        results["in_degree"] = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:top_k]

        print("Computing out-degree centrality...")
        out_degree = dict(graph.out_degree())
        results["out_degree"] = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Betweenness is expensive for large graphs
        if graph.number_of_nodes() < 10000:
            print("Computing betweenness centrality (this may take a while)...")
            betweenness = nx.betweenness_centrality(graph)
            results["betweenness"] = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return results

    def find_roots_and_leaves(self, graph: nx.DiGraph) -> Dict[str, List[str]]:
        """Find root nodes (no incoming edges) and leaf nodes (no outgoing edges)."""
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        return {
            "roots": roots,
            "leaves": leaves,
            "num_roots": len(roots),
            "num_leaves": len(leaves),
        }

    def find_strongly_connected_components(self, graph: nx.DiGraph) -> Dict:
        """Find strongly connected components (cycles if not a pure DAG)."""
        sccs = list(nx.strongly_connected_components(graph))
        non_trivial = [scc for scc in sccs if len(scc) > 1]

        return {
            "num_components": len(sccs),
            "num_non_trivial": len(non_trivial),
            "largest_non_trivial": max([len(scc) for scc in non_trivial]) if non_trivial else 0,
            "non_trivial_components": [list(scc) for scc in non_trivial[:10]],  # First 10
        }

    def compute_depth_statistics(self, graph: nx.DiGraph) -> Dict:
        """Compute depth/level statistics for a DAG."""
        if not nx.is_directed_acyclic_graph(graph):
            print("Warning: Graph has cycles, skipping depth analysis")
            return {"error": "Graph has cycles"}

        # Find roots
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]

        if not roots:
            return {"error": "No root nodes found"}

        # Compute longest path from any root to each node
        depths = {}
        for root in roots:
            for node in nx.descendants(graph, root) | {root}:
                try:
                    paths = list(nx.all_simple_paths(graph, root, node))
                    if paths:
                        max_depth = max(len(p) - 1 for p in paths)
                        depths[node] = max(depths.get(node, 0), max_depth)
                except nx.NetworkXError:
                    pass

        if not depths:
            return {"error": "Could not compute depths"}

        depth_values = list(depths.values())
        return {
            "max_depth": max(depth_values),
            "avg_depth": np.mean(depth_values),
            "depth_distribution": dict(Counter(depth_values)),
        }

    def get_subgraph(self, graph: nx.DiGraph, center_node: str, radius: int = 2) -> nx.DiGraph:
        """Extract a subgraph around a center node."""
        # Get predecessors and successors within radius
        nodes = {center_node}

        # BFS for predecessors
        current = {center_node}
        for _ in range(radius):
            new_nodes = set()
            for n in current:
                new_nodes.update(graph.predecessors(n))
            nodes.update(new_nodes)
            current = new_nodes

        # BFS for successors
        current = {center_node}
        for _ in range(radius):
            new_nodes = set()
            for n in current:
                new_nodes.update(graph.successors(n))
            nodes.update(new_nodes)
            current = new_nodes

        return graph.subgraph(nodes).copy()

    def find_theorems_by_file(self, file_pattern: str) -> List[str]:
        """Find all theorems in files matching a pattern."""
        matching = []
        for name, info in self.theorems.items():
            if file_pattern in info.get("file_path", ""):
                matching.append(name)
        return matching

    def get_dependency_chain(self, graph: nx.DiGraph, node: str, direction: str = "both") -> Dict:
        """Get the dependency chain for a node."""
        result = {"node": node}

        if direction in ["predecessors", "both"]:
            result["dependencies"] = list(graph.predecessors(node))
            result["all_ancestors"] = list(nx.ancestors(graph, node))

        if direction in ["successors", "both"]:
            result["dependents"] = list(graph.successors(node))
            result["all_descendants"] = list(nx.descendants(graph, node))

        return result

    def export_to_dot(self, graph: nx.DiGraph, output_path: Path, max_nodes: int = 500):
        """Export graph to DOT format for Graphviz visualization."""
        if graph.number_of_nodes() > max_nodes:
            print(f"Graph too large ({graph.number_of_nodes()} nodes), "
                  f"sampling {max_nodes} most connected nodes")
            # Sample most connected nodes
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            nodes = [n for n, _ in top_nodes]
            graph = graph.subgraph(nodes).copy()

        nx.drawing.nx_pydot.write_dot(graph, output_path)
        print(f"Exported to {output_path}")

    def analyze_premise_patterns(self) -> Dict:
        """Analyze patterns in premise usage."""
        if not self.theorems:
            return {"error": "No theorem data loaded"}

        premise_counts = Counter()
        premises_per_theorem = []

        for thm_name, thm_info in self.theorems.items():
            premises = thm_info.get("premises", [])
            premises_per_theorem.append(len(premises))
            premise_counts.update(premises)

        return {
            "total_theorems": len(self.theorems),
            "avg_premises_per_theorem": np.mean(premises_per_theorem) if premises_per_theorem else 0,
            "max_premises": max(premises_per_theorem) if premises_per_theorem else 0,
            "min_premises": min(premises_per_theorem) if premises_per_theorem else 0,
            "most_used_premises": premise_counts.most_common(50),
            "premise_usage_distribution": dict(Counter(premises_per_theorem)),
        }

    def run_full_analysis(self) -> Dict:
        """Run complete analysis on all graphs."""
        results = {}

        if self.file_graph:
            print("\n=== File Graph Analysis ===")
            results["file_graph"] = {
                "roots_leaves": self.find_roots_and_leaves(self.file_graph),
                "scc": self.find_strongly_connected_components(self.file_graph),
            }

            if self.file_graph.number_of_nodes() < 50000:
                results["file_graph"]["centrality"] = self.compute_centrality(self.file_graph)

        if self.theorem_graph:
            print("\n=== Theorem Graph Analysis ===")
            results["theorem_graph"] = {
                "roots_leaves": self.find_roots_and_leaves(self.theorem_graph),
                "scc": self.find_strongly_connected_components(self.theorem_graph),
            }

            if self.theorem_graph.number_of_nodes() < 50000:
                results["theorem_graph"]["centrality"] = self.compute_centrality(self.theorem_graph)

        if self.theorems:
            print("\n=== Premise Analysis ===")
            results["premise_patterns"] = self.analyze_premise_patterns()

        return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Mathlib 4 DAG")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./output",
        help="Directory with extracted graphs"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for analysis results (JSON)"
    )
    parser.add_argument(
        "--analysis",
        type=str,
        choices=["all", "centrality", "structure", "premises"],
        default="all",
        help="Type of analysis to run"
    )
    parser.add_argument(
        "--subgraph-center",
        type=str,
        default=None,
        help="Extract subgraph around this node"
    )
    parser.add_argument(
        "--subgraph-radius",
        type=int,
        default=2,
        help="Radius for subgraph extraction"
    )

    args = parser.parse_args()

    analyzer = DAGAnalyzer(Path(args.input_dir))
    analyzer.load()

    if args.subgraph_center:
        # Extract and save subgraph
        graph = analyzer.theorem_graph or analyzer.file_graph
        if graph and args.subgraph_center in graph:
            subgraph = analyzer.get_subgraph(graph, args.subgraph_center, args.subgraph_radius)
            output_path = Path(args.input_dir) / f"subgraph_{args.subgraph_center.replace('.', '_')}.json"

            data = {
                "nodes": list(subgraph.nodes()),
                "edges": list(subgraph.edges()),
                "center": args.subgraph_center,
                "radius": args.subgraph_radius,
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Subgraph saved to {output_path}")
        else:
            print(f"Node {args.subgraph_center} not found in graph")
        return

    # Run analysis
    results = analyzer.run_full_analysis()

    # Save or print results
    output_path = args.output_file or (Path(args.input_dir) / "analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    if "file_graph" in results:
        fg = results["file_graph"]
        print(f"File graph: {fg['roots_leaves']['num_roots']} roots, "
              f"{fg['roots_leaves']['num_leaves']} leaves")

    if "theorem_graph" in results:
        tg = results["theorem_graph"]
        print(f"Theorem graph: {tg['roots_leaves']['num_roots']} roots, "
              f"{tg['roots_leaves']['num_leaves']} leaves")

    if "premise_patterns" in results:
        pp = results["premise_patterns"]
        print(f"Premises: avg {pp['avg_premises_per_theorem']:.1f} per theorem")


if __name__ == "__main__":
    main()
