#!/usr/bin/env python3
"""
Extract the DAG structure from Mathlib 4 using LeanDojo.

This script traces a Mathlib 4 repository and extracts:
1. File-level dependency graph
2. Theorem-level dependency graph (premises used in proofs)
3. Declaration-level dependencies

Usage:
    python extract_mathlib_dag.py --commit <commit_hash> --output-dir ./output
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import networkx as nx
from tqdm import tqdm
from loguru import logger

from lean_dojo import LeanGitRepo, trace, TracedRepo, TracedFile, TracedTheorem


@dataclass
class TheoremInfo:
    """Information about a traced theorem."""
    full_name: str
    file_path: str
    start_line: int
    end_line: int
    premises: List[str]  # Fully qualified names of premises used
    statement: str


@dataclass
class FileInfo:
    """Information about a traced file."""
    path: str
    imports: List[str]
    theorems: List[str]  # List of theorem names in this file


class MathlibDAGExtractor:
    """Extract DAG structure from Mathlib 4 using LeanDojo."""

    MATHLIB4_URL = "https://github.com/leanprover-community/mathlib4"

    def __init__(
        self,
        commit: str,
        output_dir: Path,
        cache_dir: Optional[Path] = None,
    ):
        self.commit = commit
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir or Path.home() / ".cache" / "lean_dojo"

        self.traced_repo: Optional[TracedRepo] = None
        self.file_graph: nx.DiGraph = nx.DiGraph()
        self.theorem_graph: nx.DiGraph = nx.DiGraph()
        self.theorems: Dict[str, TheoremInfo] = {}
        self.files: Dict[str, FileInfo] = {}

    def trace_mathlib(self) -> TracedRepo:
        """Trace Mathlib 4 repository."""
        logger.info(f"Tracing Mathlib 4 at commit {self.commit}")

        repo = LeanGitRepo(self.MATHLIB4_URL, self.commit)
        # Let LeanDojo handle caching automatically (don't specify dst_dir if cache exists)
        traced = trace(repo)

        self.traced_repo = traced
        logger.info("Tracing complete")
        return traced

    def extract_file_dependencies(self) -> nx.DiGraph:
        """Extract file-level dependency graph."""
        if self.traced_repo is None:
            raise RuntimeError("Must call trace_mathlib() first")

        logger.info("Extracting file dependency graph")

        # LeanDojo provides file_dep_graph attribute
        if hasattr(self.traced_repo, 'file_dep_graph'):
            # Copy the graph structure
            for node in self.traced_repo.file_dep_graph.nodes():
                self.file_graph.add_node(node)
            for u, v in self.traced_repo.file_dep_graph.edges():
                self.file_graph.add_edge(u, v)
        else:
            # Build from traced files
            for traced_file in tqdm(self.traced_repo.traced_files, desc="Processing files"):
                file_path = str(traced_file.path)
                self.file_graph.add_node(file_path)

                # Get imports from the file
                if hasattr(traced_file, 'imports'):
                    for imp in traced_file.imports:
                        self.file_graph.add_edge(file_path, str(imp))

        logger.info(f"File graph: {self.file_graph.number_of_nodes()} nodes, "
                   f"{self.file_graph.number_of_edges()} edges")
        return self.file_graph

    def extract_theorem_dependencies(self) -> nx.DiGraph:
        """Extract theorem-level dependency graph based on premises."""
        if self.traced_repo is None:
            raise RuntimeError("Must call trace_mathlib() first")

        logger.info("Extracting theorem dependencies")

        traced_theorems = list(self.traced_repo.get_traced_theorems())

        for thm in tqdm(traced_theorems, desc="Processing theorems"):
            full_name = thm.theorem.full_name

            # Skip theorems with None full_name
            if full_name is None:
                continue

            # Get premises used in this theorem's proof
            try:
                premises = thm.get_premise_full_names()
                # Filter out None premises
                premises = [p for p in premises if p is not None]
            except Exception as e:
                logger.warning(f"Failed to get premises for {full_name}: {e}")
                premises = []

            # Store theorem info
            self.theorems[full_name] = TheoremInfo(
                full_name=full_name,
                file_path=str(thm.theorem.file_path),
                start_line=thm.theorem.start if hasattr(thm.theorem, 'start') else 0,
                end_line=thm.theorem.end if hasattr(thm.theorem, 'end') else 0,
                premises=list(premises),
                statement=str(thm.theorem.statement) if hasattr(thm.theorem, 'statement') else "",
            )

            # Add to graph
            self.theorem_graph.add_node(full_name)
            for premise in premises:
                self.theorem_graph.add_edge(full_name, premise)

        logger.info(f"Theorem graph: {self.theorem_graph.number_of_nodes()} nodes, "
                   f"{self.theorem_graph.number_of_edges()} edges")
        return self.theorem_graph

    def extract_all(self) -> Tuple[nx.DiGraph, nx.DiGraph]:
        """Extract both file and theorem dependency graphs."""
        self.trace_mathlib()
        self.extract_file_dependencies()
        self.extract_theorem_dependencies()
        return self.file_graph, self.theorem_graph

    def save_graphs(self, formats: List[str] = None):
        """Save extracted graphs in various formats."""
        if formats is None:
            formats = ["json", "edgelist", "graphml"]

        logger.info(f"Saving graphs to {self.output_dir}")

        # Save file dependency graph
        file_graph_base = self.output_dir / "file_dependencies"
        if "json" in formats:
            self._save_graph_json(self.file_graph, file_graph_base.with_suffix(".json"))
        if "edgelist" in formats:
            nx.write_edgelist(self.file_graph, file_graph_base.with_suffix(".edgelist"))
        if "graphml" in formats:
            nx.write_graphml(self.file_graph, file_graph_base.with_suffix(".graphml"))

        # Save theorem dependency graph
        thm_graph_base = self.output_dir / "theorem_dependencies"
        if "json" in formats:
            self._save_graph_json(self.theorem_graph, thm_graph_base.with_suffix(".json"))
        if "edgelist" in formats:
            nx.write_edgelist(self.theorem_graph, thm_graph_base.with_suffix(".edgelist"))
        if "graphml" in formats:
            nx.write_graphml(self.theorem_graph, thm_graph_base.with_suffix(".graphml"))

        # Save theorem metadata
        theorems_path = self.output_dir / "theorems.json"
        with open(theorems_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.theorems.items()}, f, indent=2)

        logger.info("Graphs saved successfully")

    def _save_graph_json(self, graph: nx.DiGraph, path: Path):
        """Save graph as JSON with adjacency list format."""
        data = {
            "nodes": list(graph.nodes()),
            "edges": list(graph.edges()),
            "adjacency": {node: list(graph.successors(node)) for node in graph.nodes()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def compute_statistics(self) -> Dict:
        """Compute basic statistics about the DAG."""
        stats = {
            "file_graph": {
                "num_nodes": self.file_graph.number_of_nodes(),
                "num_edges": self.file_graph.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(self.file_graph),
            },
            "theorem_graph": {
                "num_nodes": self.theorem_graph.number_of_nodes(),
                "num_edges": self.theorem_graph.number_of_edges(),
                "num_theorems_with_info": len(self.theorems),
            }
        }

        # Compute in/out degree distributions
        if self.file_graph.number_of_nodes() > 0:
            in_degrees = [d for _, d in self.file_graph.in_degree()]
            out_degrees = [d for _, d in self.file_graph.out_degree()]
            stats["file_graph"]["avg_in_degree"] = sum(in_degrees) / len(in_degrees)
            stats["file_graph"]["avg_out_degree"] = sum(out_degrees) / len(out_degrees)
            stats["file_graph"]["max_in_degree"] = max(in_degrees)
            stats["file_graph"]["max_out_degree"] = max(out_degrees)

        if self.theorem_graph.number_of_nodes() > 0:
            in_degrees = [d for _, d in self.theorem_graph.in_degree()]
            out_degrees = [d for _, d in self.theorem_graph.out_degree()]
            stats["theorem_graph"]["avg_in_degree"] = sum(in_degrees) / len(in_degrees)
            stats["theorem_graph"]["avg_out_degree"] = sum(out_degrees) / len(out_degrees)
            stats["theorem_graph"]["max_in_degree"] = max(in_degrees)
            stats["theorem_graph"]["max_out_degree"] = max(out_degrees)

            # Average premises per theorem
            premise_counts = [len(t.premises) for t in self.theorems.values()]
            if premise_counts:
                stats["theorem_graph"]["avg_premises"] = sum(premise_counts) / len(premise_counts)

        return stats


def main():
    parser = argparse.ArgumentParser(description="Extract Mathlib 4 DAG using LeanDojo")
    parser.add_argument(
        "--commit",
        type=str,
        default="v4.12.0",  # Use a recent stable tag
        help="Git commit hash or tag for Mathlib 4"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save extracted graphs"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for LeanDojo tracing"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["json", "edgelist"],
        choices=["json", "edgelist", "graphml"],
        help="Output formats for graphs"
    )
    parser.add_argument(
        "--file-deps-only",
        action="store_true",
        help="Only extract file-level dependencies (faster)"
    )

    args = parser.parse_args()

    extractor = MathlibDAGExtractor(
        commit=args.commit,
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    # Trace and extract
    extractor.trace_mathlib()
    extractor.extract_file_dependencies()

    if not args.file_deps_only:
        extractor.extract_theorem_dependencies()

    # Save results
    extractor.save_graphs(formats=args.formats)

    # Print statistics
    stats = extractor.compute_statistics()
    stats_path = Path(args.output_dir) / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n=== Extraction Statistics ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
