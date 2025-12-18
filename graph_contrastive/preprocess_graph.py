"""
Preprocessing Script for Graph-Contrastive Learning.

Converts raw JSON/JSONL data sources into optimized binary formats:
  - nodes.arrow: Apache Arrow IPC for zero-copy text access
  - edges.npy: NumPy memory-mapped array for topology

Usage:
    python preprocess_graph.py --se-graph data/SE/se_graph.json \
                               --arxiv-dag data/arxiv/extracted_envs_dag_from_theorem_papers.jsonl \
                               --output-dir data/processed
"""

import argparse
import json
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
import hashlib
import html
import re

try:
    from graph_config import (
        EdgeType, SourceType, EDGE_REGISTRY,
        normalize_arxiv_env, is_provable_env
    )
except ImportError:
    from .graph_config import (
        EdgeType, SourceType, EDGE_REGISTRY,
        normalize_arxiv_env, is_provable_env
    )


# =============================================================================
# Text Cleaning Utilities
# =============================================================================

def clean_html(text: str) -> str:
    """Remove HTML tags, decode entities, and sanitize URLs that could reveal linked posts."""
    if not text:
        return ""
    # Decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Sanitize URLs (could reveal linked posts in StackExchange)
    text = re.sub(r'https?://[^\s<>\[\]]+', '', text)
    text = re.sub(r'www\.[^\s<>\[\]]+', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_latex(text: str) -> str:
    """Clean LaTeX text and sanitize cross-references that could reveal edge structure."""
    if not text:
        return ""

    # Remove cross-references (these reveal edge/dependency structure)
    text = re.sub(r'\\eqref\{[^}]*\}', '', text)
    text = re.sub(r'\\ref\{[^}]*\}', '', text)
    text = re.sub(r'\\autoref\{[^}]*\}', '', text)
    text = re.sub(r'\\[Cc]ref\{[^}]*\}', '', text)
    text = re.sub(r'\\pageref\{[^}]*\}', '', text)
    text = re.sub(r'\\hyperref\[[^\]]*\]\{[^}]*\}', '', text)
    text = re.sub(r'\\hyperref\[[^\]]*\]', '', text)

    # Remove labels entirely (they define reference targets)
    text = re.sub(r'\\label\{[^}]*\}', '', text)

    # Remove citations (reveal bibliography links)
    text = re.sub(r'\\cite[a-z]*\{[^}]*\}', '', text)

    # Remove common LaTeX formatting commands but preserve content
    text = re.sub(r'\\(?:emph|textbf|textit|text)\{([^}]*)\}', r'\1', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_cluster_id(identifier: str) -> int:
    """Compute a deterministic cluster ID from a string identifier."""
    return int(hashlib.md5(identifier.encode()).hexdigest()[:8], 16)


# =============================================================================
# StackExchange Processing
# =============================================================================

def process_stackexchange(
    se_graph_path: str,
    start_global_id: int = 0
) -> Tuple[List[Dict], np.ndarray, Dict[str, int]]:
    """
    Process StackExchange graph JSON into nodes and edges.

    Args:
        se_graph_path: Path to se_graph.json
        start_global_id: Starting global ID for nodes

    Returns:
        nodes: List of node dicts
        edges: numpy array of shape (M, 4) [src, dst, edge_type, weight]
        node_id_map: Mapping from SE node ID (e.g., "math_123") to global_id
    """
    print(f"Loading StackExchange graph from: {se_graph_path}")
    with open(se_graph_path) as f:
        graph = json.load(f)

    nodes = []
    edges_list = []
    node_id_map = {}  # SE node ID -> global_id

    print(f"Processing {len(graph['nodes']):,} SE nodes...")

    # First pass: Create nodes
    global_id = start_global_id
    for se_node_id, node_data in tqdm(graph['nodes'].items(), desc="SE Nodes"):
        node_type = node_data.get('type', 'unknown')
        source = node_data.get('source', 'unknown')

        # Get text content
        if node_type == 'question':
            title = node_data.get('title', '')
            body = node_data.get('body', '')
            text = f"{title}\n\n{body}" if title else body
            # Cluster by first tag or question ID
            tags = node_data.get('tags', [])
            cluster_key = tags[0] if tags else se_node_id
        else:  # answer
            text = node_data.get('body', '')
            # Cluster answers with their parent question
            parent_id = node_data.get('parent_id', se_node_id)
            cluster_key = parent_id

        # Clean HTML
        text = clean_html(text)

        if not text.strip():
            continue  # Skip empty nodes

        nodes.append({
            'global_id': global_id,
            'text': text,
            'cluster_id': compute_cluster_id(cluster_key),
            'source_type': SourceType.STACKEXCHANGE,
            'original_id': se_node_id,
            'node_type': node_type,
        })

        node_id_map[se_node_id] = global_id
        global_id += 1

    print(f"Processing SE edges...")

    # Second pass: Create edges
    for src_se_id, edge_list in tqdm(graph['edges'].items(), desc="SE Edges"):
        if src_se_id not in node_id_map:
            continue

        src_global = node_id_map[src_se_id]

        for dst_se_id, label in edge_list:
            if dst_se_id not in node_id_map:
                continue

            dst_global = node_id_map[dst_se_id]

            # Map label to edge type and weight
            if label == 'accepted_answer':
                edge_type = EdgeType.SE_ANSWER
                weight = 2.0  # Higher weight for accepted
            elif label == 'voted_answer':
                edge_type = EdgeType.SE_ANSWER
                weight = 1.0
            elif label == 'duplicate':
                edge_type = EdgeType.SE_DUPLICATE
                weight = 1.0
                # Add bidirectional edge
                edges_list.append([dst_global, src_global, edge_type, weight])
            elif label == 'linked':
                edge_type = EdgeType.SE_LINKED
                weight = 0.8
            else:
                continue  # Skip unknown edge types

            edges_list.append([src_global, dst_global, edge_type, weight])

    edges = np.array(edges_list, dtype=np.float32) if edges_list else np.empty((0, 4), dtype=np.float32)

    print(f"SE Processing complete: {len(nodes):,} nodes, {len(edges):,} edges")
    return nodes, edges, node_id_map


# =============================================================================
# ArXiv DAG Processing
# =============================================================================

def process_arxiv(
    arxiv_dag_path: str,
    start_global_id: int = 0,
    max_papers: Optional[int] = None
) -> Tuple[List[Dict], np.ndarray]:
    """
    Process ArXiv DAG JSONL into nodes and edges.

    Args:
        arxiv_dag_path: Path to extracted_envs_dag_from_theorem_papers.jsonl
        start_global_id: Starting global ID for nodes
        max_papers: Maximum number of papers to process (for testing)

    Returns:
        nodes: List of node dicts
        edges: numpy array of shape (M, 4) [src, dst, edge_type, weight]
    """
    print(f"Loading ArXiv DAG from: {arxiv_dag_path}")

    nodes = []
    edges_list = []
    global_id = start_global_id

    # Count total lines for progress bar
    with open(arxiv_dag_path) as f:
        total_papers = sum(1 for _ in f)
    if max_papers:
        total_papers = min(total_papers, max_papers)

    print(f"Processing {total_papers:,} ArXiv papers...")

    with open(arxiv_dag_path) as f:
        for paper_idx, line in enumerate(tqdm(f, total=total_papers, desc="ArXiv Papers")):
            if max_papers and paper_idx >= max_papers:
                break

            paper = json.loads(line)

            if not paper.get('dag_valid', True):
                continue

            paper_nodes = paper.get('nodes', [])
            paper_edges = paper.get('edges', [])

            if len(paper_nodes) < 2:
                continue

            # Create cluster ID for this paper
            filenames = paper.get('filenames', [])
            paper_cluster_key = filenames[0] if filenames else f"paper_{paper_idx}"
            paper_cluster_id = compute_cluster_id(paper_cluster_key)

            # Map local node IDs to global IDs
            local_to_global = {}

            # First pass: Create nodes for this paper
            for node in paper_nodes:
                node_id = node.get('id', '')
                env = normalize_arxiv_env(node.get('env', 'unknown'))
                text = node.get('text', '')

                # Clean LaTeX
                text = clean_latex(text)

                if not text.strip() or len(text) < 10:
                    continue  # Skip empty or very short nodes

                nodes.append({
                    'global_id': global_id,
                    'text': text,
                    'cluster_id': paper_cluster_id,
                    'source_type': SourceType.ARXIV,
                    'original_id': node_id,
                    'node_type': env,
                })

                local_to_global[node_id] = global_id
                global_id += 1

            # Second pass: Create edges for this paper
            for edge in paper_edges:
                from_id = edge.get('from', '')
                to_id = edge.get('to', '')

                if from_id not in local_to_global or to_id not in local_to_global:
                    continue

                src_global = local_to_global[from_id]
                dst_global = local_to_global[to_id]

                # Determine edge type based on node types
                from_node = next((n for n in paper_nodes if n['id'] == from_id), None)
                to_node = next((n for n in paper_nodes if n['id'] == to_id), None)

                if not from_node or not to_node:
                    continue

                from_env = normalize_arxiv_env(from_node.get('env', ''))
                to_env = normalize_arxiv_env(to_node.get('env', ''))

                # Proof -> Theorem/Lemma edge
                if from_env == 'proof' and is_provable_env(to_env):
                    # Reverse direction: Theorem -> Proof (for training)
                    edges_list.append([
                        dst_global, src_global,
                        EdgeType.ARXIV_PROOF,
                        EDGE_REGISTRY[EdgeType.ARXIV_PROOF].weight
                    ])
                else:
                    # General dependency edge
                    edges_list.append([
                        src_global, dst_global,
                        EdgeType.ARXIV_DEP,
                        EDGE_REGISTRY[EdgeType.ARXIV_DEP].weight
                    ])

            # Third pass: Create ordinal edges (sequential in paper)
            # Sort nodes by ordinal
            sorted_nodes = sorted(
                [n for n in paper_nodes if n['id'] in local_to_global],
                key=lambda x: x.get('ordinal', 0)
            )

            for i in range(len(sorted_nodes) - 1):
                curr_id = sorted_nodes[i]['id']
                next_id = sorted_nodes[i + 1]['id']

                if curr_id in local_to_global and next_id in local_to_global:
                    edges_list.append([
                        local_to_global[curr_id],
                        local_to_global[next_id],
                        EdgeType.ARXIV_ORDINAL,
                        EDGE_REGISTRY[EdgeType.ARXIV_ORDINAL].weight
                    ])

    edges = np.array(edges_list, dtype=np.float32) if edges_list else np.empty((0, 4), dtype=np.float32)

    print(f"ArXiv Processing complete: {len(nodes):,} nodes, {len(edges):,} edges")
    return nodes, edges


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def merge_and_save(
    se_nodes: List[Dict],
    se_edges: np.ndarray,
    arxiv_nodes: List[Dict],
    arxiv_edges: np.ndarray,
    output_dir: str
):
    """
    Merge nodes and edges from all sources and save to disk.

    Creates:
        - {output_dir}/nodes.arrow: Arrow IPC file with node data
        - {output_dir}/edges.npy: NumPy array with edge data
        - {output_dir}/metadata.json: Dataset statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Merge nodes
    all_nodes = se_nodes + arxiv_nodes
    print(f"\nTotal nodes: {len(all_nodes):,}")

    # Create Arrow table
    print("Creating Arrow table...")
    table = pa.table({
        'global_id': pa.array([n['global_id'] for n in all_nodes], type=pa.int64()),
        'text': pa.array([n['text'] for n in all_nodes], type=pa.large_string()),
        'cluster_id': pa.array([n['cluster_id'] for n in all_nodes], type=pa.int64()),
        'source_type': pa.array([n['source_type'] for n in all_nodes], type=pa.int8()),
        'original_id': pa.array([n.get('original_id', '') for n in all_nodes], type=pa.string()),
        'node_type': pa.array([n.get('node_type', '') for n in all_nodes], type=pa.string()),
    })

    # Save nodes as Arrow IPC (Feather v2)
    nodes_path = output_path / "nodes.arrow"
    print(f"Saving nodes to: {nodes_path}")
    feather.write_feather(table, str(nodes_path))

    # Merge edges
    all_edges = np.vstack([se_edges, arxiv_edges]) if len(se_edges) > 0 and len(arxiv_edges) > 0 else \
                se_edges if len(se_edges) > 0 else arxiv_edges
    print(f"Total edges: {len(all_edges):,}")

    # Save edges as memory-mappable numpy
    edges_path = output_path / "edges.npy"
    print(f"Saving edges to: {edges_path}")
    np.save(str(edges_path), all_edges)

    # Compute statistics
    edge_type_counts = defaultdict(int)
    for edge in all_edges:
        edge_type_counts[int(edge[2])] += 1

    source_type_counts = defaultdict(int)
    for node in all_nodes:
        source_type_counts[int(node['source_type'])] += 1

    # Compute cluster statistics
    cluster_sizes = defaultdict(int)
    for node in all_nodes:
        cluster_sizes[node['cluster_id']] += 1

    metadata = {
        'num_nodes': len(all_nodes),
        'num_edges': len(all_edges),
        'edge_type_counts': {
            EDGE_REGISTRY[k].name if k in EDGE_REGISTRY else str(k): v
            for k, v in edge_type_counts.items()
        },
        'source_type_counts': {
            SourceType(k).name: v for k, v in source_type_counts.items()
        },
        'num_clusters': len(cluster_sizes),
        'avg_cluster_size': sum(cluster_sizes.values()) / len(cluster_sizes) if cluster_sizes else 0,
        'max_cluster_size': max(cluster_sizes.values()) if cluster_sizes else 0,
    }

    metadata_path = output_path / "metadata.json"
    print(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"  Nodes: {metadata['num_nodes']:,}")
    print(f"  Edges: {metadata['num_edges']:,}")
    print(f"  Clusters: {metadata['num_clusters']:,}")
    print(f"  Avg Cluster Size: {metadata['avg_cluster_size']:.1f}")
    print(f"\nEdge Type Distribution:")
    for edge_type, count in metadata['edge_type_counts'].items():
        print(f"    {edge_type}: {count:,}")
    print(f"\nSource Distribution:")
    for source, count in metadata['source_type_counts'].items():
        print(f"    {source}: {count:,}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess graph data for contrastive learning"
    )
    parser.add_argument(
        "--se-graph",
        type=str,
        default=None,
        help="Path to StackExchange graph JSON (data/SE/se_graph.json)"
    )
    parser.add_argument(
        "--arxiv-dag",
        type=str,
        default=None,
        help="Path to ArXiv DAG JSONL (data/arxiv/extracted_envs_dag_from_theorem_papers.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--max-arxiv-papers",
        type=int,
        default=None,
        help="Maximum number of ArXiv papers to process (for testing)"
    )
    args = parser.parse_args()

    if not args.se_graph and not args.arxiv_dag:
        parser.error("At least one of --se-graph or --arxiv-dag must be provided")

    se_nodes, se_edges = [], np.empty((0, 4), dtype=np.float32)
    arxiv_nodes, arxiv_edges = [], np.empty((0, 4), dtype=np.float32)

    # Process StackExchange
    if args.se_graph:
        se_nodes, se_edges, _ = process_stackexchange(args.se_graph)

    # Process ArXiv (starting global IDs after SE nodes)
    if args.arxiv_dag:
        arxiv_start_id = len(se_nodes)
        arxiv_nodes, arxiv_edges = process_arxiv(
            args.arxiv_dag,
            start_global_id=arxiv_start_id,
            max_papers=args.max_arxiv_papers
        )

    # Merge and save
    merge_and_save(se_nodes, se_edges, arxiv_nodes, arxiv_edges, args.output_dir)


if __name__ == "__main__":
    main()
