#!/usr/bin/env python3
"""
Download the informal Mathlib dataset from HuggingFace and map it to the
extracted formal Mathlib DAG.

This script:
1. Downloads FrenzyMath/mathlib_informal_v4.16.0 from HuggingFace
2. Maps informal descriptions to formal graph nodes (all types: theorems,
   definitions, structures, instances, etc.)
3. Creates enriched graph files with informal metadata

Usage:
    python download_informal_mathlib.py
"""

import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


def download_informal_dataset(output_dir: Path) -> pd.DataFrame:
    """Download the informal Mathlib dataset from HuggingFace."""
    print("Downloading FrenzyMath/mathlib_informal_v4.16.0 dataset...")

    dataset = load_dataset("FrenzyMath/mathlib_informal_v4.16.0", split="train")

    print(f"Dataset loaded: {len(dataset)} rows")
    print(f"Columns: {dataset.column_names}")

    # Convert to pandas for easier manipulation
    df = dataset.to_pandas()

    # Save raw dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "mathlib_informal.parquet"
    df.to_parquet(parquet_path)
    print(f"Saved dataset to {parquet_path}")

    return df


def build_informal_lookup(df: pd.DataFrame) -> dict:
    """Build a lookup dictionary from the informal dataset."""
    print("\nBuilding informal lookup for all types...")
    informal_lookup = {}

    for idx in tqdm(range(len(df)), desc="Building lookup"):
        row = df.iloc[idx]
        # Join just the name parts (not the module)
        name = ".".join(str(x) for x in row['name'])

        informal_lookup[name] = {
            'idx': int(idx),
            'module_name': list(row['module_name']),
            'kind': row['kind'],
            'name': list(row['name']),
            'signature': row['signature'],
            'type': row['type'],
            'value': row['value'] if pd.notna(row['value']) else None,
            'docstring': row['docstring'] if pd.notna(row['docstring']) else None,
            'informal_name': row['informal_name'],
            'informal_description': row['informal_description'],
        }

    print(f"Built lookup with {len(informal_lookup)} entries")
    return informal_lookup


def load_formal_graph(output_dir: Path) -> dict:
    """Load the formal theorem dependency graph."""
    graph_path = output_dir / "theorem_dependencies.json"

    if not graph_path.exists():
        raise FileNotFoundError(f"Formal graph not found at {graph_path}")

    with open(graph_path) as f:
        graph = json.load(f)

    print(f"Loaded formal graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    return graph


def create_mapping(graph: dict, informal_lookup: dict, output_dir: Path):
    """Create mapping between formal graph nodes and informal dataset."""
    print("\nMatching all graph nodes to informal dataset...")

    all_nodes = set(graph['nodes'])
    matched = {}
    matched_by_kind = Counter()

    for node in tqdm(all_nodes, desc="Matching"):
        if node in informal_lookup:
            info = informal_lookup[node]
            matched[node] = info
            matched_by_kind[info['kind']] += 1

    unmatched = all_nodes - set(matched.keys())

    print(f"\n=== Mapping Results ===")
    print(f"Total graph nodes: {len(all_nodes)}")
    print(f"Matched: {len(matched)} ({100*len(matched)/len(all_nodes):.1f}%)")
    print(f"Unmatched: {len(unmatched)} ({100*len(unmatched)/len(all_nodes):.1f}%)")

    print(f"\nMatched by kind:")
    for kind, count in matched_by_kind.most_common():
        print(f"  {kind}: {count}")

    # Create enriched graph
    print("\nCreating enriched graph...")
    enriched_nodes = {}
    for node in tqdm(graph['nodes'], desc="Enriching nodes"):
        node_data = {
            'name': node,
            'has_informal': node in matched
        }

        if node in matched:
            info = matched[node]
            node_data.update({
                'kind': info['kind'],
                'module_name': info['module_name'],
                'signature': info['signature'],
                'type': info['type'],
                'docstring': info['docstring'],
                'informal_name': info['informal_name'],
                'informal_description': info['informal_description'],
            })

        enriched_nodes[node] = node_data

    # Statistics
    stats = {
        'total_nodes': len(graph['nodes']),
        'total_edges': len(graph['edges']),
        'nodes_with_informal': len(matched),
        'coverage': len(matched) / len(graph['nodes']),
        'matched_by_kind': dict(matched_by_kind),
        'unmatched_count': len(unmatched),
    }

    # Save enriched graph
    output = {
        'nodes': list(enriched_nodes.keys()),
        'node_data': enriched_nodes,
        'edges': graph['edges'],
        'adjacency': graph['adjacency'],
        'statistics': stats
    }

    output_path = output_dir / "enriched_theorem_graph_all_types.json"
    print(f"\nSaving enriched graph to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save simplified version
    simple_output = {
        'nodes_with_descriptions': {
            name: {
                'kind': data.get('kind', 'unknown'),
                'informal_name': data.get('informal_name', ''),
                'informal_description': data.get('informal_description', ''),
                'signature': data.get('signature', ''),
            }
            for name, data in enriched_nodes.items()
            if data['has_informal']
        },
        'edges': graph['edges'],
        'statistics': stats
    }

    simple_path = output_dir / "theorem_graph_with_descriptions_all_types.json"
    print(f"Saving simplified version to {simple_path}...")
    with open(simple_path, 'w') as f:
        json.dump(simple_output, f, indent=2)
    print(f"File size: {simple_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save mapping
    mapping_path = output_dir / "formal_informal_mapping_all_types.json"
    print(f"Saving mapping to {mapping_path}...")
    with open(mapping_path, 'w') as f:
        json.dump({
            'statistics': stats,
            'matched': matched
        }, f, indent=2)
    print(f"File size: {mapping_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save statistics
    stats_path = output_dir / "mapping_statistics_all_types.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Final Summary ===")
    print(json.dumps(stats, indent=2))

    return stats


def main():
    output_dir = Path("./output")
    informal_dir = Path("./informal_data")

    # Download informal dataset
    df = download_informal_dataset(informal_dir)

    # Show dataset info
    print(f"\n=== Informal Dataset Contents ===")
    print(df['kind'].value_counts())

    # Build lookup
    informal_lookup = build_informal_lookup(df)

    # Load formal graph
    graph = load_formal_graph(output_dir)

    # Create mapping
    create_mapping(graph, informal_lookup, output_dir)


if __name__ == "__main__":
    main()
