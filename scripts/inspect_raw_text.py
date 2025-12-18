"""
Inspect raw text from preprocessed nodes to identify potential data leakage.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pyarrow.feather as feather
from pathlib import Path
import re
from collections import Counter

def sample_nodes(data_dir="data/processed", n_samples=100, seed=42):
    """Sample random nodes from each source type."""

    nodes_path = Path(data_dir) / "nodes.arrow"
    table = feather.read_table(str(nodes_path), memory_map=True)

    texts = table.column('text')
    source_types = table.column('source_type').to_numpy()
    node_types = table.column('node_type')

    # source_type: 0 = StackExchange, 1 = ArXiv
    se_indices = np.where(source_types == 0)[0]
    arxiv_indices = np.where(source_types == 1)[0]

    rng = np.random.default_rng(seed)

    se_samples = rng.choice(se_indices, size=min(n_samples, len(se_indices)), replace=False)
    arxiv_samples = rng.choice(arxiv_indices, size=min(n_samples, len(arxiv_indices)), replace=False)

    return {
        'se': [(i, texts[i].as_py(), node_types[i].as_py()) for i in se_samples],
        'arxiv': [(i, texts[i].as_py(), node_types[i].as_py()) for i in arxiv_samples],
    }


def analyze_leakage_patterns(samples):
    """Analyze text samples for potential leakage patterns."""

    patterns = {
        # LaTeX reference patterns
        'latex_ref': r'\\ref\{[^}]*\}',
        'latex_eqref': r'\\eqref\{[^}]*\}',
        'latex_cite': r'\\cite[a-z]*\{[^}]*\}',
        'latex_label': r'\\label\{[^}]*\}',
        'latex_hyperref': r'\\hyperref\[[^\]]*\]',
        'latex_autoref': r'\\autoref\{[^}]*\}',
        'latex_cref': r'\\cref\{[^}]*\}',
        'latex_pageref': r'\\pageref\{[^}]*\}',

        # URLs and links
        'url_http': r'https?://[^\s<>\[\]]+',
        'url_www': r'www\.[^\s<>\[\]]+',

        # StackExchange specific
        'se_post_id': r'(?:question|answer|post)[/\s#]?\d+',
        'se_user_link': r'/users/\d+',
        'se_question_link': r'/questions/\d+',

        # ArXiv specific
        'arxiv_id': r'arXiv:\d+\.\d+',
        'arxiv_link': r'arxiv\.org/[^\s]+',

        # Internal references in text
        'theorem_ref': r'(?:Theorem|Lemma|Proposition|Corollary|Definition|Remark)\s+\d+(?:\.\d+)*',
        'section_ref': r'(?:Section|Chapter|Appendix)\s+\d+(?:\.\d+)*',
        'equation_ref': r'\(\s*\d+(?:\.\d+)*\s*\)',
        'figure_ref': r'(?:Figure|Fig\.?|Table)\s+\d+(?:\.\d+)*',
    }

    results = {}
    for source, items in samples.items():
        results[source] = {
            'pattern_counts': Counter(),
            'pattern_examples': {},
            'total_samples': len(items),
        }

        for idx, text, node_type in items:
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    results[source]['pattern_counts'][pattern_name] += 1
                    if pattern_name not in results[source]['pattern_examples']:
                        results[source]['pattern_examples'][pattern_name] = []
                    if len(results[source]['pattern_examples'][pattern_name]) < 5:
                        results[source]['pattern_examples'][pattern_name].append({
                            'idx': idx,
                            'node_type': node_type,
                            'matches': matches[:5],
                            'context': text[:500]
                        })

    return results


def print_analysis(results):
    """Print analysis results."""

    for source in ['arxiv', 'se']:
        print(f"\n{'='*70}")
        print(f"  {source.upper()} ANALYSIS ({results[source]['total_samples']} samples)")
        print(f"{'='*70}")

        counts = results[source]['pattern_counts']
        examples = results[source]['pattern_examples']

        if not counts:
            print("No leakage patterns found.")
            continue

        print("\n--- Pattern Frequency ---")
        for pattern, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / results[source]['total_samples']
            print(f"  {pattern:20s}: {count:3d} samples ({pct:5.1f}%)")

        print("\n--- Example Matches ---")
        for pattern, exs in examples.items():
            print(f"\n[{pattern}]")
            for ex in exs[:2]:  # Show 2 examples per pattern
                print(f"  Node {ex['idx']} ({ex['node_type']}):")
                print(f"    Matches: {ex['matches']}")
                print(f"    Context: {ex['context'][:200]}...")


def print_raw_samples(samples, n=10):
    """Print raw text samples for manual inspection."""

    for source in ['arxiv', 'se']:
        print(f"\n{'='*70}")
        print(f"  RAW {source.upper()} SAMPLES (first {n})")
        print(f"{'='*70}")

        for i, (idx, text, node_type) in enumerate(samples[source][:n]):
            print(f"\n--- Sample {i+1} (node {idx}, type: {node_type}) ---")
            print(text[:1000])
            if len(text) > 1000:
                print(f"... [{len(text) - 1000} more chars]")


if __name__ == "__main__":
    print("Sampling 100 nodes from each source...")
    samples = sample_nodes(n_samples=100)

    print("\nAnalyzing leakage patterns...")
    results = analyze_leakage_patterns(samples)

    print_analysis(results)

    print("\n" + "="*70)
    print("  RAW TEXT SAMPLES FOR MANUAL INSPECTION")
    print("="*70)
    print_raw_samples(samples, n=10)
