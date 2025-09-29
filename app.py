#!/usr/bin/env python3
"""
Flask web application for browsing mathematical theorems and lemmas with LaTeX rendering.
"""

from flask import Flask, render_template, jsonify, request
import json
from pathlib import Path
from typing import Dict, List, Optional

app = Flask(__name__)

# Global storage for loaded dataset
dataset_cache = {}
current_dataset = None

def load_dataset(dataset_name: str = 'toy') -> List[Dict]:
    """Load dataset from JSONL file."""
    global dataset_cache, current_dataset

    if dataset_name in dataset_cache:
        current_dataset = dataset_cache[dataset_name]
        return current_dataset

    dataset_path = Path(f'data/lemmas_theorems/{dataset_name}.jsonl')

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")

    papers = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            paper = json.loads(line.strip())
            papers.append(paper)

    dataset_cache[dataset_name] = papers
    current_dataset = papers
    return papers

def get_paper_summary(paper: Dict) -> Dict:
    """Extract summary information from a paper."""
    # Handle both data formats
    if 'statements' in paper:
        # New format with statements array
        return {
            'arxiv_id': paper.get('arxiv_id', 'Unknown'),
            'title': paper.get('title', 'Untitled'),
            'num_theorems': len([s for s in paper.get('statements', []) if s.get('type') == 'theorem']),
            'num_lemmas': len([s for s in paper.get('statements', []) if s.get('type') == 'lemma']),
            'total_statements': len(paper.get('statements', []))
        }
    else:
        # Old format with direct theorems/lemmas arrays
        return {
            'arxiv_id': paper.get('arxiv_id', f"Paper_{id(paper)%10000:04d}"),
            'title': paper.get('title', f"Paper (T:{len(paper.get('theorems', []))}, L:{len(paper.get('lemmas', []))})"),
            'num_theorems': len(paper.get('theorems', [])),
            'num_lemmas': len(paper.get('lemmas', [])),
            'total_statements': len(paper.get('theorems', [])) + len(paper.get('lemmas', []))
        }

@app.route('/')
def index():
    """Main page with paper browser."""
    return render_template('index.html')

@app.route('/api/datasets')
def list_datasets():
    """List available datasets."""
    datasets_dir = Path('data/lemmas_theorems')
    datasets = []

    if datasets_dir.exists():
        for path in datasets_dir.glob('*.jsonl'):
            name = path.stem
            size_mb = path.stat().st_size / (1024 * 1024)
            datasets.append({
                'name': name,
                'size': f"{size_mb:.1f}MB",
                'path': str(path)
            })

    return jsonify(datasets)

@app.route('/api/load_dataset/<dataset_name>')
def load_dataset_api(dataset_name: str):
    """Load a specific dataset and return paper list."""
    try:
        papers = load_dataset(dataset_name)
        summaries = [get_paper_summary(paper) for paper in papers]
        return jsonify({
            'success': True,
            'num_papers': len(papers),
            'papers': summaries
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/paper/<int:paper_idx>')
def get_paper(paper_idx: int):
    """Get detailed information for a specific paper."""
    if current_dataset is None:
        return jsonify({'success': False, 'error': 'No dataset loaded'}), 400

    if paper_idx < 0 or paper_idx >= len(current_dataset):
        return jsonify({'success': False, 'error': 'Invalid paper index'}), 400

    paper = current_dataset[paper_idx]

    # Handle both data formats
    if 'statements' in paper:
        # New format with statements array
        theorems = []
        lemmas = []
        others = []

        for stmt in paper.get('statements', []):
            stmt_data = {
                'text': stmt.get('text', ''),
                'type': stmt.get('type', 'unknown')
            }

            if stmt['type'] == 'theorem':
                theorems.append(stmt_data)
            elif stmt['type'] == 'lemma':
                lemmas.append(stmt_data)
            else:
                others.append(stmt_data)
    else:
        # Old format with direct theorems/lemmas arrays
        theorems = [{'text': t, 'type': 'theorem'} for t in paper.get('theorems', [])]
        lemmas = [{'text': l, 'type': 'lemma'} for l in paper.get('lemmas', [])]
        others = []

    return jsonify({
        'success': True,
        'arxiv_id': paper.get('arxiv_id', f"Paper_{paper_idx:04d}"),
        'title': paper.get('title', f"Paper {paper_idx + 1} (T:{len(theorems)}, L:{len(lemmas)})"),
        'theorems': theorems,
        'lemmas': lemmas,
        'others': others
    })

@app.route('/api/search', methods=['POST'])
def search_statements():
    """Search for statements containing specific text."""
    if current_dataset is None:
        return jsonify({'success': False, 'error': 'No dataset loaded'}), 400

    query = request.json.get('query', '').lower()
    if not query:
        return jsonify({'success': False, 'error': 'Empty query'}), 400

    results = []
    for paper_idx, paper in enumerate(current_dataset):
        for stmt in paper.get('statements', []):
            if query in stmt.get('text', '').lower():
                results.append({
                    'paper_idx': paper_idx,
                    'paper_title': paper.get('title', 'Untitled'),
                    'arxiv_id': paper.get('arxiv_id', 'Unknown'),
                    'type': stmt.get('type', 'unknown'),
                    'text': stmt.get('text', '')[:200] + '...' if len(stmt.get('text', '')) > 200 else stmt.get('text', '')
                })

    return jsonify({
        'success': True,
        'num_results': len(results),
        'results': results[:100]  # Limit to 100 results
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)