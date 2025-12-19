#!/usr/bin/env python3
"""
Web-based data viewer for exploring nodes from Mathlib, SE, and arXiv sources.
Run with: python scripts/data_viewer.py
Then open http://localhost:5000 in your browser.
"""

import json
import random
from pathlib import Path
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
MATHLIB_PATH = DATA_DIR / "Mathlib" / "enriched_theorem_graph_all_types.json"
SE_PATH = DATA_DIR / "SE" / "se_graph.json"
ARXIV_PATH = DATA_DIR / "arxiv" / "extracted_envs_dag_from_theorem_papers.jsonl"

# Lazy-loaded data caches
_mathlib_data = None
_se_data = None
_arxiv_data = None


def load_mathlib():
    global _mathlib_data
    if _mathlib_data is None:
        print("Loading Mathlib data...")
        with open(MATHLIB_PATH, 'r') as f:
            _mathlib_data = json.load(f)
        print(f"Loaded {len(_mathlib_data['nodes'])} Mathlib nodes")
    return _mathlib_data


def load_se():
    global _se_data
    if _se_data is None:
        print("Loading SE data...")
        with open(SE_PATH, 'r') as f:
            _se_data = json.load(f)
        print(f"Loaded {len(_se_data['nodes'])} SE nodes")
    return _se_data


def load_arxiv():
    global _arxiv_data
    if _arxiv_data is None:
        print("Loading arXiv data...")
        _arxiv_data = []
        with open(ARXIV_PATH, 'r') as f:
            for line in f:
                _arxiv_data.append(json.loads(line))
        print(f"Loaded {len(_arxiv_data)} arXiv papers")
    return _arxiv_data


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Viewer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; margin-bottom: 20px; }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        button {
            padding: 10px 20px;
            font-size: 14px;
            cursor: pointer;
            border: none;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .source-btn {
            background: #e0e0e0;
            color: #333;
        }
        .source-btn.active {
            background: #2196F3;
            color: white;
        }
        .source-btn:hover { opacity: 0.8; }
        .fetch-btn {
            background: #4CAF50;
            color: white;
        }
        .fetch-btn:hover { background: #45a049; }
        .count-input {
            width: 80px;
            padding: 10px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 6px;
        }
        .stats {
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats span { margin-right: 20px; color: #666; }
        .stats strong { color: #333; }
        .results {
            display: grid;
            gap: 15px;
        }
        .node-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .node-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .node-title {
            font-size: 16px;
            font-weight: 600;
            color: #1a73e8;
            word-break: break-all;
        }
        .node-badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .badge-theorem { background: #e3f2fd; color: #1565c0; }
        .badge-definition { background: #f3e5f5; color: #7b1fa2; }
        .badge-question { background: #fff3e0; color: #e65100; }
        .badge-answer { background: #e8f5e9; color: #2e7d32; }
        .badge-proof { background: #fce4ec; color: #c2185b; }
        .badge-lemma { background: #e0f2f1; color: #00695c; }
        .badge-other { background: #eceff1; color: #546e7a; }
        .field {
            margin-bottom: 12px;
        }
        .field-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .field-value {
            font-size: 14px;
            color: #333;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        }
        .field-value.text-content {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .adjacency-list {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        .adjacency-item {
            background: #e3f2fd;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            color: #1565c0;
        }
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        .collapsible:hover { color: #1a73e8; }
        .collapsed { display: none; }
    </style>
</head>
<body>
    <h1>Data Viewer</h1>

    <div class="controls">
        <button class="source-btn active" data-source="mathlib">Mathlib</button>
        <button class="source-btn" data-source="se">StackExchange</button>
        <button class="source-btn" data-source="arxiv">arXiv</button>
        <span style="color: #666; margin: 0 10px;">|</span>
        <label>
            Count: <input type="number" class="count-input" id="count" value="5" min="1" max="50">
        </label>
        <button class="fetch-btn" id="fetch">Fetch Random Nodes</button>
    </div>

    <div class="stats" id="stats">
        <span>Select a source and click "Fetch Random Nodes"</span>
    </div>

    <div class="results" id="results"></div>

    <script>
        let currentSource = 'mathlib';

        document.querySelectorAll('.source-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.source-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentSource = btn.dataset.source;
            });
        });

        document.getElementById('fetch').addEventListener('click', fetchNodes);

        async function fetchNodes() {
            const count = document.getElementById('count').value;
            const results = document.getElementById('results');
            const stats = document.getElementById('stats');

            results.innerHTML = '<div class="loading">Loading...</div>';

            try {
                const response = await fetch(`/api/random/${currentSource}?count=${count}`);
                const data = await response.json();

                stats.innerHTML = `
                    <span>Source: <strong>${data.source}</strong></span>
                    <span>Total nodes: <strong>${data.total_nodes.toLocaleString()}</strong></span>
                    <span>Showing: <strong>${data.nodes.length}</strong> random nodes</span>
                `;

                results.innerHTML = data.nodes.map(node => renderNode(node, currentSource)).join('');
            } catch (err) {
                results.innerHTML = `<div class="node-card">Error: ${err.message}</div>`;
            }
        }

        function renderNode(node, source) {
            if (source === 'mathlib') return renderMathlibNode(node);
            if (source === 'se') return renderSENode(node);
            if (source === 'arxiv') return renderArxivNode(node);
        }

        function getBadgeClass(kind) {
            const map = {
                'theorem': 'badge-theorem',
                'definition': 'badge-definition',
                'question': 'badge-question',
                'answer': 'badge-answer',
                'proof': 'badge-proof',
                'lemma': 'badge-lemma',
            };
            return map[kind?.toLowerCase()] || 'badge-other';
        }

        function renderMathlibNode(node) {
            const kind = node.kind || 'unknown';
            const adjCount = node.adjacency?.length || 0;

            return `
                <div class="node-card">
                    <div class="node-header">
                        <div class="node-title">${escapeHtml(node.name)}</div>
                        <span class="node-badge ${getBadgeClass(kind)}">${kind}</span>
                    </div>
                    ${node.module_name ? `
                        <div class="field">
                            <div class="field-label">Module</div>
                            <div class="field-value">${escapeHtml(node.module_name.join('.'))}</div>
                        </div>
                    ` : ''}
                    ${node.signature ? `
                        <div class="field">
                            <div class="field-label">Signature</div>
                            <div class="field-value">${escapeHtml(node.signature)}</div>
                        </div>
                    ` : ''}
                    ${node.type ? `
                        <div class="field">
                            <div class="field-label">Type</div>
                            <div class="field-value">${escapeHtml(node.type)}</div>
                        </div>
                    ` : ''}
                    ${node.informal_name ? `
                        <div class="field">
                            <div class="field-label">Informal Name</div>
                            <div class="field-value text-content">${escapeHtml(node.informal_name)}</div>
                        </div>
                    ` : ''}
                    ${node.informal_description ? `
                        <div class="field">
                            <div class="field-label">Informal Description</div>
                            <div class="field-value text-content">${escapeHtml(node.informal_description)}</div>
                        </div>
                    ` : ''}
                    ${adjCount > 0 ? `
                        <div class="field">
                            <div class="field-label collapsible" onclick="this.nextElementSibling.classList.toggle('collapsed')">
                                Dependencies (${adjCount}) ▼
                            </div>
                            <div class="adjacency-list collapsed">
                                ${node.adjacency.map(a => `<span class="adjacency-item">${escapeHtml(a)}</span>`).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            `;
        }

        function renderSENode(node) {
            const type = node.type || 'post';
            return `
                <div class="node-card">
                    <div class="node-header">
                        <div class="node-title">${escapeHtml(node.title || node.id)}</div>
                        <span class="node-badge ${getBadgeClass(type)}">${type}</span>
                    </div>
                    ${node.source ? `
                        <div class="field">
                            <div class="field-label">Source</div>
                            <div class="field-value">${escapeHtml(node.source)}</div>
                        </div>
                    ` : ''}
                    ${node.score !== undefined ? `
                        <div class="field">
                            <div class="field-label">Score</div>
                            <div class="field-value">${node.score}</div>
                        </div>
                    ` : ''}
                    ${node.body ? `
                        <div class="field">
                            <div class="field-label">Body</div>
                            <div class="field-value text-content">${escapeHtml(stripHtml(node.body))}</div>
                        </div>
                    ` : ''}
                    ${node.url ? `
                        <div class="field">
                            <div class="field-label">URL</div>
                            <div class="field-value"><a href="${escapeHtml(node.url)}" target="_blank">${escapeHtml(node.url)}</a></div>
                        </div>
                    ` : ''}
                </div>
            `;
        }

        function renderArxivNode(node) {
            const env = node.env || 'unknown';
            return `
                <div class="node-card">
                    <div class="node-header">
                        <div class="node-title">${escapeHtml(node.id || 'Node')}</div>
                        <span class="node-badge ${getBadgeClass(env)}">${env}</span>
                    </div>
                    ${node.paper_id ? `
                        <div class="field">
                            <div class="field-label">Paper ID</div>
                            <div class="field-value">${escapeHtml(node.paper_id)}</div>
                        </div>
                    ` : ''}
                    ${node.label ? `
                        <div class="field">
                            <div class="field-label">Label</div>
                            <div class="field-value">${escapeHtml(node.label)}</div>
                        </div>
                    ` : ''}
                    ${node.file ? `
                        <div class="field">
                            <div class="field-label">File</div>
                            <div class="field-value">${escapeHtml(node.file)}</div>
                        </div>
                    ` : ''}
                    ${node.text ? `
                        <div class="field">
                            <div class="field-label">Content</div>
                            <div class="field-value">${escapeHtml(node.text)}</div>
                        </div>
                    ` : ''}
                </div>
            `;
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = String(text);
            return div.innerHTML;
        }

        function stripHtml(html) {
            const div = document.createElement('div');
            div.innerHTML = html;
            return div.textContent || div.innerText || '';
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/random/mathlib')
def random_mathlib():
    from flask import request
    count = min(int(request.args.get('count', 5)), 50)

    data = load_mathlib()
    nodes = data['nodes']
    node_data = data['node_data']
    adjacency = data['adjacency']

    # Sample random nodes
    sampled_names = random.sample(nodes, min(count, len(nodes)))

    result_nodes = []
    for name in sampled_names:
        node_info = node_data.get(name, {'name': name})
        node_info = dict(node_info)  # Copy
        node_info['adjacency'] = adjacency.get(name, [])
        result_nodes.append(node_info)

    return jsonify({
        'source': 'Mathlib',
        'total_nodes': len(nodes),
        'nodes': result_nodes
    })


@app.route('/api/random/se')
def random_se():
    from flask import request
    count = min(int(request.args.get('count', 5)), 50)

    data = load_se()
    nodes = data['nodes']

    # Sample random node IDs
    node_ids = list(nodes.keys())
    sampled_ids = random.sample(node_ids, min(count, len(node_ids)))

    result_nodes = []
    for node_id in sampled_ids:
        node_info = dict(nodes[node_id])
        node_info['id'] = node_id
        result_nodes.append(node_info)

    return jsonify({
        'source': 'StackExchange',
        'total_nodes': len(nodes),
        'nodes': result_nodes
    })


@app.route('/api/random/arxiv')
def random_arxiv():
    from flask import request
    count = min(int(request.args.get('count', 5)), 50)

    papers = load_arxiv()

    # Collect all nodes with paper reference
    all_nodes = []
    for paper in papers:
        paper_id = paper.get('arxiv_id', 'unknown')
        for node in paper.get('nodes', []):
            node_copy = dict(node)
            node_copy['paper_id'] = paper_id
            all_nodes.append(node_copy)

    # Sample random nodes
    sampled = random.sample(all_nodes, min(count, len(all_nodes)))

    return jsonify({
        'source': 'arXiv',
        'total_nodes': len(all_nodes),
        'nodes': sampled
    })


if __name__ == '__main__':
    print("Starting Data Viewer...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
