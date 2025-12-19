#!/usr/bin/env python3
"""
Web-based data viewer for exploring nodes from Mathlib, SE, and arXiv sources.
Run with: python scripts/data_viewer.py
Then open http://localhost:5000 in your browser.
"""

import json
import random
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Lazy-loaded caches
_mathlib = None
_se = None
_arxiv = None


def load_mathlib():
    global _mathlib
    if _mathlib is None:
        print("Loading Mathlib data...")
        with open(DATA_DIR / "Mathlib" / "enriched_theorem_graph_all_types.json") as f:
            data = json.load(f)
        # Build reverse adjacency (incoming edges)
        reverse_adj = {n: [] for n in data['nodes']}
        for src, targets in data['adjacency'].items():
            for tgt in targets:
                if tgt in reverse_adj:
                    reverse_adj[tgt].append(src)
        _mathlib = {**data, 'reverse_adjacency': reverse_adj}
        print(f"Loaded {len(data['nodes'])} Mathlib nodes")
    return _mathlib


def load_se():
    global _se
    if _se is None:
        print("Loading SE data...")
        with open(DATA_DIR / "SE" / "se_graph.json") as f:
            data = json.load(f)
        # Build reverse adjacency
        reverse_adj = {}
        for src, edges in data['edges'].items():
            for tgt, etype in edges:
                reverse_adj.setdefault(tgt, []).append([src, etype])
        _se = {**data, 'reverse_adjacency': reverse_adj}
        print(f"Loaded {len(data['nodes'])} SE nodes")
    return _se


def load_arxiv():
    global _arxiv
    if _arxiv is None:
        print("Loading arXiv data...")
        papers = {}
        with open(DATA_DIR / "arxiv" / "extracted_envs_dag_from_theorem_papers.jsonl") as f:
            for line in f:
                p = json.loads(line)
                pid = p.get('arxiv_id', 'unknown')
                # Build node lookup and reverse adjacency per paper
                nodes = {n['id']: {**n, 'paper_id': pid} for n in p.get('nodes', [])}
                adj = p.get('adjacency', {})
                rev_adj = {}
                for src, targets in adj.items():
                    for tgt in targets:
                        rev_adj.setdefault(tgt, []).append(src)
                papers[pid] = {'nodes': nodes, 'adjacency': adj, 'reverse_adjacency': rev_adj}
        _arxiv = papers
        print(f"Loaded {len(papers)} arXiv papers")
    return _arxiv


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Viewer</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }
        button { padding: 10px 20px; font-size: 14px; cursor: pointer; border: none; border-radius: 6px; }
        .source-btn { background: #e0e0e0; }
        .source-btn.active { background: #2196F3; color: white; }
        .fetch-btn { background: #4CAF50; color: white; }
        .count-input { width: 80px; padding: 10px; font-size: 14px; border: 1px solid #ccc; border-radius: 6px; }
        .stats { background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stats span { margin-right: 20px; color: #666; }
        .results { display: grid; gap: 15px; }
        .node-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .node-card.nested { margin: 10px 0 0 20px; border-left: 3px solid #2196F3; background: #fafafa; }
        .node-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px; gap: 10px; flex-wrap: wrap; }
        .node-title { font-size: 16px; font-weight: 600; color: #1a73e8; word-break: break-all; }
        .node-badge { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }
        .badge-theorem { background: #e3f2fd; color: #1565c0; }
        .badge-definition { background: #f3e5f5; color: #7b1fa2; }
        .badge-question { background: #fff3e0; color: #e65100; }
        .badge-answer { background: #e8f5e9; color: #2e7d32; }
        .badge-proof { background: #fce4ec; color: #c2185b; }
        .badge-lemma { background: #e0f2f1; color: #00695c; }
        .badge-other { background: #eceff1; color: #546e7a; }
        .field { margin-bottom: 12px; }
        .field-label { font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 4px; }
        .field-value { font-size: 14px; color: #333; background: #f8f9fa; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; font-family: monospace; }
        .field-value.text { font-family: system-ui, sans-serif; }
        .edge-section { margin-top: 10px; padding: 10px; background: #f0f4f8; border-radius: 6px; }
        .edge-header { font-size: 12px; color: #666; text-transform: uppercase; cursor: pointer; user-select: none; }
        .edge-header:hover { color: #1a73e8; }
        .edge-list { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
        .edge-item { background: #e3f2fd; padding: 3px 10px; border-radius: 4px; font-size: 12px; color: #1565c0; cursor: pointer; display: flex; align-items: center; gap: 4px; }
        .edge-item:hover { background: #bbdefb; }
        .edge-item.incoming { background: #fff3e0; color: #e65100; }
        .edge-item.incoming:hover { background: #ffe0b2; }
        .edge-type { font-size: 10px; opacity: 0.7; }
        .expand-icon { font-size: 10px; }
        .hidden { display: none; }
        .loading { text-align: center; padding: 20px; color: #666; }
        .nested-container { margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Data Viewer</h1>
    <div class="controls">
        <button class="source-btn active" data-source="mathlib">Mathlib</button>
        <button class="source-btn" data-source="se">StackExchange</button>
        <button class="source-btn" data-source="arxiv">arXiv</button>
        <span style="color: #666; margin: 0 10px;">|</span>
        <label>Count: <input type="number" class="count-input" id="count" value="5" min="1" max="50"></label>
        <button class="fetch-btn" id="fetch">Fetch Random Nodes</button>
    </div>
    <div class="stats" id="stats"><span>Select a source and click "Fetch Random Nodes"</span></div>
    <div class="results" id="results"></div>
<script>
let currentSource = 'mathlib';
const expandedNodes = new Set();

document.querySelectorAll('.source-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.source-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentSource = btn.dataset.source;
        expandedNodes.clear();
    });
});

document.getElementById('fetch').addEventListener('click', fetchNodes);

async function fetchNodes() {
    const count = document.getElementById('count').value;
    const results = document.getElementById('results');
    results.innerHTML = '<div class="loading">Loading...</div>';
    expandedNodes.clear();
    try {
        const response = await fetch(`/api/random/${currentSource}?count=${count}`);
        const data = await response.json();
        document.getElementById('stats').innerHTML = `
            <span>Source: <strong>${data.source}</strong></span>
            <span>Total nodes: <strong>${data.total_nodes.toLocaleString()}</strong></span>
            <span>Showing: <strong>${data.nodes.length}</strong> random nodes</span>
        `;
        results.innerHTML = data.nodes.map(n => renderNode(n, currentSource, false)).join('');
    } catch (err) {
        results.innerHTML = `<div class="node-card">Error: ${err.message}</div>`;
    }
}

async function expandNode(source, nodeId, paperId, container) {
    const key = `${source}:${paperId || ''}:${nodeId}`;
    if (expandedNodes.has(key)) {
        expandedNodes.delete(key);
        container.innerHTML = '';
        return;
    }
    expandedNodes.add(key);
    container.innerHTML = '<div class="loading">Loading...</div>';
    try {
        let url = `/api/node/${source}/${encodeURIComponent(nodeId)}`;
        if (paperId) url += `?paper_id=${encodeURIComponent(paperId)}`;
        const response = await fetch(url);
        const data = await response.json();
        container.innerHTML = renderNode(data.node, source, true);
    } catch (err) {
        container.innerHTML = `<div class="node-card nested">Error loading node</div>`;
    }
}

function renderNode(node, source, nested) {
    const cls = nested ? 'node-card nested' : 'node-card';
    if (source === 'mathlib') return renderMathlib(node, cls);
    if (source === 'se') return renderSE(node, cls);
    if (source === 'arxiv') return renderArxiv(node, cls);
}

function badge(kind) {
    const map = {theorem:'badge-theorem',definition:'badge-definition',question:'badge-question',answer:'badge-answer',proof:'badge-proof',lemma:'badge-lemma'};
    return map[kind?.toLowerCase()] || 'badge-other';
}
function esc(t) { if (!t) return ''; const d = document.createElement('div'); d.textContent = String(t); return d.innerHTML; }
function strip(html) { const d = document.createElement('div'); d.innerHTML = html; return d.textContent || ''; }
function uid() { return 'n' + Math.random().toString(36).substr(2, 9); }

function renderEdges(outgoing, incoming, source, paperId) {
    const id = uid();
    const outItems = (outgoing || []).map(e => {
        const [name, etype] = Array.isArray(e) ? e : [e, null];
        const cid = uid();
        return `<span class="edge-item" onclick="expandNode('${source}', '${esc(name)}', '${esc(paperId||'')}', document.getElementById('${cid}'))">
            <span class="expand-icon">+</span>${esc(name)}${etype ? `<span class="edge-type">(${etype})</span>` : ''}
        </span><div id="${cid}" class="nested-container"></div>`;
    }).join('');
    const inItems = (incoming || []).map(e => {
        const [name, etype] = Array.isArray(e) ? e : [e, null];
        const cid = uid();
        return `<span class="edge-item incoming" onclick="expandNode('${source}', '${esc(name)}', '${esc(paperId||'')}', document.getElementById('${cid}'))">
            <span class="expand-icon">+</span>${esc(name)}${etype ? `<span class="edge-type">(${etype})</span>` : ''}
        </span><div id="${cid}" class="nested-container"></div>`;
    }).join('');
    if (!outItems && !inItems) return '';
    return `<div class="edge-section">
        ${outItems ? `<div class="edge-header" onclick="this.nextElementSibling.classList.toggle('hidden')">Outgoing (${outgoing.length}) ▼</div><div class="edge-list">${outItems}</div>` : ''}
        ${inItems ? `<div class="edge-header" onclick="this.nextElementSibling.classList.toggle('hidden')">Incoming (${incoming.length}) ▼</div><div class="edge-list">${inItems}</div>` : ''}
    </div>`;
}

function renderMathlib(n, cls) {
    return `<div class="${cls}">
        <div class="node-header"><div class="node-title">${esc(n.name)}</div><span class="node-badge ${badge(n.kind)}">${n.kind||'unknown'}</span></div>
        ${n.module_name ? `<div class="field"><div class="field-label">Module</div><div class="field-value">${esc(n.module_name.join('.'))}</div></div>` : ''}
        ${n.signature ? `<div class="field"><div class="field-label">Signature</div><div class="field-value">${esc(n.signature)}</div></div>` : ''}
        ${n.type ? `<div class="field"><div class="field-label">Type</div><div class="field-value">${esc(n.type)}</div></div>` : ''}
        ${n.informal_name ? `<div class="field"><div class="field-label">Informal Name</div><div class="field-value text">${esc(n.informal_name)}</div></div>` : ''}
        ${n.informal_description ? `<div class="field"><div class="field-label">Informal Description</div><div class="field-value text">${esc(n.informal_description)}</div></div>` : ''}
        ${renderEdges(n.outgoing, n.incoming, 'mathlib', null)}
    </div>`;
}

function renderSE(n, cls) {
    return `<div class="${cls}">
        <div class="node-header"><div class="node-title">${esc(n.title || n.id)}</div><span class="node-badge ${badge(n.type)}">${n.type||'post'}</span></div>
        ${n.source ? `<div class="field"><div class="field-label">Source</div><div class="field-value">${esc(n.source)}</div></div>` : ''}
        ${n.score !== undefined ? `<div class="field"><div class="field-label">Score</div><div class="field-value">${n.score}</div></div>` : ''}
        ${n.body ? `<div class="field"><div class="field-label">Body</div><div class="field-value text">${esc(strip(n.body))}</div></div>` : ''}
        ${n.url ? `<div class="field"><div class="field-label">URL</div><div class="field-value"><a href="${esc(n.url)}" target="_blank">${esc(n.url)}</a></div></div>` : ''}
        ${renderEdges(n.outgoing, n.incoming, 'se', null)}
    </div>`;
}

function renderArxiv(n, cls) {
    return `<div class="${cls}">
        <div class="node-header"><div class="node-title">${esc(n.id)}</div><span class="node-badge ${badge(n.env)}">${n.env||'unknown'}</span></div>
        ${n.paper_id ? `<div class="field"><div class="field-label">Paper ID</div><div class="field-value">${esc(n.paper_id)}</div></div>` : ''}
        ${n.label ? `<div class="field"><div class="field-label">Label</div><div class="field-value">${esc(n.label)}</div></div>` : ''}
        ${n.text ? `<div class="field"><div class="field-label">Content</div><div class="field-value">${esc(n.text)}</div></div>` : ''}
        ${renderEdges(n.outgoing, n.incoming, 'arxiv', n.paper_id)}
    </div>`;
}
</script>
</body>
</html>
"""


def get_mathlib_node(name):
    data = load_mathlib()
    node = dict(data['node_data'].get(name, {'name': name}))
    node['outgoing'] = data['adjacency'].get(name, [])
    node['incoming'] = data['reverse_adjacency'].get(name, [])
    return node


def get_se_node(node_id):
    data = load_se()
    node = dict(data['nodes'].get(node_id, {}))
    node['id'] = node_id
    node['outgoing'] = data['edges'].get(node_id, [])
    node['incoming'] = data['reverse_adjacency'].get(node_id, [])
    return node


def get_arxiv_node(paper_id, node_id):
    papers = load_arxiv()
    paper = papers.get(paper_id, {})
    node = dict(paper.get('nodes', {}).get(node_id, {'id': node_id, 'paper_id': paper_id}))
    node['outgoing'] = paper.get('adjacency', {}).get(node_id, [])
    node['incoming'] = paper.get('reverse_adjacency', {}).get(node_id, [])
    return node


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/random/mathlib')
def random_mathlib():
    count = min(int(request.args.get('count', 5)), 50)
    data = load_mathlib()
    sampled = random.sample(data['nodes'], min(count, len(data['nodes'])))
    nodes = [get_mathlib_node(n) for n in sampled]
    return jsonify({'source': 'Mathlib', 'total_nodes': len(data['nodes']), 'nodes': nodes})


@app.route('/api/random/se')
def random_se():
    count = min(int(request.args.get('count', 5)), 50)
    data = load_se()
    sampled = random.sample(list(data['nodes'].keys()), min(count, len(data['nodes'])))
    nodes = [get_se_node(n) for n in sampled]
    return jsonify({'source': 'StackExchange', 'total_nodes': len(data['nodes']), 'nodes': nodes})


@app.route('/api/random/arxiv')
def random_arxiv():
    count = min(int(request.args.get('count', 5)), 50)
    papers = load_arxiv()
    all_nodes = [(pid, nid) for pid, p in papers.items() for nid in p['nodes']]
    sampled = random.sample(all_nodes, min(count, len(all_nodes)))
    nodes = [get_arxiv_node(pid, nid) for pid, nid in sampled]
    return jsonify({'source': 'arXiv', 'total_nodes': len(all_nodes), 'nodes': nodes})


@app.route('/api/node/mathlib/<path:node_id>')
def api_mathlib_node(node_id):
    return jsonify({'node': get_mathlib_node(node_id)})


@app.route('/api/node/se/<path:node_id>')
def api_se_node(node_id):
    return jsonify({'node': get_se_node(node_id)})


@app.route('/api/node/arxiv/<path:node_id>')
def api_arxiv_node(node_id):
    paper_id = request.args.get('paper_id', 'unknown')
    return jsonify({'node': get_arxiv_node(paper_id, node_id)})


if __name__ == '__main__':
    print("Starting Data Viewer...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
