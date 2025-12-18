# Dataset Format Documentation

This document describes the format of two datasets used in this project.

---

## 1. StackExchange Q&A Graph

**File:** `data/SE/se_graph.json`

**Description:** A directed graph of questions and answers from StackExchange sites (math, mathoverflow, stats). Nodes are posts (questions or answers), edges represent relationships between posts.

### Top-Level Structure

```json
{
  "nodes": { ... },
  "edges": { ... },
  "stats": { ... }
}
```

### Nodes

A dictionary mapping node IDs to node data.

**Node ID Format:** `{source}_{post_id}` (e.g., `"math_123"`, `"mathoverflow_456"`)

#### Question Node

```json
{
  "math_1": {
    "type": "question",
    "title": "What Does it Really Mean to Have Different Kinds of Infinities?",
    "body": "<p>HTML content of the question...</p>",
    "score": 201,
    "source": "math",
    "url": "https://math.stackexchange.com/questions/1",
    "tags": ["elementary-set-theory", "intuition", "infinity", "faq"],
    "creation_date": "2010-07-20T19:09:27.200",
    "last_activity_date": "2023-05-17T06:11:58.107",
    "last_edit_date": "2018-03-01T19:53:22.017",
    "view_count": 13229,
    "answer_count": 9,
    "comment_count": 3
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"question"` |
| `title` | string | Question title |
| `body` | string | Full HTML content of the question |
| `score` | int | Net upvotes (upvotes - downvotes) |
| `source` | string | Site: `"math"`, `"mathoverflow"`, or `"stats"` |
| `url` | string | Direct URL to the question |
| `tags` | list[string] | List of tags |
| `creation_date` | string | ISO 8601 timestamp |
| `last_activity_date` | string | ISO 8601 timestamp |
| `last_edit_date` | string | ISO 8601 timestamp (empty if never edited) |
| `view_count` | int | Number of views |
| `answer_count` | int | Number of answers |
| `comment_count` | int | Number of comments |

#### Answer Node

```json
{
  "math_4": {
    "type": "answer",
    "body": "<p>HTML content of the answer...</p>",
    "score": 16,
    "parent_id": "math_3",
    "source": "math",
    "url": "https://math.stackexchange.com/a/4",
    "creation_date": "2010-07-20T19:14:10.603",
    "last_activity_date": "2010-07-20T19:14:10.603",
    "last_edit_date": "",
    "comment_count": 2
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"answer"` |
| `body` | string | Full HTML content of the answer |
| `score` | int | Net upvotes |
| `parent_id` | string | Node ID of the parent question |
| `source` | string | Site: `"math"`, `"mathoverflow"`, or `"stats"` |
| `url` | string | Direct URL to the answer |
| `creation_date` | string | ISO 8601 timestamp |
| `last_activity_date` | string | ISO 8601 timestamp |
| `last_edit_date` | string | ISO 8601 timestamp (empty if never edited) |
| `comment_count` | int | Number of comments |

### Edges

A dictionary mapping source node IDs to lists of `[destination_id, edge_label]` pairs.

```json
{
  "edges": {
    "math_5": [
      ["math_7", "accepted_answer"],
      ["math_7", "voted_answer"],
      ["math_16", "voted_answer"],
      ["math_4440112", "linked"]
    ],
    "math_100": [
      ["math_200", "duplicate"]
    ]
  }
}
```

#### Edge Types

| Label | Description |
|-------|-------------|
| `accepted_answer` | Question → its accepted answer |
| `voted_answer` | Question → answer with score >= threshold (default 2) |
| `linked` | Post → another post referenced via hyperlink in body |
| `duplicate` | Question → duplicate question (marked by moderators) |

### Stats

```json
{
  "stats": {
    "num_nodes": 4622655,
    "num_edges": 2812947,
    "edges_by_type": {
      "voted_answer": 1271110,
      "accepted_answer": 1012426,
      "linked": 460745,
      "duplicate": 68666
    },
    "nodes_by_source": {
      "math": 3845470,
      "mathoverflow": 347776,
      "stats": 429409
    }
  }
}
```

### Python Usage Example

```python
import json

with open('data/SE/se_graph.json') as f:
    graph = json.load(f)

# Get a question
question = graph['nodes']['math_1']
print(question['title'])
print(question['tags'])

# Get all answers to a question
question_id = 'math_5'
answers = [
    (dst, graph['nodes'][dst])
    for dst, label in graph['edges'].get(question_id, [])
    if label in ('accepted_answer', 'voted_answer')
]

# Find all questions with a specific tag
algebra_questions = [
    (nid, node) for nid, node in graph['nodes'].items()
    if node['type'] == 'question' and 'algebra' in node.get('tags', [])
]

# Traverse linked posts
for dst, label in graph['edges'].get('math_5', []):
    if label == 'linked':
        linked_node = graph['nodes'][dst]
        print(f"Links to: {linked_node['url']}")
```

---

## 2. ArXiv Theorem DAG Dataset

**File:** `data/arxiv/extracted_envs_dag_from_theorem_papers.jsonl`

**Description:** A JSONL file where each line is a JSON object representing one arXiv paper. Each paper contains a directed acyclic graph (DAG) of mathematical environments (theorems, lemmas, proofs, etc.) and their dependencies.

**Total entries:** 424,654 papers

### Entry Structure

Each line is a JSON object:

```json
{
  "filenames": ["main.tex"],
  "nodes": [...],
  "edges": [...],
  "adjacency": {...},
  "topo_order": [...],
  "dag_valid": true,
  "unknown_refs": {...},
  "skipped_forward": 0,
  "index": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `filenames` | list[string] | Source LaTeX file(s) |
| `nodes` | list[object] | List of mathematical environment nodes |
| `edges` | list[object] | List of dependency edges |
| `adjacency` | dict | Adjacency list: `{node_id: [dependent_node_ids]}` |
| `topo_order` | list[string] | Topological ordering of node IDs |
| `dag_valid` | bool | Whether the graph is a valid DAG |
| `unknown_refs` | dict | References that couldn't be resolved |
| `skipped_forward` | int | Number of forward references skipped |
| `index` | int | Entry index in the dataset |

### Node Structure

```json
{
  "id": "def:causal_CSI",
  "env": "definition",
  "env_raw": "definition",
  "label": "def:causal_CSI",
  "file": "main.tex",
  "ordinal": 1,
  "text": "The channel state information (CSI) at the encoder is \\emph{causal} if..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique node identifier |
| `env` | string | Normalized environment type |
| `env_raw` | string | Original LaTeX environment name |
| `label` | string or null | LaTeX label (if any) |
| `file` | string | Source file name |
| `ordinal` | int | Order of appearance in the paper |
| `text` | string | LaTeX content of the environment |

#### Environment Types

| Type | Count (sample) | Description |
|------|----------------|-------------|
| `proof` | 46,022 | Proof environments |
| `lemma` | 26,091 | Lemmas |
| `theorem` | 24,292 | Theorems |
| `proposition` | 15,748 | Propositions |
| `definition` | 14,456 | Definitions |
| `remark` | 14,388 | Remarks |
| `corollary` | 8,480 | Corollaries |
| `example` | 5,383 | Examples |
| `claim` | 1,076 | Claims |
| `assumption` | 1,023 | Assumptions |

### Edge Structure

```json
{
  "from": "proof#14@main.tex:13674",
  "to": "th:noCSI_encoder_decoder"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `from` | string | Source node ID (the node that references) |
| `to` | string | Target node ID (the node being referenced) |

**Semantics:** An edge from A to B means "A references/depends on B". For example, a proof references the theorem it proves.

### Node ID Formats

Node IDs follow several patterns:

1. **Labeled nodes:** `label` (e.g., `"def:causal_CSI"`, `"th:main"`)
2. **Unlabeled nodes:** `{env}#{ordinal}@{file}` (e.g., `"lem#1@main.tex"`)
3. **Proof nodes:** `proof#{ordinal}@{file}:{char_offset}` (e.g., `"proof#14@main.tex:13674"`)

### Python Usage Example

```python
import json

# Read entries one by one (memory efficient)
with open('data/arxiv/extracted_envs_dag_from_theorem_papers.jsonl') as f:
    for line in f:
        paper = json.loads(line)

        # Get all theorems in this paper
        theorems = [n for n in paper['nodes'] if n['env'] == 'theorem']

        # Get the proof for a theorem
        theorem_id = theorems[0]['id'] if theorems else None
        if theorem_id:
            # Find edges where 'to' is this theorem (proofs reference what they prove)
            proofs = [
                e['from'] for e in paper['edges']
                if e['to'] == theorem_id
            ]

        # Use adjacency list for efficient traversal
        for node_id, dependents in paper['adjacency'].items():
            if dependents:
                print(f"{node_id} is used by: {dependents}")

        # Process in topological order
        for node_id in paper['topo_order']:
            node = next(n for n in paper['nodes'] if n['id'] == node_id)
            print(f"{node['env']}: {node['text'][:50]}...")

# Load all papers into memory (if RAM permits)
papers = []
with open('data/arxiv/extracted_envs_dag_from_theorem_papers.jsonl') as f:
    for line in f:
        papers.append(json.loads(line))
```

### Finding Dependencies

```python
def get_dependencies(paper, node_id):
    """Get all nodes that a given node depends on."""
    return [e['to'] for e in paper['edges'] if e['from'] == node_id]

def get_dependents(paper, node_id):
    """Get all nodes that depend on a given node."""
    return paper['adjacency'].get(node_id, [])

def get_node_by_id(paper, node_id):
    """Get node data by ID."""
    return next((n for n in paper['nodes'] if n['id'] == node_id), None)
```

---

## Summary Comparison

| Aspect | SE Graph | ArXiv DAG |
|--------|----------|-----------|
| Format | Single JSON | JSONL (one JSON per line) |
| Scope | Cross-document graph | Per-paper graphs |
| Node types | question, answer | theorem, lemma, proof, definition, etc. |
| Edge meaning | Q→A relationships, links | Dependency/reference relationships |
| Text format | HTML | LaTeX |
| Size | 4.6M nodes, 2.8M edges | 424K papers |
