# Data Statistics Report

This report summarizes node and edge counts across all data sources in the repository.

## Summary by Source

| Source | Nodes | Node % | Edges | Edge % |
|--------|------:|-------:|------:|-------:|
| **Mathlib** | 150,156 | 0.84% | 675,994 | 5.44% |
| **SE** | 4,622,655 | 25.93% | 2,812,947 | 22.63% |
| **arXiv** | 13,054,509 | 73.23% | 8,941,160 | 71.93% |
| **TOTAL** | 17,827,320 | 100% | 12,430,101 | 100% |

### Edge-to-Node Ratio

| Source | Edges/Node |
|--------|----------:|
| Mathlib | 4.50 |
| SE | 0.61 |
| arXiv | 0.68 |

---

## Mathlib Data

**Location:** `data/Mathlib/`

**Primary file:** `enriched_theorem_graph_all_types.json` (232 MB)

### Structure

```python
{
    'nodes': [...],           # 150,156 node names (list of strings)
    'node_data': {...},       # dict: name -> full metadata
    'edges': [...],           # 675,994 edges as [src, dst] pairs
    'adjacency': {...},       # dict: name -> list of outgoing dependencies
    'statistics': {...}       # summary stats
}
```

### Node Data Fields

For nodes with informal mappings:

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Lean declaration name | `not_le` |
| `has_informal` | Whether informal description exists | `True` |
| `kind` | Declaration type | `theorem` |
| `module_name` | Module path | `['Mathlib', 'Order', 'Defs', 'LinearOrder']` |
| `signature` | Lean signature | `: ¬a ≤ b ↔ b < a` |
| `type` | Full Lean type | `∀ {α : Type u_1} [inst : LinearOrder α] ...` |
| `docstring` | Documentation string | `None` or text |
| `informal_name` | Natural language title | `Negation of Non-Strict Inequality...` |
| `informal_description` | Natural language description | `For any two elements $a$ and $b$...` |

### Informal Coverage

- **Nodes with informal descriptions:** 125,634 / 150,156 (83.7%)
- **Nodes without informal descriptions:** 24,522 (16.3%)

### Node Types (`kind`)

| Kind | Count | % |
|------|------:|--:|
| theorem | 111,606 | 74.3% |
| unknown | 24,522 | 16.3% |
| definition | 11,309 | 7.5% |
| structure | 1,454 | 1.0% |
| abbrev | 760 | 0.5% |
| instance | 392 | 0.3% |
| inductive | 110 | 0.1% |
| classInductive | 3 | <0.1% |

### Other Mathlib Files

| File | Description |
|------|-------------|
| `theorem_dependencies.json` | Raw graph with nodes, edges, adjacency |
| `formal_informal_mapping_all_types.json` | Mapping from formal names to informal descriptions |
| `theorems.json` | Theorem metadata (file paths, premises, statements) |
| `mathlib_informal.parquet` | HuggingFace informal descriptions dataset |

---

## StackExchange Data

**Location:** `data/SE/`

**Primary file:** `se_graph.json`

### Statistics

| Metric | Value |
|--------|------:|
| Total nodes | 4,622,655 |
| Total edges | 2,812,947 |

### Nodes by Source Site

| Site | Approximate Nodes |
|------|------------------:|
| math.stackexchange.com | ~3,800,000 |
| stats.stackexchange.com | ~429,000 |
| mathoverflow.net | ~347,000 |

### Node Types

- Questions
- Answers
- Comments

### Edge Types

| Edge Type | Description |
|-----------|-------------|
| `voted_answer` | Question links to upvoted answer |
| `accepted_answer` | Question links to accepted answer |
| `linked` | Post links to another post |
| `duplicate` | Question marked as duplicate |

---

## arXiv Data

**Location:** `data/arxiv/`

**Primary file:** `extracted_envs_dag_from_theorem_papers.jsonl`

### Statistics

| Metric | Value |
|--------|------:|
| Total nodes | 13,054,509 |
| Total edges | 8,941,160 |
| Papers | ~10,000 |

### Data Format

Each line is a JSON object representing one paper:

```python
{
    'nodes': [...],        # List of theorem/proof/definition environments
    'edges': [...],        # Dependencies between statements
    'adjacency': {...},    # Adjacency list representation
    'topo_order': [...],   # Topological ordering
    'dag_valid': bool,     # Whether edges form valid DAG
    'unknown_refs': {...}  # Unresolved references
}
```

### Node Fields

| Field | Description |
|-------|-------------|
| `id` | Unique identifier within paper |
| `env` | Canonical environment type (theorem, lemma, proof, etc.) |
| `env_raw` | Raw LaTeX environment name |
| `label` | LaTeX label |
| `file` | Source file name |
| `ordinal` | Position in paper |
| `text` | Full LaTeX content |

### Node Types (Environments)

- theorem
- lemma
- proof
- definition
- proposition
- corollary
- remark
- example

---

## Preprocessed Unified Dataset

**Location:** `data/processed/`

The preprocessed data merges all sources into a unified format for training.

### Files

| File | Format | Description |
|------|--------|-------------|
| `nodes.arrow` | Apache Arrow | All nodes with unified schema |
| `edges.npy` | NumPy array | Edge tuples `[src, dst, type, weight]` |
| `metadata.json` | JSON | Dataset statistics |

### Unified Node Schema

| Column | Description |
|--------|-------------|
| `global_id` | Unique ID across all datasets |
| `text` | Node content |
| `cluster_id` | Groups related nodes (paper/thread) |
| `source_type` | 0=SE, 1=arXiv, 3=Mathlib |
| `original_id` | Source-specific identifier |
| `node_type` | question/answer/theorem/lemma/proof/definition |

### Edge Types (Preprocessed)

| ID | Name | Weight | Description |
|---:|------|-------:|-------------|
| 0 | SE_Answer | 1.0-2.0 | Question to answer |
| 1 | SE_Tag | 1.0 | Post to tag (reserved) |
| 2 | SE_Duplicate | 1.0 | Duplicate questions |
| 3 | SE_Linked | 0.8 | Hyperlink reference |
| 4 | ArXiv_Proof | 2.0 | Theorem to proof |
| 5 | ArXiv_Dep | 1.0 | Statement dependency |
| 6 | ArXiv_Ordinal | 0.5 | Sequential order |
