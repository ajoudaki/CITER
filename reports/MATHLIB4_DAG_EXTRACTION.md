# Mathlib 4 DAG Extraction Report

This document describes the process of extracting the theorem dependency graph (DAG) from Mathlib 4 using LeanDojo, along with output specifications for downstream development.

## Prerequisites

### 1. Install elan (Lean version manager)
```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

### 2. Create Python virtual environment
```bash
cd /local/home/ajoudaki/citer/Lean
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install lean-dojo networkx tqdm loguru
```

## Extraction Process

### Step 1: Ensure PATH includes elan binaries

LeanDojo requires `lake` (Lean build tool) to be available. Create a wrapper script:

```bash
#!/bin/bash
export PATH="$HOME/.elan/bin:$PATH"
source /local/home/ajoudaki/citer/Lean/.venv/bin/activate
cd /local/home/ajoudaki/citer/Lean
```

### Step 2: Run extraction

**Important:** Use a LeanDojo-compatible Mathlib 4 commit. The LeanDojo Benchmark 4 uses commit `29dcec074de168ac2bf835a77ef68bbe069194c5`.

```bash
python extract_mathlib_dag.py \
    --commit 29dcec074de168ac2bf835a77ef68bbe069194c5 \
    --output-dir ./output \
    --formats json edgelist
```

### Full wrapper script (`run_extraction.sh`)
```bash
#!/bin/bash
export PATH="$HOME/.elan/bin:$PATH"
source /local/home/ajoudaki/citer/Lean/.venv/bin/activate
cd /local/home/ajoudaki/citer/Lean

python extract_mathlib_dag.py \
    --commit 29dcec074de168ac2bf835a77ef68bbe069194c5 \
    --output-dir ./output \
    --formats json edgelist
```

## Output Statistics

### Theorem Dependency Graph
| Metric | Value |
|--------|-------|
| **Number of nodes (theorems)** | 150,156 |
| **Number of edges (dependencies)** | 675,994 |
| **Theorems with full metadata** | 126,778 |
| **Average in-degree** | 4.50 |
| **Average out-degree** | 4.50 |
| **Max in-degree** | 19,009 |
| **Max out-degree** | 114 |
| **Average premises per theorem** | 6.58 |

### File Dependency Graph
| Metric | Value |
|--------|-------|
| **Number of files** | 5,674 |
| **Is DAG** | true |

**Note:** File-level edges were not fully extracted in this run (edges = 0). The theorem-level graph is the primary output.

## Output Files

All output files are located in `./output/`.

| File | Size | Description |
|------|------|-------------|
| `theorem_dependencies.json` | 82 MB | Theorem graph in JSON format |
| `theorem_dependencies.edgelist` | 36 MB | Theorem graph as edge list |
| `theorems.json` | 51 MB | Theorem metadata dictionary |
| `file_dependencies.json` | 605 KB | File graph in JSON format |
| `file_dependencies.edgelist` | 0 B | File graph as edge list |
| `statistics.json` | 475 B | Summary statistics |

---

## File Format Specifications

### 1. `theorem_dependencies.json`

JSON object containing the theorem dependency graph.

```json
{
  "nodes": ["theorem_name_1", "theorem_name_2", ...],
  "edges": [["source_theorem", "target_premise"], ...],
  "adjacency": {
    "theorem_name_1": ["premise_1", "premise_2", ...],
    "theorem_name_2": [...]
  }
}
```

**Schema:**
| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `string[]` | List of all theorem/declaration fully qualified names |
| `edges` | `[string, string][]` | List of directed edges `[source, target]` where source depends on target |
| `adjacency` | `{string: string[]}` | Adjacency list mapping each node to its successors (premises) |

**Edge semantics:** An edge `(A, B)` means theorem `A` uses `B` as a premise in its proof.

**Example entries:**
```json
{
  "nodes": [
    "Prod.ext",
    "PProd.ext",
    "Sigma.ext",
    "rfl",
    ...
  ],
  "edges": [
    ["Prod.ext", "Prod"],
    ["Prod.ext", "rfl"],
    ...
  ],
  "adjacency": {
    "Prod.ext": ["Prod", "rfl", "rfl", "rfl"],
    ...
  }
}
```

---

### 2. `theorem_dependencies.edgelist`

Plain text edge list format, one edge per line.

```
source_theorem target_premise {}
```

**Format:** Space-separated values with an empty JSON object `{}` at the end (NetworkX edgelist format).

**Example:**
```
Prod.ext Prod {}
Prod.ext rfl {}
PProd.ext PProd {}
PProd.ext rfl {}
Sigma.ext Sigma {}
Sigma.ext HEq {}
```

**Loading with NetworkX:**
```python
import networkx as nx
G = nx.read_edgelist("theorem_dependencies.edgelist",
                      create_using=nx.DiGraph())
```

---

### 3. `theorems.json`

JSON dictionary mapping theorem names to their metadata.

```json
{
  "fully.qualified.theorem.name": {
    "full_name": "fully.qualified.theorem.name",
    "file_path": "path/to/source/file.lean",
    "start_line": 0,
    "end_line": 0,
    "premises": ["premise1", "premise2", ...],
    "statement": ""
  },
  ...
}
```

**Schema:**
| Field | Type | Description |
|-------|------|-------------|
| `full_name` | `string` | Fully qualified name of the theorem |
| `file_path` | `string` | Relative path to source file |
| `start_line` | `int` | Starting line number (0 if unavailable) |
| `end_line` | `int` | Ending line number (0 if unavailable) |
| `premises` | `string[]` | List of premises used (may contain duplicates) |
| `statement` | `string` | Theorem statement (empty if unavailable) |

**Note:** Line numbers and statements may be 0/empty for some theorems depending on LeanDojo tracing capabilities.

**Example:**
```json
{
  "Prod.ext": {
    "full_name": "Prod.ext",
    "file_path": "src/lean/Init/Ext.lean",
    "start_line": 0,
    "end_line": 0,
    "premises": ["Prod", "rfl", "rfl", "rfl"],
    "statement": ""
  }
}
```

---

### 4. `file_dependencies.json`

JSON object containing the file-level dependency graph.

```json
{
  "nodes": ["path/to/file1.lean", "path/to/file2.lean", ...],
  "edges": [["importing_file", "imported_file"], ...],
  "adjacency": {
    "path/to/file1.lean": ["imported1.lean", ...],
    ...
  }
}
```

**Schema:** Same structure as `theorem_dependencies.json`, but nodes are file paths.

---

### 5. `statistics.json`

Summary statistics for both graphs.

```json
{
  "file_graph": {
    "num_nodes": 5674,
    "num_edges": 0,
    "is_dag": true,
    "avg_in_degree": 0.0,
    "avg_out_degree": 0.0,
    "max_in_degree": 0,
    "max_out_degree": 0
  },
  "theorem_graph": {
    "num_nodes": 150156,
    "num_edges": 675994,
    "num_theorems_with_info": 126778,
    "avg_in_degree": 4.50194464423666,
    "avg_out_degree": 4.50194464423666,
    "max_in_degree": 19009,
    "max_out_degree": 114,
    "avg_premises": 6.584754452665289
  }
}
```

---

## Usage Examples

### Load theorem graph with NetworkX
```python
import json
import networkx as nx

# From JSON
with open("output/theorem_dependencies.json") as f:
    data = json.load(f)

G = nx.DiGraph()
G.add_nodes_from(data["nodes"])
G.add_edges_from(data["edges"])

# Or from edgelist
G = nx.read_edgelist("output/theorem_dependencies.edgelist",
                      create_using=nx.DiGraph())
```

### Get premises for a specific theorem
```python
import json

with open("output/theorems.json") as f:
    theorems = json.load(f)

# Get premises for a specific theorem
theorem_name = "Nat.add_comm"
if theorem_name in theorems:
    info = theorems[theorem_name]
    print(f"Premises: {info['premises']}")
    print(f"File: {info['file_path']}")
```

### Find theorems with most dependencies
```python
import json

with open("output/theorem_dependencies.json") as f:
    data = json.load(f)

# Count outgoing edges (premises used)
premise_counts = {node: len(deps) for node, deps in data["adjacency"].items()}
top_theorems = sorted(premise_counts.items(), key=lambda x: -x[1])[:10]

for name, count in top_theorems:
    print(f"{name}: {count} premises")
```

### Find most-used lemmas (highest in-degree)
```python
import networkx as nx

G = nx.read_edgelist("output/theorem_dependencies.edgelist",
                      create_using=nx.DiGraph())

in_degrees = dict(G.in_degree())
most_used = sorted(in_degrees.items(), key=lambda x: -x[1])[:10]

for name, count in most_used:
    print(f"{name}: used by {count} theorems")
```

---

## Known Issues and Notes

1. **LeanDojo version compatibility:** Not all Mathlib 4 commits are compatible with LeanDojo. Use commits from the LeanDojo Benchmark 4 for guaranteed compatibility.

2. **Memory requirements:** Full Mathlib 4 tracing requires significant memory (~100GB+ RAM). Ray's distributed tracing may kill tasks if memory is insufficient.

3. **Null handling:** Some theorems/premises may have `None` names in LeanDojo. The extraction script filters these out.

4. **File-level edges:** In this extraction, file-level edges were not fully populated. For file dependencies, consider using LeanDojo's `file_dep_graph` attribute directly if available.

5. **Premise duplicates:** The `premises` list in `theorems.json` may contain duplicates (e.g., `rfl` appearing multiple times) reflecting actual usage in proofs.

---

## Informal Mathlib Dataset Integration

### Dataset Source

The formal theorem graph has been mapped to the informal Mathlib dataset from HuggingFace:
- **Dataset:** `FrenzyMath/mathlib_informal_v4.16.0`
- **Total entries:** 187,540

### Informal Dataset Contents

The informal dataset contains human-readable descriptions for multiple declaration types:

| Kind | Count | % |
|------|-------|---|
| theorem | 136,608 | 72.8% |
| definition | 25,280 | 13.5% |
| instance | 20,621 | 11.0% |
| structure | 2,614 | 1.4% |
| abbrev | 2,066 | 1.1% |
| inductive | 269 | 0.1% |
| opaque | 76 | 0.04% |
| classInductive | 6 | 0.003% |

### Mapping Statistics

| Metric | Value |
|--------|-------|
| **Total graph nodes** | 150,156 |
| **Matched nodes** | 125,634 |
| **Graph coverage** | 83.7% |
| **Unmatched (mostly core Lean)** | 24,522 (16.3%) |

**Matched by kind:**

| Kind | Matched |
|------|---------|
| theorem | 111,606 |
| definition | 11,309 |
| structure | 1,454 |
| abbrev | 760 |
| instance | 392 |
| inductive | 110 |
| classInductive | 3 |

### Mapping Output Files

| File | Size | Description |
|------|------|-------------|
| `enriched_theorem_graph_all_types.json` | 222 MB | Full graph with all node types and informal metadata |
| `theorem_graph_with_descriptions_all_types.json` | 130 MB | Simplified graph with informal descriptions |
| `formal_informal_mapping_all_types.json` | 160 MB | Complete mapping between formal and informal entries |
| `mapping_statistics_all_types.json` | — | Mapping statistics |
| `informal_data/mathlib_informal.parquet` | 72 MB | Raw informal dataset |

### `enriched_theorem_graph_all_types.json` Schema

```json
{
  "nodes": ["node_name_1", ...],
  "node_data": {
    "node_name_1": {
      "name": "node_name_1",
      "has_informal": true,
      "kind": "theorem|definition|structure|instance|...",
      "module_name": ["Mathlib", "Order", "Defs"],
      "signature": " : ¬a ≤ b ↔ b < a",
      "type": "∀ {α : Type u_1} ...",
      "docstring": "Optional docstring",
      "informal_name": "Human-readable title",
      "informal_description": "LaTeX description with $math$"
    }
  },
  "edges": [["source", "target"], ...],
  "adjacency": {...},
  "statistics": {
    "total_nodes": 150156,
    "total_edges": 675994,
    "nodes_with_informal": 125634,
    "coverage": 0.837,
    "matched_by_kind": {"theorem": 111606, "definition": 11309, ...}
  }
}
```

### `theorem_graph_with_descriptions_all_types.json` Schema

```json
{
  "nodes_with_descriptions": {
    "node_name_1": {
      "kind": "theorem|definition|structure|...",
      "informal_name": "Human-readable title",
      "informal_description": "LaTeX description with $math$",
      "signature": " : ¬a ≤ b ↔ b < a"
    }
  },
  "edges": [["source", "target"], ...],
  "statistics": {...}
}
```

### Usage Examples

#### Get informal description for any node (theorem, definition, etc.)

```python
import json

with open("output/theorem_graph_with_descriptions_all_types.json") as f:
    data = json.load(f)

# Works for theorems
node_name = "Nat.add_comm"
if node_name in data["nodes_with_descriptions"]:
    info = data["nodes_with_descriptions"][node_name]
    print(f"Kind: {info['kind']}")
    print(f"Title: {info['informal_name']}")
    print(f"Description: {info['informal_description']}")

# Also works for definitions, structures, etc.
node_name = "MulRingNorm"  # A structure
if node_name in data["nodes_with_descriptions"]:
    info = data["nodes_with_descriptions"][node_name]
    print(f"Kind: {info['kind']}")  # "structure"
    print(f"Title: {info['informal_name']}")
```

#### Filter nodes by kind

```python
import json

with open("output/enriched_theorem_graph_all_types.json") as f:
    data = json.load(f)

# Get all definitions
definitions = [
    name for name, info in data["node_data"].items()
    if info.get("kind") == "definition"
]
print(f"Found {len(definitions)} definitions")

# Get all structures
structures = [
    name for name, info in data["node_data"].items()
    if info.get("kind") == "structure"
]
print(f"Found {len(structures)} structures")
```

### Graph Structure Notes

1. **Heterogeneous graph:** Nodes include theorems, definitions, structures, instances, inductives, etc. Edges are untyped and represent "B appears in A's proof/definition term."

2. **Edge semantics:** An edge `A → B` means declaration A directly references B in its implementation. This is **immediate dependency only**, not transitive closure.

3. **Node sources:** The formal extraction uses LeanDojo's `get_traced_theorems()` which extracts theorems as source nodes, but premises (edge targets) can be any declaration type.

### Notes on Mapping

1. **Version mismatch:** The informal dataset is from Mathlib v4.16.0, while the formal extraction is from a different commit. This causes ~16% of nodes to be unmatched.

2. **Core Lean constants:** Unmatched nodes (e.g., `rfl`, `Eq.symm`, `Bool.*`, `Decidable.*`) are from core Lean, not Mathlib, so they don't appear in the informal dataset.

3. **Name format:** The mapping uses the fully qualified Lean name (e.g., `Nat.add_comm`, `MulRingNorm`) as the key.

---

## Commit Information

- **Mathlib 4 commit:** `29dcec074de168ac2bf835a77ef68bbe069194c5`
- **Source:** LeanDojo Benchmark 4 compatible commit
- **Extraction date:** December 19, 2025
- **Informal dataset:** `FrenzyMath/mathlib_informal_v4.16.0`
