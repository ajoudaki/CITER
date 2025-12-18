import json
import numpy as np
import os

input_path = 'data/lemmas_theorems/tiny.jsonl'
sanitized_jsonl_path = 'data/lemmas_theorems/tiny_sanitized.jsonl'
output_query_path = 'tiny_queries.txt'
seed = 42

# 1. Load, Sanitize, and Save JSONL
print(f"Reading {input_path}...")
sanitized_papers = []
with open(input_path, 'r') as f:
    for line in f:
        paper = json.loads(line)
        # Sanitize
        paper['lemmas'] = [s.replace('\n', ' ') for s in paper.get('lemmas', [])]
        paper['theorems'] = [s.replace('\n', ' ') for s in paper.get('theorems', [])]
        
        statements = paper.get('lemmas', []) + paper.get('theorems', [])
        if len(statements) >= 1:
            sanitized_papers.append(paper)

print(f"Writing sanitized dataset to {sanitized_jsonl_path}...")
with open(sanitized_jsonl_path, 'w') as f:
    for paper in sanitized_papers:
        f.write(json.dumps(paper) + '\n')

# 2. Shuffle Papers (Logic from AllStatementsDataset)
print("Shuffling and extracting queries...")
rng = np.random.default_rng(seed)
indices = np.arange(len(sanitized_papers))
rng.shuffle(indices)

# 3. Extract Statements in Order from Sanitized Data
queries = []
for i in indices:
    paper = sanitized_papers[i]
    # Lemmas first
    for stmt in paper.get('lemmas', []):
        queries.append(stmt) # Already sanitized
    # Theorems second
    for stmt in paper.get('theorems', []):
        queries.append(stmt)

# 4. Write to query file
with open(output_query_path, 'w') as f:
    for q in queries:
        f.write(q + '\n')

print(f"Wrote {len(queries)} queries to {output_query_path}")
