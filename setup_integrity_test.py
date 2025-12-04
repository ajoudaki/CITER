import json
import os

# Ensure directory exists
os.makedirs('data/lemmas_theorems', exist_ok=True)

# 1. Read some real data to be realistic
source_path = 'data/lemmas_theorems/toy.jsonl'
stmts = []
try:
    with open(source_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            stmts.extend(data.get('lemmas', []))
            stmts.extend(data.get('theorems', []))
            if len(stmts) >= 20:
                break
except:
    # Fallback if file read fails
    stmts = [f"Statement {i} for integrity test" for i in range(20)]

stmts = [s.replace('\n', ' ') for s in stmts[:20]]

# Split into lemmas and theorems for the JSONL
lemmas = stmts[:10]
theorems = stmts[10:]

# 2. Create integrity_test.jsonl (Single paper)
jsonl_path = 'data/lemmas_theorems/integrity_test.jsonl'
with open(jsonl_path, 'w') as f:
    record = {
        "title": "Integrity Test Paper",
        "arxiv_id": "0000.00000",
        "lemmas": lemmas,
        "theorems": theorems
    }
    f.write(json.dumps(record) + '\n')
print(f"Created {jsonl_path}")

# 3. Create temp_queries.txt (Exact same order: lemmas then theorems)
query_path = 'temp_queries.txt'
with open(query_path, 'w') as f:
    for s in lemmas:
        f.write(s + '\n')
    for s in theorems:
        f.write(s + '\n')
print(f"Created {query_path}")
