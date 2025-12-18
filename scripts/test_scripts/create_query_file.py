import json

input_path = 'data/lemmas_theorems/toy.jsonl'
output_path = 'temp_queries.txt'
limit = 50  # Limit to 50 statements for speed

queries = []
try:
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            queries.extend(data.get('lemmas', []))
            queries.extend(data.get('theorems', []))
            if len(queries) >= limit:
                break
except Exception as e:
    print(f"Error reading {input_path}: {e}")

# Filter out empty or very short strings if any
queries = [q.replace('\n', ' ') for q in queries if q and len(q) > 10][:limit]

with open(output_path, 'w') as f:
    for q in queries:
        f.write(q + '\n')

print(f"Wrote {len(queries)} queries to {output_path}")

