#!/usr/bin/env python3
"""Simple analysis of theorem/lemma token lengths"""

import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Collect lengths
lengths = []

with open('data/lemmas_theorems.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i % 1000 == 0:
            print(f"Processing paper {i}...")
        paper = json.loads(line)
        statements = paper.get('lemmas', []) + paper.get('theorems', [])

        for stmt in statements:
            if stmt:  # Skip empty statements
                tokens = tokenizer(stmt, truncation=False, add_special_tokens=True)
                lengths.append(len(tokens['input_ids']))

# Basic stats
print(f"Total statements: {len(lengths)}")
print(f"Min length: {min(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Mean length: {sum(lengths)/len(lengths):.1f}")
print(f"Median length: {sorted(lengths)[len(lengths)//2]}")

# Percentiles
sorted_lengths = sorted(lengths)
print(f"\nPercentiles:")
for p in [50, 75, 90, 95, 99]:
    idx = int(len(sorted_lengths) * p / 100)
    print(f"  {p}%: {sorted_lengths[idx]} tokens")

# How many fit in common sizes
for max_len in [128, 256, 512]:
    fit = sum(1 for l in lengths if l <= max_len)
    print(f"\nFit in {max_len}: {fit}/{len(lengths)} ({100*fit/len(lengths):.1f}%)")

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, edgecolor='black')
plt.axvline(x=256, color='r', linestyle='--', label='Current MAX_LENGTH=256')
plt.axvline(x=512, color='g', linestyle='--', label='BERT max=512')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.title('Distribution of Theorem/Lemma Token Lengths')
plt.legend()
plt.savefig('token_lengths.png')
print("\nHistogram saved to token_lengths.png")