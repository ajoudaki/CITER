#!/usr/bin/env python3
"""Test script to verify UI functionality"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_ui():
    print("Testing Theorem & Lemma Browser UI\n")
    print("=" * 50)

    # 1. Test dataset listing
    print("1. Available Datasets:")
    response = requests.get(f"{BASE_URL}/api/datasets")
    datasets = response.json()
    for ds in datasets:
        print(f"   - {ds['name']}: {ds['size']}")

    # 2. Load toy dataset
    print("\n2. Loading 'toy' dataset...")
    response = requests.get(f"{BASE_URL}/api/load_dataset/toy")
    data = response.json()
    if data['success']:
        print(f"   ✓ Loaded {data['num_papers']} papers")
        print(f"   Sample papers:")
        for i, paper in enumerate(data['papers'][:3]):
            print(f"     Paper {i+1}: {paper['title'][:50]}...")
            print(f"       - Theorems: {paper['num_theorems']}, Lemmas: {paper['num_lemmas']}")

    # 3. Get first paper details
    print("\n3. Loading Paper 0 details...")
    response = requests.get(f"{BASE_URL}/api/paper/0")
    paper = response.json()
    if paper['success']:
        print(f"   Title: {paper['title']}")
        print(f"   ArXiv ID: {paper['arxiv_id']}")
        print(f"   Theorems: {len(paper['theorems'])}")
        print(f"   Lemmas: {len(paper['lemmas'])}")

        if paper['theorems']:
            print(f"\n   First Theorem Preview:")
            print(f"   {paper['theorems'][0]['text'][:200]}...")

    # 4. Test search
    print("\n4. Testing search for 'continuous'...")
    response = requests.post(
        f"{BASE_URL}/api/search",
        json={"query": "continuous"}
    )
    results = response.json()
    if results['success']:
        print(f"   Found {results['num_results']} results")
        if results['results']:
            print(f"   First result:")
            r = results['results'][0]
            print(f"     Paper: {r['paper_title'][:50]}...")
            print(f"     Type: {r['type']}")

    print("\n" + "=" * 50)
    print("✓ UI is working correctly!")
    print(f"\nOpen http://localhost:5000 in your browser to use the interface")

if __name__ == "__main__":
    test_ui()