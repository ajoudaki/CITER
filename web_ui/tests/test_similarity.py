#!/usr/bin/env python3
"""Test script to verify model loading and similarity search"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_similarity():
    print("Testing Model Loading and Similarity Search\n")
    print("=" * 50)

    # 1. Load toy dataset
    print("1. Loading 'toy' dataset...")
    response = requests.get(f"{BASE_URL}/api/load_dataset/toy")
    data = response.json()
    if data['success']:
        print(f"   ✓ Loaded {data['num_papers']} papers")
    else:
        print(f"   ✗ Failed: {data.get('error', 'Unknown error')}")
        return

    # 2. List available models
    print("\n2. Available models:")
    response = requests.get(f"{BASE_URL}/api/models")
    models = response.json()
    for model in models:
        print(f"   - {model['display']} ({model['name']})")

    if not models:
        print("   No models available. Please ensure model files exist in outputs/")
        return

    # 3. Load first available model
    model_to_load = models[0]['name']
    print(f"\n3. Loading model '{model_to_load}'...")
    response = requests.get(f"{BASE_URL}/api/load_model/{model_to_load}")
    data = response.json()
    if data['success']:
        print(f"   ✓ Model loaded successfully")
    else:
        print(f"   ✗ Failed: {data.get('error', 'Unknown error')}")
        return

    # 4. Find similar statements
    print("\n4. Finding similar statements to Paper 0, Theorem 0...")
    response = requests.post(
        f"{BASE_URL}/api/find_similar",
        json={
            "paper_idx": 0,
            "stmt_idx": 0,
            "stmt_type": "theorem"
        }
    )
    data = response.json()

    if data['success']:
        print(f"   ✓ Found {len(data['results'])} similar statements")
        print(f"\n   Query Statement:")
        print(f"   Type: {data['query']['type']}")
        print(f"   Text: {data['query']['text'][:100]}...")

        print(f"\n   Top 5 Similar Statements:")
        for i, result in enumerate(data['results'][:5]):
            print(f"\n   {i+1}. Paper {result['paper_idx']+1} - {result['type']}")
            print(f"      Similarity: {result['similarity']*100:.2f}%")
            print(f"      Text: {result['text'][:100]}...")
    else:
        print(f"   ✗ Failed: {data.get('error', 'Unknown error')}")

    print("\n" + "=" * 50)
    print("✓ Model loading and similarity search working!")
    print("\nTo use the UI:")
    print("1. Open http://localhost:5000 in your browser")
    print("2. Load the 'toy' dataset")
    print("3. Select and load a model (BERT or Qwen)")
    print("4. Click on any paper")
    print("5. Click 'Find Similar' next to any theorem/lemma")

if __name__ == "__main__":
    test_similarity()