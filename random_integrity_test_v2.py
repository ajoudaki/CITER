import json
import os
import subprocess
import torch
import sys
import numpy as np
import argparse

def get_cmd_prefix(nproc):
    if nproc <= 1:
        return "python"
    return f"torchrun --nproc_per_node={nproc}"

def run_randomized_integrity_test_v2(dataset_size='small', split='eval', num_samples=50, model_path='outputs/demo/big_run_qwen-7b', nproc=1):
    print(f"--- Randomized Integrity Test V2 (Using Metadata) ---")
    print(f"Dataset: {dataset_size} | Split: {split} | Samples: {num_samples} | GPUs: {nproc}")
    
    cmd_prefix = get_cmd_prefix(nproc)

    # 1. Check for Metadata File
    # ... (rest of logic same) ...
    
    metadata_path = os.path.join(model_path, 'embeddings', dataset_size, f'{split}_metadata.jsonl')
    emb_path = os.path.join(model_path, 'embeddings', dataset_size, f'{split}_embeddings.pt')
    
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at {metadata_path}.")
        print("Re-computing embeddings to generate metadata... (This might take a moment)")
        
        cmd = (
            f"{cmd_prefix} theorem_contrastive_training.py "
            f"model=qwen-2.5-math-7b "
            f"+training.compute_embeddings=true "
            f"dataset.size={dataset_size} "
            f"+dataset.split={split} "
            f"+training.load_model_path={model_path} "
            f"training.max_length=256 " # Keep it fast for test
            f"training.micro_batch_size=8 "
            f"training.quantization.enabled=true "
            f"wandb.enabled=false "
            f"output.save_dir={model_path}" # Save back to the same place
        )
        
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            print("Re-computation failed!")
            print(res.stderr)
            sys.exit(1)
        else:
            print("Re-computation successful. Metadata generated.")

    # ... (Logic for Loading Embeddings and Metadata - Unchanged) ...
    print(f"Loading embeddings from {emb_path}...")
    full_embeddings = torch.load(emb_path).float()
    print(f"Full embeddings shape: {full_embeddings.shape}")
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))
    
    if len(metadata) != full_embeddings.shape[0]:
        print(f"CRITICAL ERROR: Metadata count ({len(metadata)}) != Embedding count ({full_embeddings.shape[0]})")
        sys.exit(1)
        
    # 3. Randomly Select Samples using the Metadata
    indices = np.random.choice(len(metadata), num_samples, replace=False)
    indices.sort()
    
    selected_metadata = [metadata[i] for i in indices]
    selected_embeddings = full_embeddings[indices]

    # DEBUG: Check for duplicate texts
    texts = [m['text'] for m in selected_metadata]
    unique_texts = set(texts)
    if len(unique_texts) < len(texts):
        print(f"WARNING: {len(texts) - len(unique_texts)} duplicate texts found!")
        from collections import Counter
        text_counts = Counter(texts)
        for t, c in text_counts.items():
            if c > 1:
                print(f"  Duplicate ({c}x): {t[:60]}...")
    else:
        print(f"All {len(texts)} selected texts are unique.")

    print(f"Selected {len(selected_metadata)} samples.")
    
    # 4. Create Temp Dataset from Selected Texts
    temp_dataset_name = f"temp_integrity_v2_{dataset_size}_{split}"
    temp_jsonl_path = f"data/lemmas_theorems/{temp_dataset_name}.jsonl"
    
    with open(temp_jsonl_path, 'w') as f:
        for item in selected_metadata:
            # We must ensure the text is processed exactly as before.
            # The `compute_embeddings` logic reads from JSONL and puts it into 'lemmas'/'theorems'.
            # We'll use the 'lemmas' list to store the text.
            record = {"lemmas": [item['text']], "theorems": []}
            f.write(json.dumps(record) + '\n')
            
    # 5. Compute New Embeddings
    print("Computing new embeddings for verification...")
    cmd_verify = (
        f"{cmd_prefix} theorem_contrastive_training.py "
        f"model=qwen-2.5-math-7b "
        f"+training.compute_embeddings=true "
        f"dataset.size={temp_dataset_name} "
        f"+dataset.split=all "
        f"+training.load_model_path={model_path} "
        f"training.max_length=256 "
        f"training.micro_batch_size=8 "
        f"training.quantization.enabled=true "
        f"wandb.enabled=false "
        # Note: We don't need to override save_dir, it defaults to outputs/
    )
    
    res = subprocess.run(cmd_verify, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("Verification computation failed!")
        print(res.stderr)
        sys.exit(1)
        
    # ... (Rest of script unchanged) ...
    # 6. Compare
    # New embeddings will be in outputs/embeddings/{temp_dataset_name}/all_embeddings.pt
    new_emb_path = f"outputs/embeddings/{temp_dataset_name}/all_embeddings.pt"
    if not os.path.exists(new_emb_path):
         # Check model path just in case config behaved differently
         new_emb_path = os.path.join(model_path, "embeddings", temp_dataset_name, "all_embeddings.pt")

    if not os.path.exists(new_emb_path):
        print("Error: New embeddings file not found.")
        sys.exit(1)

    new_embeddings = torch.load(new_emb_path).float()

    # --- FIX: Handle Shuffling ---
    # The new run shuffled the temp dataset, so new_embeddings are in random order.
    # We must use the new metadata to align them back to selected_metadata order.
    
    new_metadata_path = os.path.join(os.path.dirname(new_emb_path), "all_metadata.jsonl")
    if not os.path.exists(new_metadata_path):
        print(f"Error: New metadata file not found at {new_metadata_path}")
        sys.exit(1)
        
    print(f"Loading new metadata from {new_metadata_path}...")
    new_metadata = []
    with open(new_metadata_path, 'r') as f:
        for line in f:
            new_metadata.append(json.loads(line))
            
    if len(new_metadata) != len(new_embeddings):
         print("Error: New metadata length mismatch.")
         sys.exit(1)
         
    # Map text -> embedding vector
    text_to_emb = {}
    for idx, item in enumerate(new_metadata):
        text_to_emb[item['text']] = new_embeddings[idx]
        
    # Re-construct new_embeddings in the expected order
    ordered_new_embeddings = []
    for item in selected_metadata:
        text = item['text']
        if text not in text_to_emb:
            print(f"Error: Text not found in new output: {text[:50]}...")
            sys.exit(1)
        ordered_new_embeddings.append(text_to_emb[text])
        
    new_embeddings = torch.stack(ordered_new_embeddings)

    # DEBUG: Check for zero or near-zero embeddings before normalization
    old_norms = torch.norm(selected_embeddings, dim=1)
    new_norms = torch.norm(new_embeddings, dim=1)
    if (old_norms < 1e-6).any() or (new_norms < 1e-6).any():
        print("WARNING: Zero or near-zero embeddings detected!")
        print(f"  Old zeros: {(old_norms < 1e-6).sum().item()}, New zeros: {(new_norms < 1e-6).sum().item()}")
    else:
        print(f"Embedding norms OK - Old: min={old_norms.min():.4f}, max={old_norms.max():.4f} | New: min={new_norms.min():.4f}, max={new_norms.max():.4f}")

    # DEBUG: Check direct cosine similarity between old and new for each sample
    # This tells us if individual embeddings match (before looking at pair-wise sim matrix)
    old_norm = torch.nn.functional.normalize(selected_embeddings, p=2, dim=1)
    new_norm = torch.nn.functional.normalize(new_embeddings, p=2, dim=1)
    direct_cosine = (old_norm * new_norm).sum(dim=1)
    print(f"Direct old-vs-new cosine per sample: min={direct_cosine.min():.4f}, max={direct_cosine.max():.4f}, mean={direct_cosine.mean():.4f}")
    low_match_indices = (direct_cosine < 0.99).nonzero(as_tuple=True)[0]
    if len(low_match_indices) > 0:
        print(f"WARNING: {len(low_match_indices)} samples have low direct match (cosine < 0.99):")
        for idx in low_match_indices[:5]:  # Show first 5
            print(f"  Sample {idx.item()}: cosine={direct_cosine[idx]:.4f}, text={selected_metadata[idx.item()]['text'][:80]}...")

    # Compare Similarity Matrices
    # Normalize
    selected_embeddings = torch.nn.functional.normalize(selected_embeddings, p=2, dim=1)
    new_embeddings = torch.nn.functional.normalize(new_embeddings, p=2, dim=1)
    
    sim_old = torch.mm(selected_embeddings, selected_embeddings.t())
    sim_new = torch.mm(new_embeddings, new_embeddings.t())
    
    diff = (sim_old - sim_new).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # DEBUG: Print worst pair details
    if max_diff > 0.01:
        worst_idx = diff.argmax().item()
        i, j = worst_idx // diff.shape[1], worst_idx % diff.shape[1]
        print(f"\n--- DEBUG: Worst Pair Analysis ---")
        print(f"Worst pair indices: ({i}, {j})")
        print(f"  sim_old[{i},{j}] = {sim_old[i,j]:.4f}")
        print(f"  sim_new[{i},{j}] = {sim_new[i,j]:.4f}")
        print(f"  Text i: {selected_metadata[i]['text'][:100]}...")
        print(f"  Text j: {selected_metadata[j]['text'][:100]}...")
        # Check if texts are the same (duplicate issue)
        if selected_metadata[i]['text'] == selected_metadata[j]['text']:
            print(f"  *** DUPLICATE TEXT DETECTED between indices {i} and {j}! ***")
        # Also check direct embedding comparison for these indices
        cos_sim_direct_old = torch.nn.functional.cosine_similarity(
            selected_embeddings[i:i+1], selected_embeddings[j:j+1]).item()
        cos_sim_direct_new = torch.nn.functional.cosine_similarity(
            new_embeddings[i:i+1], new_embeddings[j:j+1]).item()
        print(f"  Direct cosine (old): {cos_sim_direct_old:.4f}, Direct cosine (new): {cos_sim_direct_new:.4f}")

    print(f"\n--- Results ---")
    print(f"Max Difference: {max_diff:.6f}")
    print(f"Mean Difference: {mean_diff:.6f}")
    
    # Clean up
    if os.path.exists(temp_jsonl_path): os.remove(temp_jsonl_path)

    if max_diff < 2e-3:
        print("SUCCESS: Integrity Check V2 Passed.")
    else:
        print("FAILURE: Significant difference found.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--nproc", type=int, default=1, help="Number of GPUs/processes to use")
    args = parser.parse_args()
    
    run_randomized_integrity_test_v2(args.size, args.split, args.samples, nproc=args.nproc)
