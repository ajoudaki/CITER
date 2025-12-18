import json
import numpy as np
import torch
import os
import subprocess
import sys
import argparse
import uuid

def load_and_reconstruct_data(dataset_size, split, seed=42, train_ratio=0.8):
    """
    Reconstructs the exact list of statements corresponding to the pre-computed embeddings
    by mimicking AllStatementsDataset logic.
    """
    jsonl_path = f"data/lemmas_theorems/{dataset_size}.jsonl"
    print(f"Loading original data from {jsonl_path}...")
    
    all_papers_data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            paper = json.loads(line)
            statements = paper.get('lemmas', []) + paper.get('theorems', [])
            if len(statements) >= 1:
                all_papers_data.append(paper)
    
    # Reproducible Shuffle
    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_papers_data))
    rng.shuffle(indices)
    
    n_train = int(len(all_papers_data) * train_ratio)
    
    if split == 'train':
        split_indices = indices[:n_train]
    elif split == 'eval':
        split_indices = indices[n_train:]
    elif split == 'all':
        split_indices = indices
    else:
        raise ValueError(f"Unknown split: {split}")
    
    selected_papers = [all_papers_data[i] for i in split_indices]
    
    # Flatten exactly as AllStatementsDataset does
    all_statements = []
    for paper in selected_papers:
        for stmt in paper.get('lemmas', []):
            all_statements.append(stmt)
        for stmt in paper.get('theorems', []):
            all_statements.append(stmt)
            
    return all_statements

def run_integrity_test(dataset_size='small', split='eval', num_samples=50, model_path='outputs/demo/big_run_qwen-7b'):
    print(f"--- Randomized Integrity Test ---")
    print(f"Dataset: {dataset_size} | Split: {split} | Samples: {num_samples}")
    
    # 1. Load Pre-computed Embeddings
    emb_filename = f"{split}_embeddings.pt" if split != 'train' else "train_embeddings.pt"
    # Note: based on ls output, 'embeddings.pt' might be train or all?
    # Let's stick to standard naming if possible, or check existence.
    # The listing showed 'eval_embeddings.pt'. For train it might be different.
    # Let's assume standard pattern or specific check.
    
    emb_path = os.path.join(model_path, 'embeddings', dataset_size, emb_filename)
    if not os.path.exists(emb_path):
        # Fallback checks
        if split == 'eval' and os.path.exists(os.path.join(model_path, 'embeddings', dataset_size, 'eval_embeddings.pt')):
             emb_path = os.path.join(model_path, 'embeddings', dataset_size, 'eval_embeddings.pt')
        elif os.path.exists(os.path.join(model_path, 'embeddings', dataset_size, 'embeddings.pt')):
             emb_path = os.path.join(model_path, 'embeddings', dataset_size, 'embeddings.pt')
             print(f"Warning: Using 'embeddings.pt' as fallback.")
        else:
             print(f"Error: Could not find embedding file for {dataset_size}/{split}")
             sys.exit(1)
             
    print(f"Loading embeddings from {emb_path}...")
    full_embeddings = torch.load(emb_path).float()
    print(f"Full embeddings shape: {full_embeddings.shape}")
    
    # 2. Reconstruct Data
    all_statements = load_and_reconstruct_data(dataset_size, split)
    print(f"Reconstructed {len(all_statements)} statements.")
    
    if len(all_statements) != full_embeddings.shape[0]:
        print(f"CRITICAL WARNING: Metadata count ({len(all_statements)}) != Embedding count ({full_embeddings.shape[0]})")
        # If size differs, we can't trust indices.
        # However, if compute_embeddings trimmed padding, they should match exactly.
        # If they don't match, we might be looking at the wrong split or random seed.
        # Proceeding but truncating to min length to avoid crash, though test will likely fail.
        min_len = min(len(all_statements), full_embeddings.shape[0])
        all_statements = all_statements[:min_len]
        full_embeddings = full_embeddings[:min_len]
    
    # 3. Select Random Samples
    indices = np.random.choice(len(all_statements), num_samples, replace=False)
    indices.sort() # Keep original order for easier debug
    
    selected_texts = [all_statements[i] for i in indices]
    selected_embeddings = full_embeddings[indices]
    
    # 4. Create Temporary Dataset
    temp_id = str(uuid.uuid4())[:8]
    temp_dataset_name = f"temp_integrity_{temp_id}"
    temp_jsonl_path = f"data/lemmas_theorems/{temp_dataset_name}.jsonl"
    
    print(f"Creating temporary dataset: {temp_jsonl_path}")
    with open(temp_jsonl_path, 'w') as f:
        for text in selected_texts:
            # Wrap each in a dummy paper. 
            # We put it in 'lemmas' to ensure it's read.
            record = {"lemmas": [text], "theorems": []}
            f.write(json.dumps(record) + '\n')
            
    # 5. Run Compute Embeddings on Temp Dataset
    print("Running compute_embeddings on temporary dataset...")
    cmd = (
        f"python theorem_contrastive_training.py "
        f"model=qwen-2.5-math-7b "
        f"+training.compute_embeddings=true "
        f"dataset.size={temp_dataset_name} "
        f"+dataset.split=all "
        f"+training.load_model_path={model_path} "
        f"training.micro_batch_size=8 "
        f"training.quantization.enabled=true "
        f"wandb.enabled=false"
    )
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("Generation Command Failed!")
            print(result.stderr)
            sys.exit(1)
    finally:
        # Cleanup dataset immediately? No, keep for debug if fail.
        pass

    # 6. Load New Embeddings
    new_emb_path = os.path.join("outputs/embeddings", temp_dataset_name, "all_embeddings.pt")
    if not os.path.exists(new_emb_path): # Check default output location
         # Try model path
         new_emb_path = os.path.join(model_path, "embeddings", temp_dataset_name, "all_embeddings.pt")
    
    if not os.path.exists(new_emb_path):
        print(f"Error: Could not find generated embeddings at {new_emb_path}")
        # Clean up jsonl
        if os.path.exists(temp_jsonl_path): os.remove(temp_jsonl_path)
        sys.exit(1)
        
    new_embeddings = torch.load(new_emb_path).float()
    print(f"New embeddings shape: {new_embeddings.shape}")
    
    # 7. Compare Similarity Matrices
    # Normalize first (just in case)
    selected_embeddings = torch.nn.functional.normalize(selected_embeddings, p=2, dim=1)
    new_embeddings = torch.nn.functional.normalize(new_embeddings, p=2, dim=1)
    
    sim_old = torch.mm(selected_embeddings, selected_embeddings.t())
    sim_new = torch.mm(new_embeddings, new_embeddings.t())
    
    diff = (sim_old - sim_new).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n--- Results ---")
    print(f"Max Difference: {max_diff:.6f}")
    print(f"Mean Difference: {mean_diff:.6f}")
    
    # Cleanup
    if os.path.exists(temp_jsonl_path): os.remove(temp_jsonl_path)
    # Also remove the output directory for temp dataset if possible
    # shutil.rmtree(os.path.dirname(new_emb_path)) # Be careful with this
    
    if max_diff < 1e-3:
        print("SUCCESS: Integrity Check Passed.")
    else:
        print("FAILURE: Matrices differ significantly.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", help="Dataset size (tiny, small, etc)")
    parser.add_argument("--split", default="eval", help="Split (train, eval, all)")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    args = parser.parse_args()
    
    run_integrity_test(args.size, args.split, args.samples)
