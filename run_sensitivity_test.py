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

def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("Command failed!")
        print(res.stderr)
        sys.exit(1)
    return res

def run_sensitivity_test(nproc=1, num_samples=50, model_path='outputs/demo/big_run_qwen-7b'):
    np.random.seed(42)
    print(f"--- Sensitivity / Robustness Test ---")
    print(f"Samples: {num_samples} | GPUs: {nproc}")
    
    # 1. Load Source Data (Tiny)
    source_path = "data/lemmas_theorems/tiny.jsonl"
    print(f"Loading samples from {source_path}...")
    all_stmts = []
    with open(source_path, 'r') as f:
        for line in f:
            p = json.loads(line)
            all_stmts.extend(p.get('lemmas', []))
            all_stmts.extend(p.get('theorems', []))
            
    # Sample
    if len(all_stmts) > num_samples:
        indices = np.random.choice(len(all_stmts), num_samples, replace=False)
        samples = [all_stmts[i] for i in indices]
    else:
        samples = all_stmts
        
    # 2. Prepare Datasets
    # A. Original
    # B. Appended (Add suffix)
    # C. Truncated (Remove last few words)
    
    suffix = " This is a minor addition."
    
    data_orig = []
    data_append = []
    data_trunc = []
    
    for s in samples:
        # Original
        data_orig.append(s)
        
        # Append
        data_append.append(s + suffix)
        
        # Truncate
        words = s.split()
        if len(words) > 5:
            truncated = " ".join(words[:-3]) # Remove last 3 words
        else:
            truncated = s # Too short to truncate
        data_trunc.append(truncated)
        
    datasets = {
        "sens_orig": data_orig,
        "sens_append": data_append,
        "sens_trunc": data_trunc
    }
    
    embeddings = {}
    cmd_prefix = get_cmd_prefix(nproc)
    
    # 3. Run Compute Loop
    for name, data_list in datasets.items():
        print(f"\n--- Processing {name} ---")
        # Save JSONL
        jsonl_path = f"data/lemmas_theorems/{name}.jsonl"
        with open(jsonl_path, 'w') as f:
            for text in data_list:
                # Wrap in lemma to ensure it's read
                rec = {"lemmas": [text], "theorems": []}
                f.write(json.dumps(rec) + '\n')
                
        # Compute
        cmd = (
            f"{cmd_prefix} theorem_contrastive_training.py "
            f"model=qwen-2.5-math-7b "
            f"+training.compute_embeddings=true "
            f"dataset.size={name} "
            f"+dataset.split=all "
            f"+training.load_model_path={model_path} "
            f"training.max_length=256 "
            f"training.micro_batch_size=8 "
            f"training.quantization.enabled=true "
            f"wandb.enabled=false "
            # Default save dir is outputs/
        )
        run_cmd(cmd)
        
        # Load Result
        # Result path: outputs/embeddings/{name}/all_embeddings.pt
        # Note: Metadata logic in previous test creates all_metadata.jsonl.
        # We need to use metadata to ensure alignment because of shuffling!
        
        emb_path = f"outputs/embeddings/{name}/all_embeddings.pt"
        meta_path = f"outputs/embeddings/{name}/all_metadata.jsonl"
        
        if not os.path.exists(emb_path):
            print(f"Output not found: {emb_path}")
            sys.exit(1)
            
        raw_embs = torch.load(emb_path).float()
        
        # Load metadata to Re-Align
        # Since we created the jsonl with 1 statement per line (wrapped in paper), 
        # and AllStatementsDataset shuffles papers, we MUST re-align.
        
        with open(meta_path, 'r') as f:
            meta = [json.loads(line) for line in f]
            
        # Map text -> emb
        # NOTE: In truncated/append versions, texts are unique (mostly).
        # But 'samples' list is our Ground Truth order.
        
        text_to_emb = {}
        for idx, item in enumerate(meta):
            text_to_emb[item['text']] = raw_embs[idx]
            
        # Re-align to match `data_list` order
        aligned_embs = []
        for text in data_list:
            if text not in text_to_emb:
                # Fallback for truncated potentially creating dups or issues?
                # Should be fine usually.
                print("Error: Text mismatch during alignment.")
                sys.exit(1)
            aligned_embs.append(text_to_emb[text])
            
        embeddings[name] = torch.stack(aligned_embs)
        
        # Cleanup jsonl
        if os.path.exists(jsonl_path): os.remove(jsonl_path)

    # 4. Compare
    print(f"\n--- Similarity Analysis ---")
    
    orig = torch.nn.functional.normalize(embeddings["sens_orig"], p=2, dim=1)
    append = torch.nn.functional.normalize(embeddings["sens_append"], p=2, dim=1)
    trunc = torch.nn.functional.normalize(embeddings["sens_trunc"], p=2, dim=1)
    
    # Compute pairwise cosine similarity (row vs row)
    # (N, D) * (N, D) -> sum(dim=1) -> (N,)
    
    sim_append = (orig * append).sum(dim=1)
    sim_trunc = (orig * trunc).sum(dim=1)
    
    print(f"\n1. Original vs Appended ('{suffix}')")
    print(f"   Mean Similarity: {sim_append.mean().item():.4f}")
    print(f"   Min Similarity:  {sim_append.min().item():.4f}")
    print(f"   Max Similarity:  {sim_append.max().item():.4f}")
    
    print(f"\n2. Original vs Truncated (Remove last 3 words)")
    print(f"   Mean Similarity: {sim_trunc.mean().item():.4f}")
    print(f"   Min Similarity:  {sim_trunc.min().item():.4f}")
    print(f"   Max Similarity:  {sim_trunc.max().item():.4f}")

    # Detailed Analysis of Failures
    sorted_indices = torch.argsort(sim_trunc) # Ascending
    
    print(f"\n--- Top 10 Most Sensitive Samples (Lowest Similarity after Truncation) ---")
    for i in range(min(10, num_samples)):
        idx = sorted_indices[i].item()
        score = sim_trunc[idx].item()
        original = datasets["sens_orig"][idx]
        truncated = datasets["sens_trunc"][idx]
        
        # Calculate what was removed
        # Simple diff for display
        words_orig = original.split()
        words_trunc = truncated.split()
        removed_words = words_orig[len(words_trunc):]
        removed_str = " ".join(removed_words)
        
        print(f"\n{i+1}. Similarity: {score:.4f}")
        print(f"   Original:  ...{original[-100:] if len(original)>100 else original}")
        print(f"   Truncated: ...{truncated[-100:] if len(truncated)>100 else truncated}")
        print(f"   Removed:   [{removed_str}]")

    # Threshold check
    # We expect high similarity. If it drops below e.g. 0.8, it's "sensitive".
    # But this is just a report for the user.
    
    if sim_append.mean().item() > 0.85 and sim_trunc.mean().item() > 0.85:
        print("\nCONCLUSION: Model is ROBUST to small perturbations (High Similarity).")
    else:
        print("\nCONCLUSION: Model is SENSITIVE to small perturbations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    
    run_sensitivity_test(nproc=args.nproc, num_samples=args.samples)
