import json
import os
import subprocess
import torch
import sys
import shutil
import numpy as np

# Configuration
MODEL_PATH = "outputs/demo/big_run_qwen-7b"
DATASET_NAME = "distributed_test"
JSONL_PATH = f"data/lemmas_theorems/{DATASET_NAME}.jsonl"
NUM_STATEMENTS = 100

# 1. Create Dataset
print(f"--- Creating {DATASET_NAME} dataset with ~{NUM_STATEMENTS} statements ---")
# We'll just create dummy data for speed and consistency
lemmas = [f"Lemma {i}: This is a distributed test lemma." for i in range(NUM_STATEMENTS // 2)]
theorems = [f"Theorem {i}: This is a distributed test theorem." for i in range(NUM_STATEMENTS // 2)]

# Create a single paper containing all of them to avoid split complexity (train_ratio etc)
# actually, AllStatementsDataset shuffles papers. 
# To verify order strictly, let's create multiple papers.
papers = []
for i in range(10): # 10 papers
    p_lemmas = lemmas[i*5 : (i+1)*5] # 5 lemmas per paper
    p_theorems = theorems[i*5 : (i+1)*5] # 5 theorems per paper
    papers.append({
        "title": f"Paper {i}",
        "arxiv_id": f"dist.{i}",
        "lemmas": p_lemmas,
        "theorems": p_theorems
    })

os.makedirs("data/lemmas_theorems", exist_ok=True)
with open(JSONL_PATH, 'w') as f:
    for p in papers:
        f.write(json.dumps(p) + '\n')

print(f"Created {JSONL_PATH}")

# 2. Define Commands
# Common args
common_args = (
    f"model=qwen-2.5-math-7b "
    f"+training.compute_embeddings=true "
    f"dataset.size={DATASET_NAME} "
    f"+dataset.split=all "
    f"+training.load_model_path={MODEL_PATH} "
    f"training.max_length=128 " # Short length for speed
    f"training.micro_batch_size=8 "
    f"training.quantization.enabled=true "
    f"wandb.enabled=false "
    # Force seed to be identical
    f"dataset.seed=42 "
)

# Output directories (we need to override save_dir to separate them)
single_dir = "outputs/dist_test/single"
multi_dir = "outputs/dist_test/multi"

# Clean previous runs
if os.path.exists(single_dir): shutil.rmtree(single_dir)
if os.path.exists(multi_dir): shutil.rmtree(multi_dir)

cmd_single = (
    f"python theorem_contrastive_training.py "
    f"{common_args} "
    f"output.save_dir={single_dir}"
)

cmd_multi = (
    f"torchrun --nproc_per_node=2 theorem_contrastive_training.py "
    f"{common_args} "
    f"output.save_dir={multi_dir}"
)

# 3. Run Single GPU
print("\n--- Running Single GPU ---")
print(cmd_single)
res = subprocess.run(cmd_single, shell=True, capture_output=True, text=True)
if res.returncode != 0:
    print("SINGLE GPU FAILED")
    print(res.stderr)
    sys.exit(1)
else:
    print("Single GPU finished successfully.")

# 4. Run Multi GPU
print("\n--- Running Multi GPU (2 Processes) ---")
print(cmd_multi)
res = subprocess.run(cmd_multi, shell=True, capture_output=True, text=True)
if res.returncode != 0:
    print("MULTI GPU FAILED")
    print(res.stderr)
    # sys.exit(1) # Don't exit yet, let's see if it produced output
else:
    print("Multi GPU finished successfully.")

# 5. Compare
print("\n--- Comparing Results ---")

# Paths to embeddings
# Note: compute_embeddings saves to {save_dir}/embeddings/{dataset_size}/{split}_embeddings.pt
path_single = os.path.join(single_dir, "embeddings", DATASET_NAME, "all_embeddings.pt")
path_multi = os.path.join(multi_dir, "embeddings", DATASET_NAME, "all_embeddings.pt")

if not os.path.exists(path_single):
    print(f"Single GPU output not found at {path_single}")
    sys.exit(1)
if not os.path.exists(path_multi):
    print(f"Multi GPU output not found at {path_multi}")
    sys.exit(1)

emb_single = torch.load(path_single).float()
emb_multi = torch.load(path_multi).float()

print(f"Single Shape: {emb_single.shape}")
print(f"Multi Shape:  {emb_multi.shape}")

if emb_single.shape != emb_multi.shape:
    print("FAILURE: Shapes differ!")
    sys.exit(1)

# Check values
diff = (emb_single - emb_multi).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"Max Difference: {max_diff:.6f}")
print(f"Mean Difference: {mean_diff:.6f}")

if max_diff < 1e-3:
    print("SUCCESS: Embeddings are identical (within tolerance).")
else:
    # Detailed debugging
    # Find first mismatch
    mismatch_indices = (diff.max(dim=1).values > 1e-3).nonzero(as_tuple=True)[0]
    print(f"Found {len(mismatch_indices)} mismatching rows.")
    if len(mismatch_indices) > 0:
        first_idx = mismatch_indices[0].item()
        print(f"First mismatch at index {first_idx}:")
        print(f"  Single: {emb_single[first_idx, :5]}...")
        print(f"  Multi:  {emb_multi[first_idx, :5]}...")
        
    print("FAILURE: Embeddings differ significantly.")
    sys.exit(1)
