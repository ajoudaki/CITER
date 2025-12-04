import os
import subprocess
import torch
import numpy as np
import sys

# Configuration
MODEL_PATH = "outputs/demo/big_run_qwen-7b"
DATASET_SIZE = "integrity_test"
QUERY_FILE = "temp_queries.txt"

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("COMMAND FAILED")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)
    else:
        print("Command successful.")

# 1. Compute Embeddings
cmd1 = (
    f"python theorem_contrastive_training.py "
    f"model=qwen-2.5-math-7b "
    f"+training.compute_embeddings=true "
    f"dataset.size={DATASET_SIZE} "
    f"+dataset.split=all "
    f"+training.load_model_path={MODEL_PATH} "
    f"training.max_length=512 "
    f"training.micro_batch_size=4 "
    f"training.quantization.enabled=true "
    f"wandb.enabled=false"
)

# 2. Compute Pairwise Similarities
cmd2 = (
    f"python theorem_contrastive_training.py "
    f"model=qwen-2.5-math-7b "
    f"+training.compute_pairwise_similarities=true "
    f"+training.query_file={QUERY_FILE} "
    f"+training.load_model_path={MODEL_PATH} "
    f"training.max_length=512 "
    f"training.micro_batch_size=4 "
    f"training.quantization.enabled=true "
    f"wandb.enabled=false"
)

print("--- Step 1: Compute Embeddings ---")
run_command(cmd1)

print("\n--- Step 2: Compute Pairwise Similarities ---")
run_command(cmd2)

print("\n--- Step 3: Verification ---")

# Load Embeddings
embed_path = f"outputs/embeddings/{DATASET_SIZE}/all_embeddings.pt"
if not os.path.exists(embed_path):
    print(f"Error: Embedding file not found at {embed_path}")
    sys.exit(1)

embeddings = torch.load(embed_path).float() # Load as float
print(f"Loaded embeddings shape: {embeddings.shape}")

# Compute Similarity Matrix Manually
# embeddings are already normalized in the model output usually, but let's check norms
norms = torch.norm(embeddings, dim=1)
print(f"Embedding norms (min/max): {norms.min().item():.4f}, {norms.max().item():.4f}")
# If normalized, mm is cosine similarity
manual_sim = torch.mm(embeddings, embeddings.t())

# Load Automatic Similarity Matrix
sim_matrix_path = "outputs/similarity_matrix.npy"
if not os.path.exists(sim_matrix_path):
    print(f"Error: Similarity matrix file not found at {sim_matrix_path}")
    sys.exit(1)

auto_sim = np.load(sim_matrix_path)
auto_sim_tensor = torch.from_numpy(auto_sim).float()
print(f"Loaded similarity matrix shape: {auto_sim_tensor.shape}")

# Comparison
if manual_sim.shape != auto_sim_tensor.shape:
    print(f"Shape mismatch! Manual: {manual_sim.shape}, Auto: {auto_sim_tensor.shape}")
    sys.exit(1)

# Check if they are close
# We use a slightly loose tolerance because of float16/bfloat16 accumulation differences
# manual_sim was computed from loaded float tensors (which might have been saved as half)
# auto_sim was computed inside the script (likely in float32 after casting)
diff = (manual_sim - auto_sim_tensor).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"Max difference: {max_diff:.6f}")
print(f"Mean difference: {mean_diff:.6f}")

if max_diff < 1e-3:
    print("SUCCESS: Matrices are effectively identical.")
else:
    print("FAILURE: Matrices differ significantly.")
    sys.exit(1)
