# test_equivalence.py
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import copy

# Import the module to be tested
from distributed_clip import distributed_train_step, trivial_contrastive_step

# ===================================================================
# Test Configuration
# ===================================================================

TEST_CONFIG = {
    'D_INPUT': 64,
    'D_EMBED': 32,
    'GLOBAL_BATCH_SIZE': 128, # N
    'MICRO_BATCH_SIZE': 16,   # B
    'STREAM_CHUNK_SIZE': 32,  # M
    'TAU': 0.1,               # τ
    'LR': 0.1,                
}

# ===================================================================
# Toy Model Definitions (For Testing)
# ===================================================================

class ToyEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)

class ToyContrastiveModel(nn.Module):
    def __init__(self, encoder_x, encoder_y):
        super().__init__()
        # Adheres to the required interface
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        
    def forward(self, x, y):
        return self.encoder_x(x), self.encoder_y(y)

# ===================================================================
# DDP Setup and Test Utilities
# ===================================================================

def setup_DDP():
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA.")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if "LOCAL_RANK" not in os.environ:
         raise RuntimeError("LOCAL_RANK not set. Please run using torchrun.")
         
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device

def init_model_deterministic(config, device, dtype=torch.float32):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    encoder_x = ToyEncoder(config['D_INPUT'], config['D_EMBED'])
    encoder_y = ToyEncoder(config['D_INPUT'], config['D_EMBED'])
    model = ToyContrastiveModel(encoder_x, encoder_y).to(device=device, dtype=dtype)
    
    # Use SGD for verification stability
    optimizer = torch.optim.SGD(model.parameters(), lr=config['LR'])
    return model, optimizer

def generate_fixed_batch(config, device, dtype=torch.float32):
    torch.manual_seed(1337)
    N = config['GLOBAL_BATCH_SIZE']
    D_in = config['D_INPUT']
    X = torch.randn(N, D_in, device=device, dtype=dtype)
    Y = X * 0.9 + torch.randn(N, D_in, device=device, dtype=dtype) * 0.1
    return X, Y

def compare_tensors(t1, t2, tolerance=1e-6):
    if t1.shape != t2.shape:
        return False, float('inf')
    diff = (t1.detach() - t2.detach()).abs().max().item()
    return diff <= tolerance, diff

def compare_models_and_grads(model1, model2, tolerance=1e-6):
    match_weights, max_diff_weights = True, 0.0
    match_grads, max_diff_grads = True, 0.0

    m1 = model1.module if hasattr(model1, 'module') else model1
    m2 = model2.module if hasattr(model2, 'module') else model2

    for (name1, p1), (name2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        is_match_w, diff_w = compare_tensors(p1.data, p2.data, tolerance)
        max_diff_weights = max(max_diff_weights, diff_w)
        if not is_match_w: match_weights = False
            
        if (p1.grad is None) != (p2.grad is None):
            match_grads = False
            max_diff_grads = float('inf')
        elif p1.grad is not None:
            is_match_g, diff_g = compare_tensors(p1.grad, p2.grad, tolerance)
            max_diff_grads = max(max_diff_grads, diff_g)
            if not is_match_g: match_grads = False

    return match_weights, max_diff_weights, match_grads, max_diff_grads

# ===================================================================
# Main Test Logic
# ===================================================================

def test_equivalence(rank, local_rank, world_size, device):
    config = TEST_CONFIG
    N = config['GLOBAL_BATCH_SIZE']
    
    test_dtype = torch.float32
    tolerance = 1e-5 
    
    if rank == 0:
        print(f"\n--- Running DDP Equivalence Test ---")
        print(f"Config: N={N}, P={world_size}, B={config['MICRO_BATCH_SIZE']}, M={config['STREAM_CHUNK_SIZE']}")

    if N % world_size != 0:
        if rank == 0: print(f"FAILURE: N ({N}) must be divisible by P ({world_size})."); return

    # 1. Initialize Base Model
    model_base, _ = init_model_deterministic(config, device, dtype=test_dtype)
    
    # 2. Prepare Trivial Run (Ground Truth) - Rank 0 only
    if rank == 0:
        model_trivial = copy.deepcopy(model_base)
        optimizer_trivial = torch.optim.SGD(model_trivial.parameters(), lr=config['LR'])

    # 3. Prepare Distributed Run
    model_dist = copy.deepcopy(model_base)
    optimizer_dist = torch.optim.SGD(model_dist.parameters(), lr=config['LR'])
    model_dist = DDP(model_dist, device_ids=[local_rank])

    # 4. Generate Fixed Data
    global_X, global_Y = generate_fixed_batch(config, device, dtype=test_dtype)

    # 5. Run Trivial Step (Ground Truth) - Rank 0 ONLY
    loss_trivial = 0.0
    if rank == 0:
        print("\nRunning Trivial (Ground Truth) Step on Rank 0...")
        loss_trivial = trivial_contrastive_step(model_trivial, optimizer_trivial, global_X, global_Y, config)

    dist.barrier()

    # 6. Run Distributed Step - All Ranks
    if rank == 0: print("Running Distributed Step across all ranks...")
    
    C = N // world_size
    start_idx = rank * C
    local_X = global_X[start_idx:start_idx+C]
    local_Y = global_Y[start_idx:start_idx+C]

    # Execute the distributed step
    loss_dist = distributed_train_step(model_dist, optimizer_dist, local_X, local_Y, config)

    dist.barrier()
    
    # 7. Verification (Rank 0 only)
    if rank == 0:
        print("\nComparing Results on Rank 0...")

        # Compare Models and Gradients
        match_w, diff_w, match_g, diff_g = compare_models_and_grads(model_trivial, model_dist, tolerance)

        # Compare Loss
        diff_loss = abs(loss_trivial - loss_dist)
        match_loss = diff_loss <= tolerance

        print(f"Loss Match:         {'PASS' if match_loss else 'FAIL'} (Trivial: {loss_trivial:.6f}, Dist: {loss_dist:.6f}, Diff: {diff_loss:.4e})")
        print(f"Gradient Match:     {'PASS' if match_g else 'FAIL'} (Max Diff: {diff_g:.4e})")
        print(f"Final Weight Match: {'PASS' if match_w else 'FAIL'} (Max Diff: {diff_w:.4e})")

        if match_loss and match_g and match_w:
            print("\n--- Test PASSED: Implementations are mathematically equivalent. ---")
        else:
            print("\n--- Test FAILED: Implementations are NOT equivalent. ---")

if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=N test_equivalence.py
    try:
        rank, local_rank, world_size, device = setup_DDP()
    except RuntimeError as e:
        print(f"DDP setup failed: {e}")
        sys.exit(1)

    test_equivalence(rank, local_rank, world_size, device)
    dist.destroy_process_group()
