import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from contextlib import nullcontext
import copy
import math # Imported for math.ceil

# ===================================================================
# Configuration
# ===================================================================

# Default configuration for training
CONFIG = {
    'D_INPUT': 512, 'D_EMBED': 128, 'GLOBAL_BATCH_SIZE': 2048, 'MICRO_BATCH_SIZE': 64,
    'STREAM_CHUNK_SIZE': 512, 'TAU': 0.07, 'LR': 1e-4, 'EPOCHS': 5,
    'DATASET_SIZE': 50000,
}

# Configuration used specifically for the equivalence test
TEST_CONFIG = {
    'D_INPUT': 64, 'D_EMBED': 32, 'GLOBAL_BATCH_SIZE': 128, 'MICRO_BATCH_SIZE': 16,
    'STREAM_CHUNK_SIZE': 32, 'TAU': 0.1, 'LR': 0.1,
}

# ===================================================================
# Model Definition (Modular)
# ===================================================================

class SwappableEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, simple=False):
        super().__init__()
        if simple:
            self.net = nn.Linear(input_dim, embed_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, embed_dim * 2), nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )

    def forward(self, x):
        x = self.net(x)
        # L2 normalization is crucial for the VJP calculation in Phase D.
        return F.normalize(x, p=2, dim=-1)

class ContrastiveModel(nn.Module):
    def __init__(self, encoder_x, encoder_y):
        super().__init__()
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
    def forward(self, x, y):
        return self.encoder_x(x), self.encoder_y(y)

# ===================================================================
# Distributed Utilities
# ===================================================================

def setup_DDP():
    if not torch.cuda.is_available():
        raise RuntimeError("This implementation requires CUDA.")
        
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

def all_gather_embeddings(local_embeddings):
    world_size = dist.get_world_size()
    if world_size == 1:
        return local_embeddings.detach()
        
    local_embeddings = local_embeddings.contiguous()
    gathered = [torch.zeros_like(local_embeddings) for _ in range(world_size)]
    dist.all_gather(gathered, local_embeddings)
    
    return torch.cat(gathered, dim=0).detach()

# ===================================================================
# Core Recipe Implementation Helpers (Phases B & C)
# ===================================================================

def compute_streaming_log_normalizers(Z_x, Z_y, tau, M):
    """Phase B: Computes row (a) and column (b) log-normalizers."""
    N = Z_x.shape[0]
    a = torch.empty(N, device=Z_x.device, dtype=Z_x.dtype)
    b = torch.empty(N, device=Z_x.device, dtype=Z_x.dtype)

    # 1. Row normalizers (a).
    Z_y_T = Z_y.T.contiguous()
    for start in range(0, N, M):
        end = min(start + M, N)
        S_block = torch.matmul(Z_x[start:end], Z_y_T) / tau
        a[start:end] = torch.logsumexp(S_block, dim=1)

    # 2. Column normalizers (b).
    Z_x_T = Z_x.T.contiguous()
    for start in range(0, N, M):
        end = min(start + M, N)
        # S.T_C,: = (Z_y[C] @ Z_x.T) / tau
        ST_block = torch.matmul(Z_y[start:end], Z_x_T) / tau
        b[start:end] = torch.logsumexp(ST_block, dim=1)

    return a.detach(), b.detach()

def compute_streaming_gradients(Z_x, Z_y, Z_x_M, Z_y_M, a, b, a_M, b_M, tau, N, M_stream):
    """
    Phase C: Computes embedding gradients G_x_M and G_y_M.
    Implements the MATHEMATICALLY CORRECT gradients for symmetric CLIP loss.
    
    G_x = 1/(2Nτ) * [ (P @ Z_y) + (Q @ Z_y) - 2*Z_y ]  <-- FIX 1: Q@Zy instead of Q.T@Zy
    G_y = 1/(2Nτ) * [ (P.T @ Z_x) + (Q.T @ Z_x) - 2*Z_x ]
    """
    B, D = Z_x_M.shape
    device = Z_x_M.device
    scale = 1.0 / (2 * N * tau)

    # Initialize accumulators (all BxD)
    # G_x components
    P_M_Zy = torch.zeros(B, D, device=device, dtype=Z_x.dtype)
    Q_M_Zy = torch.zeros(B, D, device=device, dtype=Z_x.dtype) # Accumulator for (Q @ Zy)_M
    # G_y components
    PT_M_Zx = torch.zeros(B, D, device=device, dtype=Z_x.dtype)
    QT_M_Zx = torch.zeros(B, D, device=device, dtype=Z_x.dtype)

    # Stream over the global dimension N (Chunks R/C)
    for start in range(0, N, M_stream):
        end = min(start + M_stream, N)
        
        Z_x_Chunk = Z_x[start:end]
        Z_y_Chunk = Z_y[start:end]
        a_Chunk = a[start:end]
        b_Chunk = b[start:end] # Needed for Q[M,:]

        # === 1. Calculations involving S_M,C (B x M_stream) ===
        # Used for G_x terms: P[M,:]@Zy and Q[M,:]@Zy
        
        # S_M,C = (Z_x_M @ Z_y[C].T) / tau.
        S_MC = torch.matmul(Z_x_M, Z_y_Chunk.T) / tau
        
        # Term 1: P[M,:]@Zy. Needs row normalizer a[M] (a_M).
        P_MC = torch.exp(S_MC - a_M.unsqueeze(1))
        P_M_Zy += torch.matmul(P_MC, Z_y_Chunk)
        
        # Term 2: Q[M,:]@Zy. Needs column normalizer b[C] (b_Chunk). (FIX 1 APPLIED)
        # Q_MC = exp(S_MC - b_Chunk). Broadcast b_Chunk as (1, M_stream).
        Q_MC = torch.exp(S_MC - b_Chunk.unsqueeze(0))
        Q_M_Zy += torch.matmul(Q_MC, Z_y_Chunk)

        # === 2. Calculations involving S_R,M (M_stream x B) ===
        # Used for G_y terms: P[:,M].T@Zx and Q[:,M].T@Zx
        
        # S_R,M = (Z_x[R] @ Z_y_M.T) / tau.
        S_RM = torch.matmul(Z_x_Chunk, Z_y_M.T) / tau

        # Term 3: P[:,M].T@Zx. Needs row normalizer a[R] (a_Chunk).
        P_RM = torch.exp(S_RM - a_Chunk.unsqueeze(1))
        PT_M_Zx += torch.matmul(P_RM.T, Z_x_Chunk)

        # Term 4: Q[:,M].T@Zx. Needs column normalizer b[M] (b_M).
        Q_RM = torch.exp(S_RM - b_M.unsqueeze(0))
        QT_M_Zx += torch.matmul(Q_RM.T, Z_x_Chunk)

    # Finalize Gradients
    G_x_M = (P_M_Zy + Q_M_Zy - 2 * Z_y_M) * scale
    G_y_M = (PT_M_Zx + QT_M_Zx - 2 * Z_x_M) * scale
    
    return G_x_M.detach(), G_y_M.detach()

# ===================================================================
# Distributed Training Step (Phases A-E) (UPDATED)
# ===================================================================

def distributed_train_step(model, optimizer, local_x, local_y, config, rank, world_size):
    """Implements the 5-phase distributed training step with Loss Scaling."""
    
    N = config['GLOBAL_BATCH_SIZE']
    B = config['MICRO_BATCH_SIZE']
    M_stream = config['STREAM_CHUNK_SIZE']
    TAU = config['TAU']
    C = local_x.shape[0] # Local chunk size (N/P)
    P = world_size       # World size (P)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # === Phase A: Forward embeddings (no grad, microbatched) ===
    module = model.module if hasattr(model, 'module') else model
    local_Z_x, local_Z_y = [], []

    with torch.no_grad():
        for i in range(0, C, B):
            z_x = module.encoder_x(local_x[i:i+B])
            z_y = module.encoder_y(local_y[i:i+B])
            local_Z_x.append(z_x)
            local_Z_y.append(z_y)

    local_Z_x = torch.cat(local_Z_x, dim=0)
    local_Z_y = torch.cat(local_Z_y, dim=0)

    # Sync point 1: All-gather embeddings
    Z_x = all_gather_embeddings(local_Z_x)
    Z_y = all_gather_embeddings(local_Z_y)

    # === Phase B: Global normalizers (no grad, streamed) ===
    with torch.no_grad():
        a, b = compute_streaming_log_normalizers(Z_x, Z_y, TAU, M_stream)

    # === Phase C & D: Local gradient construction and VJP (microbatched) ===
    start_idx_global = rank * C
    
    # Calculate the number of microbatches for DDP synchronization handling
    num_microbatches = math.ceil(C / B)

    for i_mb in range(num_microbatches):
        i = i_mb * B
        mb_start, mb_end = i, min(i + B, C)
        global_start = start_idx_global + mb_start
        global_end = start_idx_global + mb_end

        # DDP Synchronization Control: Only sync on the last microbatch (Gradient Accumulation)
        is_last_microbatch = (i_mb == num_microbatches - 1)
        # Use no_sync() if it's not the last microbatch and we are in a distributed setting
        context = model.no_sync() if (not is_last_microbatch and P > 1 and hasattr(model, 'no_sync')) else nullcontext()

        with context:
            # --- Phase C: Local gradient construction (Fully Streamed) ---
            with torch.no_grad():
                # Uses the mathematically corrected implementation (FIX 1)
                G_x_M, G_y_M = compute_streaming_gradients(
                    Z_x, Z_y, 
                    Z_x[global_start:global_end], Z_y[global_start:global_end], 
                    a, b, 
                    a[global_start:global_end], b[global_start:global_end], 
                    TAU, N, M_stream
                )

            # --- Phase D: Recompute with grad + apply VJP ---
            # 5. Recompute (must call DDP-wrapped model)
            mb_x = local_x[mb_start:mb_end]
            mb_y = local_y[mb_start:mb_end]
            Z_x_M_grad, Z_y_M_grad = model(mb_x, mb_y)

            # 6. Surrogate micro-loss (VJP)
            surrogate_loss = (Z_x_M_grad * G_x_M).sum() + (Z_y_M_grad * G_y_M).sum()

            # FIX 2: Loss Scaling for correct DDP aggregation.
            # Since L_M includes 1/N, and DDP applies 1/P (AVG), we scale by P 
            # so the final gradient scaling is (1/N) * P * (1/P) = 1/N (SUM).
            scaled_surrogate_loss = surrogate_loss * P

            # 7. Backward pass
            # Sync Point 2 happens here automatically if is_last_microbatch=True.
            scaled_surrogate_loss.backward()

    # === Phase E: Optimizer step ===
    # 8. Optimizer step
    optimizer.step()

# ===================================================================
# Trivial (Ground Truth) Implementation
# ===================================================================

def trivial_contrastive_step(model, optimizer, global_x, global_y, config):
    """Implements the standard symmetric CLIP loss."""
    N = global_x.shape[0]
    TAU = config['TAU']

    model.train()
    optimizer.zero_grad(set_to_none=True)

    Z_x, Z_y = model(global_x, global_y)
    S = torch.matmul(Z_x, Z_y.T) / TAU
    labels = torch.arange(N, device=Z_x.device)
    
    # Row-wise CE (Image-to-Text)
    loss_x = F.cross_entropy(S, labels, reduction='sum')
    # Column-wise CE (Text-to-Image), implemented as Row-wise CE on S.T
    loss_y = F.cross_entropy(S.T, labels, reduction='sum')

    # Total loss L = 1/(2N) * (L_x + L_y)
    total_loss = (loss_x + loss_y) / (2 * N)

    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

# ===================================================================
# Equivalence Test Harness (DDP based)
# ===================================================================

# (Test utilities remain the same, included for completeness)

def init_model_deterministic(config, device, use_simple_encoder, use_sgd=False, dtype=torch.float32):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    encoder_x = SwappableEncoder(config['D_INPUT'], config['D_EMBED'], simple=use_simple_encoder)
    encoder_y = SwappableEncoder(config['D_INPUT'], config['D_EMBED'], simple=use_simple_encoder)
    model = ContrastiveModel(encoder_x, encoder_y).to(device=device, dtype=dtype)
    
    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['LR'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'])
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

def test_equivalence(rank, local_rank, world_size, device):
    config = TEST_CONFIG
    N = config['GLOBAL_BATCH_SIZE']
    
    use_simple_encoder = True
    use_sgd = True
    test_dtype = torch.float32
    tolerance = 1e-5 # Tolerance for float32 comparison
    
    if rank == 0:
        print(f"\n--- Running DDP Equivalence Test (N={N}, P={world_size}, dtype={test_dtype}) ---")
    
    if N % world_size != 0:
        if rank == 0: print(f"FAILURE: N must be divisible by P."); return

    # 1. Initialize Base Model (Identical across ranks)
    model_base, _ = init_model_deterministic(config, device, use_simple_encoder, use_sgd=use_sgd, dtype=test_dtype)
    
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
    if rank == 0:
        print("Running Trivial (Ground Truth) Step on Rank 0...")
        trivial_contrastive_step(model_trivial, optimizer_trivial, global_X, global_Y, config)

    dist.barrier()

    # 6. Run Distributed Step - All Ranks
    if rank == 0: print("Running Distributed Step across all ranks...")
    
    C = N // world_size
    start_idx = rank * C
    local_X = global_X[start_idx:start_idx+C]
    local_Y = global_Y[start_idx:start_idx+C]

    distributed_train_step(model_dist, optimizer_dist, local_X, local_Y, config, rank, world_size)

    dist.barrier()
    
    # 7. Verification (Rank 0 only)
    if rank == 0:
        print("Comparing Results on Rank 0...")

        match_w, diff_w, match_g, diff_g = compare_models_and_grads(model_trivial, model_dist, tolerance)

        print(f"Gradient Match:     {'PASS' if match_g else 'FAIL'} (Max Diff: {diff_g:.4e}, Tol: {tolerance:.1e})")
        print(f"Final Weight Match: {'PASS' if match_w else 'FAIL'} (Max Diff: {diff_w:.4e}, Tol: {tolerance:.1e})")

        if match_g and match_w:
            print("--- Test PASSED: Implementations are mathematically equivalent. ---")
        else:
            print("--- Test FAILED: Implementations are NOT equivalent. ---")

# ===================================================================
# Main Training Loop (Boilerplate)
# ===================================================================

class SyntheticPairedDataset(Dataset):
    def __init__(self, num_samples, input_dim):
        self.num_samples = num_samples
        self.input_dim = input_dim
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        x = torch.randn(self.input_dim)
        y = x * 0.9 + torch.randn(self.input_dim) * 0.1
        return x, y

def run_training(rank, local_rank, world_size, device):
    config = CONFIG
    N, B = config['GLOBAL_BATCH_SIZE'], config['MICRO_BATCH_SIZE']
    
    if N % world_size != 0:
        if rank == 0: print("Error: N must be divisible by P."); return
        
    C = N // world_size
    if C < B:
        if rank == 0: print(f"Warning: C ({C}) < B ({B}). Adjusting B=C.")
        config['MICRO_BATCH_SIZE'] = C; B = C

    if rank == 0:
        print(f"Starting Distributed Training...")
        print(f"World Config: N={N}, P={world_size}, C={C}, B={B}, M={config['STREAM_CHUNK_SIZE']}")

    # 1. Model Setup
    model, optimizer = init_model_deterministic(config, device, use_simple_encoder=False, use_sgd=False, dtype=torch.float32)
    model = DDP(model, device_ids=[local_rank])

    # 2. Data Setup
    dataset = SyntheticPairedDataset(config['DATASET_SIZE'], config['D_INPUT'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=C, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

    # 3. Training Loop
    for epoch in range(config['EPOCHS']):
        sampler.set_epoch(epoch)
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            distributed_train_step(model, optimizer, batch_x, batch_y, config, rank, world_size)

            if rank == 0 and batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{config['EPOCHS']}, Step {batch_idx}")

# ===================================================================
# Entry Point
# ===================================================================

if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=N script_name.py [train|test]
    
    try:
        rank, local_rank, world_size, device = setup_DDP()
    except RuntimeError as e:
        print(f"DDP setup failed: {e}")
        sys.exit(1)

    mode = 'train'
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            mode = 'test'
        elif sys.argv[1] != 'train':
            if rank == 0:
                print("Invalid argument. Usage: torchrun ... script_name.py [train|test]")
            dist.destroy_process_group()
            sys.exit(1)

    if mode == 'test':
        test_equivalence(rank, local_rank, world_size, device)
    else:
        run_training(rank, local_rank, world_size, device)

    dist.destroy_process_group()