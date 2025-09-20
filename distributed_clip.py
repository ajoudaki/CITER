# distributed_clip.py
import torch
import torch.nn.functional as F
import torch.distributed as dist
import math
from contextlib import nullcontext
from typing import Dict, Tuple

# ===================================================================
# Distributed Utilities
# ===================================================================

def all_gather_embeddings(local_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Gathers embeddings from all ranks (Sync Point 1).
    Ensures the result is detached to prevent cross-rank autograd edges.
    """
    if not dist.is_initialized():
        # Fallback for non-distributed execution
        return local_embeddings.detach()
        
    world_size = dist.get_world_size()
    if world_size == 1:
        return local_embeddings.detach()
        
    local_embeddings = local_embeddings.contiguous()
    gathered = [torch.zeros_like(local_embeddings) for _ in range(world_size)]
    
    # Perform the all_gather operation
    dist.all_gather(gathered, local_embeddings)
    
    return torch.cat(gathered, dim=0).detach()

# ===================================================================
# Core Recipe Implementation Helpers (Phases B & C, Loss)
# ===================================================================

def compute_streaming_log_normalizers(Z_x: torch.Tensor, Z_y: torch.Tensor, tau: float, M: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phase B: Computes row (a) and column (b) log-normalizers memory-efficiently.
    """
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
        ST_block = torch.matmul(Z_y[start:end], Z_x_T) / tau
        b[start:end] = torch.logsumexp(ST_block, dim=1)

    return a.detach(), b.detach()

def calculate_global_loss(Z_x: torch.Tensor, Z_y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, tau: float, N: int) -> float:
    """
    Calculates the global symmetric contrastive loss using the log-normalizers.
    L = 1/(2N) * Sum_i (a_i + b_i - 2*S_ii)
    """
    # Calculate S_ii (diagonal of the similarity matrix)
    # S_ii = (Z_x[i] @ Z_y[i]) / tau
    S_diag = (Z_x * Z_y).sum(dim=1) / tau
    
    # Calculate the total loss
    # We use .item() to return a Python float, ensuring it's detached.
    loss = (a.sum() + b.sum() - 2 * S_diag.sum()) / (2 * N)
    return loss.item()

def compute_streaming_gradients(
    Z_x: torch.Tensor, Z_y: torch.Tensor, 
    Z_x_M: torch.Tensor, Z_y_M: torch.Tensor, 
    a: torch.Tensor, b: torch.Tensor, 
    a_M: torch.Tensor, b_M: torch.Tensor, 
    tau: float, N: int, M_stream: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phase C: Computes embedding gradients G_x_M and G_y_M using a fully streamed approach.
    """
    B, D = Z_x_M.shape
    device = Z_x_M.device
    scale = 1.0 / (2 * N * tau)

    # Initialize accumulators
    P_M_Zy = torch.zeros(B, D, device=device, dtype=Z_x.dtype)
    Q_M_Zy = torch.zeros(B, D, device=device, dtype=Z_x.dtype)
    PT_M_Zx = torch.zeros(B, D, device=device, dtype=Z_x.dtype)
    QT_M_Zx = torch.zeros(B, D, device=device, dtype=Z_x.dtype)

    # Stream over the global dimension N
    for start in range(0, N, M_stream):
        end = min(start + M_stream, N)
        
        Z_x_Chunk = Z_x[start:end]
        Z_y_Chunk = Z_y[start:end]
        a_Chunk = a[start:end]
        b_Chunk = b[start:end]

        # === 1. Calculations involving S_M,C (B x M_stream) ===
        S_MC = torch.matmul(Z_x_M, Z_y_Chunk.T) / tau
        
        # Term 1: P[M,:]@Zy.
        P_MC = torch.exp(S_MC - a_M.unsqueeze(1))
        P_M_Zy += torch.matmul(P_MC, Z_y_Chunk)
        
        # Term 2: Q[M,:]@Zy.
        Q_MC = torch.exp(S_MC - b_Chunk.unsqueeze(0))
        Q_M_Zy += torch.matmul(Q_MC, Z_y_Chunk)

        # === 2. Calculations involving S_R,M (M_stream x B) ===
        S_RM = torch.matmul(Z_x_Chunk, Z_y_M.T) / tau

        # Term 3: P[:,M].T@Zx.
        P_RM = torch.exp(S_RM - a_Chunk.unsqueeze(1))
        PT_M_Zx += torch.matmul(P_RM.T, Z_x_Chunk)

        # Term 4: Q[:,M].T@Zx.
        Q_RM = torch.exp(S_RM - b_M.unsqueeze(0))
        QT_M_Zx += torch.matmul(Q_RM.T, Z_x_Chunk)

    # Finalize Gradients
    G_x_M = (P_M_Zy + Q_M_Zy - 2 * Z_y_M) * scale
    G_y_M = (PT_M_Zx + QT_M_Zx - 2 * Z_x_M) * scale
    
    return G_x_M.detach(), G_y_M.detach()

# ===================================================================
# Public Interface: Distributed Training Step (Phases A-E)
# ===================================================================

def distributed_train_step(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    local_x: torch.Tensor, 
    local_y: torch.Tensor, 
    config: Dict
) -> float:
    """
    Implements the 5-phase distributed, memory-efficient training step.
    
    Assumes DDP is initialized if running in a distributed setting.

    Returns:
        float: The global batch loss value.
    """
    
    # Determine distributed context
    if dist.is_initialized():
        P = dist.get_world_size()
        rank = dist.get_rank()
    else:
        P = 1
        rank = 0

    N = config['GLOBAL_BATCH_SIZE']
    B = config['MICRO_BATCH_SIZE']
    M_stream = config['STREAM_CHUNK_SIZE']
    TAU = config['TAU']
    C = local_x.shape[0] # Local chunk size
    
    if N != C * P:
         raise ValueError(f"Global batch size N={N} must equal C*P ({C}*{P}).")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # === Phase A: Forward embeddings (no grad, microbatched) ===
    module = model.module if hasattr(model, 'module') else model
    
    # Interface check for optimization
    if not hasattr(module, 'encoder_x') or not hasattr(module, 'encoder_y'):
        raise AttributeError("Model (or model.module) must expose 'encoder_x' and 'encoder_y' for optimized Phase A.")

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

    # === Phase B: Global normalizers and Loss Calculation (no grad, streamed) ===
    with torch.no_grad():
        a, b = compute_streaming_log_normalizers(Z_x, Z_y, TAU, M_stream)
        # Calculate the global loss (redundantly on all GPUs, no communication needed)
        global_loss = calculate_global_loss(Z_x, Z_y, a, b, TAU, N)


    # === Phase C & D: Local gradient construction and VJP (microbatched) ===
    start_idx_global = rank * C
    num_microbatches = math.ceil(C / B)

    for i_mb in range(num_microbatches):
        i = i_mb * B
        mb_start, mb_end = i, min(i + B, C)
        global_start = start_idx_global + mb_start
        global_end = start_idx_global + mb_end

        # DDP Synchronization Control
        is_last_microbatch = (i_mb == num_microbatches - 1)
        context = model.no_sync() if (not is_last_microbatch and P > 1 and hasattr(model, 'no_sync')) else nullcontext()

        with context:
            # --- Phase C: Local gradient construction (Fully Streamed) ---
            with torch.no_grad():
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

            # Loss Scaling for correct DDP aggregation (SUM instead of AVG).
            scaled_surrogate_loss = surrogate_loss * P

            # 7. Backward pass
            # Sync Point 2 happens here automatically if is_last_microbatch=True.
            scaled_surrogate_loss.backward()

    # === Phase E: Optimizer step ===
    optimizer.step()
    
    return global_loss

# ===================================================================
# Public Interface: Trivial (Ground Truth) Implementation
# ===================================================================

def trivial_contrastive_step(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    global_x: torch.Tensor, 
    global_y: torch.Tensor, 
    config: Dict
) -> float:
    """
    A standard, mathematically equivalent implementation of symmetric CLIP loss.
    """
    N = global_x.shape[0]
    TAU = config['TAU']

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Forward pass
    Z_x, Z_y = model(global_x, global_y)

    # Compute Similarity Matrix (N x N)
    S = torch.matmul(Z_x, Z_y.T) / TAU

    # Compute Symmetric Loss
    labels = torch.arange(N, device=Z_x.device)
    
    loss_x = F.cross_entropy(S, labels, reduction='sum')
    loss_y = F.cross_entropy(S.T, labels, reduction='sum')

    # Total loss L = 1/(2N) * (L_x + L_y)
    total_loss = (loss_x + loss_y) / (2 * N)

    # Backward pass
    total_loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    return total_loss.item()