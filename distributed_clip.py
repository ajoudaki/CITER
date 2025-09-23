# distributed_clip.py
import torch
import torch.nn.functional as F
import torch.distributed as dist
import math
from contextlib import nullcontext
from typing import Dict, Tuple, Optional
from torch.cuda.amp import autocast
from tqdm import tqdm

# ===================================================================
# Distributed Utilities
# ===================================================================

def all_gather_embeddings(local_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Gathers embeddings from all ranks (Sync Point 1).
    Ensures the result is detached to prevent cross-rank autograd edges.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local_embeddings.detach()
        
    world_size = dist.get_world_size()
    local_embeddings = local_embeddings.contiguous()
    gathered = [torch.zeros_like(local_embeddings) for _ in range(world_size)]
    
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

    Z_y_T = Z_y.T.contiguous()
    for start in range(0, N, M):
        end = min(start + M, N)
        S_block = torch.matmul(Z_x[start:end], Z_y_T) / tau
        a[start:end] = torch.logsumexp(S_block, dim=1)

    Z_x_T = Z_x.T.contiguous()
    for start in range(0, N, M):
        end = min(start + M, N)
        ST_block = torch.matmul(Z_y[start:end], Z_x_T) / tau
        b[start:end] = torch.logsumexp(ST_block, dim=1)

    return a.detach(), b.detach()

def calculate_global_loss(Z_x: torch.Tensor, Z_y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, tau: float, N: int) -> float:
    """
    Calculates the global symmetric contrastive loss using the log-normalizers.
    """
    S_diag = (Z_x * Z_y).sum(dim=1) / tau
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
    P_M_Zy = torch.zeros(B, D, device=device, dtype=Z_x.dtype)
    Q_M_Zy = torch.zeros(B, D, device=device, dtype=Z_x.dtype)
    PT_M_Zx = torch.zeros(B, D, device=device, dtype=Z_x.dtype)
    QT_M_Zx = torch.zeros(B, D, device=device, dtype=Z_x.dtype)

    for start in range(0, N, M_stream):
        end = min(start + M_stream, N)
        Z_x_Chunk, Z_y_Chunk = Z_x[start:end], Z_y[start:end]
        a_Chunk, b_Chunk = a[start:end], b[start:end]

        S_MC = torch.matmul(Z_x_M, Z_y_Chunk.T) / tau
        P_MC = torch.exp(S_MC - a_M.unsqueeze(1))
        P_M_Zy += torch.matmul(P_MC, Z_y_Chunk)
        Q_MC = torch.exp(S_MC - b_Chunk.unsqueeze(0))
        Q_M_Zy += torch.matmul(Q_MC, Z_y_Chunk)

        S_RM = torch.matmul(Z_x_Chunk, Z_y_M.T) / tau
        P_RM = torch.exp(S_RM - a_Chunk.unsqueeze(1))
        PT_M_Zx += torch.matmul(P_RM.T, Z_x_Chunk)
        Q_RM = torch.exp(S_RM - b_M.unsqueeze(0))
        QT_M_Zx += torch.matmul(Q_RM.T, Z_x_Chunk)

    G_x_M = (P_M_Zy + Q_M_Zy - 2 * Z_y_M) * scale
    G_y_M = (PT_M_Zx + QT_M_Zx - 2 * Z_x_M) * scale
    
    return G_x_M.detach(), G_y_M.detach()

# ===================================================================
# Public Interfaces
# ===================================================================

def distributed_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    config: Dict,
    scaler: Optional[object] = None
) -> float:
    """
    Implements the 5-phase distributed, memory-efficient training step.
    Supports automatic mixed precision when scaler is provided.
    """
    if dist.is_initialized():
        P, rank = dist.get_world_size(), dist.get_rank()
    else:
        P, rank = 1, 0

    N, B, M_stream, TAU = config['GLOBAL_BATCH_SIZE'], config['MICRO_BATCH_SIZE'], config['STREAM_CHUNK_SIZE'], config['TAU']
    C = local_x.shape[0]
    
    if N != C * P:
         raise ValueError(f"Global batch size N={N} must equal C*P ({C}*{P}).")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    module = model.module if hasattr(model, 'module') else model
    if not hasattr(module, 'encoder_x') or not hasattr(module, 'encoder_y'):
        raise AttributeError("Model must expose 'encoder_x' and 'encoder_y'.")

    # Use autocast context if scaler is provided, otherwise no-op
    amp_context = autocast() if scaler is not None else nullcontext()

    local_Z_x, local_Z_y = [], []
    with torch.no_grad(), amp_context:
        for i in range(0, C, B):
            z_x = module.encoder_x(local_x[i:i+B])
            z_y = module.encoder_y(local_y[i:i+B])
            local_Z_x.append(z_x)  # Use float16 for all-gather embeddings 
            local_Z_y.append(z_y)
    local_Z_x, local_Z_y = torch.cat(local_Z_x, dim=0), torch.cat(local_Z_y, dim=0)

    Z_x, Z_y = all_gather_embeddings(local_Z_x), all_gather_embeddings(local_Z_y)

    with torch.no_grad():
        a, b = compute_streaming_log_normalizers(Z_x, Z_y, TAU, M_stream)
        global_loss = calculate_global_loss(Z_x, Z_y, a, b, TAU, N)

    start_idx_global = rank * C
    num_microbatches = math.ceil(C / B)

    for i_mb in range(num_microbatches):
        i = i_mb * B
        mb_start, mb_end = i, min(i + B, C)
        global_start, global_end = start_idx_global + mb_start, start_idx_global + mb_end
        is_last_microbatch = (i_mb == num_microbatches - 1)
        context = model.no_sync() if (not is_last_microbatch and P > 1 and hasattr(model, 'no_sync')) else nullcontext()

        with context:
            with torch.no_grad():
                G_x_M, G_y_M = compute_streaming_gradients(
                    Z_x, Z_y, Z_x[global_start:global_end], Z_y[global_start:global_end],
                    a, b, a[global_start:global_end], b[global_start:global_end],
                    TAU, N, M_stream)

            with amp_context:
                Z_x_M_grad, Z_y_M_grad = model(local_x[mb_start:mb_end], local_y[mb_start:mb_end])
                surrogate_loss = (Z_x_M_grad.float() * G_x_M).sum() + (Z_y_M_grad.float() * G_y_M).sum()
                scaled_surrogate_loss = surrogate_loss * P

            if scaler is not None:
                scaler.scale(scaled_surrogate_loss).backward()
            else:
                scaled_surrogate_loss.backward()

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return global_loss

def distributed_validate_step(
    model: torch.nn.Module,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    config: Dict
) -> Tuple[float, Dict[str, float]]:
    """
    Performs a distributed, memory-efficient validation step with progress bars
    for both embedding calculation and metric computation.
    """
    model.eval()
    
    if dist.is_initialized():
        P, rank = dist.get_world_size(), dist.get_rank()
    else:
        P, rank = 1, 0

    N, B, M_stream, TAU = config['GLOBAL_BATCH_SIZE'], config['MICRO_BATCH_SIZE'], config['STREAM_CHUNK_SIZE'], config['TAU']
    C = local_x.shape[0]

    if N != C * P:
        raise ValueError(f"Global batch size N={N} must equal local_chunk_size*P ({C}*{P}).")

    with torch.no_grad():
        module = model.module if hasattr(model, 'module') else model
        if not hasattr(module, 'encoder_x') or not hasattr(module, 'encoder_y'):
            raise AttributeError("Model must expose 'encoder_x' and 'encoder_y'.")
        
        # --- Phase A: Compute local embeddings with a progress bar ---
        local_Z_x, local_Z_y = [], []
        
        # --- NEW PROGRESS BAR START ---
        # This wrapper will show progress for the embedding calculation loop.
        embedding_pbar = tqdm(range(0, C, B), desc="Val Embeddings", disable=(rank != 0))
        # --- NEW PROGRESS BAR END ---

        for i in embedding_pbar:
            z_x = module.encoder_x(local_x[i:i+B])
            z_y = module.encoder_y(local_y[i:i+B])
            local_Z_x.append(z_x)
            local_Z_y.append(z_y)
        
        local_Z_x = torch.cat(local_Z_x, dim=0)
        local_Z_y = torch.cat(local_Z_y, dim=0)

        # --- Sync Point 1: All-gather to get global embeddings ---
        Z_x = all_gather_embeddings(local_Z_x)
        Z_y = all_gather_embeddings(local_Z_y)

        # --- Phase B: Compute global loss (this part is fast enough not to need a progress bar) ---
        a, b = compute_streaming_log_normalizers(Z_x, Z_y, TAU, M_stream)
        global_loss = calculate_global_loss(Z_x, Z_y, a, b, TAU, N)
        
        # --- Streamed Top-K Accuracy and MRR Calculation ---
        start_idx_global = rank * C
        labels = torch.arange(start_idx_global, start_idx_global + C, device=local_x.device)
        k_vals = [1, 5, 10]
        max_k = min(max(k_vals), N)
        topk_for_mrr = min(N, Z_y.shape[0]) 

        val_micro_batch_size = config.get('MICRO_BATCH_SIZE', 128)
        num_local_chunks = math.ceil(C / val_micro_batch_size)

        all_reciprocal_ranks = []
        all_correct_counts = {k: 0 for k in k_vals}

        metric_pbar = tqdm(range(num_local_chunks), desc="Validation", disable=(rank != 0))

        for i in metric_pbar:
            start = i * val_micro_batch_size
            end = min((i + 1) * val_micro_batch_size, C)
            if start >= end: continue

            local_Z_x_chunk = local_Z_x[start:end]
            chunk_labels = labels[start:end]
            S_chunk = torch.matmul(local_Z_x_chunk, Z_y.T) / TAU
            _, topk_indices_chunk = S_chunk.topk(k=topk_for_mrr, dim=1)

            for k in k_vals:
                if k <= max_k:
                    correct = (topk_indices_chunk[:, :k] == chunk_labels.unsqueeze(1)).any(dim=1).sum()
                    all_correct_counts[k] += correct.item()

            for j in range(len(chunk_labels)):
                positions = (topk_indices_chunk[j] == chunk_labels[j]).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    rank_val = positions[0].item() + 1
                    all_reciprocal_ranks.append(1.0 / rank_val)
                else:
                    all_reciprocal_ranks.append(0.0)
        
        # --- Sync Point 2: Aggregate metrics from all GPUs ---
        correct_counts = [all_correct_counts[k] for k in k_vals]
        local_mrr_sum = sum(all_reciprocal_ranks)
        metrics_tensor = torch.tensor(correct_counts + [local_mrr_sum], device=local_x.device, dtype=torch.float32)
        
        if P > 1:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        total_metrics = metrics_tensor.cpu().numpy()
        total_correct = total_metrics[:len(k_vals)]
        total_mrr_sum = total_metrics[-1]

        topk_acc = {k: count / N for k, count in zip(k_vals, total_correct)}
        topk_acc['MRR'] = total_mrr_sum / N

    return global_loss, topk_acc


def trivial_contrastive_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_x: torch.Tensor,
    global_y: torch.Tensor,
    config: Dict,
    scaler: Optional[object] = None
) -> float:
    """
    A standard, mathematically equivalent implementation of symmetric CLIP loss.
    Supports automatic mixed precision when scaler is provided.
    """
    N, TAU = global_x.shape[0], config['TAU']
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Use autocast if scaler is provided
    amp_context = autocast() if scaler is not None else nullcontext()

    with amp_context:
        Z_x, Z_y = model(global_x, global_y)
        S = torch.matmul(Z_x, Z_y.T) / TAU
        labels = torch.arange(N, device=Z_x.device)
        loss_x = F.cross_entropy(S, labels, reduction='sum')
        loss_y = F.cross_entropy(S.T, labels, reduction='sum')
        total_loss = (loss_x + loss_y) / (2 * N)

    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        optimizer.step()
    return total_loss.item()

