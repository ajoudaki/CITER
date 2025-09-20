
### User Guide: `distributed_clip.py` Interface

This guide explains how to use the `distributed_clip.py` module for efficient, large-batch contrastive training in a PyTorch Distributed Data Parallel (DDP) environment.

#### Overview

The module provides a function, `distributed_train_step`, which performs a single optimization step for symmetric contrastive learning (e.g., CLIP). It is designed to handle the complexities of distributed computation, memory efficiency for large batches, and correct gradient synchronization, allowing users to scale training effectively.

#### Prerequisites

1.  **Environment:** Requires PyTorch and a CUDA-enabled environment.
2.  **Execution:** Training must be launched using `torchrun`.
3.  **DDP Setup:** The distributed process group (`torch.distributed.init_process_group`) must be initialized before calling the training functions.

#### Interface: `distributed_train_step`

```python
from distributed_clip import distributed_train_step

def distributed_train_step(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    local_x: torch.Tensor, 
    local_y: torch.Tensor, 
    config: Dict
) -> float:
```

##### Parameters

  * **`model`** (`torch.nn.Module`): The contrastive model. See Model Requirements below.
  * **`optimizer`** (`torch.optim.Optimizer`): The optimizer (e.g., AdamW, SGD).
  * **`local_x`**, **`local_y`** (`torch.Tensor`): The local data chunks assigned to the current GPU rank. If the global batch size is $N$ and there are $P$ GPUs, these tensors have a batch dimension of $N/P$.
  * **`config`** (`Dict`): A dictionary containing the following keys:
      * `'GLOBAL_BATCH_SIZE'` (N): The total effective batch size across all GPUs.
      * `'MICRO_BATCH_SIZE'` (B): The size of microbatches used for local processing (gradient accumulation).
      * `'STREAM_CHUNK_SIZE'` (M): The chunk size used for streaming the similarity matrix calculations (memory optimization parameter).
      * `'TAU'` ($\\tau$): The temperature parameter for the contrastive loss.

##### Returns

  * `float`: The calculated global batch loss value, suitable for logging during training.

#### Model Requirements

Your PyTorch model must adhere to a specific interface:

1.  **DDP Wrapping:** The model must be wrapped in `torch.nn.parallel.DistributedDataParallel` (DDP).
2.  **Forward Method:** The model's `forward` method must accept `(x, y)` and return normalized embeddings `(z_x, z_y)`. Embeddings **must** be L2 normalized.
3.  **Encoder Access:** The underlying module (i.e., `model.module` when DDP wrapped) must expose the two encoders as attributes named `encoder_x` and `encoder_y`.

**Example Model Structure:**

```python
import torch.nn as nn
import torch.nn.functional as F

class MyContrastiveModel(nn.Module):
    def __init__(self, encoder1, encoder2):
        super().__init__()
        # Must be named encoder_x and encoder_y
        self.encoder_x = encoder1 
        self.encoder_y = encoder2

    def forward(self, x, y):
        z_x = self.encoder_x(x)
        z_y = self.encoder_y(y)
        # Ensure embeddings are normalized
        z_x = F.normalize(z_x, p=2, dim=-1)
        z_y = F.normalize(z_y, p=2, dim=-1)
        return z_x, z_y
```

#### Example Usage (Training Loop)

```python
# Assuming DDP is initialized, model is wrapped, optimizer is set up, 
# and dataloader with DistributedSampler is ready.

# Define configuration
TRAIN_CONFIG = { ... }

# Training loop
for epoch in range(num_epochs):
    dataloader.sampler.set_epoch(epoch)
    for batch_idx, (local_x, local_y) in enumerate(dataloader):
        local_x = local_x.to(device)
        local_y = local_y.to(device)

        # Perform the distributed training step
        loss = distributed_train_step(
            model, optimizer, local_x, local_y, TRAIN_CONFIG
        )

        if dist.get_rank() == 0:
            print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss:.4f}")
```

-----

### In-Depth Mathematical Description of the Distributed Algorithm

This document provides a rigorous, self-contained mathematical description of the implemented distributed symmetric contrastive learning algorithm. The method ensures exact mathematical equivalence to a global batch computation while minimizing synchronization overhead and memory footprint.

#### 1\. Problem Formulation

We aim to optimize a model parameterized by $\\theta$ using a global batch of $N$ paired samples ${(x\_i, y\_i)}\_{i=1}^N$. The model produces unit-norm embeddings $Z\_x, Z\_y \\in \\mathbb{R}^{N \\times d}$.

The similarity matrix $S \\in \\mathbb{R}^{N \\times N}$ is defined as $S = \\frac{1}{\\tau} Z\_x Z\_y^\\top$.
We define row-softmax $P = \\mathrm{softmax}*{\\text{row}}(S)$ and column-softmax $Q = \\mathrm{softmax}*{\\text{col}}(S)$.

The objective is the symmetric InfoNCE loss:

$\\mathcal{L}(\\theta) = \\frac{1}{2N} \\sum\_{i=1}^N \\left( -\\log P\_{ii} - \\log Q\_{ii} \\right)$

#### 2\. Gradient Derivation and Strategy

The optimization relies on the gradients of $\\mathcal{L}$ with respect to the embeddings:

$G\_x = \\frac{\\partial \\mathcal{L}}{\\partial Z\_x} = \\frac{1}{2N\\tau} \\left( P Z\_y + Q Z\_y - 2 Z\_y \\right)$
$G\_y = \\frac{\\partial \\mathcal{L}}{\\partial Z\_y} = \\frac{1}{2N\\tau} \\left( P^\\top Z\_x + Q^\\top Z\_x - 2 Z\_x \\right)$

The core strategy is to decouple the calculation of these embedding gradients ($G\_x, G\_y$), which require global information, from the backpropagation to the parameters ($\\nabla\_\\theta \\mathcal{L}$), using the Vector-Jacobian Product (VJP).

#### 3\. The 5-Phase Distributed Algorithm

The computation is distributed across $P$ GPUs, each handling a local chunk $I\_g$ of size $C=N/P$. The algorithm avoids $O(N^2)$ memory complexity and limits synchronization to two points.

##### Phase A: Forward Embeddings and Synchronization 1 (All-Gather)

1.  **Local Forward:** Each GPU $g$ computes its local embeddings ${z\_i^x, z\_i^y}\_{i\\in I\_g}$ without gradient tracking (`no_grad`).
2.  **Synchronization 1:** An All-Gather operation collects these local embeddings. Every GPU now possesses the full, detached global matrices $Z\_x$ and $Z\_y$.

##### Phase B: Global Normalizers and Loss Calculation (Streaming)

We require the log-normalizers (LogSumExp) for $P$ and $Q$:
$a\_i = \\log \\sum\_j \\exp(S\_{ij}) \\quad \\text{(Row LSE)} \\qquad b\_j = \\log \\sum\_i \\exp(S\_{ij}) \\quad \\text{(Column LSE)}$

**Memory Efficiency (Streaming):** To avoid materializing $S$, we compute $a$ and $b$ by streaming over blocks of size $M$. For row blocks $R$:
$S\_{R,:} = (Z\_x[R] @ Z\_y^\\top)/\\tau$
$a\_R = \\text{logsumexp}(S\_{R,:}, \\text{dim}=1)$
$S\_{R,:}$ is discarded after use.

**Global Loss Calculation:**
We can express the loss $\\mathcal{L}$ using the normalizers, as $\\log P\_{ii} = S\_{ii} - a\_i$ and $\\log Q\_{ii} = S\_{ii} - b\_i$:

$\\mathcal{L} = \\frac{1}{2N} \\sum\_{i=1}^N \\left( (a\_i - S\_{ii}) + (b\_i - S\_{ii}) \\right) = \\frac{1}{2N} \\sum\_{i=1}^N \\left( a\_i + b\_i - 2S\_{ii} \\right)$

The diagonal elements $S\_{ii} = \\frac{1}{\\tau} (z\_i^x \\cdot z\_i^y)$ are computed efficiently. Since $Z\_x, Z\_y, a, b$ are globally available, this loss calculation requires no further communication.

##### Phase C: Local Gradient Precomputation (Streaming)

On GPU $g$, we process a microbatch $M \\subset I\_g$ (size $B$) to compute $G\_x[M]$ and $G\_y[M]$.

**Memory Efficiency (Streaming):** We must avoid materializing the $B \\times N$ probability slices (e.g., $P\_{M,:}$). We leverage the distributive property and stream over the global dimension $N$ in chunks $C'$:

$(P Z\_y)*M = P*{M,:} Z\_y = \\sum\_{\\text{Chunks } C'} P\_{M,C'} Z\_y[C']$

In each streaming step:

1.  Compute similarity slice $S\_{M,C'}$.
2.  Compute probability slice $P\_{M,C'} = \\exp(S\_{M,C'} - a\_M \\mathbf{1}^\\top)$ using precomputed $a\_M$.
3.  Accumulate $P\_{M,C'} @ Z\_y[C']$.
4.  Discard $S\_{M,C'}$ and $P\_{M,C'}$.

This results in the detached embedding gradients $G\_x[M]$ and $G\_y[M]$.

##### Phase D: Recomputation and Vector-Jacobian Product (VJP)

We now compute the parameter gradients $\\frac{\\partial \\mathcal{L}}{\\partial \\theta}$ using the precomputed $G[M]$.

1.  **Recomputation:** Rerun the encoders on the microbatch $M$ *with gradient tracking*, yielding $Z'[M]$ tied to the computation graph.
2.  **VJP via Surrogate Loss:** Define a surrogate loss $\\mathcal{L}\_M$:

$\\mathcal{L}\_M = \\langle Z\_x'[M], G\_x[M] \\rangle + \\langle Z\_y'[M], G\_y[M] \\rangle$

Calling `backward()` on $\\mathcal{L}\_M$ computes the exact local parameter gradient for this microbatch via the chain rule.

##### Phase E: Synchronization 2 (All-Reduce) and Loss Scaling

We must aggregate local parameter gradients. The global loss $\\mathcal{L}$ definition includes a $1/N$ factor, which is already incorporated into $G\_x$ and $G\_y$. Therefore, the global parameter gradient is the **SUM** of the local gradients.

PyTorch DDP, by default, performs an All-Reduce with an **AVERAGE** operation (dividing by $P$).

**Loss Scaling:** To achieve a SUM using DDP's averaging mechanism, we scale the surrogate loss by $P$ in Phase D before backpropagation:

$\\mathcal{L}\_M^{\\text{Scaled}} = P \\cdot \\mathcal{L}\_M$

**Synchronization 2:** DDP performs the All-Reduce (Sync Point 2). The resulting synchronized gradient is:

$\\nabla\_\\theta \\mathcal{L} = \\underbrace{\\frac{1}{P}}*{\\text{DDP Avg.}} \\sum*{g=1}^P \\frac{\\partial \\mathcal{L}*{M\_g}^{\\text{Scaled}}}{\\partial \\theta} = \\frac{1}{P} \\sum*{g=1}^P \\left( P \\cdot \\frac{\\partial \\mathcal{L}*{M\_g}}{\\partial \\theta} \\right) = \\sum*{g=1}^P \\frac{\\partial \\mathcal{L}\_{M\_g}}{\\partial \\theta}$

This yields the exact global gradient, ensuring mathematical equivalence. The optimizer then updates the parameters.