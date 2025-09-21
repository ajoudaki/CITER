
###  Guide: `distributed_clip.py` Interface

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
      * `'TAU'` ($\tau$): The temperature parameter for the contrastive loss.

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

#### 1. Problem Formulation

We aim to optimize a model parameterized by $\theta$ using a global batch of $N$ paired samples ${(x_i, y_i)}_{i=1}^N$. The model produces unit-norm embeddings $Z_x, Z_y \in \mathbb{R}^{N \times d}$.

The similarity matrix $S \in \mathbb{R}^{N \times N}$ is defined as $S = \frac{1}{\tau} Z_x Z_y^\top$.
We define row-softmax $P = \mathrm{softmax}*{\text{row}}(S)$ and column-softmax $Q = \mathrm{softmax}*{\text{col}}(S)$.

The objective is the symmetric InfoNCE loss:

$$\mathcal{L}(\theta) = \frac{1}{2N} \sum_{i=1}^N \left( -\log P_{ii} - \log Q_{ii} \right)$$

#### 2. Gradient Derivation and Strategy

The optimization relies on the gradients of $\mathcal{L}$ with respect to the embeddings:

$$G_x = \frac{\partial \mathcal{L}}{\partial Z_x} = \frac{1}{2N\tau} \left( P Z_y + Q Z_y - 2 Z_y \right)$$
$$G_y = \frac{\partial \mathcal{L}}{\partial Z_y} = \frac{1}{2N\tau} \left( P^\top Z_x + Q^\top Z_x - 2 Z_x \right)$$

The core strategy is to decouple the calculation of these embedding gradients ($G_x, G_y$), which require global information, from the backpropagation to the parameters ($\nabla_\theta \mathcal{L}$), using the Vector-Jacobian Product (VJP).

#### 3. The 5-Phase Distributed Algorithm

The computation is distributed across $P$ GPUs, each handling a local chunk $I_g$ of size $C=N/P$. The algorithm avoids $O(N^2)$ memory complexity and limits synchronization to two points.

##### Phase A: Forward Embeddings and Synchronization 1 (All-Gather)

1.  **Local Forward:** Each GPU $g$ computes its local embeddings ${z_i^x, z_i^y}_{i\in I_g}$ without gradient tracking (`no_grad`).
2.  **Synchronization 1:** An All-Gather operation collects these local embeddings. Every GPU now possesses the full, detached global matrices $Z_x$ and $Z_y$.

##### Phase B: Global Normalizers and Loss Calculation (Streaming)

We require the log-normalizers (LogSumExp) for $P$ and $Q$:
$$a_i = \log \sum_j \exp(S_{ij}) \quad \text{(Row LSE)} \qquad b_j = \log \sum_i \exp(S_{ij}) \quad \text{(Column LSE)}$$

**Memory Efficiency (Streaming):** To avoid materializing $S$, we compute $a$ and $b$ by streaming over blocks of size $M$. For row blocks $R$:
$S_{R,:} = (Z_x[R] @ Z_y^\top)/\tau$
$a_R = \text{logsumexp}(S_{R,:}, \text{dim}=1)$
$S_{R,:}$ is discarded after use.

**Global Loss Calculation:**
We can express the loss $\mathcal{L}$ using the normalizers, as $\log P_{ii} = S_{ii} - a_i$ and $\log Q_{ii} = S_{ii} - b_i$:

$$\mathcal{L} = \frac{1}{2N} \sum_{i=1}^N \left( (a_i - S_{ii}) + (b_i - S_{ii}) \right) = \frac{1}{2N} \sum_{i=1}^N \left( a_i + b_i - 2S_{ii} \right)$$

The diagonal elements $S_{ii} = \frac{1}{\tau} (z_i^x \cdot z_i^y)$ are computed efficiently. Since $Z_x, Z_y, a, b$ are globally available, this loss calculation requires no further communication.

##### Phase C: Local Gradient Precomputation (Streaming)

On GPU $g$, we process a microbatch $M \subset I_g$ (size $B$) to compute $G_x[M]$ and $G_y[M]$.

**Memory Efficiency (Streaming):** We must avoid materializing the $B \times N$ probability slices (e.g., $P_{M,:}$). We leverage the distributive property and stream over the global dimension $N$ in chunks $C'$:

$$(P Z_y)*M = P*{M,:} Z_y = \sum_{\text{Chunks } C'} P_{M,C'} Z_y[C']$$

In each streaming step:

1.  Compute similarity slice $S_{M,C'}$.
2.  Compute probability slice $P_{M,C'} = \exp(S_{M,C'} - a_M \mathbf{1}^\top)$ using precomputed $a_M$.
3.  Accumulate $P_{M,C'} @ Z_y[C']$.
4.  Discard $S_{M,C'}$ and $P_{M,C'}$.

This results in the detached embedding gradients $G_x[M]$ and $G_y[M]$.

##### Phase D: Recomputation and Vector-Jacobian Product (VJP)

We now compute the parameter gradients $\frac{\partial \mathcal{L}}{\partial \theta}$ using the precomputed $G[M]$.

1.  **Recomputation:** Rerun the encoders on the microbatch $M$ *with gradient tracking*, yielding $Z'[M]$ tied to the computation graph.
2.  **VJP via Surrogate Loss:** Define a surrogate loss $\mathcal{L}_M$:

$$\mathcal{L}_M = \langle Z_x'[M], G_x[M] \rangle + \langle Z_y'[M], G_y[M] \rangle$$

Calling `backward()` on $\mathcal{L}_M$ computes the exact local parameter gradient for this microbatch via the chain rule.

##### Phase E: Synchronization 2 (All-Reduce) and Loss Scaling

We must aggregate local parameter gradients. The global loss $\mathcal{L}$ definition includes a $1/N$ factor, which is already incorporated into $G_x$ and $G_y$. Therefore, the global parameter gradient is the **SUM** of the local gradients.

PyTorch DDP, by default, performs an All-Reduce with an **AVERAGE** operation (dividing by $P$).

**Loss Scaling:** To achieve a SUM using DDP's averaging mechanism, we scale the surrogate loss by $P$ in Phase D before backpropagation:

$\mathcal{L}_M^{\text{Scaled}} = P \cdot \mathcal{L}_M$

**Synchronization 2:** DDP performs the All-Reduce (Sync Point 2). The resulting synchronized gradient is:

$\nabla_\theta \mathcal{L} = \underbrace{\frac{1}{P}}*{\text{DDP Avg.}} \sum*{g=1}^P \frac{\partial \mathcal{L}*{M_g}^{\text{Scaled}}}{\partial \theta} = \frac{1}{P} \sum*{g=1}^P \left( P \cdot \frac{\partial \mathcal{L}*{M_g}}{\partial \theta} \right) = \sum*{g=1}^P \frac{\partial \mathcal{L}_{M_g}}{\partial \theta}$

This yields the exact global gradient, ensuring mathematical equivalence. The optimizer then updates the parameters.
