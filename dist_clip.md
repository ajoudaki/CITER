This document provides a self-contained mathematical description of an efficient and scalable implementation for distributed symmetric contrastive learning. This approach ensures mathematical equivalence to a global batch computation while minimizing synchronization overhead to only two points and avoiding the materialization of the $N \times N$ similarity matrix.

### 1. Mathematical Formulation

#### 1.1 Setup and Notation

We consider a global batch of $N$ paired samples $\{(x_i, y_i)\}_{i=1}^N$. Two encoders, $f_x$ and $f_y$ (parameterized by $\theta$), process these samples to produce unit-norm embeddings $Z_x, Z_y \in \mathbb{R}^{N \times d}$, where $d$ is the embedding dimension and $\|z_i\|_2 = 1$.

The similarity matrix $S \in \mathbb{R}^{N \times N}$ is defined as:
$S = \frac{1}{\tau} Z_x Z_y^\top$
where $\tau > 0$ is the temperature parameter.

We define the row-softmax probabilities $P$ and column-softmax probabilities $Q$:
$P = \mathrm{softmax}_{\text{row}}(S) \qquad Q = \mathrm{softmax}_{\text{col}}(S)$

#### 1.2 Symmetric Contrastive Loss

The objective is the symmetric InfoNCE (CLIP) loss, averaged over the global batch:

$\mathcal{L}(\theta) = \frac{1}{2N} \sum_{i=1}^N \left( -\log P_{ii} - \log Q_{ii} \right)$

#### 1.3 Gradient Derivation

To optimize $\mathcal{L}(\theta)$, we first derive the gradients with respect to the embeddings $Z_x$ and $Z_y$. The derivation, based on the standard cross-entropy gradient, yields the following closed-form expressions:

$G_x = \frac{\partial \mathcal{L}}{\partial Z_x} = \frac{1}{2N\tau} \left( P Z_y + Q Z_y - 2 Z_y \right)$

$G_y = \frac{\partial \mathcal{L}}{\partial Z_y} = \frac{1}{2N\tau} \left( P^\top Z_x + Q^\top Z_x - 2 Z_x \right)$

These formulas are the foundation of the distributed strategy, allowing us to calculate the exact embedding gradients if we have access to the global $Z_x, Z_y, P,$ and $Q$.

### 2. Distributed Strategy

The training is distributed across $P$ GPUs. The global batch $N$ is partitioned such that each GPU $g$ handles a disjoint local chunk $I_g$ of size $C=N/P$. The strategy decouples the global gradient calculation from the local backpropagation.

### 3. The 5-Phase Algorithm

The algorithm proceeds in five phases, utilizing streaming for memory efficiency and specific synchronization strategies. $B$ denotes the local microbatch size.

#### Phase A: Forward Embeddings and Synchronization 1

1.  **Local Computation:** Each GPU $g$ computes the embeddings for its local chunk $\{z_i^x, z_i^y\}_{i\in I_g}$ *without gradient tracking* (`no_grad`).
2.  **Synchronization 1 (All-Gather):** The local embeddings are gathered. Every GPU now holds the complete, detached global matrices $Z_x$ and $Z_y$. Detaching prevents cross-rank autograd edges.

#### Phase B: Global Log-Normalizers (Streaming)

We require $P$ and $Q$ to compute the gradients. We compute the necessary components memory-efficiently by first calculating the log-normalizers:

$a_i = \log \sum_j \exp(S_{ij}) \quad \text{(Row LSE)} \qquad b_j = \log \sum_i \exp(S_{ij}) \quad \text{(Column LSE)}$

We avoid materializing the $O(N^2)$ matrix $S$. We stream the computation using blocks of size $M$.

*   **For $a$:** Iterate over row blocks $R \subset \{1,\dots,N\}$. Compute $S_{R,:} = (Z_x[R] @ Z_y^\top)/\tau$. Calculate $a_R = \text{logsumexp}(S_{R,:}, \text{dim}=1)$. Discard $S_{R,:}$.
*   **For $b$:** Computed similarly by streaming over rows of $Z_y$ (columns of $S$).

#### Phase C: Local Gradient Precomputation (Streaming)

On GPU $g$, we iterate over a microbatch $M \subset I_g$. We aim to compute the embedding gradients $G_x[M]$ and $G_y[M]$.

We again employ streaming to avoid materializing $B \times N$ or $N \times B$ probability slices. We stream over the global dimension $N$ in chunks $C$.

Consider the term $T = P_{M,:} Z_y$ from the $G_x[M]$ calculation. We utilize the distributive property:
$T = \sum_{\text{Chunks } C} P_{M,C} Z_y[C]$

1. Compute the similarity slice $S_{M,C} = (Z_x[M] @ Z_y[C]^\top)/\tau$.
2. Compute the probability slice $P_{M,C} = \exp(S_{M,C} - a_M \mathbf{1}^\top)$.
3. Accumulate the contribution $P_{M,C} @ Z_y[C]$.
4. Discard $S_{M,C}$ and $P_{M,C}$.

All terms in $G_x[M]$ and $G_y[M]$ are computed this way, keeping memory complexity dominated by $O(N \cdot d)$.

#### Phase D: Recomputation and Vector-Jacobian Product (VJP)

We now connect the precomputed, detached embedding gradients $G[M]$ (from Phase C) to the model parameters $\theta$.

1.  **Recomputation:** The encoders are rerun on the microbatch $M$, this time *with gradient tracking*, yielding $Z_x'[M]$ and $Z_y'[M]$ connected to the computation graph.
2.  **Surrogate Loss (VJP):** We define a surrogate loss $\mathcal{L}_M$ as the inner product of the recomputed embeddings and the precomputed gradients:

$\mathcal{L}_M = \langle Z_x'[M], G_x[M] \rangle + \langle Z_y'[M], G_y[M] \rangle$

By the chain rule, calling `backward()` on $\mathcal{L}_M$ executes the Vector-Jacobian Product, yielding the exact local parameter gradients $\frac{\partial \mathcal{L}_M}{\partial \theta}$.

#### Phase E: Synchronization 2 and Loss Scaling

We must aggregate the local parameter gradients. The loss definition $\mathcal{L}$ includes a $1/N$ factor, which is already present in the gradients $G_x, G_y$. Therefore, the global parameter gradient is the **SUM** of the local parameter gradients.

However, PyTorch Distributed Data Parallel (DDP) defaults to **AVERAGING** gradients (dividing by $P$).

To achieve a SUM using DDP's synchronization, we employ **Loss Scaling**. In Phase D, we scale the surrogate loss by the world size $P$ before the backward pass:

$\mathcal{L}_M^{\text{Scaled}} = P \cdot \mathcal{L}_M$

**Synchronization 2 (All-Reduce):** DDP automatically performs an All-Reduce on the gradients. The resulting global gradient is:

$\nabla_\theta \mathcal{L} = \underbrace{\frac{1}{P}}_{\text{DDP Avg.}} \sum_{g=1}^P \frac{\partial \mathcal{L}_{M_g}^{\text{Scaled}}}{\partial \theta} = \frac{1}{P} \sum_{g=1}^P \left( P \cdot \frac{\partial \mathcal{L}_{M_g}}{\partial \theta} \right) = \sum_{g=1}^P \frac{\partial \mathcal{L}_{M_g}}{\partial \theta}$

This yields the mathematically correct global gradient. Finally, the optimizer step is executed.
