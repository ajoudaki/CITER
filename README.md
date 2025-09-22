# Mathematical Theorem Contrastive Learning System

## Table of Contents
1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Quick Start](#quick-start)
4. [Configuration System](#configuration-system)
5. [Model Architecture](#model-architecture)
6. [Distributed Training Algorithm](#distributed-training-algorithm)
7. [Advanced Usage](#advanced-usage)
8. [Monitoring & Logging](#monitoring--logging)
9. [Implementation Details](#implementation-details)

## Overview

This project implements a state-of-the-art contrastive learning system for mathematical theorems and lemmas. Using CLIP-style symmetric contrastive learning, the system learns to create embeddings where theorems/lemmas from the same mathematical paper are close in the embedding space, while those from different papers are far apart.

### Key Goals
- **Learn semantic relationships** between mathematical statements through self-supervised learning
- **Scale efficiently** to large batches (1024+) using distributed training
- **Minimize memory footprint** through streaming computation techniques
- **Support multiple model architectures** (BERT, Qwen, etc.) with LoRA fine-tuning

### Core Innovation
The system implements a memory-efficient distributed algorithm that achieves exact mathematical equivalence to global batch computation while:
- Limiting synchronization to only 2 points (vs. continuous communication)
- Avoiding O(N²) memory complexity through streaming
- Supporting gradient accumulation for large effective batch sizes

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input: Mathematical Papers              │
│                  (Theorems & Lemmas in JSONL)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                    │
│   • Pairs theorems/lemmas from same paper                   │
│   • Tokenizes text (max_length: 256 tokens)                 │
│   • Creates train/validation splits                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Architecture                       │
│   ┌──────────────┐              ┌──────────────┐            │
│   │  Encoder X   │              │  Encoder Y   │            │
│   │  (Shared)    │              │  (Shared)    │            │
│   └──────┬───────┘              └──────┬───────┘            │
│          │                              │                   │
│          ▼                              ▼                   │
│   ┌──────────────┐              ┌──────────────┐            │
│   │  Projection  │              │  Projection  │            │
│   │   (Shared)   │              │   (Shared)   │            │
│   └──────┬───────┘              └───────┬──────┘            │
│          │                              │                   │
│          ▼                              ▼                   │
│      L2 Norm                        L2 Norm                 │
│          │                              │                   │
│          └──────────────┬───────────────┘                   │
│                         │                                   │
│                         ▼                                   │
│              Symmetric InfoNCE Loss                         │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation
```bash
# Install dependencies
pip install torch transformers peft hydra-core wandb tqdm

# Clone the repository
git clone <repository-url>
cd paperGPT
```

### Basic Training

#### Single GPU
```bash
python theorem_contrastive_training.py
```

#### Multi-GPU (Distributed)
```bash
# Using 2 GPUs
torchrun --nproc_per_node=2 theorem_contrastive_training.py

# Using 4 GPUs with custom config
torchrun --nproc_per_node=4 theorem_contrastive_training.py \
    dataset.size=small \
    training.global_batch_size=512 \
    training.num_epochs=10
```

### Quick Configurations

```bash
# Use tiny dataset for testing
python theorem_contrastive_training.py dataset.size=tiny

# Disable wandb logging
python theorem_contrastive_training.py wandb.mode=disabled

# Use different model
python theorem_contrastive_training.py model=bert-base

# Custom batch size and learning rate
python theorem_contrastive_training.py \
    training.global_batch_size=256 \
    training.lr=1e-4
```

## Configuration System

The project uses Hydra for configuration management with a hierarchical structure:

### Configuration Hierarchy
```
configs/
├── config.yaml           # Main config with defaults
├── dataset/
│   └── lemmas_theorems.yaml  # Dataset configurations
├── model/
│   ├── bert-base.yaml    # BERT model config
│   └── qwen-1.5b.yaml    # Qwen model config
└── training/
    ├── default.yaml      # Default training params
    └── small_model.yaml  # Lightweight config
```

### Key Configuration Parameters

#### Dataset Configuration (`configs/dataset/`)
```yaml
name: lemmas_theorems
size: small  # Options: toy, tiny, small, medium, full
train_ratio: 0.8
seed: 42

# Dataset sizes:
# toy:    ~20MB  - Quick testing
# tiny:   ~4MB   - Very small subset
# small:  ~40MB  - Development
# medium: ~400MB - Medium scale
# full:   ~1.7GB - Complete dataset
```

#### Model Configuration (`configs/model/`)
```yaml
# Example: bert-base.yaml
name: bert-base
model_name: bert-base-uncased
model_type: cls_pooling  # or last_token_pooling
hidden_dim: 768
lora_target_modules: ["query", "key", "value"]
```

#### Training Configuration (`configs/training/`)
```yaml
global_batch_size: [64, 64, 1024]  # Can be list for scheduling
micro_batch_size: 6     # Per-GPU batch for gradient accumulation
stream_chunk_size: 256  # Memory optimization parameter
tau: 0.07              # Temperature for contrastive loss
lr: 2e-4
num_epochs: 20
max_length: 256        # Max token length
output_dim: 2048       # Embedding dimension

# LoRA configuration
lora:
  enabled: true
  r: 8                 # LoRA rank
  lora_alpha: 16
  lora_dropout: 0.05
```

### Warmup

There is a warmup that gows up linearly, and a cosine lr decay. can be set via `training.warmup_steps` config. 

## Model Architecture

### Encoder Types

#### 1. CLS Pooling Encoder
Uses the [CLS] token embedding (BERT-style):
```python
class CLSPoolingEncoder:
    def forward(self, input_ids, attention_mask):
        outputs = base_model(input_ids, attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return normalize(projection(cls_embedding))
```

#### 2. Last Token Encoder
Uses the last non-padding token (GPT-style):
```python
class LastTokenEncoder:
    def forward(self, input_ids, attention_mask):
        outputs = base_model(input_ids, attention_mask)
        last_positions = attention_mask.sum(dim=1) - 1
        last_token = outputs[range(batch), last_positions]
        return normalize(projection(last_token))
```

### Parameter Sharing
Both encoders (X and Y) share the same weights for efficiency:
```python
self.encoder_x = base_encoder
self.encoder_y = self.encoder_x  # Shared parameters
```

### LoRA Integration
When enabled, LoRA adapters are applied to specified modules:
- **BERT**: targets `["query", "key", "value"]`
- **Qwen**: targets `["q_proj", "v_proj", "k_proj", "o_proj"]`

Only LoRA parameters and projection layers are trainable, keeping base model frozen.

## Distributed Training Algorithm

### Overview
The distributed algorithm implements a 5-phase approach that achieves exact mathematical equivalence to global batch training while minimizing memory usage and synchronization overhead.

### Mathematical Foundation

Given N paired samples {(x_i, y_i)}, the symmetric InfoNCE loss is:

```
L = 1/2N Σ[-log P_ii - log Q_ii]
```

Where:
- S = (1/τ) * Z_x @ Z_y^T  (similarity matrix)
- P = softmax_row(S)
- Q = softmax_col(S)

### The 5-Phase Algorithm

#### Phase A: Forward & Sync 1 (All-Gather)
```python
# Each GPU computes local embeddings
local_Z_x = encoder_x(local_data_x)  # No gradients
local_Z_y = encoder_y(local_data_y)

# Synchronization Point 1: All-Gather
global_Z_x = all_gather(local_Z_x).detach()
global_Z_y = all_gather(local_Z_y).detach()
```

#### Phase B: Global Loss (Streaming)
```python
# Compute log-normalizers without materializing S
for chunk in stream_chunks(global_Z_x, M):
    S_chunk = chunk @ global_Z_y.T / tau
    a[chunk_idx] = logsumexp(S_chunk, dim=1)
    # S_chunk is discarded

# Similarly for column normalizers (b)
loss = (a.sum() + b.sum() - 2*diag.sum()) / 2N
```

#### Phase C: Gradient Precomputation (Streaming)
```python
# Compute embedding gradients G_x, G_y
for microbatch in local_data:
    for chunk in stream_chunks(global_data, M):
        S_chunk = microbatch @ chunk.T / tau
        P_chunk = exp(S_chunk - a)
        accumulate(P_chunk @ chunk)
    G_x[microbatch] = accumulated_gradient
```

#### Phase D: VJP & Backpropagation
```python
# Recompute with gradients enabled
Z_x_grad = encoder_x(microbatch_x)  # With gradients
Z_y_grad = encoder_y(microbatch_y)

# Surrogate loss for VJP
surrogate_loss = (Z_x_grad * G_x).sum() + (Z_y_grad * G_y).sum()
scaled_loss = surrogate_loss * world_size  # Compensate for DDP averaging
scaled_loss.backward()
```

#### Phase E: Sync 2 (All-Reduce)
```python
# DDP automatically performs All-Reduce
# Gradients are averaged across GPUs
optimizer.step()
```

### Memory Efficiency

The streaming approach ensures:
- **Space Complexity**: O(N*d + M*N/P) instead of O(N²)
- **Only 2 sync points** vs continuous communication
- **Exact gradients** - mathematically equivalent to global batch

Key parameters:
- `STREAM_CHUNK_SIZE (M)`: Controls memory vs computation trade-off
- `MICRO_BATCH_SIZE (B)`: Controls gradient accumulation granularity

## Advanced Usage

### Custom Model Integration

To add a new model type:

1. Create encoder class:
```python
class CustomEncoder(nn.Module):
    def __init__(self, model_name, hidden_dim, output_dim):
        super().__init__()
        self.base_model = load_your_model(model_name)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        features = self.base_model(input_ids, attention_mask)
        return F.normalize(self.projection(features), p=2, dim=-1)
```

2. Register in `ENCODER_REGISTRY`:
```python
ENCODER_REGISTRY = {
    'custom_pooling': CustomEncoder,
    # ... other encoders
}
```

3. Create config file `configs/model/custom.yaml`:
```yaml
name: custom
model_name: path/to/model
model_type: custom_pooling
hidden_dim: 1024
lora_target_modules: ["attention", "mlp"]
```

### Performance Tuning

#### Batch Size Selection
- Start with `global_batch_size = 8 * num_gpus` for testing
- Scale up gradually: [64, 256, 1024]
- Monitor GPU memory with larger batches

#### Stream Chunk Size
- Smaller values (128-256): Less memory, more computation
- Larger values (512-1024): More memory, less computation
- Default 256 works well for most cases

#### Gradient Accumulation
```bash
# Large effective batch with limited GPU memory
python theorem_contrastive_training.py \
    training.global_batch_size=1024 \
    training.micro_batch_size=4  # Only 4 samples per GPU at once
```

### Multi-Node Training

For training across multiple nodes:
```bash
# On node 0
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<node0_ip> --master_port=29500 \
    theorem_contrastive_training.py

# On node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<node0_ip> --master_port=29500 \
    theorem_contrastive_training.py
```

## Monitoring & Logging

### Weights & Biases Integration

The system integrates with W&B for experiment tracking:

```bash
# Login (one-time)
wandb login

# Run with logging
python theorem_contrastive_training.py \
    wandb.project=my-experiments \
    wandb.name=baseline-run
```

#### Logged Metrics
- **Training**: loss, learning_rate, batch_size (per step)
- **Validation**: loss, MRR, top-1/5/10 accuracy (per epoch)
- **Model**: trainable_params, total_params, trainable_percent
- **System**: GPU utilization, memory usage

#### Offline Mode
```bash
# Train offline
python theorem_contrastive_training.py wandb.mode=offline

# Sync later
wandb sync
```

### Output Structure
```
outputs/
├── 2024-01-15/
│   └── 14-30-00/
│       ├── .hydra/           # Hydra configs
│       ├── theorem_contrastive_training.log
│       ├── qwen-1.5b_lora_adapters/  # Saved LoRA
│       └── qwen-1.5b_projection.pt   # Projection layer
```

## Implementation Details

### Numerical Stability

The implementation ensures numerical stability through:

1. **LogSumExp for normalizers**: Prevents overflow in softmax
```python
a = torch.logsumexp(S, dim=1)  # Instead of log(sum(exp(S)))
```

2. **Detached operations**: Prevents gradient graph explosion
```python
global_Z = all_gather(local_Z).detach()
```

3. **Streaming computation**: Avoids large intermediate tensors

### Gradient Correctness

The algorithm guarantees exact gradients through:

1. **Vector-Jacobian Product (VJP)**: Separates gradient computation from backprop
2. **Loss scaling**: Compensates for DDP averaging
3. **Proper synchronization**: All-Gather for embeddings, All-Reduce for gradients

### Error Recovery

The system includes robustness features:

```python
# Checkpoint saving
if epoch % save_interval == 0:
    save_checkpoint(model, optimizer, epoch)

# Automatic mixed precision (if enabled)
with autocast():
    loss = distributed_train_step(...)
```

### Testing & Validation

Run the test suite:
```bash
# Test gradient equivalence
python test_equivalence.py

# Analyze token lengths for dataset
python analyze_lengths.py
```

## Troubleshooting

### Common Issues

#### OOM Errors
- Reduce `micro_batch_size`
- Reduce `stream_chunk_size`
- Enable gradient checkpointing

#### Slow Training
- Increase `stream_chunk_size` if memory allows
- Ensure data loading isn't bottleneck (`num_workers`)
- Check network bandwidth for multi-node

#### NaN Loss
- Reduce learning rate
- Check for empty/corrupted data samples
- Ensure embeddings are normalized

### Debug Mode
```bash
# Verbose logging
HYDRA_FULL_ERROR=1 python theorem_contrastive_training.py \
    hydra.verbose=true
```

## Citation

If you use this codebase, please cite:
```bibtex
@software{theorem_contrastive_2024,
  title={Distributed Contrastive Learning for Mathematical Theorems},
  author={Your Name},
  year={2024},
  url={repository-url}
}
```

## License

[Specify your license here]

---

*For questions or issues, please open a GitHub issue or contact the maintainers.*
