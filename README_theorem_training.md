# Theorem Contrastive Learning with CLIP

Implementation of CLIP-style contrastive learning for mathematical theorems and lemmas using BERT encoders.

## Overview

This implementation trains a model to learn representations of mathematical theorems and lemmas using contrastive learning. The key idea is that theorems/lemmas from the same paper should have similar embeddings.

## Files

- `theorem_clip_simple.py` - Simple, clean single-GPU implementation
- `theorem_contrastive_training.py` - Full implementation with distributed training support
- `test_theorem_training.py` - Test suite for components
- `run_training.sh` - Launch script for training

## How It Works

1. **Data Loading**: Pairs of theorems/lemmas are sampled from the same paper
2. **Encoding**: BERT encodes each text, using the CLS token embedding
3. **Projection**: Embeddings are projected and L2-normalized
4. **Contrastive Loss**: InfoNCE loss aligns embeddings from the same paper

## Usage

### Simple Training (Single GPU)
```bash
python theorem_clip_simple.py
```

### Full Training (Single GPU)
```bash
./run_training.sh
```

### Distributed Training (Multiple GPUs)
```bash
./run_training.sh --distributed 4  # Use 4 GPUs
```

## Model Architecture

```
Text Input → BERT → CLS Token → Projection → L2 Norm → Contrastive Loss
```

- **Encoder**: BERT-base-uncased
- **Embedding Dim**: 256
- **Temperature (τ)**: 0.07
- **Batch Size**: 8 (configurable)

## Key Features

- Clean separation of concerns
- Compatible with `distributed_clip` framework
- Supports both single and multi-GPU training
- Memory-efficient streaming for large batches
- Parameter sharing between encoders

## Training Details

The model minimizes the symmetric InfoNCE loss:
```
L = -1/2N Σ[log p(yi|xi) + log p(xi|yi)]
```

Where positive pairs (xi, yi) are theorems/lemmas from the same paper.