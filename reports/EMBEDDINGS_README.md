# Pre-computing Embeddings

This directory contains scripts for pre-computing and caching dataset embeddings for faster inference.

## Directory Structure

```
outputs/demo/
└── qwen-1.5b/
    ├── projection.pt                    # Projection layer weights
    ├── lora_adapters/                   # LoRA adapter weights
    │   ├── adapter_config.json
    │   ├── adapter_model.bin
    │   └── ...
    └── embeddings/                      # Pre-computed embeddings
        ├── toy/
        │   ├── embeddings.pt            # Embedding tensor [N, 2048]
        │   ├── metadata.json            # Statement metadata
        │   └── info.json                # Dataset info
        ├── tiny/
        ├── small/
        ├── medium/
        └── full/
```

## Computing Embeddings

### Single Dataset

To compute embeddings for a single dataset size:

```bash
# Using 4 GPUs with batch size 64
torchrun --nproc_per_node=4 compute_embeddings.py \
    --model_dir outputs/demo/qwen-1.5b \
    --dataset_size toy \
    --batch_size 64
```

### All Dataset Sizes

To compute embeddings for all dataset sizes at once:

```bash
# Automatically detects number of GPUs
./scripts/compute_all_embeddings.sh outputs/demo/qwen-1.5b

# Or specify number of GPUs
./scripts/compute_all_embeddings.sh outputs/demo/qwen-1.5b 4
```

## Features

- **Distributed Training**: Uses PyTorch DDP to utilize all available GPUs
- **Mixed Precision**: FP16 inference for faster computation and lower memory usage
- **Large Batches**: Automatically maximizes batch size based on available memory
- **Caching**: Embeddings are cached on disk and loaded instantly by the web UI

## Web UI Integration

The web UI (`web_ui/app.py`) automatically:
1. Checks for pre-computed embeddings in `outputs/demo/{model_name}/embeddings/{dataset_size}/`
2. Loads cached embeddings if available (instant)
3. Falls back to on-the-fly computation if not found

## Performance

Pre-computing embeddings provides significant speedup:

| Dataset Size | Statements | On-the-fly | Pre-computed | Speedup |
|--------------|-----------|------------|--------------|---------|
| toy          | ~100      | ~5s        | <1s          | ~5x     |
| tiny         | ~1K       | ~30s       | <1s          | ~30x    |
| small        | ~10K      | ~5min      | <1s          | ~300x   |
| medium       | ~50K      | ~20min     | <1s          | ~1200x  |
| full         | ~500K     | ~3hr       | <1s          | ~10000x |

## Organizing Models for Demo

To prepare a model for the demo:

```bash
# Create demo directory structure
mkdir -p outputs/demo/qwen-1.5b

# Copy model weights
cp outputs/qwen-1.5b_projection.pt outputs/demo/qwen-1.5b/projection.pt
cp -r outputs/qwen-1.5b_lora_adapters outputs/demo/qwen-1.5b/lora_adapters

# Compute embeddings for all datasets
./scripts/compute_all_embeddings.sh outputs/demo/qwen-1.5b
```

## Requirements

- PyTorch with CUDA support
- Multiple GPUs (recommended)
- Sufficient disk space for embeddings:
  - toy: ~1 MB
  - tiny: ~10 MB
  - small: ~100 MB
  - medium: ~500 MB
  - full: ~4 GB
