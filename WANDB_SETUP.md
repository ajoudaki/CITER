# Weights & Biases Setup Guide

## Installation

If you haven't installed wandb yet:
```bash
pip install wandb
```

## Initial Setup

1. **Login to wandb** (one-time setup):
```bash
wandb login
```
This will prompt you for your API key, which you can find at https://wandb.ai/authorize

2. **Set your entity** (optional):
Edit `configs/config.yaml` and set your wandb entity/username:
```yaml
wandb:
  entity: your-username  # Replace with your wandb username or team name
```

## Configuration

The wandb configuration is in `configs/config.yaml`:

```yaml
wandb:
  enabled: true                    # Toggle wandb on/off
  project: theorem-contrastive      # Project name in wandb
  entity: null                      # Your username or team
  name: null                        # Run name (auto-generated if null)
  tags:                            # Tags for organizing runs
    - contrastive
    - ${model.name}                # Automatically includes model name
    - ${dataset.size}              # Automatically includes dataset size
  group: null                      # Group related runs
  notes: null                      # Additional notes
  mode: online                     # Options: online, offline, disabled
```

## Usage Examples

### Basic usage:
```bash
python theorem_contrastive_training.py
```

### Disable wandb for testing:
```bash
python theorem_contrastive_training.py wandb.enabled=false
```

### Custom project and tags:
```bash
python theorem_contrastive_training.py \
    wandb.project=my-experiment \
    wandb.tags=[bert,large-batch,experiment-1]
```

### Offline mode (sync later):
```bash
python theorem_contrastive_training.py wandb.mode=offline
```
To sync offline runs later:
```bash
wandb sync
```

### Custom run name and group:
```bash
python theorem_contrastive_training.py \
    wandb.name=bert-baseline \
    wandb.group=ablation-study \
    wandb.notes="Testing effect of learning rate"
```

## Logged Metrics

The following metrics are automatically logged:

### Training Metrics (logged every step):
- `train/loss` - Training loss
- `train/learning_rate` - Current learning rate
- `train/batch_size` - Current batch size

### Validation Metrics (logged every epoch):
- `val/loss` - Validation loss
- `val/mrr` - Mean Reciprocal Rank
- `val/top1_acc` - Top-1 accuracy
- `val/top5_acc` - Top-5 accuracy
- `val/top10_acc` - Top-10 accuracy

### Model Information:
- `trainable_params` - Number of trainable parameters
- `total_params` - Total number of parameters
- `trainable_percent` - Percentage of trainable parameters

### Configuration:
All configuration parameters are logged automatically, including:
- Model architecture
- Dataset configuration
- Training hyperparameters
- LoRA settings (if enabled)

## Multi-GPU Training

When using distributed training, only rank 0 will log to wandb to avoid duplicate logging:
```bash
torchrun --nproc_per_node=2 theorem_contrastive_training.py
```

## Tips

1. **Organize experiments**: Use groups to organize related runs:
   ```bash
   python theorem_contrastive_training.py wandb.group=lr-sweep
   ```

2. **Add custom tags**: Tags help filter runs in the wandb UI:
   ```bash
   python theorem_contrastive_training.py wandb.tags=[baseline,final]
   ```

3. **Disable for debugging**: When debugging, disable wandb:
   ```bash
   python theorem_contrastive_training.py wandb.mode=disabled
   ```

4. **Resume runs**: To resume a crashed run, use the same run ID:
   ```bash
   python theorem_contrastive_training.py wandb.id=your-run-id wandb.resume=true
   ```

## Viewing Results

After running experiments, view your results at:
- https://wandb.ai/YOUR_ENTITY/theorem-contrastive

The wandb dashboard provides:
- Real-time metrics visualization
- Hyperparameter comparison
- System metrics (GPU usage, memory, etc.)
- Model checkpoints (if configured)
- Experiment comparison tools