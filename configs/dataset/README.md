# Dataset Configuration

## Available Datasets

### lemmas_theorems
Mathematical lemmas and theorems dataset with multiple size options:

| Size   | File Size | Description                      | Usage                           |
|--------|-----------|----------------------------------|---------------------------------|
| toy    | ~20MB     | Toy dataset for quick testing   | `dataset.size=toy`             |
| tiny   | ~4MB      | Very small subset                | `dataset.size=tiny`            |
| small  | ~40MB     | Small subset for development     | `dataset.size=small` (default) |
| medium | ~400MB    | Medium subset                    | `dataset.size=medium`          |
| full   | ~1.7GB    | Complete dataset                 | `dataset.size=full`            |

## Usage Examples

```bash
# Use tiny dataset
python theorem_contrastive_training.py dataset.size=tiny

# Use full dataset with custom train ratio
python theorem_contrastive_training.py dataset.size=full dataset.train_ratio=0.9

# Use medium dataset with different seed
python theorem_contrastive_training.py dataset.size=medium dataset.seed=123

# Combine with other configs
python theorem_contrastive_training.py \
    model=bert-base \
    training=fast \
    dataset.size=tiny \
    training.num_epochs=2
```

## Adding New Datasets

1. Create a directory: `data/<dataset_name>/`
2. Add size variants: `tiny.jsonl`, `small.jsonl`, etc.
3. Create config: `configs/dataset/<dataset_name>.yaml`
4. Use: `python theorem_contrastive_training.py dataset=<dataset_name> dataset.size=<size>`