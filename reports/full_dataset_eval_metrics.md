# Validation Metrics Report: Full Dataset

**Date:** December 16, 2025
**Model:** Qwen2.5-Math-7B-Instruct (Fine-tuned with Contrastive Learning)
**Model Path:** `outputs/big_run_qwen-7b`
**Dataset:** Full (1.7GB)
**Split:** eval

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-Math-7B-Instruct |
| Pooling Strategy | last_token_pooling |
| Hidden Dimension | 3584 |
| Output Dimension | 2048 |
| LoRA Enabled | Yes (r=8, alpha=16, dropout=0.05) |
| LoRA Target Modules | q_proj, v_proj, k_proj, o_proj |
| Quantization | 4-bit (nf4, double quant) |
| Trainable Parameters | 12,388,352 (0.32%) |

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Train Papers | 277,205 |
| Train Theorem/Lemma Pairs | 1,630,988 |
| Eval Papers | 69,302 |
| Eval Theorem/Lemma Pairs | 406,745 |
| Total Eval Statements | 787,684 |
| Unique Paper IDs | 72,867 |

---

## Query Statistics (True Positives per Query)

| Statistic | Value |
|-----------|-------|
| Mean | 18.86 |
| Median | 14.00 |
| Std Dev | 18.50 |
| Min | 0 |
| Max | 286 |

**Percentiles:**
| P50 | P75 | P90 | P95 | P99 |
|-----|-----|-----|-----|-----|
| 14 | 23 | 37 | 49 | 91 |

---

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| **MRR** | **0.8204** |
| **MAP** | **44.27%** |

### Precision & Recall at K

| K | Precision@K | Recall@K |
|---|-------------|----------|
| 1 | 77.30% | 7.58% |
| 5 | 60.66% | 25.15% |
| 10 | 48.13% | 35.90% |
| 50 | 18.66% | 57.03% |
| 100 | 10.98% | 63.98% |
| 1000 | 1.50% | 82.01% |

---

## Key Observations

1. **High MRR (0.82)**: The model ranks relevant documents highly on average, with the first relevant result typically appearing near the top.

2. **Strong Precision@1 (77.3%)**: Over 3/4 of queries have a relevant document as the top result.

3. **Good Recall Scaling**: Recall increases steadily with K:
   - @10: 35.9%
   - @100: 64.0%
   - @1000: 82.0%

4. **MAP (44.27%)**: Indicates moderate ranking quality across all relevant documents.

5. **Large Evaluation Scale**: Metrics computed over 787,684 queries with ~19 relevant documents per query on average.

---

## Commands Used

### 1. Compute Embeddings (8 GPUs)

```bash
torchrun --nproc_per_node=8 theorem_contrastive_training.py \
    model=qwen-2.5-math-7b \
    dataset.size=full \
    +training.compute_embeddings=true \
    +training.load_model_path=outputs/big_run_qwen-7b \
    output.save_dir=outputs/big_run_qwen-7b \
    +dataset.split=eval
```

### 2. Compute Metrics from Embeddings

```bash
torchrun --nproc_per_node=1 theorem_contrastive_training.py \
    model=qwen-2.5-math-7b \
    dataset.size=full \
    +training.compute_metrics_from_embeddings=true \
    +training.load_model_path=outputs/big_run_qwen-7b \
    output.save_dir=outputs/big_run_qwen-7b \
    +dataset.split=eval
```

---

## Training Configuration

### Training Command

```bash
torchrun --nproc_per_node=8 theorem_contrastive_training.py \
    model=qwen-2.5-math-7b \
    dataset.size=full \
    training.warmup_steps=200 \
    training.lr=0.001 \
    training.quantization.enabled=true \
    training.global_batch_size=16384 \
    training.micro_batch_size=8 \
    training.gradient_checkpointing=true
```

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Global Batch Size | 16,384 |
| Micro Batch Size | 8 |
| Learning Rate | 0.001 |
| Weight Decay | 0.01 |
| Warmup Steps | 200 |
| Number of Epochs | 10 |
| Max Sequence Length | 256 |
| Temperature (tau) | 0.07 |
| Stream Chunk Size | 256 |
| Gradient Checkpointing | Enabled |
| Mixed Precision (AMP) | Enabled |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Enabled | Yes |
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target Modules | q_proj, v_proj, k_proj, o_proj |

### Quantization Configuration

| Parameter | Value |
|-----------|-------|
| Enabled | Yes |
| Quant Type | nf4 |
| Double Quant | Yes |
| Compute Dtype | bfloat16 |

### Full Training Config (YAML)

```yaml
model:
  name: qwen-2.5-math-7b
  model_name: Qwen/Qwen2.5-Math-7B-Instruct
  model_type: last_token_pooling
  hidden_dim: 3584
  output_dim: 2048
  lora_target_modules: [q_proj, v_proj, k_proj, o_proj]

training:
  global_batch_size: 16384
  micro_batch_size: 8
  stream_chunk_size: 256
  tau: 0.07
  lr: 0.001
  weight_decay: 0.01
  num_epochs: 10
  max_length: 256
  warmup_steps: 200
  gradient_checkpointing: true
  use_amp: true
  drop_last: true
  validation_interval: 0
  k_vals: [1, 5, 10, 50, 100]
  lora:
    enabled: true
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
  quantization:
    enabled: true
    bnb_4bit_quant_type: nf4
    bnb_4bit_use_double_quant: true
    bnb_4bit_compute_dtype: bfloat16

dataset:
  name: lemmas_theorems
  base_path: data/lemmas_theorems
  size: full
  train_ratio: 0.8
  seed: 42
  sampling: stratified

output:
  save_dir: ./outputs
  save_lora: true
```
