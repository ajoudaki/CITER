#!/bin/bash

# This script generates the complete Hydra configuration directory structure and files
# required for the refactored theorem_contrastive_training.py script.

# --- Stop on any error ---
set -e

# --- Main function to create all configs ---
create_configs() {
    echo "🚀 Starting Hydra config setup..."

    # 1. Create directory structure
    echo "    - Creating directory structure under ./configs/"
    mkdir -p configs/model/addons
    mkdir -p configs/training
    mkdir -p configs/optimizer
    mkdir -p configs/dataset

    # 2. Create the main config file
    echo "    - Creating main config.yaml"
cat << 'EOF' > configs/config.yaml
defaults:
  - model: qwen-1.5b
  - training: default
  - optimizer: adamw
  - dataset: lemmas_theorems
  - _self_

# Dataset configuration is loaded from the dataset file.
# You can override specific values here if needed, e.g., dataset.size=tiny
dataset:
  base_path: ${dataset.base_path} # Inherits base_path from the dataset config

# Weights & Biases configuration
wandb:
  enabled: true
  project: "theorem-contrastive"
  entity: null # Set to your wandb entity/username
  name: null # Auto-generated: ${model.name}_${dataset.size}
  tags:
    - contrastive
    - ${model.name}
    - ${dataset.size}
  group: null
  notes: null
  mode: "online" # online, offline, or disabled

# Output configuration
output:
  save_dir: "./outputs/checkpoints"
  save_lora: true

# Runtime configuration
runtime:
  distributed: false
  num_workers: 4

# Hydra configuration
hydra:
  run:
    dir: outputs/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: outputs/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
  job:
    chdir: false
EOF

    # 3. Create dataset config
    echo "    - Creating dataset/lemmas_theorems.yaml"
cat << 'EOF' > configs/dataset/lemmas_theorems.yaml
# Lemmas and Theorems dataset configuration
name: lemmas_theorems
base_path: data/lemmas_theorems
size: small  # Default size
train_ratio: 0.8
seed: 42
sampling: stratified  # Options: 'random' or 'stratified'

# Available sizes with descriptions
sizes:
  toy: "20MB - Quick testing"
  tiny: "4MB - Very small subset"
  small: "40MB - Small subset for development"
  medium: "400MB - Medium subset"
  full: "1.7GB - Complete dataset"
EOF

    # 4. Create training config
    echo "    - Creating training/default.yaml"
cat << 'EOF' > configs/training/default.yaml
# Parameters controlling the training loop and execution

# Batching & Data
global_batch_size: 1024
micro_batch_size: 8
stream_chunk_size: 256
max_length: 256

# Training schedule
num_epochs: 10

# Model & Loss
tau: 0.07 # Temperature for contrastive loss

# Performance & Memory
use_amp: true               # Mixed precision
gradient_checkpointing: false # Set to true to trade compute for memory

# Validation
validation_interval: 0  # 0 means validate only at epoch end

# Scheduler Configuration
scheduler:
  _target_: theorem_contrastive_training.get_cosine_schedule_with_warmup
  num_warmup_steps: 100
  # num_training_steps is passed programmatically
EOF

    # 5. Create optimizer configs
    echo "    - Creating optimizer configs (adamw, adamw_8bit)"
cat << 'EOF' > configs/optimizer/adamw.yaml
_target_: torch.optim.AdamW
lr: 1.0e-4
weight_decay: 0.01
betas: [0.9, 0.999]
eps: 1.0e-8
EOF

cat << 'EOF' > configs/optimizer/adamw_8bit.yaml
_target_: bitsandbytes.optim.AdamW8bit
lr: 1.0e-4
weight_decay: 0.01
betas: [0.9, 0.999]
eps: 1.0e-8
EOF

    # 6. Create model addon configs
    echo "    - Creating model/addons (lora, quantization)"
cat << 'EOF' > configs/model/addons/lora.yaml
# Addon to enable LoRA fine-tuning.
# Usage: python train.py model=qwen-1.5b model/addons=lora

lora:
  _target_: peft.LoraConfig
  enabled: true
  task_type: "FEATURE_EXTRACTION"
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  # target_modules are inherited from the base model config
EOF

cat << 'EOF' > configs/model/addons/quantization.yaml
# Addon to enable 4-bit quantization.
# Usage: python train.py model=qwen-1.5b model/addons=quantization optimizer=adamw_8bit

quantization_config:
  _target_: transformers.BitsAndBytesConfig
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: "torch.bfloat16"
EOF

    # 7. Create all base model configs
    echo "    - Creating all base model configs"

# --- bert-base ---
cat << 'EOF' > configs/model/bert-base.yaml
_target_: theorem_contrastive_training.CLSPoolingEncoder
name: bert-base
model_name: bert-base-uncased
hidden_dim: 768
output_dim: 2048

lora_target_modules:
- query
- key
- value

lora:
  enabled: false
EOF

# --- qwen-1.5b ---
cat << 'EOF' > configs/model/qwen-1.5b.yaml
_target_: theorem_contrastive_training.LastTokenEncoder
name: qwen-1.5b
model_name: deepseek-ai/deepseek-r1-distill-qwen-1.5b
hidden_dim: 1536
output_dim: 2048

lora_target_modules:
- q_proj
- v_proj
- k_proj
- o_proj

lora:
  enabled: false
EOF

# --- qwen-2.5-math-7b ---
cat << 'EOF' > configs/model/qwen-2.5-math-7b.yaml
_target_: theorem_contrastive_training.LastTokenEncoder
name: qwen-2.5-math-7b
model_name: Qwen/Qwen2.5-Math-7B-Instruct
hidden_dim: 3584
output_dim: 2048

lora_target_modules:
- q_proj
- v_proj
- k_proj
- o_proj

lora:
  enabled: false
EOF

# --- qwen-14b ---
cat << 'EOF' > configs/model/qwen-14b.yaml
_target_: theorem_contrastive_training.LastTokenEncoder
name: qwen-14b
model_name: Qwen/Qwen3-14B
hidden_dim: 5120
output_dim: 2048

lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

lora:
  enabled: false
EOF

    # 8. Create baseline model configs
    echo "    - Creating baseline model configs (jaccard, minhash, bm25)"

# --- jaccard-baseline ---
cat << 'EOF' > configs/model/jaccard-baseline.yaml
_target_: theorem_contrastive_training.JaccardEncoder
name: jaccard-baseline
model_name: bert-base-uncased # Placeholder, not used
output_dim: 2048
EOF

# --- minhash-baseline ---
cat << 'EOF' > configs/model/minhash-baseline.yaml
_target_: theorem_contrastive_training.MinHashEncoder
name: minhash-baseline
model_name: bert-base-uncased # Placeholder, not used
output_dim: 2048
EOF

# --- bm25-baseline ---
cat << 'EOF' > configs/model/bm25-baseline.yaml
_target_: theorem_contrastive_training.BM25Encoder
name: bm25-baseline
model_name: bert-base-uncased # Placeholder, not used
output_dim: 2048
EOF

    echo -e "\n✅ All configuration files have been created successfully in the 'configs' directory."
    echo "You can now run your training script, for example:"
    echo "    python theorem_contrastive_training.py model=bert-base"
    echo "    python theorem_contrastive_training.py model=qwen-1.5b model/addons=lora"
}

# --- Execute the function ---
create_configs


