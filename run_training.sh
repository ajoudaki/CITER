#!/bin/bash
# run_training.sh - Launch theorem contrastive training

# Check for distributed flag
if [ "$1" = "--distributed" ]; then
    # Get number of GPUs (default to 2 if not specified)
    NUM_GPUS="${2:-2}"

    echo "Starting distributed training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS theorem_contrastive_training.py
else
    echo "Starting single device training..."
    python theorem_contrastive_training.py
fi