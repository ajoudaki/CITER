#!/bin/bash
# Helper script to compute embeddings for all dataset sizes
# Usage: ./scripts/compute_all_embeddings.sh outputs/demo/qwen-1.5b [num_gpus]

MODEL_DIR=${1:-"outputs/demo/qwen-1.5b"}
NUM_GPUS=${2:-$(nvidia-smi -L | wc -l)}

echo "Computing embeddings for all dataset sizes"
echo "Model directory: $MODEL_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "----------------------------------------"

DATASET_SIZES=("toy" "tiny" "small" "medium" "full")

for SIZE in "${DATASET_SIZES[@]}"; do
    echo ""
    echo "Processing dataset: $SIZE"
    echo "----------------------------------------"

    # Skip if dataset doesn't exist
    if [ ! -f "data/lemmas_theorems/${SIZE}.jsonl" ]; then
        echo "Dataset ${SIZE}.jsonl not found, skipping..."
        continue
    fi

    # Compute batch size based on dataset size and available memory
    case $SIZE in
        "toy")
            BATCH_SIZE=128
            ;;
        "tiny")
            BATCH_SIZE=96
            ;;
        "small")
            BATCH_SIZE=64
            ;;
        "medium")
            BATCH_SIZE=48
            ;;
        "full")
            BATCH_SIZE=32
            ;;
        *)
            BATCH_SIZE=64
            ;;
    esac

    torchrun --nproc_per_node=$NUM_GPUS compute_embeddings.py \
        --model_dir "$MODEL_DIR" \
        --dataset_size "$SIZE" \
        --batch_size $BATCH_SIZE

    if [ $? -eq 0 ]; then
        echo "✓ Successfully computed embeddings for $SIZE"
    else
        echo "✗ Failed to compute embeddings for $SIZE"
        exit 1
    fi
done

echo ""
echo "========================================="
echo "✓ All embeddings computed successfully!"
echo "========================================="
