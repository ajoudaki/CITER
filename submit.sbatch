#!/bin/bash
#SBATCH --job-name=llmservice_nemo_reasoning-math-retrieval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --partition=interactive
#SBATCH --account=llmservice_nemo_reasoning
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/contrastive_test/out.log
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/contrastive_test/err.log

set -euo pipefail

# ---------- Container & mounts (host paths must exist) ----------
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/siddjain/containers/citer_v1.sqsh"
CONTAINER_MOUNTS="/lustre/fsw/portfolios/llmservice/users/igitman/hf_models:/hf_models,/lustre/fsw/portfolios/llmservice/users/igitman/nemo_models:/nemo_models,/lustre/fsw/portfolios/llmservice/users/siddjain/workspace:/workspace,/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output:/output,/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/archive:/archive,/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data:/data,/lustre/fsw/portfolios/llmservice/users/siddjain/my_models:/my_models,/lustre:/lustre,/home/siddjain/.netrc:/root/.netrc"

# Verify mounts exist before launching (avoid silent failures)
for SRC in /lustre/fsw/portfolios/llmservice/users/igitman/hf_models /lustre/fsw/portfolios/llmservice/users/igitman/nemo_models /lustre/fsw/portfolios/llmservice/users/siddjain/workspace /lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output /lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/archive /lustre/fsw/portfolios/llmservice/users/siddjain/llm/data /lustre/fsw/portfolios/llmservice/users/siddjain/my_models /lustre /home/siddjain/.netrc; do
  if [ ! -e "$SRC" ]; then echo "ERROR: Missing mount source on host: $SRC"; exit 1; fi
done

# ---------- Distributed defaults (8 GPUs/node) ----------
NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
NODE_RANK="${SLURM_NODEID:-0}"
MASTER_ADDR="127.0.0.1"
if [ "$NNODES" -gt 1 ]; then MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"; fi
# Prefer a high port (firewalls often block low ports). Generate once and export to all tasks.
MASTER_PORT="${MASTER_PORT:-$(( 40000 + (RANDOM % 20000) ))}"
# Derive workers per node from allocated GPUs (fallback to 8)
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"   # one worker per GPU

# Optional NCCL/network tuning
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker,virbr,vmnet,vboxnet}"

echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"

# Single-line launch (inside container). We keep torchrun so it scales if you change --nodes later.
srun --ntasks=${NNODES} --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --container-image="${CONTAINER_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
  bash -lc "set -e; cd /workspace/CITER && \
    echo Host=\$(hostname) PROCID=\${SLURM_PROCID} NODEID=\${SLURM_NODEID:-NA} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NNODES} NPROC=${NPROC_PER_NODE} && \
    torchrun \
      --nproc_per_node=${NPROC_PER_NODE} \
      --nnodes=${NNODES} \
      --node_rank=\${SLURM_PROCID} \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      train_cluster.py"
