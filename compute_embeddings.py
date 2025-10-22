#!/usr/bin/env python3
"""
Compute and cache embeddings for entire datasets using distributed training.

Usage:
    torchrun --nproc_per_node=<num_gpus> compute_embeddings.py \
        --model_dir outputs/demo/qwen-1.5b \
        --dataset_size toy \
        --batch_size 64

This script:
1. Loads model and projection weights from model_dir
2. Processes the dataset using all available GPUs with DDP
3. Caches embeddings to model_dir/embeddings/<dataset_size>/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel


class StatementDataset(Dataset):
    """Dataset of mathematical statements from papers."""

    def __init__(self, jsonl_path: Path):
        self.statements = []
        self.metadata = []

        # Load all statements from JSONL
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for paper_idx, line in enumerate(f):
                paper = json.loads(line.strip())

                # Handle both data formats
                if 'statements' in paper:
                    statements = paper['statements']
                else:
                    statements = []
                    for theorem in paper.get('theorems', []):
                        statements.append({'text': theorem, 'type': 'theorem'})
                    for lemma in paper.get('lemmas', []):
                        statements.append({'text': lemma, 'type': 'lemma'})

                for stmt in statements:
                    self.statements.append(stmt.get('text', ''))
                    self.metadata.append({
                        'paper_idx': paper_idx,
                        'paper_title': paper.get('title', 'Untitled'),
                        'arxiv_id': paper.get('arxiv_id', 'Unknown'),
                        'type': stmt.get('type', 'unknown'),
                        'text': stmt.get('text', '')
                    })

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        return {
            'text': self.statements[idx],
            'idx': idx
        }


class SimpleEncoder(nn.Module):
    """Encoder combining base model with projection layer."""

    def __init__(self, base_model, projection, model_type: str):
        super().__init__()
        self.base_model = base_model
        self.projection = projection
        self.model_type = model_type

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Pretrained embedding models may return embeddings directly
        if self.model_type == 'pretrained' and hasattr(outputs, 'embeddings'):
            return F.normalize(outputs.embeddings, p=2, dim=-1)

        # Use CLS token for BERT, last token for Qwen
        if 'bert' in self.model_type.lower():
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(sequence_lengths), device=sequence_lengths.device)
            embedding = outputs.last_hidden_state[batch_indices, sequence_lengths, :]

        return F.normalize(self.projection(embedding), p=2, dim=-1)


def collate_fn(batch, tokenizer, max_length=256):
    """Collate function for batching statements."""
    texts = [item['text'] for item in batch]
    indices = [item['idx'] for item in batch]

    inputs = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'indices': torch.tensor(indices)
    }


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def load_model(model_dir: Path, local_rank: int, pretrained_model: str = None):
    """Load model, LoRA adapters, and projection layer."""

    # Use pretrained model directly if specified
    if pretrained_model:
        if local_rank == 0:
            print(f"Loading pretrained model: {pretrained_model}")
        base_model = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
        hidden_dim = base_model.config.hidden_size
        projection = nn.Identity()  # No projection for pretrained embedding models
        model_type = 'pretrained'
    else:
        # Detect model type from directory name
        model_dir_name = model_dir.name.lower()

        # Check for LoRA adapters to determine exact model
        lora_dirs = list(model_dir.glob('*lora_adapters'))
        lora_hint = lora_dirs[0].name.lower() if lora_dirs else ''

        if 'bert' in model_dir_name:
            base_model_name = 'bert-base-uncased'
            hidden_dim = 768
            model_type = 'bert'
        elif '7b' in model_dir_name or '7b' in lora_hint or 'math-7b' in lora_hint:
            # Qwen 7B models
            if 'math' in lora_hint:
                base_model_name = 'Qwen/Qwen2.5-Math-7B-Instruct'
            else:
                base_model_name = 'Qwen/Qwen2.5-7B'
            hidden_dim = 3584
            model_type = 'qwen-7b'
        elif 'qwen' in model_dir_name or '1.5b' in model_dir_name:
            base_model_name = 'Qwen/Qwen2.5-1.5B'
            hidden_dim = 1536
            model_type = 'qwen'
        else:
            raise ValueError(f"Cannot determine model type from directory name: {model_dir_name}")

        # Load base model with quantization for 7B models
        use_quantization = '7b' in model_type.lower()

        if use_quantization:
            if local_rank == 0:
                print(f"Loading base model: {base_model_name} with 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            base_model = AutoModel.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map={'': local_rank}  # Map to specific GPU for DDP
            )
        else:
            if local_rank == 0:
                print(f"Loading base model: {base_model_name}")
            base_model = AutoModel.from_pretrained(base_model_name)

        # Load LoRA adapters if they exist
        lora_paths = [
            model_dir / 'lora_adapters',
            model_dir / f'{model_dir.name}_lora_adapters'
        ]

        for lora_path in lora_paths:
            if lora_path.exists():
                if local_rank == 0:
                    print(f"Loading LoRA adapters from: {lora_path}")
                base_model = PeftModel.from_pretrained(base_model, str(lora_path))
                break

        # Load projection layer
        projection_paths = [
            model_dir / 'projection.pt',
            model_dir / f'{model_dir.name}_projection.pt'
        ]

        projection = nn.Linear(hidden_dim, 2048)
        for proj_path in projection_paths:
            if proj_path.exists():
                if local_rank == 0:
                    print(f"Loading projection from: {proj_path}")
                projection.load_state_dict(torch.load(proj_path, map_location='cpu'))
                break

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Create encoder
    encoder = SimpleEncoder(base_model, projection, model_type)

    # For quantized models, only move projection to device
    # Base model is already on correct device via device_map
    if 'use_quantization' in locals() and use_quantization:
        encoder.projection = encoder.projection.to(local_rank).half()
        # Don't wrap quantized models with DDP - they're already on the right device
    else:
        encoder = encoder.to(local_rank)
        encoder = encoder.half()  # fp16
        # Wrap with DDP for non-quantized models
        encoder = DDP(encoder, device_ids=[local_rank])

    encoder.eval()

    return encoder, tokenizer


def compute_embeddings(
    model_dir: Path,
    dataset_size: str,
    batch_size: int,
    local_rank: int,
    world_size: int,
    pretrained_model: str = None
):
    """Compute embeddings for entire dataset."""

    # Load dataset
    dataset_path = Path(f'data/lemmas_theorems/{dataset_size}.jsonl')
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if local_rank == 0:
        print(f"Loading dataset: {dataset_path}")

    dataset = StatementDataset(dataset_path)

    if local_rank == 0:
        print(f"Total statements: {len(dataset)}")

    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False
    )

    # Load model
    encoder, tokenizer = load_model(model_dir, local_rank, pretrained_model)

    # Create dataloader with custom collate function
    from functools import partial
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    # Compute embeddings
    all_embeddings = []
    all_indices = []

    if local_rank == 0:
        print(f"Computing embeddings with batch_size={batch_size} on {world_size} GPUs...")
        pbar = tqdm(total=len(dataloader), desc="Processing batches")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            indices = batch['indices']

            # Compute embeddings with autocast
            with autocast(device_type='cuda', enabled=True):
                embeddings = encoder(input_ids, attention_mask)

            # Store embeddings on GPU for now (will gather on GPU, then move to CPU)
            all_embeddings.append(embeddings.float())
            all_indices.append(indices.to(local_rank))

            if local_rank == 0:
                pbar.update(1)

    if local_rank == 0:
        pbar.close()

    # Concatenate embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    # Gather from all GPUs to rank 0 (gather on GPU, then move to CPU)
    if world_size > 1:
        # Gather embeddings (on GPU)
        gathered_embeddings = [torch.zeros_like(all_embeddings, device=local_rank) for _ in range(world_size)] if local_rank == 0 else None
        dist.gather(all_embeddings, gathered_embeddings, dst=0)

        # Gather indices (on GPU)
        gathered_indices = [torch.zeros_like(all_indices, device=local_rank) for _ in range(world_size)] if local_rank == 0 else None
        dist.gather(all_indices, gathered_indices, dst=0)

        if local_rank == 0:
            all_embeddings = torch.cat(gathered_embeddings, dim=0).cpu()
            all_indices = torch.cat(gathered_indices, dim=0).cpu()
    else:
        # Single GPU - just move to CPU
        all_embeddings = all_embeddings.cpu()
        all_indices = all_indices.cpu()

    # Save embeddings (only on rank 0)
    if local_rank == 0:
        # Create output directory
        if pretrained_model:
            model_name = pretrained_model.replace('/', '_')
            output_dir = Path('outputs/demo') / model_name / 'embeddings' / dataset_size
        else:
            output_dir = model_dir / 'embeddings' / dataset_size
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort by indices to get original order
        sorted_idx = torch.argsort(all_indices)
        all_embeddings = all_embeddings[sorted_idx]

        # Save embeddings
        embeddings_path = output_dir / 'embeddings.pt'
        print(f"Saving embeddings to: {embeddings_path}")
        torch.save(all_embeddings, embeddings_path)

        # Save metadata
        metadata_path = output_dir / 'metadata.json'
        print(f"Saving metadata to: {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(dataset.metadata, f)

        # Save info
        info_path = output_dir / 'info.json'
        info = {
            'dataset_size': dataset_size,
            'num_statements': len(dataset),
            'embedding_dim': all_embeddings.shape[1],
            'model_dir': str(model_dir),
            'dataset_path': str(dataset_path)
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"✓ Saved {len(all_embeddings)} embeddings ({all_embeddings.shape})")


def main():
    parser = argparse.ArgumentParser(description='Compute embeddings for dataset')
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing model weights (e.g., outputs/demo/qwen-1.5b)')
    parser.add_argument('--pretrained_model', type=str,
                        help='HuggingFace model name (e.g., Qwen/Qwen3-Embedding-0.6B)')
    parser.add_argument('--dataset_size', type=str, required=True,
                        choices=['toy', 'tiny', 'small', 'medium', 'full'],
                        help='Size of dataset to process')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per GPU')

    args = parser.parse_args()

    if not args.model_dir and not args.pretrained_model:
        parser.error('Either --model_dir or --pretrained_model must be specified')
    if args.model_dir and args.pretrained_model:
        parser.error('Cannot specify both --model_dir and --pretrained_model')

    # Setup distributed
    local_rank = setup_distributed()
    world_size = dist.get_world_size()

    model_dir = Path(args.model_dir) if args.model_dir else Path('.')

    if local_rank == 0:
        print(f"=" * 80)
        print(f"Computing Embeddings")
        print(f"=" * 80)
        if args.pretrained_model:
            print(f"Pretrained model: {args.pretrained_model}")
        else:
            print(f"Model directory: {model_dir}")
        print(f"Dataset size: {args.dataset_size}")
        print(f"Batch size: {args.batch_size} per GPU")
        print(f"World size: {world_size} GPUs")
        print(f"=" * 80)

    try:
        compute_embeddings(
            model_dir=model_dir,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            local_rank=local_rank,
            world_size=world_size,
            pretrained_model=args.pretrained_model
        )
    finally:
        cleanup_distributed()

    if local_rank == 0:
        print("✓ Done!")


if __name__ == '__main__':
    main()
