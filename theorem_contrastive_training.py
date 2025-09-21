import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from typing import Dict
from tqdm import tqdm
from pathlib import Path

# Set TOKENIZERS_PARALLELISM before any other imports to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

# Import distributed functions
from distributed_clip import distributed_train_step, trivial_contrastive_step, distributed_validate_step

# ===================================================================
# Dataset (unchanged)
# ===================================================================
class TheoremLemmaDataset(Dataset):
    """Dataset that loads theorem/lemma pairs from JSONL file"""
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512, split: str = 'train', train_ratio: float = 0.8, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.data = []

        all_papers = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                all_statements = paper.get('lemmas', []) + paper.get('theorems', [])
                if len(all_statements) >= 2:
                    all_papers.append(all_statements)

        random.seed(seed)
        random.shuffle(all_papers)
        random.seed()

        n_train = int(len(all_papers) * train_ratio)
        self.data = all_papers[:n_train] if split == 'train' else all_papers[n_train:]
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{split.upper()} set: {len(self.data)} papers")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        statements = self.data[idx]
        if self.split == 'eval':
            text_x, text_y = (statements[0], statements[1]) if len(statements) >= 2 else (statements[0], statements[0])
        else:
            text_x, text_y = random.sample(statements, 2) if len(statements) >= 2 else (statements[0], statements[0])

        tokens_x = self.tokenizer(text_x, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        tokens_y = self.tokenizer(text_y, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids_x': tokens_x['input_ids'].squeeze(0), 'attention_mask_x': tokens_x['attention_mask'].squeeze(0),
            'input_ids_y': tokens_y['input_ids'].squeeze(0), 'attention_mask_y': tokens_y['attention_mask'].squeeze(0)
        }

# ===================================================================
# Model Architectures
# ===================================================================
class CLSPoolingEncoder(nn.Module):
    """Encoder that uses the [CLS] token embedding."""
    def __init__(self, model_name: str, hidden_dim: int, output_dim: int):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(self.projection(cls_embedding), p=2, dim=-1)

class LastTokenEncoder(nn.Module):
    """Encoder that uses the last non-padding token's hidden state."""
    def __init__(self, model_name: str, hidden_dim: int, output_dim: int):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(sequence_lengths), device=sequence_lengths.device)
        last_token_embedding = last_hidden_state[batch_indices, sequence_lengths, :]
        return F.normalize(self.projection(last_token_embedding), p=2, dim=-1)

ENCODER_REGISTRY = {
    'cls_pooling': CLSPoolingEncoder,
    'last_token_pooling': LastTokenEncoder,
}

class EncoderWrapper(nn.Module):
    def __init__(self, base_encoder, max_length):
        super().__init__()
        self.base_encoder = base_encoder
        self.max_length = max_length

    def forward(self, packed_tensor):
        input_ids = packed_tensor[:, :self.max_length].long()
        attention_mask = packed_tensor[:, self.max_length:].long()
        return self.base_encoder(input_ids, attention_mask)

class TheoremContrastiveModel(nn.Module):
    def __init__(self, base_encoder, max_length):
        super().__init__()
        self.encoder_x = EncoderWrapper(base_encoder, max_length)
        self.encoder_y = self.encoder_x

    def forward(self, x_packed, y_packed):
        return self.encoder_x(x_packed), self.encoder_y(y_packed)

# ===================================================================
# Helper Functions
# ===================================================================
def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || "
          f"trainable%: {100 * trainable_params / all_param:.2f}")

def setup_model(cfg: DictConfig, device):
    """Setup model with optional LoRA."""
    encoder_class = ENCODER_REGISTRY.get(cfg.model.model_type)
    if encoder_class is None:
        raise ValueError(f"Unknown model type: {cfg.model.model_type}")

    base_encoder = encoder_class(
        model_name=cfg.model.model_name,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=cfg.training.output_dim
    )

    # Apply LoRA if enabled
    if cfg.training.lora.enabled:
        # Freeze base model parameters
        for param in base_encoder.base_model.parameters():
            param.requires_grad = False

        # Create LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.training.lora.r,
            lora_alpha=cfg.training.lora.lora_alpha,
            lora_dropout=cfg.training.lora.lora_dropout,
            target_modules=cfg.model.lora_target_modules,
        )

        # Wrap with LoRA adapters
        base_encoder.base_model = get_peft_model(base_encoder.base_model, peft_config)

    # Projection head is always trainable
    for param in base_encoder.projection.parameters():
        param.requires_grad = True

    model = TheoremContrastiveModel(base_encoder, max_length=cfg.training.max_length)
    return model.to(device)

def prepare_validation_data(val_dataset, device, distributed, rank):
    """Prepare and distribute validation data."""
    if distributed:
        val_objects = [None, None]
        if rank == 0:
            print("Preparing and distributing validation dataset...")
            val_data = [val_dataset[i] for i in range(len(val_dataset))]
            val_x = torch.stack([d['input_ids_x'] for d in val_data])
            val_mx = torch.stack([d['attention_mask_x'] for d in val_data])
            val_y = torch.stack([d['input_ids_y'] for d in val_data])
            val_my = torch.stack([d['attention_mask_y'] for d in val_data])
            val_x_packed = torch.cat([val_x, val_mx], dim=1)
            val_y_packed = torch.cat([val_y, val_my], dim=1)
            val_objects = [val_x_packed, val_y_packed]
        dist.broadcast_object_list(val_objects, src=0)
        val_x_packed, val_y_packed = val_objects
    else:
        print("Preparing validation dataset...")
        val_data = [val_dataset[i] for i in range(len(val_dataset))]
        val_x = torch.stack([d['input_ids_x'] for d in val_data])
        val_mx = torch.stack([d['attention_mask_x'] for d in val_data])
        val_y = torch.stack([d['input_ids_y'] for d in val_data])
        val_my = torch.stack([d['attention_mask_y'] for d in val_data])
        val_x_packed = torch.cat([val_x, val_mx], dim=1)
        val_y_packed = torch.cat([val_y, val_my], dim=1)

    return val_x_packed, val_y_packed

def save_model(cfg: DictConfig, model, rank):
    """Save model or LoRA adapters."""
    if rank != 0:
        return

    output_dir = Path(cfg.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model

    if cfg.training.lora.enabled and cfg.output.save_lora:
        # Save LoRA adapters and projection
        adapter_path = output_dir / f'{cfg.model.name}_lora_adapters'
        projection_path = output_dir / f'{cfg.model.name}_projection.pt'

        model_to_save.encoder_x.base_encoder.base_model.save_pretrained(str(adapter_path))
        torch.save(model_to_save.encoder_x.base_encoder.projection.state_dict(), projection_path)
        print(f"LoRA adapters saved to '{adapter_path}' and projection to '{projection_path}'")
    else:
        # Save full model
        model_path = output_dir / f'{cfg.model.name}_contrastive_model.pt'
        torch.save(model_to_save.state_dict(), model_path)
        print(f"Model saved to {model_path}")

# ===================================================================
# Training Function
# ===================================================================
def get_epoch_batch_size(batch_size_config, epoch):
    """Get batch size for current epoch from config (single value or list)."""
    # Check if it's a list-like object (handles both list and OmegaConf ListConfig)
    if OmegaConf.is_list(batch_size_config) or isinstance(batch_size_config, list):
        # Use epoch-specific batch size or last value for remaining epochs
        idx = min(epoch, len(batch_size_config) - 1)
        return batch_size_config[idx]
    return batch_size_config

def train(cfg: DictConfig, rank: int = 0, world_size: int = 1, distributed: bool = False):
    """Main training function."""
    device = torch.device(f'cuda:{rank}') if distributed and torch.cuda.is_available() else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if distributed:
        torch.cuda.set_device(rank)

    if rank == 0:
        print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        print(f"Using model: {cfg.model.model_name} with {cfg.model.model_type} strategy.")
        if cfg.training.lora.enabled:
            print("LoRA is ENABLED.")

        # Show batch size schedule if using list
        if OmegaConf.is_list(cfg.training.global_batch_size):
            print(f"Batch size schedule: {list(cfg.training.global_batch_size)}")

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Construct dataset path from name and size
    dataset_path = Path(cfg.dataset.base_path) / f"{cfg.dataset.size}.jsonl"

    if rank == 0:
        print(f"Loading dataset: {dataset_path}")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Create datasets
    train_dataset = TheoremLemmaDataset(
        str(dataset_path), tokenizer,
        max_length=cfg.training.max_length,
        split='train',
        train_ratio=cfg.dataset.train_ratio,
        seed=cfg.dataset.seed
    )
    val_dataset = TheoremLemmaDataset(
        str(dataset_path), tokenizer,
        max_length=cfg.training.max_length,
        split='eval',
        train_ratio=cfg.dataset.train_ratio,
        seed=cfg.dataset.seed
    )

    # Initial data loader setup (will be recreated each epoch if batch size changes)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None

    # Setup model
    model = setup_model(cfg, device)

    if rank == 0:
        print_trainable_parameters(model)

    if distributed:
        model = DDP(model, device_ids=[rank])

    # Setup optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr
    )

    # Prepare validation data
    val_x_packed, val_y_packed = prepare_validation_data(val_dataset, device, distributed, rank)

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        # Get batch size for current epoch
        epoch_batch_size = get_epoch_batch_size(cfg.training.global_batch_size, epoch)
        batch_size = epoch_batch_size // world_size

        # Create data loader with current epoch's batch size
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=cfg.runtime.num_workers,
            pin_memory=True,
            drop_last=cfg.training.drop_last
        )

        if rank == 0 and (epoch == 0 or (OmegaConf.is_list(cfg.training.global_batch_size) and
                                          epoch < len(cfg.training.global_batch_size))):
            print(f"Epoch {epoch+1}: Global batch size = {epoch_batch_size}")

        model.train()
        if distributed and train_sampler:
            train_sampler.set_epoch(epoch)

        total_loss, num_batches = 0, 0

        pbar = tqdm(train_loader, disable=(rank!=0), desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        for batch in pbar:
            x_packed = torch.cat([batch['input_ids_x'].to(device), batch['attention_mask_x'].to(device)], dim=1)
            y_packed = torch.cat([batch['input_ids_y'].to(device), batch['attention_mask_y'].to(device)], dim=1)
            actual_batch_size = x_packed.shape[0] * (world_size if distributed else 1)

            if cfg.training.drop_last and actual_batch_size < epoch_batch_size:
                if rank == 0:
                    print(f"Skipping incomplete batch of size {actual_batch_size}")
                continue

            # Create config dict for compatibility with existing functions
            train_config = {
                'GLOBAL_BATCH_SIZE': actual_batch_size,
                'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
                'STREAM_CHUNK_SIZE': cfg.training.stream_chunk_size,
                'TAU': cfg.training.tau
            }

            if distributed:
                loss = distributed_train_step(model, optimizer, x_packed, y_packed, train_config)
            else:
                loss = trivial_contrastive_step(model, optimizer, x_packed, y_packed, train_config)

            total_loss += loss
            num_batches += 1
            if rank == 0:
                pbar.set_postfix(loss=f'{loss:.4f}')

        # Validation
        N_val = val_x_packed.shape[0]
        val_world_size = world_size if distributed else 1
        C_val = N_val // val_world_size

        if N_val % val_world_size != 0 and rank == 0:
            print(f"Warning: Val set size {N_val} not divisible by world size {val_world_size}")

        start, end = rank * C_val, (rank + 1) * C_val
        local_val_x = val_x_packed[start:end].to(device)
        local_val_y = val_y_packed[start:end].to(device)

        val_config = {
            'GLOBAL_BATCH_SIZE': C_val * val_world_size,
            'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
            'STREAM_CHUNK_SIZE': cfg.training.stream_chunk_size,
            'TAU': cfg.training.tau
        }

        val_loss, topk_acc = distributed_validate_step(model, local_val_x, local_val_y, val_config)

        if rank == 0:
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"\nEpoch [{epoch+1}/{cfg.training.num_epochs}] Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  MRR:        {topk_acc.get('MRR', 0):.4f}")
            print(f"  Top@1 Acc:  {topk_acc.get(1, 0)*100:.2f}%")
            print(f"  Top@5 Acc:  {topk_acc.get(5, 0)*100:.2f}%")
            print(f"  Top@10 Acc: {topk_acc.get(10, 0)*100:.2f}%\n")

        if distributed:
            dist.barrier()

    # Save model
    save_model(cfg, model, rank)

# ===================================================================
# Main Entry Point with Hydra
# ===================================================================
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""

    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Running in distributed mode. Rank: {rank}, World Size: {world_size}")
        train(cfg, rank, world_size, distributed=True)
        dist.destroy_process_group()
    else:
        print("Running in single device mode")
        train(cfg, distributed=False)

if __name__ == "__main__":
    main()