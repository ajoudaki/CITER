import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from typing import Dict, Optional
from tqdm import tqdm
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler

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
import wandb

# Import distributed functions
from distributed_clip import distributed_train_step, trivial_contrastive_step, distributed_validate_step

# ===================================================================
# Dataset Classes
# ===================================================================
class StratifiedTheoremDataset(Dataset):
    """Stratified dataset that ensures each theorem/lemma appears ~once per epoch"""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512,
                 split: str = 'train', train_ratio: float = 0.8, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed

        # Load papers
        all_papers = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                statements = paper.get('lemmas', []) + paper.get('theorems', [])
                if len(statements) >= 2:
                    all_papers.append(statements)

        # Split train/eval
        random.seed(seed)
        random.shuffle(all_papers)
        n_train = int(len(all_papers) * train_ratio)
        self.papers = all_papers[:n_train] if split == 'train' else all_papers[n_train:]

        # Initialize paper states for stratified sampling
        self.reset_epoch()

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{split.upper()} set: {len(self.papers)} papers, "
                  f"~{self.total_pairs} theorem/lemma pairs")

    def reset_epoch(self):
        """Reset paper states for new epoch - call at epoch start"""
        self.rng = random.Random(self.seed + (self.current_epoch if hasattr(self, 'current_epoch') else 0))
        self.current_epoch = getattr(self, 'current_epoch', 0) + 1

        # Prepare each paper's theorem pairs
        self.paper_queues = []
        self.total_pairs = 0

        for paper_statements in self.papers:
            # Shuffle statements for this epoch
            statements = paper_statements.copy()
            self.rng.shuffle(statements)

            # Make even by repeating one if odd
            if len(statements) % 2 == 1:
                statements.append(self.rng.choice(statements))

            # Create pairs queue for this paper
            pairs = [(statements[i], statements[i+1])
                     for i in range(0, len(statements), 2)]
            self.paper_queues.append(pairs)
            self.total_pairs += len(pairs)

        # Track active papers (those with remaining pairs)
        self.active_papers = set(range(len(self.papers)))
        self.consumed_pairs = 0

    def __len__(self):
        """Return number of batches we can create"""
        return self.total_pairs if self.split == 'train' else len(self.papers)

    def __getitem__(self, idx):
        """Get a single pair - DataLoader will batch these"""
        if self.split == 'eval':
            # For eval, use fixed first two statements
            statements = self.papers[idx % len(self.papers)]
            text_x, text_y = (statements[0], statements[1]) if len(statements) >= 2 else (statements[0], statements[0])
        else:
            # For train, return next unconsumed pair
            # Select random active paper
            if not self.active_papers:
                # Shouldn't happen if DataLoader respects __len__
                self.reset_epoch()

            paper_idx = self.rng.choice(list(self.active_papers))
            if self.paper_queues[paper_idx]:
                text_x, text_y = self.paper_queues[paper_idx].pop(0)
                if not self.paper_queues[paper_idx]:
                    self.active_papers.remove(paper_idx)
            else:
                # Fallback - shouldn't happen
                text_x = text_y = self.papers[0][0]

        # Tokenize
        tokens_x = self.tokenizer(text_x, padding='max_length', truncation=True,
                                   max_length=self.max_length, return_tensors='pt')
        tokens_y = self.tokenizer(text_y, padding='max_length', truncation=True,
                                   max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids_x': tokens_x['input_ids'].squeeze(0),
            'attention_mask_x': tokens_x['attention_mask'].squeeze(0),
            'input_ids_y': tokens_y['input_ids'].squeeze(0),
            'attention_mask_y': tokens_y['attention_mask'].squeeze(0)
        }

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
def init_wandb(cfg: DictConfig, rank: int = 0) -> Optional[object]:
    """Initialize Weights & Biases logging."""
    if not cfg.wandb.enabled or rank != 0:
        return None

    # Resolve interpolations in tags
    tags = OmegaConf.to_container(cfg.wandb.tags, resolve=True)

    # Auto-generate run name if not provided
    run_name = cfg.wandb.name
    if run_name is None:
        run_name = f"{cfg.model.name}_{cfg.dataset.size}_bs{cfg.training.global_batch_size}"

    # Convert config to dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Initialize wandb
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        tags=tags,
        group=cfg.wandb.group,
        notes=cfg.wandb.notes,
        mode=cfg.wandb.mode,
        config=config_dict
    )

    print(f"Weights & Biases initialized: {wandb.run.url}")
    return run

def log_metrics(metrics: Dict, step: int, prefix: str = ""):
    """Log metrics to wandb if enabled."""
    if wandb.run is not None:
        log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        wandb.log(log_dict, step=step)

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || "
          f"trainable%: {100 * trainable_params / all_param:.2f}")

    # Log to wandb
    if wandb.run is not None:
        wandb.config.update({
            "trainable_params": trainable_params,
            "total_params": all_param,
            "trainable_percent": 100 * trainable_params / all_param
        })

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
        # Convert target_modules to list if it's a ListConfig
        target_modules = list(cfg.model.lora_target_modules) if OmegaConf.is_list(cfg.model.lora_target_modules) else cfg.model.lora_target_modules

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.training.lora.r,
            lora_alpha=cfg.training.lora.lora_alpha,
            lora_dropout=cfg.training.lora.lora_dropout,
            target_modules=target_modules,
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
class WarmupScheduler:
    """Simple warmup scheduler for batch size transitions."""
    def __init__(self, warmup_steps, main_batch_size):
        self.warmup_steps = warmup_steps or []
        self.main_batch_size = main_batch_size
        self.current_step = 0
        self.warmup_phase = 0
        self.total_warmup_steps = sum(steps for _, steps in self.warmup_steps)

    def get_batch_size(self):
        """Get current batch size based on training step."""
        if not self.warmup_steps:
            return self.main_batch_size

        # Check if we're still in warmup
        steps_so_far = 0
        for batch_size, steps in self.warmup_steps:
            if self.current_step < steps_so_far + steps:
                return batch_size
            steps_so_far += steps

        # Warmup complete, use main batch size
        return self.main_batch_size

    def step(self):
        """Increment step counter."""
        self.current_step += 1
        return self.get_batch_size()

    def is_warmup_complete(self):
        """Check if warmup phase is complete."""
        return self.current_step >= self.total_warmup_steps

def train(cfg: DictConfig, rank: int = 0, world_size: int = 1, distributed: bool = False):
    """Main training function (SIMPLIFIED for fixed batch size)."""
    device = torch.device(f'cuda:{rank}') if distributed and torch.cuda.is_available() else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if distributed:
        torch.cuda.set_device(rank)

    # Initialize wandb (only on rank 0)
    wandb_run = init_wandb(cfg, rank)

    if rank == 0:
        print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        print(f"Using model: {cfg.model.model_name} with {cfg.model.model_type} strategy.")
        if cfg.training.lora.enabled:
            print("LoRA is ENABLED.")

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

    # Choose dataset class based on sampling strategy
    sampling_strategy = cfg.dataset.get('sampling', 'random')
    DatasetClass = StratifiedTheoremDataset if sampling_strategy == 'stratified' else TheoremLemmaDataset

    if rank == 0:
        print(f"Using {sampling_strategy} sampling strategy")

    # Create datasets
    train_dataset = DatasetClass(
        str(dataset_path), tokenizer,
        max_length=cfg.training.max_length,
        split='train',
        train_ratio=cfg.dataset.train_ratio,
        seed=cfg.dataset.seed
    )
    val_dataset = DatasetClass(
        str(dataset_path), tokenizer,
        max_length=cfg.training.max_length,
        split='eval',
        train_ratio=cfg.dataset.train_ratio,
        seed=cfg.dataset.seed
    )

    # Calculate the per-GPU batch size once.
    if distributed:
        if cfg.training.global_batch_size % world_size != 0:
            raise ValueError(f"Global batch size ({cfg.training.global_batch_size}) must be divisible by world size ({world_size}).")
        local_batch_size = cfg.training.global_batch_size // world_size
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        local_batch_size = cfg.training.global_batch_size
        train_sampler = None

    # DataLoader now creates perfectly sized batches for each GPU directly.
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None and not isinstance(train_dataset, StratifiedTheoremDataset)),
        num_workers=cfg.runtime.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Setup model
    model = setup_model(cfg, device)

    if rank == 0:
        print_trainable_parameters(model)

    if distributed:
        model = DDP(model, device_ids=[rank])

    # Setup optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.get('weight_decay', 0.01)
    )

    # Setup mixed precision scaler for GPU training
    use_amp = cfg.training.get('use_amp', True) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if rank == 0 and use_amp:
        print("Mixed precision training ENABLED (fp16)")
    elif rank == 0:
        print("Mixed precision training DISABLED (fp32)")

    # Prepare validation data
    val_x_packed, val_y_packed = prepare_validation_data(val_dataset, device, distributed, rank)

    # Track global step for wandb
    global_step = 0

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        # Reset stratified dataset for new epoch if applicable
        if isinstance(train_dataset, StratifiedTheoremDataset):
            train_dataset.reset_epoch()
            if rank == 0:
                print(f"Epoch {epoch+1}: Reset stratified sampling, "
                      f"{len(train_dataset.active_papers)} active papers")

        model.train()
        if distributed and train_sampler:
            train_sampler.set_epoch(epoch)

        total_loss, num_batches = 0, 0
        pbar = tqdm(train_loader, disable=(rank!=0), desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")

        # Simplified training loop
        for batch in pbar:
            # DataLoader provides a perfectly sized batch for this GPU. No accumulation needed.
            x_packed = torch.cat([batch['input_ids_x'], batch['attention_mask_x']], dim=1).to(device)
            y_packed = torch.cat([batch['input_ids_y'], batch['attention_mask_y']], dim=1).to(device)

            train_config = {
                'GLOBAL_BATCH_SIZE': cfg.training.global_batch_size,
                'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
                'STREAM_CHUNK_SIZE': cfg.training.stream_chunk_size,
                'TAU': cfg.training.tau
            }

            if distributed:
                loss = distributed_train_step(model, optimizer, x_packed, y_packed, train_config, scaler)
            else:
                loss = trivial_contrastive_step(model, optimizer, x_packed, y_packed, train_config, scaler)

            total_loss += loss
            num_batches += 1
            global_step += 1

            # Log training metrics
            if rank == 0:
                pbar.set_postfix(loss=f'{loss:.4f}')
                log_metrics({
                    'loss': loss,
                    'learning_rate': cfg.training.lr,
                    'batch_size': cfg.training.global_batch_size,
                }, step=global_step, prefix='train')

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

            # Log validation metrics
            log_metrics({
                'loss': val_loss,
                'mrr': topk_acc.get('MRR', 0),
                'top1_acc': topk_acc.get(1, 0),
                'top5_acc': topk_acc.get(5, 0),
                'top10_acc': topk_acc.get(10, 0)
            }, step=global_step, prefix='val')

            # Log epoch summary
            log_metrics({
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss
            }, step=global_step)

        if distributed:
            dist.barrier()

    # Save model
    save_model(cfg, model, rank)

    # Close wandb
    if wandb_run is not None:
        wandb.finish()

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