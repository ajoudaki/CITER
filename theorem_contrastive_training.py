import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import math
from typing import Dict, Optional
from tqdm import tqdm
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from numba import njit
import torch.utils.checkpoint as checkpoint

# Set TOKENIZERS_PARALLELISM before any other imports to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes.optim as bnb_optim

import wandb

# Import distributed functions
from distributed_clip import distributed_train_step, trivial_contrastive_step, distributed_validate_step

# Global rank for simplified logging
global_rank = 0

# ===================================================================
# Numba-Optimized Helper Function for Pair Generation (Unchanged)
# ===================================================================
@njit(cache=True)
def _generate_shuffled_pairs_numba(paper_lengths: np.ndarray, seed: int) -> np.ndarray:
    """A Numba-JIT compiled function to generate and shuffle theorem pairs efficiently."""
    np.random.seed(seed)
    total_num_pairs = 0
    for length in paper_lengths:
        total_num_pairs += (length + 1) // 2

    all_pairs_arr = np.empty((total_num_pairs, 2, 2), dtype=np.int64)
    current_pair_idx = 0

    for paper_idx, num_stmts in enumerate(paper_lengths):
        if num_stmts % 2 != 0:
            stmt_indices = np.empty(num_stmts + 1, dtype=np.int64)
            random_choice = np.random.randint(0, num_stmts)
            stmt_indices[num_stmts] = random_choice
        else:
            stmt_indices = np.empty(num_stmts, dtype=np.int64)

        for i in range(num_stmts):
            stmt_indices[i] = i

        np.random.shuffle(stmt_indices)
        num_paper_pairs = len(stmt_indices) // 2
        for i in range(num_paper_pairs):
            stmt_idx1 = stmt_indices[2 * i]
            stmt_idx2 = stmt_indices[2 * i + 1]
            all_pairs_arr[current_pair_idx][0][0] = paper_idx
            all_pairs_arr[current_pair_idx][0][1] = stmt_idx1
            all_pairs_arr[current_pair_idx][1][0] = paper_idx
            all_pairs_arr[current_pair_idx][1][1] = stmt_idx2
            current_pair_idx += 1

    np.random.shuffle(all_pairs_arr)
    return all_pairs_arr


# ===================================================================
# Dataset Class (Unchanged)
# ===================================================================
class StratifiedTheoremDataset(Dataset):
    """Stratified dataset that ensures each theorem/lemma appears ~once per epoch."""
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512,
                 split: str = 'train', train_ratio: float = 0.8, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed
        self.epoch = 0
        all_papers_statements = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                statements = paper.get('lemmas', []) + paper.get('theorems', [])
                if len(statements) >= 2:
                    all_papers_statements.append(statements)

        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(all_papers_statements))
        rng.shuffle(indices)
        n_train = int(len(all_papers_statements) * train_ratio)
        split_indices = indices[:n_train] if split == 'train' else indices[n_train:]
        self.papers = [all_papers_statements[i] for i in split_indices]
        self.paper_lengths = np.array([len(p) for p in self.papers], dtype=np.int64)
        self.epoch_pairs = np.array([])
        self.reset_epoch()
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{self.split.upper()} set: {len(self.papers)} papers, "
                  f"{len(self.epoch_pairs)} theorem/lemma pairs per epoch.")

    def reset_epoch(self):
        epoch_seed = self.seed + self.epoch
        self.epoch_pairs = _generate_shuffled_pairs_numba(self.paper_lengths, epoch_seed)
        self.epoch += 1

    def __len__(self):
        return len(self.epoch_pairs) if self.split == 'train' else len(self.papers)

    def __getitem__(self, idx: int):
        if self.split == 'eval':
            statements = self.papers[idx % len(self.papers)]
            text_x, text_y = statements[0], statements[1]
        else:
            pair_indices = self.epoch_pairs[idx]
            paper_idx_x, stmt_idx_x = pair_indices[0]
            paper_idx_y, stmt_idx_y = pair_indices[1]
            text_x = self.papers[paper_idx_x][stmt_idx_x]
            text_y = self.papers[paper_idx_y][stmt_idx_y]

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


# ===================================================================
# Model Architectures (Refactored)
# ===================================================================
class CLSPoolingEncoder(nn.Module):
    """Encoder that uses the [CLS] token embedding."""
    def __init__(self, model_name: str, hidden_dim: int, output_dim: int, quantization_config: Optional[DictConfig] = None, **kwargs):
        super().__init__()
        q_config = None
        if quantization_config:
            if isinstance(quantization_config, DictConfig):
                # Handle the dtype conversion for quantization config
                q_config_dict = OmegaConf.to_container(quantization_config, resolve=True)
                if 'bnb_4bit_compute_dtype' in q_config_dict and isinstance(q_config_dict['bnb_4bit_compute_dtype'], str):
                    # Convert string dtype to torch dtype
                    if 'bfloat16' in q_config_dict['bnb_4bit_compute_dtype']:
                        q_config_dict['bnb_4bit_compute_dtype'] = torch.bfloat16
                    elif 'float16' in q_config_dict['bnb_4bit_compute_dtype']:
                        q_config_dict['bnb_4bit_compute_dtype'] = torch.float16
                q_config = hydra.utils.instantiate(q_config_dict)
            else:
                # Already instantiated by Hydra
                q_config = quantization_config
        model_kwargs = {
            "quantization_config": q_config,
            "device_map": "auto" if q_config else None,
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        self.base_model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        # REFACTOR: Forward pass is now simpler. Gradient checkpointing is enabled outside.
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(self.projection(cls_embedding), p=2, dim=-1)

class LastTokenEncoder(nn.Module):
    """Encoder that uses the last non-padding token's hidden state."""
    def __init__(self, model_name: str, hidden_dim: int, output_dim: int, quantization_config: Optional[DictConfig] = None, **kwargs):
        super().__init__()
        q_config = None
        if quantization_config:
            if isinstance(quantization_config, DictConfig):
                # Handle the dtype conversion for quantization config
                q_config_dict = OmegaConf.to_container(quantization_config, resolve=True)
                if 'bnb_4bit_compute_dtype' in q_config_dict and isinstance(q_config_dict['bnb_4bit_compute_dtype'], str):
                    # Convert string dtype to torch dtype
                    if 'bfloat16' in q_config_dict['bnb_4bit_compute_dtype']:
                        q_config_dict['bnb_4bit_compute_dtype'] = torch.bfloat16
                    elif 'float16' in q_config_dict['bnb_4bit_compute_dtype']:
                        q_config_dict['bnb_4bit_compute_dtype'] = torch.float16
                q_config = hydra.utils.instantiate(q_config_dict)
            else:
                # Already instantiated by Hydra
                q_config = quantization_config
        model_kwargs = {"quantization_config": q_config}
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        self.base_model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        # REFACTOR: Forward pass is now simpler.
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(sequence_lengths), device=sequence_lengths.device)
        last_token_embedding = last_hidden_state[batch_indices, sequence_lengths, :]
        return F.normalize(self.projection(last_token_embedding), p=2, dim=-1)

# Baseline encoders (Jaccard, MinHash, BM25) are unchanged as they are stateless
class JaccardEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768, **kwargs):
        super().__init__()
        self.vocab_size = min(30000, 50000)
        self.output_dim = output_dim
    def forward(self, input_ids, attention_mask): # ... (implementation unchanged)
        batch_size, device = input_ids.shape[0], input_ids.device
        bow_vectors = torch.zeros(batch_size, self.vocab_size, device=device)
        for i in range(batch_size):
            valid_tokens = input_ids[i][attention_mask[i] == 1]
            valid_tokens = valid_tokens[(valid_tokens > 100) & (valid_tokens < self.vocab_size)]
            if len(valid_tokens) > 0:
                bow_vectors[i].scatter_(0, valid_tokens, 1.0)
        bow_vectors = F.normalize(bow_vectors, p=2, dim=-1)
        if self.vocab_size != self.output_dim:
            if not hasattr(self, 'projection'):
                self.projection = nn.Linear(self.vocab_size, self.output_dim, bias=False).to(device)
                nn.init.orthogonal_(self.projection.weight)
                self.projection.requires_grad_(False)
            bow_vectors = self.projection(bow_vectors)
            bow_vectors = F.normalize(bow_vectors, p=2, dim=-1)
        return bow_vectors

class MinHashEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768, **kwargs):
        super().__init__()
        self.vocab_size = min(30000, 50000)
        self.num_hashes = output_dim
        self.prime = self._find_next_prime(self.vocab_size)
        self.a, self.b, self.hashed_vocab = None, None, None
    def _find_next_prime(self, n: int) -> int: # ... (implementation unchanged)
        def is_prime(num):
            if num <= 1: return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0: return False
            return True
        candidate = n
        while not is_prime(candidate): candidate += 1
        return candidate
    def _init_hash_functions(self, device): # ... (implementation unchanged)
        if self.a is None:
            rand_max = self.prime - 1
            self.a = torch.randint(1, rand_max, (self.num_hashes, 1), device=device, dtype=torch.int64)
            self.b = torch.randint(0, rand_max, (self.num_hashes, 1), device=device, dtype=torch.int64)
            vocab_indices = torch.arange(self.vocab_size, device=device, dtype=torch.int64).unsqueeze(0)
            self.hashed_vocab = (self.a * vocab_indices + self.b) % self.prime
    def forward(self, input_ids, attention_mask): # ... (implementation unchanged)
        batch_size, device = input_ids.shape[0], input_ids.device
        self._init_hash_functions(device)
        bow_vectors = torch.zeros(batch_size, self.vocab_size, device=device)
        for i in range(batch_size):
            valid_tokens = input_ids[i][attention_mask[i] == 1]
            valid_tokens = valid_tokens[(valid_tokens > 100) & (valid_tokens < self.vocab_size)]
            if len(valid_tokens) > 0:
                bow_vectors[i].scatter_(0, valid_tokens, 1.0)
        hashed_vocab_expanded = self.hashed_vocab.unsqueeze(0).expand(batch_size, -1, -1)
        masked_hashes = torch.where(bow_vectors.unsqueeze(1) == 1, hashed_vocab_expanded.float(), float('inf'))
        signatures, _ = torch.min(masked_hashes, dim=2)
        signatures = F.normalize(signatures, p=2, dim=-1)
        return signatures

class BM25Encoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768, **kwargs):
        super().__init__()
        self.vocab_size = min(30000, 50000)
        self.output_dim = output_dim
        self.k1, self.b = 1.2, 0.75
    def forward(self, input_ids, attention_mask): # ... (implementation unchanged)
        batch_size, device = input_ids.shape[0], input_ids.device
        bm25_vectors = torch.zeros(batch_size, self.vocab_size, device=device)
        for i in range(batch_size):
            valid_tokens = input_ids[i][attention_mask[i] == 1]
            valid_tokens = valid_tokens[(valid_tokens > 100) & (valid_tokens < self.vocab_size)]
            if len(valid_tokens) > 0:
                doc_len = len(valid_tokens)
                unique_tokens, counts = torch.unique(valid_tokens, return_counts=True)
                tf = counts.float()
                length_norm = 1.0 + self.b * (doc_len / 100.0 - 1.0)
                bm25_scores = (tf * (self.k1 + 1)) / (tf + self.k1 * max(length_norm, 0.5))
                bm25_vectors[i, unique_tokens] = bm25_scores
        bm25_vectors = F.normalize(bm25_vectors + 1e-10, p=2, dim=-1)
        if self.vocab_size != self.output_dim:
            if not hasattr(self, 'projection'):
                self.projection = nn.Linear(self.vocab_size, self.output_dim, bias=False).to(device)
                nn.init.orthogonal_(self.projection.weight)
                self.projection.requires_grad_(False)
            bm25_vectors = self.projection(bm25_vectors)
            bm25_vectors = F.normalize(bm25_vectors, p=2, dim=-1)
        return bm25_vectors

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
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def init_wandb(cfg: DictConfig, rank: int = 0) -> Optional[object]:
    if not cfg.wandb.enabled or rank != 0: return None
    tags = OmegaConf.to_container(cfg.wandb.tags, resolve=True)
    run_name = cfg.wandb.name or f"{cfg.model.name}_{cfg.dataset.size}_bs{cfg.training.global_batch_size}"
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=run_name,
                     tags=tags, group=cfg.wandb.group, notes=cfg.wandb.notes,
                     mode=cfg.wandb.mode, config=config_dict)
    print(f"Weights & Biases initialized: {wandb.run.url}")
    return run

def log_metrics(metrics: Dict, step: int, prefix: str = ""):
    if wandb.run is not None:
        log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        wandb.log(log_dict, step=step)

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    if all_param > 0:
        print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || "
              f"trainable%: {100 * trainable_params / all_param:.2f}")
    else:
        print("Model has no parameters (baseline model)")
    if wandb.run is not None:
        wandb.config.update({"trainable_params": trainable_params, "total_params": all_param,
                             "trainable_percent": 100 * trainable_params / all_param if all_param > 0 else 0})

def setup_model(cfg: DictConfig, device):
    """REFACTORED: Setup model using direct instantiation from Hydra configs."""
    if global_rank == 0:
        print(f"Instantiating model using target: {cfg.model._target_}")

    # Create a new config with only the necessary fields for model instantiation
    model_config = {'_target_': cfg.model._target_}

    # Add fields that exist in the config
    for field in ['model_name', 'hidden_dim', 'output_dim']:
        if field in cfg.model:
            model_config[field] = cfg.model[field]

    model_config = OmegaConf.create(model_config)

    # If quantization addon is present, add it to the model config
    if 'addons' in cfg.model and 'quantization_config' in cfg.model.addons:
        model_config['quantization_config'] = cfg.model.addons.quantization_config

    # Directly instantiate the base encoder from the model config.
    # Hydra passes all parameters from the model config to the class __init__.
    base_encoder = hydra.utils.instantiate(model_config)

    # Apply LoRA if the config is present and enabled
    lora_enabled = False
    lora_config_dict = None

    # Check for LoRA in the old structure (model.lora.enabled)
    if 'lora' in cfg.model and hasattr(cfg.model.lora, 'enabled') and cfg.model.lora.enabled:
        lora_enabled = True
        lora_config_dict = OmegaConf.to_container(cfg.model.lora, resolve=True)
    # Check for LoRA in the new addons structure
    elif 'addons' in cfg.model and 'lora' in cfg.model.addons:
        # Check if lora_enabled flag is set
        if 'lora_enabled' in cfg.model.addons and cfg.model.addons.lora_enabled:
            lora_enabled = True
            lora_config_dict = OmegaConf.to_container(cfg.model.addons.lora, resolve=True)

    if lora_enabled and lora_config_dict:
        if global_rank == 0:
            print("LoRA is ENABLED.")
        for param in base_encoder.base_model.parameters():
            param.requires_grad = False

        # Add the target modules to the LoRA config from the base model config
        lora_config_dict['target_modules'] = list(cfg.model.lora_target_modules)
        # Remove unwanted keys before instantiation
        for key in ['_target_', 'enabled']:
            if key in lora_config_dict:
                del lora_config_dict[key]
        from peft import LoraConfig
        lora_config = LoraConfig(**lora_config_dict)

        base_encoder.base_model = get_peft_model(base_encoder.base_model, lora_config)

    # Enable gradient checkpointing if specified in the training config
    if cfg.training.gradient_checkpointing:
        if hasattr(base_encoder, 'base_model'):
            base_encoder.base_model.gradient_checkpointing_enable()
            if global_rank == 0:
                print("Gradient checkpointing is ENABLED.")
        else:
            if global_rank == 0:
                print("Warning: Gradient checkpointing enabled but model has no 'base_model' attribute.")

    # Ensure projection layer is trainable if it exists
    if hasattr(base_encoder, 'projection'):
        for param in base_encoder.projection.parameters():
            param.requires_grad = True

    model = TheoremContrastiveModel(base_encoder, max_length=cfg.training.max_length)
    return model.to(device)

def prepare_validation_data(val_dataset, device, distributed, rank):
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
        if distributed:
            val_objects = [val_x_packed, val_y_packed]
    if distributed:
        dist.broadcast_object_list(val_objects, src=0)
        val_x_packed, val_y_packed = val_objects
    return val_x_packed, val_y_packed

def save_model(cfg: DictConfig, model, rank):
    if rank != 0: return
    output_dir = Path(cfg.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    if 'lora' in cfg.model and cfg.model.lora.enabled and cfg.output.save_lora:
        adapter_path = output_dir / f'{cfg.model.name}_lora_adapters'
        projection_path = output_dir / f'{cfg.model.name}_projection.pt'
        model_to_save.encoder_x.base_encoder.base_model.save_pretrained(str(adapter_path))
        torch.save(model_to_save.encoder_x.base_encoder.projection.state_dict(), projection_path)
        print(f"LoRA adapters saved to '{adapter_path}' and projection to '{projection_path}'")
    else:
        model_path = output_dir / f'{cfg.model.name}_contrastive_model.pt'
        torch.save(model_to_save.state_dict(), model_path)
        print(f"Model saved to {model_path}")

# ===================================================================
# Training Function (Refactored)
# ===================================================================
def train(cfg: DictConfig, rank: int = 0, world_size: int = 1, distributed: bool = False):
    device = torch.device(f'cuda:{rank}') if distributed and torch.cuda.is_available() else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if distributed: torch.cuda.set_device(rank)
    global global_rank
    global_rank = rank

    wandb_run = init_wandb(cfg, rank)
    if rank == 0:
        print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    dataset_path = Path(cfg.dataset.base_path) / f"{cfg.dataset.size}.jsonl"
    train_dataset = StratifiedTheoremDataset(str(dataset_path), tokenizer, max_length=cfg.training.max_length, split='train')
    val_dataset = StratifiedTheoremDataset(str(dataset_path), tokenizer, max_length=cfg.training.max_length, split='eval')

    if distributed:
        local_batch_size = cfg.training.global_batch_size // world_size
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        local_batch_size = cfg.training.global_batch_size
        train_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=local_batch_size, sampler=train_sampler,
                              shuffle=False, num_workers=cfg.runtime.num_workers, pin_memory=True, drop_last=True)

    model = setup_model(cfg, device)
    if rank == 0: print_trainable_parameters(model)
    if distributed: model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = None
    if len(trainable_params) > 0:
        # REFACTOR: Instantiate optimizer directly from config
        if rank == 0: print(f"Using optimizer: {cfg.optimizer._target_}")
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=trainable_params)
    else:
        if rank == 0: print("No trainable parameters found. Running validation only.")

    scheduler = None
    if optimizer is not None:
        # REFACTOR: Instantiate scheduler directly from config
        num_training_steps = len(train_loader) * cfg.training.num_epochs
        if rank == 0: print(f"Scheduler: {cfg.training.scheduler._target_} over {num_training_steps} steps.")
        scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer, num_training_steps=num_training_steps)

    use_amp = cfg.training.use_amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if rank == 0: print(f"Mixed precision training {'ENABLED' if use_amp else 'DISABLED'}")

    val_x_packed, val_y_packed = prepare_validation_data(val_dataset, device, distributed, rank)

    if optimizer is None: # Baseline model validation-only logic (unchanged)
        if rank == 0: print("No trainable parameters - running validation only.")
        model.eval()
        N_val, val_world_size = val_x_packed.shape[0], (world_size if distributed else 1)
        C_val = N_val // val_world_size
        start, end = rank * C_val, (rank + 1) * C_val
        val_config = {'GLOBAL_BATCH_SIZE': N_val, 'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
                      'STREAM_CHUNK_SIZE': cfg.training.stream_chunk_size, 'TAU': cfg.training.tau}
        val_loss, val_metrics = distributed_validate_step(model, val_x_packed[start:end].to(device),
                                                          val_y_packed[start:end].to(device), val_config,
                                                          k_vals=cfg.training.get('k_vals', [1, 5, 10]))
        if rank == 0:
            print(f"\nValidation Results:\nVal Loss: {val_loss:.4f}")
            print(f"MRR: {val_metrics.get('MRR', 0):.4f}")
            # Print all k values in a formatted way
            k_vals = cfg.training.get('k_vals', [1, 5, 10])
            for k in k_vals:
                if k in val_metrics:
                    print(f"Top@{k}: {val_metrics[k]*100:.2f}%")
        return

    global_step = 0
    validation_interval = cfg.training.get('validation_interval', 0)

    # In the `train` function, find the `run_validation` helper function
    def run_validation(step_num, epoch_num=None): # Validation helper (unchanged)
        model.eval()
        N_val, val_world_size = val_x_packed.shape[0], (world_size if distributed else 1)
        C_val = N_val // val_world_size
        start, end = rank * C_val, (rank + 1) * C_val
        val_config = {'GLOBAL_BATCH_SIZE': C_val * val_world_size, 'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
                      'STREAM_CHUNK_SIZE': cfg.training.stream_chunk_size, 'TAU': cfg.training.tau}
        
        # FIX: Add the `autocast` context manager here
        with torch.no_grad(), autocast(enabled=use_amp):
            val_loss, topk_acc = distributed_validate_step(model, val_x_packed[start:end].to(device),
                                                           val_y_packed[start:end].to(device), val_config,
                                                           k_vals=cfg.training.get('k_vals', [1, 5, 10]))
        
        if rank == 0:
            prefix = f"\n[Epoch {epoch_num+1}] Validation" if epoch_num is not None else "\nValidation"
            print(f"{prefix} at step {step_num}:")
            print(f"  Val Loss:    {val_loss:.4f}\n  MRR:         {topk_acc.get('MRR', 0):.4f}")
            # Print all configured k values
            k_vals_to_report = cfg.training.get('k_vals', [1, 5, 10])
            acc_str = "  ".join([f"Top@{k}: {topk_acc.get(k, 0)*100:.2f}%" for k in k_vals_to_report if k in topk_acc])
            print(f"  {acc_str}\n")
            # Log all k values to wandb
            metrics = {'loss': val_loss, 'mrr': topk_acc.get('MRR', 0)}
            for k in k_vals_to_report:
                if k in topk_acc:
                    metrics[f'top{k}_acc'] = topk_acc.get(k, 0)
            log_metrics(metrics, step=step_num, prefix='val')
        model.train()
        return val_loss, topk_acc

    for epoch in range(cfg.training.num_epochs): # Main training loop (unchanged)
        train_dataset.reset_epoch()
        model.train()
        if distributed and train_sampler: train_sampler.set_epoch(epoch)
        total_loss, num_batches = 0, 0
        pbar = tqdm(train_loader, disable=(rank!=0), desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        for batch in pbar:
            x_packed = torch.cat([batch['input_ids_x'], batch['attention_mask_x']], dim=1).to(device)
            y_packed = torch.cat([batch['input_ids_y'], batch['attention_mask_y']], dim=1).to(device)
            train_config = {'GLOBAL_BATCH_SIZE': cfg.training.global_batch_size, 'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
                            'STREAM_CHUNK_SIZE': cfg.training.stream_chunk_size, 'TAU': cfg.training.tau}
            loss = distributed_train_step(model, optimizer, x_packed, y_packed, train_config, scaler) if distributed else \
                   trivial_contrastive_step(model, optimizer, x_packed, y_packed, train_config, scaler)
            scheduler.step()
            total_loss += loss
            num_batches += 1
            global_step += 1
            if rank == 0:
                pbar.set_postfix(loss=f'{loss:.4f}', lr=f'{scheduler.get_last_lr()[0]:.2e}')
                log_metrics({'loss': loss, 'learning_rate': scheduler.get_last_lr()[0]}, step=global_step, prefix='train')
            if validation_interval > 0 and global_step % validation_interval == 0:
                run_validation(global_step, epoch_num=epoch)
                if distributed: dist.barrier()

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        val_loss, topk_acc = run_validation(global_step, epoch_num=epoch)
        if rank == 0:
            print(f"\nEpoch [{epoch+1}/{cfg.training.num_epochs}] Summary:\n  Train Loss: {avg_train_loss:.4f}\n  Val Loss:   {val_loss:.4f}")
            print(f"  MRR:        {topk_acc.get('MRR', 0):.4f}")
            # Print key accuracy metrics
            k_vals_to_report = cfg.training.get('k_vals', [1, 5, 10])
            key_ks = [k for k in [1, 5, 10] if k in k_vals_to_report and k in topk_acc]
            if key_ks:
                acc_str = "  ".join([f"Top@{k}: {topk_acc.get(k, 0)*100:.2f}%" for k in key_ks])
                print(f"  {acc_str}\n")
            log_metrics({'epoch': epoch + 1, 'avg_train_loss': avg_train_loss}, step=global_step)
        if distributed: dist.barrier()

    save_model(cfg, model, rank)
    if wandb_run is not None: wandb.finish()


# ===================================================================
# Main Entry Point with Hydra (Unchanged)
# ===================================================================
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        print(f"Running in distributed mode. Rank: {rank}, World Size: {world_size}")
        train(cfg, rank, world_size, distributed=True)
        dist.destroy_process_group()
    else:
        print("Running in single device mode")
        train(cfg, distributed=False)

if __name__ == "__main__":
    main()