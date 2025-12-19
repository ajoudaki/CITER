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
import numpy as np  # Import numpy for the new implementation


# Set TOKENIZERS_PARALLELISM before any other imports to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import bitsandbytes.optim as bnb_optim

import wandb


# Import distributed functions
from distributed_clip import (
    distributed_train_step,
    trivial_contrastive_step,
    distributed_validate_step,
    validate_metrics,
    compute_and_gather_embeddings,
    compute_retrieval_metrics as compute_retrieval_metrics # Use alias to fix name mismatch
)

# Import preprocessed graph dataset
from graph_contrastive.dataset import PreprocessedGraphDataset

from numba import njit # Import the Numba JIT compiler
import torch.utils.checkpoint as checkpoint # <-- ADD THIS IMPORT


# ===================================================================
# Numba-Optimized Helper Function for Pair Generation
# ===================================================================
@njit(cache=True)
def _generate_shuffled_pairs_numba(paper_lengths: np.ndarray, seed: int) -> np.ndarray:
    """
    A Numba-JIT compiled function to generate and shuffle theorem pairs efficiently.
    ... (Numba function content is unchanged) ...
    """
    # Numba requires its own random seed initialization
    np.random.seed(seed)

    # 1. Pre-calculate the total number of pairs to pre-allocate the final array
    total_num_pairs = 0
    for length in paper_lengths:
        # This is an efficient way to calculate ceil(length / 2) using integer division
        total_num_pairs += (length + 1) // 2

    # 2. Pre-allocate the memory for the output array. This is much faster than appending.
    all_pairs_arr = np.empty((total_num_pairs, 2, 2), dtype=np.int64)
    current_pair_idx = 0

    # 3. Iterate through each paper to generate its pairs
    for paper_idx, num_stmts in enumerate(paper_lengths):

        # If there's an odd number of statements, we need to add a duplicate
        if num_stmts % 2 != 0:
            stmt_indices = np.empty(num_stmts + 1, dtype=np.int64)
            # Pick a random statement to duplicate
            random_choice = np.random.randint(0, num_stmts)
            stmt_indices[num_stmts] = random_choice
        else:
            stmt_indices = np.empty(num_stmts, dtype=np.int64)

        # Populate the indices array
        for i in range(num_stmts):
            stmt_indices[i] = i

        # Shuffle the indices in-place and form pairs
        np.random.shuffle(stmt_indices)

        num_paper_pairs = len(stmt_indices) // 2
        for i in range(num_paper_pairs):
            stmt_idx1 = stmt_indices[2 * i]
            stmt_idx2 = stmt_indices[2 * i + 1]

            # Store the pair directly into the pre-allocated array
            all_pairs_arr[current_pair_idx][0][0] = paper_idx
            all_pairs_arr[current_pair_idx][0][1] = stmt_idx1
            all_pairs_arr[current_pair_idx][1][0] = paper_idx
            all_pairs_arr[current_pair_idx][1][1] = stmt_idx2

            current_pair_idx += 1

    # 4. Shuffle the entire array of pairs globally
    np.random.shuffle(all_pairs_arr)
    return all_pairs_arr


# ===================================================================
# Dataset Class (Re-implemented for efficiency)
# ===================================================================
class StratifiedTheoremDataset(Dataset):
    """
    Stratified dataset that ensures each theorem/lemma appears ~once per epoch.
    ... (This class is unchanged) ...
    """
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512,
                 split: str = 'train', train_ratio: float = 0.8, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed
        self.epoch = 0

        # Load papers' statements into a list of lists
        all_papers_statements = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                statements = paper.get('lemmas', []) + paper.get('theorems', [])
                if len(statements) >= 2:
                    all_papers_statements.append(statements)

        # Split train/eval using a reproducible shuffle
        # Use numpy's older RandomState for compatibility if needed, but default_rng is fine here
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(all_papers_statements))
        rng.shuffle(indices)
        n_train = int(len(all_papers_statements) * train_ratio)
        split_indices = indices[:n_train] if split == 'train' else indices[n_train:]

        self.papers = [all_papers_statements[i] for i in split_indices]

        # Create a NumPy array of paper lengths for the Numba function
        self.paper_lengths = np.array([len(p) for p in self.papers], dtype=np.int64)

        self.epoch_pairs = np.array([]) # Will be populated in reset_epoch
        self.reset_epoch()

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{self.split.upper()} set: {len(self.papers)} papers, "
                  f"{len(self.epoch_pairs)} theorem/lemma pairs per epoch.")

    def reset_epoch(self):
        """
        Generates and shuffles all possible statement pairs for the upcoming epoch
        by calling the high-performance Numba-compiled helper function.
        """
        # The core logic is now offloaded to the fast, compiled function
        epoch_seed = self.seed + self.epoch
        self.epoch_pairs = _generate_shuffled_pairs_numba(self.paper_lengths, epoch_seed)
        self.epoch += 1

    def __len__(self):
        """Return the total number of pairs available for the epoch."""
        return len(self.epoch_pairs) if self.split == 'train' else len(self.papers)

    def __getitem__(self, idx: int):
        """
        Retrieves a pre-computed pair by index and tokenizes it.
        This method is now a simple, fast lookup.
        """
        if self.split == 'eval':
            # For eval, use fixed first two statements for consistency
            statements = self.papers[idx % len(self.papers)]
            text_x, text_y = statements[0], statements[1]
        else:
            # For train, retrieve the pre-shuffled pair indices
            pair_indices = self.epoch_pairs[idx]
            paper_idx_x, stmt_idx_x = pair_indices[0]
            paper_idx_y, stmt_idx_y = pair_indices[1]
            text_x = self.papers[paper_idx_x][stmt_idx_x]
            text_y = self.papers[paper_idx_y][stmt_idx_y]

        # Tokenize the selected text pair
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
# New Dataset for All-vs-All Validation
# ===================================================================
# (In theorem_contrastive_training.py)

class AllStatementsDataset(Dataset):
    """
    A dataset that loads *all* individual statements (lemmas/theorems)
    from a given split ('train', 'eval', or 'all') and returns each
    statement with its paper_id.
    
    --- MODIFIED ---
    Now stores full metadata for retrieval.
    """
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512,
                 split: str = 'eval', train_ratio: float = 0.8, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed

        # --- MODIFIED: Load full paper objects, not just statements ---
        all_papers_data = [] 
        with open(jsonl_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                statements = paper.get('lemmas', []) + paper.get('theorems', [])
                if len(statements) >= 1:
                    all_papers_data.append(paper) # Store the whole paper dict
        # --- END MODIFIED ---

        # Split train/eval/all using a reproducible shuffle
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(all_papers_data)) # Use new list
        rng.shuffle(indices)
        n_train = int(len(all_papers_data) * train_ratio)
        
        if split == 'train':
            split_indices = indices[:n_train]
        elif split == 'eval':
            split_indices = indices[n_train:]
        elif split == 'all':
            split_indices = indices
        else:
            raise ValueError(f"Unknown split '{split}'. Must be 'train', 'eval', or 'all'.")

        # self.papers is now a list of paper *objects* (dicts)
        self.papers = [all_papers_data[i] for i in split_indices]
        
        # --- MODIFIED: Flatten statements and store all metadata ---
        self.metadata = [] 
        
        for paper_id, paper in enumerate(self.papers):
            paper_title = paper.get('title', 'Untitled')
            arxiv_id = paper.get('arxiv_id', 'Unknown')
            
            for i, stmt in enumerate(paper.get('lemmas', [])):
                self.metadata.append({
                    'text': stmt,
                    'paper_id': paper_id, # Local split paper_id
                    'paper_title': paper_title,
                    'arxiv_id': arxiv_id,
                    'type': 'lemma',
                    'stmt_idx': i,
                    'uid': f"{paper_id}.lemma.{i}"
                })
            for i, stmt in enumerate(paper.get('theorems', [])):
                self.metadata.append({
                    'text': stmt,
                    'paper_id': paper_id,
                    'paper_title': paper_title,
                    'arxiv_id': arxiv_id,
                    'type': 'theorem',
                    'stmt_idx': i,
                    'uid': f"{paper_id}.theorem.{i}"
                })
        # --- END MODIFIED ---

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[AllStatementsDataset] {self.split.upper()} set: "
                  f"{len(self.papers)} papers, "
                  f"{len(self.metadata)} total statements.") # Use self.metadata

    def __len__(self):
        """Return the total number of individual statements."""
        return len(self.metadata) # Use self.metadata

    def __getitem__(self, idx: int):
        """
        Retrieves a single statement and its paper_id.
        """
        item_data = self.metadata[idx] # Get metadata dict
        text = item_data['text']
        paper_id = item_data['paper_id']

        # Tokenize the statement text
        tokens = self.tokenizer(text, padding='max_length', truncation=True,
                                max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'paper_id': torch.tensor(paper_id, dtype=torch.long)
        }

# ===================================================================
# Model Architectures 
# ===================================================================
class CLSPoolingEncoder(nn.Module):
    """Encoder that uses the [CLS] token embedding with optional gradient checkpointing."""
    def __init__(self, model_name: str, hidden_dim: int, output_dim: int, use_gradient_checkpointing: bool = False, quantization_config=None):
        super().__init__()
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto" if quantization_config else None,
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        self.base_model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, input_ids, attention_mask):
        if self.use_gradient_checkpointing:
            outputs = checkpoint.checkpoint(
                self.base_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_reentrant=False
            )
        else:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # --- ADDED .half() for fp16 consistency ---
        projected = self.projection(cls_embedding.half())
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized.half() # Return fp16

class LastTokenEncoder(nn.Module):
    """Encoder that uses the last non-padding token's hidden state with optional gradient checkpointing."""
    def __init__(self, model_name: str, hidden_dim: int, output_dim: int, use_gradient_checkpointing: bool = False, quantization_config=None):
        super().__init__()
        model_kwargs = {
            "quantization_config": quantization_config,
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        self.base_model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, input_ids, attention_mask):
        if self.use_gradient_checkpointing:
            outputs = checkpoint.checkpoint(
                self.base_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_reentrant=False
            )
        else:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(sequence_lengths), device=sequence_lengths.device)
        last_token_embedding = last_hidden_state[batch_indices, sequence_lengths, :]
        
        # This code already contains the fp16 fix
        projected = self.projection(last_token_embedding.half())
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized.half()


class JaccardEncoder(nn.Module):
    """... (Baseline class unchanged) ..."""
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768, **kwargs):
        super().__init__()
        self.vocab_size = min(30000, 50000)
        self.output_dim = output_dim
    def forward(self, input_ids, attention_mask):
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
    """... (Baseline class unchanged) ..."""
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768, **kwargs):
        super().__init__()
        self.vocab_size = min(30000, 50000)
        self.num_hashes = output_dim
        self.prime = self._find_next_prime(self.vocab_size)
        self.a = None
        self.b = None
        self.hashed_vocab = None
    def _find_next_prime(self, n: int) -> int:
        def is_prime(num):
            if num <= 1: return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0: return False
            return True
        candidate = n
        while not is_prime(candidate):
            candidate += 1
        return candidate
    def _init_hash_functions(self, device):
        if self.a is None:
            rand_max = self.prime - 1
            self.a = torch.randint(1, rand_max, (self.num_hashes, 1), device=device, dtype=torch.int64)
            self.b = torch.randint(0, rand_max, (self.num_hashes, 1), device=device, dtype=torch.int64)
            vocab_indices = torch.arange(self.vocab_size, device=device, dtype=torch.int64).unsqueeze(0)
            self.hashed_vocab = (self.a * vocab_indices + self.b) % self.prime
    def forward(self, input_ids, attention_mask):
        batch_size, device = input_ids.shape[0], input_ids.device
        self._init_hash_functions(device)
        bow_vectors = torch.zeros(batch_size, self.vocab_size, device=device)
        for i in range(batch_size):
            valid_tokens = input_ids[i][attention_mask[i] == 1]
            valid_tokens = valid_tokens[(valid_tokens > 100) & (valid_tokens < self.vocab_size)]
            if len(valid_tokens) > 0:
                bow_vectors[i].scatter_(0, valid_tokens, 1.0)
        hashed_vocab_expanded = self.hashed_vocab.unsqueeze(0).expand(batch_size, -1, -1)
        masked_hashes = torch.where(
            bow_vectors.unsqueeze(1) == 1,
            hashed_vocab_expanded.float(),
            float('inf')
        )
        signatures, _ = torch.min(masked_hashes, dim=2)
        signatures = F.normalize(signatures, p=2, dim=-1)
        return signatures


class BM25Encoder(nn.Module):
    """... (Baseline class unchanged) ..."""
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768, **kwargs):
        super().__init__()
        self.vocab_size = min(30000, 50000)
        self.output_dim = output_dim
        self.k1 = 1.2
        self.b = 0.75
    def forward(self, input_ids, attention_mask):
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


ENCODER_REGISTRY = {
    'cls_pooling': CLSPoolingEncoder,
    'last_token_pooling': LastTokenEncoder,
    'jaccard': JaccardEncoder,
    'minhash': MinHashEncoder,
    'bm25': BM25Encoder,
}


class EncoderWrapper(nn.Module):
    """... (Class unchanged) ..."""
    def __init__(self, base_encoder, max_length):
        super().__init__()
        self.base_encoder = base_encoder
        self.max_length = max_length

    def forward(self, packed_tensor):
        input_ids = packed_tensor[:, :self.max_length].long()
        attention_mask = packed_tensor[:, self.max_length:].long()
        return self.base_encoder(input_ids, attention_mask)

class TheoremContrastiveModel(nn.Module):
    """... (Class unchanged) ..."""
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
    """... (Function unchanged) ..."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def init_wandb(cfg: DictConfig, rank: int = 0) -> Optional[object]:
    """... (Function unchanged) ..."""
    if not cfg.wandb.enabled or rank != 0:
        return None
    tags = OmegaConf.to_container(cfg.wandb.tags, resolve=True)
    run_name = cfg.wandb.name or f"{cfg.model.name}_{cfg.dataset.size}_bs{cfg.training.global_batch_size}"
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        project=cfg.wandb.project, entity=cfg.wandb.entity, name=run_name,
        tags=tags, group=cfg.wandb.group, notes=cfg.wandb.notes,
        mode=cfg.wandb.mode, config=config_dict
    )
    print(f"Weights & Biases initialized: {wandb.run.url}")
    return run

def log_metrics(metrics: Dict, step: int, prefix: str = ""):
    """... (Function unchanged) ..."""
    if wandb.run is not None:
        log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        wandb.log(log_dict, step=step)

def print_trainable_parameters(model):
    """... (Function unchanged) ..."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    if all_param > 0:
        print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || "
              f"trainable%: {100 * trainable_params / all_param:.2f}")
    else:
        print(f"Model has no parameters (baseline model)")
    if wandb.run is not None:
        wandb.config.update({
            "trainable_params": trainable_params, "total_params": all_param,
            "trainable_percent": 100 * trainable_params / all_param if all_param > 0 else 0
        })

def setup_model(cfg: DictConfig, device):
    """... (Function unchanged) ..."""
    encoder_class = ENCODER_REGISTRY.get(cfg.model.model_type)
    if encoder_class is None:
        raise ValueError(f"Unknown model type: {cfg.model.model_type}")
    model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs = {
        'model_name': model_config_dict.get('model_name'),
        'hidden_dim': model_config_dict.get('hidden_dim'),
        'output_dim': model_config_dict.get('output_dim', cfg.training.get('output_dim', 2048)),
        'use_gradient_checkpointing': cfg.training.get('gradient_checkpointing', False),
    }
    if cfg.training.get("quantization") and cfg.training.quantization.get("enabled"):
        if global_rank == 0:
            print("Quantization is ENABLED.")
        compute_dtype_str = cfg.training.quantization.get("bnb_4bit_compute_dtype", "bfloat16")
        compute_dtype = getattr(torch, compute_dtype_str)
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.training.quantization.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.training.quantization.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif global_rank == 0:
        print("Quantization is DISABLED.")
    base_encoder = encoder_class(**model_kwargs)
    if cfg.training.lora.enabled:
        for param in base_encoder.base_model.parameters():
            param.requires_grad = False
        target_modules = list(cfg.model.lora_target_modules)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=cfg.training.lora.r,
            lora_alpha=cfg.training.lora.lora_alpha, lora_dropout=cfg.training.lora.lora_dropout,
            target_modules=target_modules,
        )
        base_encoder.base_model = get_peft_model(base_encoder.base_model, peft_config)
    if hasattr(base_encoder, 'projection'):
        for param in base_encoder.projection.parameters():
            param.requires_grad = True
    model = TheoremContrastiveModel(base_encoder, max_length=cfg.training.max_length)
    return model.to(device)

def prepare_validation_data(val_dataset, device, distributed, rank, reshuffle=False,
                           max_val_samples=50000, val_batch_size=512, num_workers=4):
    """Prepare validation data using DataLoader for fast parallel loading.

    Args:
        val_dataset: Validation dataset
        device: Target device
        distributed: Whether using distributed training
        rank: Process rank
        reshuffle: Whether to reshuffle the dataset
        max_val_samples: Maximum number of validation samples (from config)
        val_batch_size: Batch size for validation loading (from config)
        num_workers: Number of dataloader workers (from config)
    """
    val_objects = [None, None]
    if rank == 0:
        if reshuffle and hasattr(val_dataset, 'reset_epoch'):
            val_dataset.reset_epoch()

        # Limit validation size for efficiency
        n_samples = min(len(val_dataset), max_val_samples)
        print(f"Preparing validation dataset ({n_samples:,} samples)...")

        # Use DataLoader with workers for parallel tokenization
        from torch.utils.data import DataLoader, Subset
        subset = Subset(val_dataset, range(n_samples))
        loader = DataLoader(subset, batch_size=val_batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

        val_x_list, val_mx_list, val_y_list, val_my_list = [], [], [], []
        for batch in tqdm(loader, desc="Loading validation", disable=False):
            val_x_list.append(batch['input_ids_x'])
            val_mx_list.append(batch['attention_mask_x'])
            val_y_list.append(batch['input_ids_y'])
            val_my_list.append(batch['attention_mask_y'])

        val_x_packed = torch.cat([torch.cat([x, mx], dim=1) for x, mx in zip(val_x_list, val_mx_list)], dim=0)
        val_y_packed = torch.cat([torch.cat([y, my], dim=1) for y, my in zip(val_y_list, val_my_list)], dim=0)

        if distributed:
            val_objects = [val_x_packed, val_y_packed]
    if distributed:
        dist.broadcast_object_list(val_objects, src=0)
        val_x_packed, val_y_packed = val_objects
    return val_x_packed, val_y_packed

def prepare_all_statements_data(val_dataset, device, distributed, rank, world_size: int = 1):
    """... (Function unchanged) ..."""
    objects_to_broadcast = [None, None]
    if rank == 0:
        print("Preparing and distributing all validation statements for all-vs-all eval...")
        data = [val_dataset[i] for i in range(len(val_dataset))]
        all_input_ids = torch.stack([d['input_ids'] for d in data])
        all_attention_masks = torch.stack([d['attention_mask'] for d in data])
        all_paper_ids = torch.stack([d['paper_id'] for d in data])
        all_statements_packed = torch.cat([all_input_ids, all_attention_masks], dim=1)
        num_statements = all_statements_packed.shape[0]
        if distributed and num_statements % world_size != 0:
            remainder = num_statements % world_size
            num_to_pad = world_size - remainder
            print(f"Padding validation set with {num_to_pad} dummy statements "
                  f"to be divisible by world size {world_size}.")
            padding_statements = all_statements_packed[0:1].repeat(num_to_pad, 1)
            padding_ids = torch.full((num_to_pad,), -1, 
                                     dtype=all_paper_ids.dtype, 
                                     device=all_paper_ids.device)
            all_statements_packed = torch.cat([all_statements_packed, padding_statements], dim=0)
            all_paper_ids = torch.cat([all_paper_ids, padding_ids], dim=0)
        if distributed:
            objects_to_broadcast = [all_statements_packed, all_paper_ids]
    if distributed:
        dist.broadcast_object_list(objects_to_broadcast, src=0)
        all_statements_packed, all_paper_ids = objects_to_broadcast
    if not distributed and rank == 0:
        pass
    elif not distributed:
        return None, None
    return all_statements_packed, all_paper_ids

def save_model(cfg: DictConfig, model, rank, epoch: Optional[int] = None):
    """... (Function unchanged) ..."""
    if rank != 0: return
    output_dir = Path(cfg.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    suffix = f'_epoch{epoch}' if epoch is not None else ''
    if cfg.training.lora.enabled and cfg.output.save_lora:
        adapter_path = output_dir / f'{cfg.model.name}_lora_adapters{suffix}'
        projection_path = output_dir / f'{cfg.model.name}_projection{suffix}.pt'
        model_to_save.encoder_x.base_encoder.base_model.save_pretrained(str(adapter_path))
        torch.save(model_to_save.encoder_x.base_encoder.projection.state_dict(), projection_path)
        print(f"LoRA adapters saved to '{adapter_path}' and projection to '{projection_path}'")
    else:
        model_path = output_dir / f'{cfg.model.name}_contrastive_model{suffix}.pt'
        torch.save(model_to_save.state_dict(), model_path)
        print(f"Model saved to {model_path}")

def load_saved_weights(model, load_path: Path, cfg: DictConfig, rank: int = 0):
    """... (Function unchanged) ..."""
    if rank == 0:
        print(f"Loading saved weights from: {load_path}")
    model_to_load = model.module if hasattr(model, 'module') else model
    if cfg.training.lora.enabled:
        adapter_path = load_path / f'{cfg.model.name}_lora_adapters'
        projection_path = load_path / f'{cfg.model.name}_projection.pt'
        if rank == 0:
            print(f"Loading LoRA adapter weights from: {adapter_path}")
        model_to_load.encoder_x.base_encoder.base_model.load_adapter(str(adapter_path), adapter_name="default")
        if rank == 0:
            print(f"Loading projection from: {projection_path}")
        model_to_load.encoder_x.base_encoder.projection.load_state_dict(
            torch.load(projection_path, map_location='cpu', weights_only=False)
        )
        model_to_load.encoder_x.base_encoder.projection = model_to_load.encoder_x.base_encoder.projection.half()
    else:
        model_path = load_path / f'{cfg.model.name}_contrastive_model.pt'
        if rank == 0:
            print(f"Loading full model from: {model_path}")
        model_to_load.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))

    # Barrier to ensure all ranks have loaded weights before proceeding
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("All ranks synchronized after loading weights.")


def compute_embeddings(model, tokenizer, cfg: DictConfig, device, rank: int = 0, world_size: int = 1, distributed: bool = False):
    """... (Function unchanged) ..."""
    import json
    from tqdm import tqdm
    model.eval()
    data_split = cfg.dataset.get('split', 'eval')
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"COMPUTE EMBEDDINGS MODE (Split: {data_split})")
        print(f"{'='*80}")
    dataset_path = Path(cfg.dataset.base_path) / f"{cfg.dataset.size}.jsonl"
    dataset_all = AllStatementsDataset(
        str(dataset_path), tokenizer,
        max_length=cfg.training.max_length,
        split=data_split,
        train_ratio=cfg.dataset.get('train_ratio', 0.8),
        seed=cfg.dataset.get('seed', 42)
    )
    all_statements_packed, all_paper_ids = prepare_all_statements_data(
        dataset_all, device, distributed, rank, world_size
    )
    N_global = all_statements_packed.shape[0]
    C_local = N_global // world_size
    start, end = rank * C_local, (rank + 1) * C_local
    use_amp = cfg.training.get('use_amp', True) and torch.cuda.is_available()
    compute_config = {
        'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
        'USE_AMP': use_amp,
        'DEBUG_GATHER': True  # TEMP: Hardcoded for debugging
    }
    _, Z_all, paper_ids_all = compute_and_gather_embeddings(
        model,
        all_statements_packed[start:end].to(device),
        all_paper_ids[start:end].to(device),
        compute_config,
        rank
    )
    if rank == 0:
        num_original_statements = len(dataset_all)
        if N_global > num_original_statements:
            print(f"Trimming {N_global - num_original_statements} padding statements before saving.")
            Z_all = Z_all[:num_original_statements]
            paper_ids_all = paper_ids_all[:num_original_statements]
        output_dir = Path(cfg.output.save_dir) / 'embeddings' / cfg.dataset.size
        output_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = output_dir / f'{data_split}_embeddings.pt'
        paper_ids_path = output_dir / f'{data_split}_paper_ids.pt'
        info_path = output_dir / f'{data_split}_info.json'
        print(f"Saving embeddings to: {embeddings_path}")
        torch.save(Z_all.cpu(), embeddings_path)
        print(f"Saving paper IDs to: {paper_ids_path}")
        torch.save(paper_ids_all.cpu(), paper_ids_path)
        
        metadata_path = output_dir / f'{data_split}_metadata.jsonl'
        print(f"Saving metadata to: {metadata_path}")
        with open(metadata_path, 'w') as f:
            for i in range(len(Z_all)): # Z_all is already trimmed to num_original_statements
                # Safety check in case dataset changed (unlikely in this scope)
                if i >= len(dataset_all.metadata):
                    break
                f.write(json.dumps(dataset_all.metadata[i]) + '\n')
        
        info = {
            'dataset_size': cfg.dataset.size,
            'split': data_split,
            'num_statements': len(Z_all),
            'embedding_dim': Z_all.shape[1],
            'model_dir': str(cfg.output.save_dir),
            'dataset_path': str(dataset_path)
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ Saved {len(Z_all)} embeddings ({Z_all.shape})")
        print(f"{'='*80}\n")

# ===================================================================
# NEW: Top-K Retrieval Function
# ===================================================================
def retrieve_top_k(model, tokenizer, cfg: DictConfig, device, rank: int):
    """
    Loads queries from a file and retrieves the top-K most similar
    items from the pre-computed embedding database.
    
    This function runs *only* on Rank 0.
    """
    if rank != 0:
        return

    K = cfg.training.retrieve_top_k
    query_file = cfg.training.get('query_file')
    if not query_file:
        raise ValueError("Must provide `+training.query_file=path/to/queries.txt` for retrieval.")
    
    query_file_path = Path(query_file)
    if not query_file_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_file_path}")

    data_split = cfg.dataset.get('split', 'eval')
    
    print(f"\n{'='*80}")
    print(f"RETRIEVAL MODE (K={K}, Split: {data_split})")
    print(f"{'='*80}")

    # --- 1. Load Database Metadata ---
    # We re-load the dataset to get the text and metadata.
    # The modified AllStatementsDataset now loads everything we need.
    print(f"Loading database metadata for split '{data_split}'...")
    dataset_path = Path(cfg.dataset.base_path) / f"{cfg.dataset.size}.jsonl"
    db_dataset = AllStatementsDataset(
        str(dataset_path), tokenizer,
        max_length=cfg.training.max_length,
        split=data_split,
        train_ratio=cfg.dataset.get('train_ratio', 0.8),
        seed=cfg.dataset.get('seed', 42)
    )
    db_metadata = db_dataset.metadata

    # --- 2. Load Database Embeddings ---
    load_path = Path(cfg.training.load_model_path)
    embeddings_path = load_path / 'embeddings' / cfg.dataset.size / f'{data_split}_embeddings.pt'
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Database embeddings not found: {embeddings_path}")
        
    print(f"Loading database embeddings from {embeddings_path}...")
    db_embeddings = torch.load(embeddings_path, map_location=device, weights_only=True).half()

    if len(db_metadata) != len(db_embeddings):
        raise ValueError(
            f"Metadata count ({len(db_metadata)}) does not match "
            f"embedding count ({len(db_embeddings)}). Is your split ('{data_split}') correct?"
        )

    # Filter out zero vectors
    norms = torch.norm(db_embeddings, p=2, dim=1)
    valid_mask = norms > 1e-6
    db_embeddings = db_embeddings[valid_mask]
    db_metadata = [db_metadata[i] for i in range(len(db_metadata)) if valid_mask[i]]

    print(f"Loaded {len(db_embeddings)} database entries (filtered {(~valid_mask).sum().item()} zero vectors).")

    # --- 3. Load and Embed Queries ---
    print(f"Reading queries from: {query_file_path}")
    with open(query_file_path, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    print(f"Embedding {len(queries)} queries...")
    
    model_to_use = model.module if hasattr(model, 'module') else model
    model_to_use.eval()
    
    query_embeddings = []
    use_amp = cfg.training.get('use_amp', True) and torch.cuda.is_available()
    batch_size = cfg.training.micro_batch_size
    
    with torch.no_grad(), autocast(enabled=use_amp):
        for i in tqdm(range(0, len(queries), batch_size), desc="Embedding Queries"):
            batch_queries = queries[i:i+batch_size]
            
            tokens = tokenizer(
                batch_queries, 
                padding='max_length', 
                truncation=True,
                max_length=cfg.training.max_length, 
                return_tensors='pt'
            )
            
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            # Manually pack the tensor [B, 2 * max_length]
            packed_batch = torch.cat([input_ids, attention_mask], dim=1)
            
            # Get embeddings (will be fp16 from our modified encoder)
            q_embeds = model_to_use.encoder_x(packed_batch)
            query_embeddings.append(q_embeds)
            
    query_embeddings = torch.cat(query_embeddings, dim=0)

    # --- 4. Perform Search ---
    print("Searching...")
    # Compute cosine similarity
    # [num_queries, D] @ [D, num_db] -> [num_queries, num_db]
    scores = torch.matmul(query_embeddings, db_embeddings.T)
    
    top_k_scores, top_k_indices = scores.topk(k=K, dim=1)
    
    # Move results to CPU for processing
    top_k_scores = top_k_scores.cpu()
    top_k_indices = top_k_indices.cpu()

    # --- 5. Format and Print Results ---
    results_list = []
    for i in range(len(queries)):
        query_text = queries[i]
        hits = []
        for j in range(K):
            db_index = top_k_indices[i, j].item()
            score = top_k_scores[i, j].item()
            metadata = db_metadata[db_index]
            
            hits.append({
                "rank": j + 1,
                "score": score,
                "text": metadata['text'],
                "type": metadata['type'],
                "paper_id": metadata['paper_id'], # This is the local split ID
                "paper_title": metadata['paper_title'],
                "arxiv_id": metadata['arxiv_id']
            })
        
        results_list.append({
            "query": query_text,
            "top_k_hits": hits
        })

        # Print to console
        print(f"\n{'='*80}")
        print(f"QUERY: {query_text}")
        print(f"{'-'*80}")
        for hit in hits:
            print(f"  RANK {hit['rank']} (Score: {hit['score']:.4f})")
            print(f"  Type: {hit['type']} | Paper: {hit['paper_title']} ({hit['arxiv_id']})")
            print(f"  Text: {hit['text'][:200]}...") # Truncate for readability
        print(f"{'='*80}")
    
    # --- 6. Save Full Results to JSON ---
    output_file = Path(cfg.output.save_dir) / f"retrieval_results_{data_split}_k{K}.json"
    print(f"\nSaving full results for {len(results_list)} queries to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results_list, f, indent=2)


# ===================================================================
# NEW: Pairwise Query Similarity Function
# ===================================================================
def compute_pairwise_similarities(model, tokenizer, cfg: DictConfig, device, rank: int):
    """
    Loads queries from a file, embeds them all, and computes pairwise similarities.

    This function runs *only* on Rank 0.
    """
    if rank != 0:
        return

    query_file = cfg.training.get('query_file')
    if not query_file:
        raise ValueError("Must provide `+training.query_file=path/to/queries.txt` for pairwise similarity computation.")

    query_file_path = Path(query_file)
    if not query_file_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_file_path}")

    print(f"\n{'='*80}")
    print(f"PAIRWISE SIMILARITY MODE")
    print(f"{'='*80}")

    # --- 1. Load Queries ---
    print(f"\nReading queries from: {query_file_path}")
    with open(query_file_path, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(queries)} queries")

    # --- 2. Embed All Queries ---
    print(f"\nEmbedding {len(queries)} queries...")

    model_to_use = model.module if hasattr(model, 'module') else model
    model_to_use.eval()

    query_embeddings = []
    use_amp = cfg.training.get('use_amp', True) and torch.cuda.is_available()
    batch_size = cfg.training.micro_batch_size

    with torch.no_grad(), autocast(enabled=use_amp):
        for i in tqdm(range(0, len(queries), batch_size), desc="Embedding Queries"):
            batch_queries = queries[i:i+batch_size]

            tokens = tokenizer(
                batch_queries,
                padding='max_length',
                truncation=True,
                max_length=cfg.training.max_length,
                return_tensors='pt'
            )

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            # Manually pack the tensor [B, 2 * max_length]
            packed_batch = torch.cat([input_ids, attention_mask], dim=1)

            # Get embeddings (will be fp16 from our modified encoder)
            q_embeds = model_to_use.encoder_x(packed_batch)
            query_embeddings.append(q_embeds)

    query_embeddings = torch.cat(query_embeddings, dim=0)  # [num_queries, D]
    print(f"Embedded {len(query_embeddings)} queries with shape {query_embeddings.shape}")

    # --- 3. Compute Pairwise Similarities ---
    print("\nComputing pairwise similarities...")
    # [num_queries, D] @ [D, num_queries] -> [num_queries, num_queries]
    similarity_matrix = torch.matmul(query_embeddings, query_embeddings.T)

    # Move to CPU and convert to float for processing
    similarity_matrix = similarity_matrix.float().cpu()

    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Min similarity: {similarity_matrix.min().item():.4f}")
    print(f"Max similarity: {similarity_matrix.max().item():.4f}")
    print(f"Mean similarity (excluding diagonal): {(similarity_matrix.sum() - similarity_matrix.trace()).item() / (len(queries) * (len(queries) - 1)):.4f}")

    # --- 4. Format and Print Results ---
    print(f"\n{'='*80}")
    print("PAIRWISE SIMILARITY RESULTS")
    print(f"{'='*80}\n")

    # Create results structure
    results = {
        'num_queries': len(queries),
        'queries': queries,
        'similarity_matrix': similarity_matrix.tolist(),
        'pairwise_results': []
    }

    # Print and save detailed pairwise comparisons
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):  # Only upper triangle (avoid duplicates and self-comparisons)
            similarity = similarity_matrix[i, j].item()

            results['pairwise_results'].append({
                'query_1_idx': i,
                'query_2_idx': j,
                'query_1': queries[i][:100] + '...' if len(queries[i]) > 100 else queries[i],
                'query_2': queries[j][:100] + '...' if len(queries[j]) > 100 else queries[j],
                'similarity': similarity
            })

    # Sort by similarity (highest first)
    results['pairwise_results'].sort(key=lambda x: x['similarity'], reverse=True)

    # Print top 10 most similar pairs
    print("Top 10 Most Similar Query Pairs:")
    print("-" * 80)
    for idx, pair in enumerate(results['pairwise_results'][:10], 1):
        print(f"\n{idx}. Similarity: {pair['similarity']:.4f}")
        print(f"   Query {pair['query_1_idx']}: {pair['query_1']}")
        print(f"   Query {pair['query_2_idx']}: {pair['query_2']}")

    # Print bottom 10 least similar pairs
    print(f"\n{'='*80}")
    print("Top 10 Least Similar Query Pairs:")
    print("-" * 80)
    for idx, pair in enumerate(results['pairwise_results'][-10:], 1):
        print(f"\n{idx}. Similarity: {pair['similarity']:.4f}")
        print(f"   Query {pair['query_1_idx']}: {pair['query_1']}")
        print(f"   Query {pair['query_2_idx']}: {pair['query_2']}")

    # --- 5. Save Results to JSON ---
    output_file = Path(cfg.output.save_dir) / f"pairwise_similarities.json"
    print(f"\n{'='*80}")
    print(f"Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Also save similarity matrix as numpy array for easier analysis
    import numpy as np
    matrix_file = Path(cfg.output.save_dir) / f"similarity_matrix.npy"
    np.save(matrix_file, similarity_matrix.numpy())
    print(f"Saved similarity matrix to: {matrix_file}")

    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"  Total queries: {len(queries)}")
    print(f"  Total pairs: {len(results['pairwise_results'])}")
    print(f"  Highest similarity: {results['pairwise_results'][0]['similarity']:.4f}")
    print(f"  Lowest similarity: {results['pairwise_results'][-1]['similarity']:.4f}")
    print(f"{'='*80}\n")


# ===================================================================
# Main Training Function
# ===================================================================

def train(cfg: DictConfig, rank: int = 0, world_size: int = 1, distributed: bool = False):
    """Main training function with learning rate scheduler."""
    device = torch.device(f'cuda:{rank}') if distributed and torch.cuda.is_available() else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if distributed:
        torch.cuda.set_device(rank)
    global global_rank
    global_rank = rank

    wandb_run = init_wandb(cfg, rank)

    if rank == 0:
        print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        print(f"Using model: {cfg.model.model_name} with {cfg.model.model_type} strategy.")
        if cfg.training.lora.enabled:
            print("LoRA is ENABLED.")
        if cfg.training.get('gradient_checkpointing', False):
            print("Gradient checkpointing is ENABLED (saves memory, slower training).")
        else:
            print("Gradient checkpointing is DISABLED (uses more memory, faster training).")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = Path(cfg.dataset.base_path)  # Directory for preprocessed data
    if rank == 0:
        print(f"Loading preprocessed dataset from: {dataset_path}")
        nodes_file = cfg.dataset.get('nodes_file', 'nodes.arrow')
        if not (dataset_path / nodes_file).exists():
            raise FileNotFoundError(f"Preprocessed dataset not found: {dataset_path}/{nodes_file}")

    # --- Dataset configuration from config ---
    dataset_train_ratio = cfg.dataset.get('train_ratio', 0.9)
    dataset_seed = cfg.dataset.get('seed', 42)
    dataset_use_prompts = cfg.dataset.get('use_prompts', True)
    dataset_nodes_file = cfg.dataset.get('nodes_file', 'nodes.arrow')
    dataset_edges_file = cfg.dataset.get('edges_file', 'edges.npy')

    # --- Datasets for TRAINING (using preprocessed graph data) ---
    DatasetClass = PreprocessedGraphDataset
    train_dataset = DatasetClass(
        str(dataset_path), tokenizer,
        max_length=cfg.training.max_length,
        split='train',
        train_ratio=dataset_train_ratio,
        seed=dataset_seed,
        use_prompts=dataset_use_prompts,
        nodes_file=dataset_nodes_file,
        edges_file=dataset_edges_file
    )
    val_dataset = DatasetClass(
        str(dataset_path), tokenizer,
        max_length=cfg.training.max_length,
        split='eval',
        train_ratio=dataset_train_ratio,
        seed=dataset_seed,
        use_prompts=dataset_use_prompts,
        nodes_file=dataset_nodes_file,
        edges_file=dataset_edges_file
    )

    # --- DataLoader configuration from config ---
    dataloader_cfg = cfg.dataset.get('dataloader', {})
    dl_num_workers = dataloader_cfg.get('num_workers', cfg.runtime.num_workers)
    dl_pin_memory = dataloader_cfg.get('pin_memory', True)
    dl_drop_last = dataloader_cfg.get('drop_last', True)
    dl_prefetch_factor = dataloader_cfg.get('prefetch_factor', 2)

    # --- Validation configuration from config ---
    validation_cfg = cfg.dataset.get('validation', {})
    val_max_samples = validation_cfg.get('max_samples', 50000)
    val_batch_size = validation_cfg.get('batch_size', 512)

    if distributed:
        if cfg.training.global_batch_size % world_size != 0:
            raise ValueError(f"Global batch size ({cfg.training.global_batch_size}) must be divisible by world size ({world_size}).")
        local_batch_size = cfg.training.global_batch_size // world_size
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        local_batch_size = cfg.training.global_batch_size
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=local_batch_size, sampler=train_sampler,
        shuffle=False,
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        drop_last=dl_drop_last,
        prefetch_factor=dl_prefetch_factor if dl_num_workers > 0 else None
    )

    model = setup_model(cfg, device)
    if rank == 0:
        print_trainable_parameters(model)
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # --- Define the data split to use for compute modes ---
    data_split = cfg.dataset.get('split', 'eval')


    # --- BLOCK 1: compute_embeddings ---
    if cfg.training.get('compute_embeddings', False):
        load_path = Path(cfg.training.get('load_model_path', cfg.output.save_dir))
        load_saved_weights(model, load_path, cfg, rank)
        compute_embeddings(model, tokenizer, cfg, device, rank, world_size, distributed)
        return

    # --- AMP/SCALER DEFINITION ---
    use_amp = cfg.training.get('use_amp', True) and torch.cuda.is_available()
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    if rank == 0:
        print(f"Mixed precision training {'ENABLED' if use_amp else 'DISABLED'}")

    
    # --- BLOCK 2: compute_metrics_from_embeddings ---
    if cfg.training.get('compute_metrics_from_embeddings', False):
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"COMPUTE METRICS FROM EMBEDDINGS MODE (Split: {data_split})")
            print(f"{'='*80}")
        
        objects_to_broadcast = [None, None]
        Z_all, paper_ids_all = None, None
        
        if rank == 0:
            load_path = Path(cfg.training.load_model_path)
            embeddings_path = load_path / 'embeddings' / cfg.dataset.size / f'{data_split}_embeddings.pt'
            paper_ids_path = load_path / 'embeddings' / cfg.dataset.size / f'{data_split}_paper_ids.pt'

            if not embeddings_path.exists() or not paper_ids_path.exists():
                raise FileNotFoundError(
                    f"Could not find cached embeddings for split '{data_split}'. Searched for: \n"
                    f"- {embeddings_path}\n"
                    f"- {paper_ids_path}\n"
                    f"Please run `+training.compute_embeddings=true dataset.split={data_split}` first."
                )

            print(f"Loading cached embeddings from: {embeddings_path}")
            Z_all_cached = torch.load(embeddings_path, map_location='cpu', weights_only=True)
            print(f"Loading cached paper IDs from: {paper_ids_path}")
            paper_ids_all_cached = torch.load(paper_ids_path, map_location='cpu', weights_only=True)
            
            if distributed:
                objects_to_broadcast = [Z_all_cached, paper_ids_all_cached]
            else:
                Z_all = Z_all_cached
                paper_ids_all = paper_ids_all_cached
        
        if distributed:
            dist.broadcast_object_list(objects_to_broadcast, src=0)
            Z_all, paper_ids_all = objects_to_broadcast
        
        Z_all = Z_all.to(device)
        paper_ids_all = paper_ids_all.to(device)

        N_global = Z_all.shape[0]
        if N_global % world_size != 0:
            raise ValueError(f"Loaded {N_global} embeddings, which is not divisible by world size {world_size}.")

        C_local = N_global // world_size
        start, end = rank * C_local, (rank + 1) * C_local
        
        local_Z = Z_all[start:end]
        local_paper_ids = paper_ids_all[start:end]

        val_config = {
            'GLOBAL_BATCH_SIZE': N_global,
            'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
            'STREAM_CHUNK_SIZE': cfg.training.stream_chunk_size,
            'TAU': cfg.training.tau,
            'USE_AMP': use_amp
        }
        k_vals = cfg.training.get('k_vals', [1, 5, 10,])

        val_loss, val_metrics = compute_retrieval_metrics(
            local_Z, local_paper_ids, Z_all, paper_ids_all,
            val_config, rank, k_vals
        )
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Metrics Results (Split: {data_split}, from Cached Embeddings)")
            print(f"{'='*80}")
            print(f"Val Loss: {val_loss:.4f}") 
            for k, v in val_metrics.items():
                if k == 'MRR':
                    print(f"MRR: {v:.4f}")
                else:
                    print(f"{k}: {v*100:.2f}%")
            print(f"{'='*80}")
        return
    
    # --- BLOCK 3: compute_metrics ---
    if cfg.training.get('compute_metrics', False):
        load_path = Path(cfg.training.get('load_model_path', cfg.output.save_dir))
        load_saved_weights(model, load_path, cfg, rank)

        if rank == 0:
            print(f"\n[COMPUTE METRICS MODE] (Split: {data_split})")
        
        dataset_all = AllStatementsDataset(
            str(dataset_path), tokenizer, 
            max_length=cfg.training.max_length, 
            split=data_split,
            train_ratio=cfg.dataset.get('train_ratio', 0.8),
            seed=cfg.dataset.get('seed', 42)
        )
        
        val_world_size = world_size if distributed else 1
        all_statements_packed, all_paper_ids = prepare_all_statements_data(
            dataset_all, device, distributed, rank, val_world_size
        )

        model.eval()
        
        N_val_global = all_statements_packed.shape[0]
        C_val_local = N_val_global // val_world_size
        start, end = rank * C_val_local, (rank + 1) * C_val_local

        val_config = {
            'GLOBAL_BATCH_SIZE': N_val_global,
            'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
            'STREAM_CHUNK_SIZE': cfg.training.stream_chunk_size,
            'TAU': cfg.training.tau,
            'USE_AMP': use_amp
        }

        with torch.no_grad(), autocast(enabled=use_amp):
            val_loss, val_metrics = validate_metrics(
                model, 
                all_statements_packed[start:end].to(device), 
                all_paper_ids[start:end].to(device), 
                val_config,
                k_vals=cfg.training.get('k_vals', [1, 5, 10])
            )

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Metrics Results (Split: {data_split}, Model-Computed)")
            print(f"{'='*80}")
            print(f"Val Loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                if k == 'MRR':
                    print(f"MRR: {v:.4f}")
                else:
                    print(f"{k}: {v*100:.2f}%")
            print(f"{'='*80}")
        return

    # --- NEW BLOCK 4: retrieve_top_k ---
    if cfg.training.get('retrieve_top_k', 0) > 0:
        if distributed:
            print("WARNING: Retrieval mode does not support multi-GPU. Running on Rank 0 only.")

        # Load model weights
        load_path = Path(cfg.training.get('load_model_path', cfg.output.save_dir))
        load_saved_weights(model, load_path, cfg, rank)

        # Call the new helper function
        retrieve_top_k(model, tokenizer, cfg, device, rank)

        return # Exit after retrieval

    # --- NEW BLOCK 5: compute_pairwise_similarities ---
    if cfg.training.get('compute_pairwise_similarities', False):
        if distributed:
            print("WARNING: Pairwise similarity mode does not support multi-GPU. Running on Rank 0 only.")

        # Load model weights
        load_path = Path(cfg.training.get('load_model_path', cfg.output.save_dir))
        load_saved_weights(model, load_path, cfg, rank)

        # Call the new helper function
        compute_pairwise_similarities(model, tokenizer, cfg, device, rank)

        return # Exit after computation

    # --- Standard Training Setup ---
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if len(trainable_params) > 0:
        use_quantized_optim = cfg.training.get("quantization") and cfg.training.quantization.get("enabled")
        if use_quantized_optim:
            if rank == 0: print("Using 8-bit AdamW optimizer.")
            optimizer = bnb_optim.AdamW8bit(
                trainable_params, lr=cfg.training.lr,
                weight_decay=cfg.training.get('weight_decay', 0.01)
            )
        else:
            if rank == 0: print("Using standard AdamW optimizer.")
            optimizer = torch.optim.AdamW(
                trainable_params, lr=cfg.training.lr,
                weight_decay=cfg.training.get('weight_decay', 0.01)
            )
    else:
        optimizer = None

    if optimizer is not None:
        num_training_steps = len(train_loader) * cfg.training.num_epochs
        warmup_steps = cfg.training.get('warmup_steps', 2000)
        validation_interval = cfg.training.get('validation_interval', 0)
        if rank == 0:
            print(f"Scheduler: Cosine decay with {warmup_steps} warmup steps over {num_training_steps} total steps.")
            if validation_interval > 0:
                print(f"Validation: Will run every {validation_interval} steps + at epoch end")
            else:
                print(f"Validation: Will run only at epoch end")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
    else:
        scheduler = None
        validation_interval = 0

    val_x_packed, val_y_packed = prepare_validation_data(
        val_dataset, device, distributed, rank,
        max_val_samples=val_max_samples,
        val_batch_size=val_batch_size,
        num_workers=dl_num_workers
    )

    if len(trainable_params) == 0:
        if rank == 0:
            print("No trainable parameters - running in-training-style validation only.")
        model.eval()
        N_val = val_x_packed.shape[0]
        val_world_size = world_size if distributed else 1
        C_val = N_val // val_world_size
        start, end = rank * C_val, (rank + 1) * C_val
        local_val_x_packed = val_x_packed[start:end].to(device)
        local_val_y_packed = val_y_packed[start:end].to(device)
        val_config = {
            'GLOBAL_BATCH_SIZE': N_val,
            'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
            'STREAM_CHUNK_SIZE': cfg.training.stream_chunk_size,
            'TAU': cfg.training.tau
        }
        val_loss, val_metrics = distributed_validate_step(
            model, local_val_x_packed, local_val_y_packed, val_config
        )
        if rank == 0:
            print(f"\nValidation Results:")
            print(f"Val Loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                if k == 'MRR': print(f"MRR: {v:.4f}")
                else: print(f"{k}: {v*100:.2f}%")
        return

    global_step = 0

    def run_validation(step_num, epoch_num=None):
        model.eval()
        N_val, val_world_size = val_x_packed.shape[0], (world_size if distributed else 1)
        C_val = N_val // val_world_size
        start, end = rank * C_val, (rank + 1) * C_val
        val_config = {
            'GLOBAL_BATCH_SIZE': C_val * val_world_size, 
            'MICRO_BATCH_SIZE': cfg.training.micro_batch_size,
            'STREAM_CHUNK_SIZE': cfg.training.stream_chunk_size, 
            'TAU': cfg.training.tau,
            'USE_AMP': use_amp
        }
        with torch.no_grad(), autocast(enabled=use_amp):
            val_loss, topk_acc = distributed_validate_step(
                model, val_x_packed[start:end].to(device),
                val_y_packed[start:end].to(device), val_config,
                k_vals=cfg.training.get('k_vals', [1, 5, 10])
            )
        if rank == 0:
            prefix = f"\n[Epoch {epoch_num+1}] Validation" if epoch_num is not None else "\nValidation"
            print(f"{prefix} at step {step_num}:")
            print(f"  Val Loss:    {val_loss:.4f}\n  MRR:         {topk_acc.get('MRR', 0):.4f}")
            k_vals_to_report = cfg.training.get('k_vals', [1, 5, 10])
            acc_str = "  ".join([f"Top@{k}: {topk_acc.get(k, 0)*100:.2f}%" for k in k_vals_to_report if k in topk_acc])
            print(f"  {acc_str}\n")
            metrics = {'loss': val_loss, 'mrr': topk_acc.get('MRR', 0)}
            for k in k_vals_to_report:
                if k in topk_acc:
                    metrics[f'top{k}_acc'] = topk_acc.get(k, 0)
            log_metrics(metrics, step=step_num, prefix='val')
        model.train()
        return val_loss, topk_acc

    # --- MAIN TRAINING LOOP ---
    for epoch in range(cfg.training.num_epochs):
        train_dataset.reset_epoch()
        
        model.train()
        if distributed and train_sampler:
            train_sampler.set_epoch(epoch)

        total_loss, num_batches = 0, 0
        pbar = tqdm(train_loader, disable=(rank!=0), desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")

        for batch in pbar:
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
            
            scheduler.step()

            total_loss += loss
            num_batches += 1
            global_step += 1

            if rank == 0:
                pbar.set_postfix(loss=f'{loss:.4f}', lr=f'{scheduler.get_last_lr()[0]:.2e}')
                log_metrics({
                    'loss': loss,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'batch_size': cfg.training.global_batch_size,
                }, step=global_step, prefix='train')

            if validation_interval > 0 and global_step % validation_interval == 0:
                val_loss, topk_acc = run_validation(global_step, epoch_num=epoch)
                if distributed:
                    dist.barrier()

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        val_loss, topk_acc = run_validation(global_step, epoch_num=epoch)

        if rank == 0:
            print(f"\nEpoch [{epoch+1}/{cfg.training.num_epochs}] Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  MRR:        {topk_acc.get('MRR', 0):.4f}")
            k_vals_to_report = cfg.training.get('k_vals', [1, 5, 10])
            for k in k_vals_to_report:
                if k in topk_acc:
                    print(f"  Top@{k} Acc:  {topk_acc.get(k, 0)*100:.2f}%")
            print("")

            log_metrics({'epoch': epoch + 1, 'avg_train_loss': avg_train_loss}, step=global_step)

        save_model(cfg, model, rank, epoch=epoch+1)

        if distributed:
            dist.barrier()

    save_model(cfg, model, rank)
    if wandb_run is not None:
        wandb.finish()

# ===================================================================
# Main Entry Point with Hydra (Unchanged)
# ===================================================================
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""
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