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
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes.optim as bnb_optim

import wandb

# Import distributed functions
from distributed_clip import distributed_train_step, trivial_contrastive_step, distributed_validate_step

from numba import njit # Import the Numba JIT compiler
import torch.utils.checkpoint as checkpoint # <-- ADD THIS IMPORT


# ===================================================================
# Numba-Optimized Helper Function for Pair Generation
# ===================================================================
@njit(cache=True)
def _generate_shuffled_pairs_numba(paper_lengths: np.ndarray, seed: int) -> np.ndarray:
    """
    A Numba-JIT compiled function to generate and shuffle theorem pairs efficiently.

    Args:
        paper_lengths: A 1D NumPy array where each element is the number of statements in a paper.
        seed: An integer for the random number generator.

    Returns:
        A 3D NumPy array of shape (total_pairs, 2, 2) containing the shuffled pairs.
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
    This implementation pre-computes an array of all pairs for the epoch using a
    fast, Numba-compiled helper function.
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
# Model Architectures 
# ===================================================================
class CLSPoolingEncoder(nn.Module):
    """Encoder that uses the [CLS] token embedding with optional gradient checkpointing."""
    def __init__(self, model_name: str, hidden_dim: int, output_dim: int, use_gradient_checkpointing: bool = False, quantization_config=None):
        super().__init__()
        # Pass quantization_config if it exists, and set device_map only when quantizing
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto" if quantization_config else None,
        }
        # Filter out None values to keep the from_pretrained call clean
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
        return F.normalize(self.projection(cls_embedding), p=2, dim=-1)

class LastTokenEncoder(nn.Module):
    """Encoder that uses the last non-padding token's hidden state with optional gradient checkpointing."""
    def __init__(self, model_name: str, hidden_dim: int, output_dim: int, use_gradient_checkpointing: bool = False, quantization_config=None):
        super().__init__()
        # Pass quantization_config if it exists, and set device_map only when quantizing
        model_kwargs = {
            "quantization_config": quantization_config,
            # "device_map": "auto", #if quantization_config else None,
        }
        # Filter out None values to keep the from_pretrained call clean
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
        return F.normalize(self.projection(last_token_embedding), p=2, dim=-1)


class JaccardEncoder(nn.Module):
    """Bag-of-tokens encoder using Jaccard similarity (no training needed)."""
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768, **kwargs):
        super().__init__()
        # Just use a fixed vocab size instead of loading tokenizer
        self.vocab_size = min(30000, 50000)  # Cap at 30k for memory
        self.output_dim = output_dim

    def forward(self, input_ids, attention_mask):
        batch_size, device = input_ids.shape[0], input_ids.device

        # Create binary bag-of-words vectors
        bow_vectors = torch.zeros(batch_size, self.vocab_size, device=device)

        for i in range(batch_size):
            # Get valid tokens (non-padding)
            valid_tokens = input_ids[i][attention_mask[i] == 1]
            # Filter special tokens (typically < 1000) and cap at vocab_size
            valid_tokens = valid_tokens[(valid_tokens > 100) & (valid_tokens < self.vocab_size)]
            if len(valid_tokens) > 0:
                # Set 1 for present tokens (bag-of-words)
                bow_vectors[i].scatter_(0, valid_tokens, 1.0)

        # L2 normalize for cosine similarity (equivalent to Jaccard for binary vectors)
        bow_vectors = F.normalize(bow_vectors, p=2, dim=-1)

        # Project to output dimension if needed
        if self.vocab_size != self.output_dim:
            # Simple random projection for dimension matching
            if not hasattr(self, 'projection'):
                self.projection = nn.Linear(self.vocab_size, self.output_dim, bias=False).to(device)
                # Initialize with random orthogonal matrix
                nn.init.orthogonal_(self.projection.weight)
                self.projection.requires_grad_(False)
            bow_vectors = self.projection(bow_vectors)
            bow_vectors = F.normalize(bow_vectors, p=2, dim=-1)

        return bow_vectors


class MinHashEncoder(nn.Module):
    """MinHash encoder using Hamming distance (no training needed)."""
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768, **kwargs):
        super().__init__()
        self.vocab_size = min(30000, 50000)
        self.num_hashes = output_dim  # Use output_dim as number of hash functions

        # Find prime for hash functions
        self.prime = self._find_next_prime(self.vocab_size)

        # Initialize hash parameters (will be set on first forward pass to get device)
        self.a = None
        self.b = None
        self.hashed_vocab = None

    def _find_next_prime(self, n: int) -> int:
        """Find next prime >= n."""
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
        """Initialize hash functions on the correct device."""
        if self.a is None:
            rand_max = self.prime - 1
            self.a = torch.randint(1, rand_max, (self.num_hashes, 1), device=device, dtype=torch.int64)
            self.b = torch.randint(0, rand_max, (self.num_hashes, 1), device=device, dtype=torch.int64)

            # Pre-compute hash values for all vocabulary tokens
            vocab_indices = torch.arange(self.vocab_size, device=device, dtype=torch.int64).unsqueeze(0)
            self.hashed_vocab = (self.a * vocab_indices + self.b) % self.prime

    def forward(self, input_ids, attention_mask):
        batch_size, device = input_ids.shape[0], input_ids.device

        # Initialize hash functions on first call
        self._init_hash_functions(device)

        # Create binary bag-of-words vectors
        bow_vectors = torch.zeros(batch_size, self.vocab_size, device=device)
        for i in range(batch_size):
            valid_tokens = input_ids[i][attention_mask[i] == 1]
            valid_tokens = valid_tokens[(valid_tokens > 100) & (valid_tokens < self.vocab_size)]
            if len(valid_tokens) > 0:
                bow_vectors[i].scatter_(0, valid_tokens, 1.0)

        # Compute MinHash signatures
        # Expand hashed vocab for batch processing
        hashed_vocab_expanded = self.hashed_vocab.unsqueeze(0).expand(batch_size, -1, -1)

        # Mask non-present tokens with infinity
        masked_hashes = torch.where(
            bow_vectors.unsqueeze(1) == 1,
            hashed_vocab_expanded.float(),
            float('inf')
        )

        # Get MinHash signature (minimum hash for each hash function)
        signatures, _ = torch.min(masked_hashes, dim=2)

        # Normalize to unit vectors for cosine similarity
        # (This makes cosine similarity approximate Hamming distance ranking)
        signatures = F.normalize(signatures, p=2, dim=-1)

        return signatures


class BM25Encoder(nn.Module):
    """BM25 scoring encoder (no training needed)."""
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768, **kwargs):
        super().__init__()
        self.vocab_size = min(30000, 50000)
        self.output_dim = output_dim
        # BM25 parameters
        self.k1 = 1.2  # Term frequency saturation
        self.b = 0.75  # Length normalization

    def forward(self, input_ids, attention_mask):
        batch_size, device = input_ids.shape[0], input_ids.device

        # Create BM25 weighted vectors
        bm25_vectors = torch.zeros(batch_size, self.vocab_size, device=device)

        for i in range(batch_size):
            valid_tokens = input_ids[i][attention_mask[i] == 1]
            valid_tokens = valid_tokens[(valid_tokens > 100) & (valid_tokens < self.vocab_size)]

            if len(valid_tokens) > 0:
                doc_len = len(valid_tokens)
                # Vectorized BM25 scoring
                unique_tokens, counts = torch.unique(valid_tokens, return_counts=True)
                tf = counts.float()
                # BM25 scoring: (tf * (k1 + 1)) / (tf + k1 * norm)
                length_norm = 1.0 + self.b * (doc_len / 100.0 - 1.0)
                bm25_scores = (tf * (self.k1 + 1)) / (tf + self.k1 * max(length_norm, 0.5))
                bm25_vectors[i, unique_tokens] = bm25_scores

        # L2 normalize for cosine similarity
        bm25_vectors = F.normalize(bm25_vectors + 1e-10, p=2, dim=-1)

        # Project to output dimension if needed
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
# Helper Functions (Unchanged)
# ===================================================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
    """
    Create a learning rate schedule with a linear warmup phase followed by a cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def init_wandb(cfg: DictConfig, rank: int = 0) -> Optional[object]:
    """Initialize Weights & Biases logging."""
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
    """Log metrics to wandb if enabled."""
    if wandb.run is not None:
        log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        wandb.log(log_dict, step=step)

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
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
    """Setup model with optional LoRA, quantization, and gradient checkpointing."""
    encoder_class = ENCODER_REGISTRY.get(cfg.model.model_type)
    if encoder_class is None:
        raise ValueError(f"Unknown model type: {cfg.model.model_type}")

    # Convert model config to dict
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)

    # Handle quantization if enabled in training config
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

    # Create encoder with unpacked model config
    base_encoder = encoder_class(**model_kwargs)

    if cfg.training.lora.enabled:
        # Prepare for LoRA. This works correctly whether the model is quantized or not.
        for param in base_encoder.base_model.parameters():
            param.requires_grad = False

        target_modules = list(cfg.model.lora_target_modules)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=cfg.training.lora.r,
            lora_alpha=cfg.training.lora.lora_alpha, lora_dropout=cfg.training.lora.lora_dropout,
            target_modules=target_modules,
        )
        base_encoder.base_model = get_peft_model(base_encoder.base_model, peft_config)

    # Only set projection parameters if encoder has projection layer
    if hasattr(base_encoder, 'projection'):
        for param in base_encoder.projection.parameters():
            param.requires_grad = True
        
    model = TheoremContrastiveModel(base_encoder, max_length=cfg.training.max_length)
    
    return model.to(device)

def prepare_validation_data(val_dataset, device, distributed, rank):
    """Prepare and distribute validation data."""
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
    """Save model or LoRA adapters."""
    if rank != 0: return
    output_dir = Path(cfg.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    if cfg.training.lora.enabled and cfg.output.save_lora:
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
# Training Function (Unchanged)
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

    dataset_path = Path(cfg.dataset.base_path) / f"{cfg.dataset.size}.jsonl"
    if rank == 0:
        print(f"Loading dataset: {dataset_path}")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    DatasetClass = StratifiedTheoremDataset # Hard-coding to the improved class
    train_dataset = DatasetClass(str(dataset_path), tokenizer, max_length=cfg.training.max_length, split='train')
    val_dataset = DatasetClass(str(dataset_path), tokenizer, max_length=cfg.training.max_length, split='eval')

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
        # No need for DataLoader shuffle; dataset is pre-shuffled per epoch
        shuffle=False, 
        num_workers=cfg.runtime.num_workers, pin_memory=True, drop_last=True
    )

    model = setup_model(cfg, device)
    if rank == 0:
        print_trainable_parameters(model)
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Check if model has trainable parameters
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    # Only create optimizer if we have trainable parameters
    if len(trainable_params) > 0:
        use_quantized_optim = cfg.training.get("quantization") and cfg.training.quantization.get("enabled")

        if use_quantized_optim:
            if rank == 0:
                print("Using 8-bit AdamW optimizer.")
            optimizer = bnb_optim.AdamW8bit(
                trainable_params,
                lr=cfg.training.lr,
                weight_decay=cfg.training.get('weight_decay', 0.01)
            )
        else:
            if rank == 0:
                print("Using standard AdamW optimizer.")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=cfg.training.lr,
                weight_decay=cfg.training.get('weight_decay', 0.01)
            )
    else:
        optimizer = None  # No optimizer needed for baseline models

    # Only setup scheduler if we have an optimizer
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

    use_amp = cfg.training.get('use_amp', True) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if rank == 0:
        print(f"Mixed precision training {'ENABLED' if use_amp else 'DISABLED'}")

    val_x_packed, val_y_packed = prepare_validation_data(val_dataset, device, distributed, rank)

    # For baseline models with no parameters, run validation only
    if len(trainable_params) == 0:
        if rank == 0:
            print("No trainable parameters - running validation only.")

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
                if k == 'MRR':
                    print(f"MRR: {v:.4f}")
                else:
                    print(f"top-{k}: {v:.4f}")
        return  # Exit early for baseline models

    global_step = 0

    # Helper function for validation
    def run_validation(step_num, epoch_num=None):
        model.eval()
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

        with torch.no_grad():
            val_loss, topk_acc = distributed_validate_step(model, local_val_x, local_val_y, val_config)

        if rank == 0:
            if epoch_num is not None:
                print(f"\n[Epoch {epoch_num+1}] Validation at step {step_num}:")
            else:
                print(f"\nValidation at step {step_num}:")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  MRR:        {topk_acc.get('MRR', 0):.4f}")
            print(f"  Top@1 Acc:  {topk_acc.get(1, 0)*100:.2f}%")
            print(f"  Top@5 Acc:  {topk_acc.get(5, 0)*100:.2f}%")
            print(f"  Top@10 Acc: {topk_acc.get(10, 0)*100:.2f}%\n")

            log_metrics({'loss': val_loss, 'mrr': topk_acc.get('MRR', 0), 'top1_acc': topk_acc.get(1, 0)}, step=step_num, prefix='val')

        model.train()
        return val_loss, topk_acc

    for epoch in range(cfg.training.num_epochs):
        # The dataset's epoch must be synced with the training loop's epoch
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

            # Run validation at specified intervals during training
            if validation_interval > 0 and global_step % validation_interval == 0:
                val_loss, topk_acc = run_validation(global_step, epoch_num=epoch)
                if distributed:
                    dist.barrier()

        # End of epoch validation (always run)
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        val_loss, topk_acc = run_validation(global_step, epoch_num=epoch)

        if rank == 0:
            print(f"\nEpoch [{epoch+1}/{cfg.training.num_epochs}] Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  MRR:        {topk_acc.get('MRR', 0):.4f}")
            print(f"  Top@1 Acc:  {topk_acc.get(1, 0)*100:.2f}%")
            print(f"  Top@5 Acc:  {topk_acc.get(5, 0)*100:.2f}%")
            print(f"  Top@10 Acc: {topk_acc.get(10, 0)*100:.2f}%\n")

            log_metrics({'epoch': epoch + 1, 'avg_train_loss': avg_train_loss}, step=global_step)

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
