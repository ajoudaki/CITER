# train_math_app.py
import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from typing import Dict

# Import the core distributed training logic
try:
    # distributed_clip must be importable (e.g., in the same directory)
    from distributed_clip import distributed_train_step
except ImportError:
    print("Error: distributed_clip.py not found.")
    sys.exit(1)

# ===================================================================
# Configuration
# ===================================================================

# Centralized Configuration
# Note on Batch Sizes: The sample dataset has 7 usable papers. 
# GLOBAL_BATCH_SIZE (N) must be <= 7 and divisible by the number of GPUs (P).
# If running with P=2, N=6 is suitable.

APP_CONFIG = {
    'DATA_PATH': 'data/lemmas_theorems.jsonl',
    # Model (Swappable)
    'MODEL_NAME': 'bert-base-uncased', # Can be swapped, e.g., 'allenai/scibert_scivocab_uncased'
    'MAX_SEQ_LEN': 256,
    # Training Hyperparameters
    'LR': 2e-5,
    'EPOCHS': 5,
    'NUM_WORKERS': 4,
    'SEED': 42,
    
    # Distributed Training Parameters (Passed to distributed_clip.py)
    'DIST_CONFIG': {
        'GLOBAL_BATCH_SIZE': 1024,  # N: Total number of papers processed globally per step
        'MICRO_BATCH_SIZE': 32,   # B: Number of papers processed locally per microbatch
        'STREAM_CHUNK_SIZE': 32,  # M: Chunk size for streaming computation
        'TAU': 0.07,             # Temperature
    }
}

# ===================================================================
# DDP Setup Utility
# ===================================================================

def setup_DDP(seed: int):
    """Initializes the distributed process group for torchrun."""
    if not torch.cuda.is_available():
        raise RuntimeError("This application requires CUDA.")
        
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if "LOCAL_RANK" not in os.environ:
         raise RuntimeError("LOCAL_RANK not set. Please run using torchrun.")
         
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    
    return rank, local_rank, world_size, device

# ===================================================================
# Model Definition (Modular)
# ===================================================================

class TransformerEncoder(nn.Module):
    """
    A swappable encoder based on HuggingFace Transformers (e.g., BERT).
    Uses the CLS token embedding as the representation.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the CLS token embedding (index 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # CRITICAL: L2 normalization must be applied here.
        # This ensures embeddings generated during Phase A (no_grad) by calling 
        # the encoder directly are correctly normalized.
        return F.normalize(cls_embedding, p=2, dim=-1)

class ContrastiveModel(nn.Module):
    """
    Wraps the encoders. Adheres to the interface required by distributed_clip.py.
    """
    def __init__(self, encoder_x: nn.Module, encoder_y: nn.Module):
        super().__init__()
        # Interface requirement: expose encoder_x and encoder_y
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y

    def forward(self, tokens_x: Dict[str, torch.Tensor], tokens_y: Dict[str, torch.Tensor]) -> tuple:
        # Process inputs (tokens_x/y are dictionaries from the tokenizer)
        # The ** unpacks the dictionary for the encoder's forward method
        z_x = self.encoder_x(**tokens_x)
        z_y = self.encoder_y(**tokens_y)
        return z_x, z_y

# ===================================================================
# Data Handling
# ===================================================================

class MathPaperDataset(Dataset):
    """
    Dataset for sampling pairs of statements from the same paper.
    """
    def __init__(self, data_path: str):
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        loaded_data = []
        try:
            with open(data_path, 'r') as f:
                for line in f:
                    paper = json.loads(line)
                    statements = paper.get('lemmas', []) + paper.get('theorems', [])
                    # The data file is pre-filtered, but we ensure the condition holds
                    if len(statements) >= 2:
                        loaded_data.append(statements)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_path}")
        return loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Samples two distinct statements from the paper at index idx.
        """
        statements = self.data[idx]
        # random.sample ensures distinctness
        sample_x, sample_y = random.sample(statements, 2)
        return sample_x, sample_y

def get_collate_fn(tokenizer, max_length):
    """
    Creates a collate function that tokenizes the input strings dynamically.
    """
    def collate_fn(batch):
        # Unzip the batch [(x1, y1), (x2, y2), ...]
        batch_x, batch_y = zip(*batch)
        
        # Tokenize both lists of strings
        tokens_x = tokenizer(
            list(batch_x), padding=True, truncation=True, 
            max_length=max_length, return_tensors="pt"
        )
        tokens_y = tokenizer(
            list(batch_y), padding=True, truncation=True, 
            max_length=max_length, return_tensors="pt"
        )
        return tokens_x, tokens_y
    return collate_fn

# ===================================================================
# Training Application
# ===================================================================

def main():
    config = APP_CONFIG
    dist_config = config['DIST_CONFIG']

    # 1. Setup DDP
    try:
        rank, local_rank, world_size, device = setup_DDP(config['SEED'])
    except RuntimeError as e:
        print(f"DDP setup failed: {e}")
        return

    N = dist_config['GLOBAL_BATCH_SIZE']
    B = dist_config['MICRO_BATCH_SIZE']

    # 2. Configuration Checks
    if N % world_size != 0:
        if rank == 0: print(f"Error: N ({N}) must be divisible by P ({world_size}).")
        return
    C = N // world_size # Local chunk size

    if C < B:
        if rank == 0: print(f"Warning: C ({C}) < B ({B}). Adjusting B=C.")
        dist_config['MICRO_BATCH_SIZE'] = C
        B = C

    # 3. Data Setup
    try:
        dataset = MathPaperDataset(config['DATA_PATH'])
    except FileNotFoundError as e:
        if rank == 0: print(e)
        return

    if len(dataset) < N:
         if rank == 0: 
             print(f"Error: Dataset size ({len(dataset)}) is smaller than Global Batch Size ({N}).")
         return

    # Initialize tokenizer and collate function
    tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])
    collate_fn = get_collate_fn(tokenizer, config['MAX_SEQ_LEN'])

    # Setup Distributed Sampler (drop_last=True is crucial for fixed N)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    
    dataloader = DataLoader(
        dataset, batch_size=C, sampler=sampler, collate_fn=collate_fn, 
        num_workers=config['NUM_WORKERS'], pin_memory=True
    )

    # 4. Model Setup (Modular)
    # Initialize the encoder. We use a shared encoder for both views (x and y).
    encoder = TransformerEncoder(config['MODEL_NAME'])
    
    # Initialize the contrastive model structure
    model = ContrastiveModel(encoder_x=encoder, encoder_y=encoder).to(device)
    
    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank])

    # 5. Optimizer Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'])

    if rank == 0:
        print(f"--- Starting MathCLIP Distributed Training ---")
        print(f"Model: {config['MODEL_NAME']}. Dataset Size: {len(dataset)}.")
        print(f"World Config: N={N}, P={world_size}, C={C}, B={B}, Tau={dist_config['TAU']}")

    # 6. Training Loop
    for epoch in range(config['EPOCHS']):
        # Ensure shuffling works correctly across epochs
        sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (tokens_x, tokens_y) in enumerate(dataloader):
            # Efficiency: Move the entire local chunk (C) to the GPU immediately.
            # This ensures that microbatching within distributed_train_step involves no further CPU-GPU transfer.
            local_x = {k: v.to(device, non_blocking=True) for k, v in tokens_x.items()}
            local_y = {k: v.to(device, non_blocking=True) for k, v in tokens_y.items()}

            # Execute the distributed training step
            loss = distributed_train_step(
                model, optimizer, local_x, local_y, dist_config
            )
            
            epoch_loss += loss
            num_batches += 1

            if rank == 0:
                print(f"Epoch {epoch+1}/{config['EPOCHS']}, Step {batch_idx+1}, Global Loss: {loss:.4f}")

        if rank == 0 and num_batches > 0:
            print(f"--- Epoch {epoch+1} Finished. Avg Loss: {epoch_loss/num_batches:.4f} ---")

    if rank == 0:
        print("Training Complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    # Avoid tokenizer parallelism issues when using fork in DataLoader/DDP
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Ensure that the script is run using torchrun
    if "LOCAL_RANK" not in os.environ:
        print("Please run this script using torchrun. E.g.:")
        print("torchrun --nproc_per_node=2 train_math_app.py")
    else:
        main()
