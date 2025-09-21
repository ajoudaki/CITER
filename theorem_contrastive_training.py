# theorem_contrastive_training.py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
import random
import numpy as np
from typing import List, Tuple, Dict
import os
from distributed_clip import distributed_train_step, trivial_contrastive_step

# ===================================================================
# Dataset for Theorems and Lemmas
# ===================================================================

class TheoremLemmaDataset(Dataset):
    """Dataset that loads theorem/lemma pairs from JSONL file"""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Load all papers from JSONL
        with open(jsonl_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                lemmas = paper.get('lemmas', [])
                theorems = paper.get('theorems', [])

                # Combine lemmas and theorems from same paper
                all_statements = lemmas + theorems

                # Create pairs from same paper (positive pairs)
                if len(all_statements) >= 2:
                    self.data.append(all_statements)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get statements from same paper
        statements = self.data[idx]

        # Sample two different statements from the same paper
        if len(statements) >= 2:
            sampled = random.sample(statements, 2)
            text_x, text_y = sampled[0], sampled[1]
        else:
            # If only one statement, duplicate it
            text_x = text_y = statements[0]

        # Tokenize
        tokens_x = self.tokenizer(
            text_x,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        tokens_y = self.tokenizer(
            text_y,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids_x': tokens_x['input_ids'].squeeze(0),
            'attention_mask_x': tokens_x['attention_mask'].squeeze(0),
            'input_ids_y': tokens_y['input_ids'].squeeze(0),
            'attention_mask_y': tokens_y['attention_mask'].squeeze(0)
        }

# ===================================================================
# BERT-based Text Encoder
# ===================================================================

class BERTEncoder(nn.Module):
    """BERT encoder that extracts CLS token embedding"""

    def __init__(self, model_name: str = 'bert-base-uncased', hidden_dim: int = 768, output_dim: int = 256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract CLS token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Project to desired dimension
        projected = self.projection(cls_embedding)

        # L2 normalize
        normalized = F.normalize(projected, p=2, dim=-1)

        return normalized

# ===================================================================
# Contrastive Model Wrapper
# ===================================================================

class EncoderWrapper(nn.Module):
    """Wrapper to make encoder compatible with distributed_clip interface"""

    def __init__(self, base_encoder, max_length):
        super().__init__()
        self.base_encoder = base_encoder
        self.max_length = max_length

    def forward(self, packed_tensor):
        # Unpack the tensor (assumes input_ids and attention_mask are concatenated)
        input_ids = packed_tensor[:, :self.max_length].long()
        attention_mask = packed_tensor[:, self.max_length:].long()
        return self.base_encoder(input_ids, attention_mask)


class TheoremContrastiveModel(nn.Module):
    """Model wrapper that follows distributed_clip interface"""

    def __init__(self, bert_encoder, max_length=512):
        super().__init__()
        # Create wrapped encoders (sharing the same base encoder)
        self.encoder_x = EncoderWrapper(bert_encoder, max_length)
        self.encoder_y = self.encoder_x  # Share parameters

    def forward(self, x_packed, y_packed):
        z_x = self.encoder_x(x_packed)
        z_y = self.encoder_y(y_packed)
        return z_x, z_y

# ===================================================================
# Training Configuration
# ===================================================================

TRAIN_CONFIG = {
    'GLOBAL_BATCH_SIZE': 8*1024,    # Desired total batch size across all GPUs
    'MICRO_BATCH_SIZE': 64,       # Micro batch size for gradient accumulation
    'STREAM_CHUNK_SIZE': 64,      # Streaming chunk size for memory efficiency
    'TAU': 0.07,                 # Temperature parameter
    'LR': 0.00015,                  # Learning rate
    'NUM_EPOCHS': 10,            # Number of training epochs
    'MAX_LENGTH': 256,           # Max token length for BERT
    'OUTPUT_DIM': 768,           # Output embedding dimension
}

# ===================================================================
# Training Function
# ===================================================================

def train(rank: int = 0, world_size: int = 1, distributed: bool = False):
    """Main training function"""

    # Set device
    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Create dataset
    dataset = TheoremLemmaDataset(
        'data/lemmas_theorems.jsonl',
        tokenizer,
        max_length=TRAIN_CONFIG['MAX_LENGTH']
    )

    # Create data loader
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        batch_size = TRAIN_CONFIG['GLOBAL_BATCH_SIZE'] // world_size
    else:
        sampler = None
        batch_size = TRAIN_CONFIG['GLOBAL_BATCH_SIZE']

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True
    )

    # Create model
    bert_encoder = BERTEncoder(
        model_name='bert-base-uncased',
        hidden_dim=768,
        output_dim=TRAIN_CONFIG['OUTPUT_DIM']
    )
    model = TheoremContrastiveModel(bert_encoder, max_length=TRAIN_CONFIG['MAX_LENGTH']).to(device)

    # Wrap in DDP if distributed
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['LR'])

    # Training loop
    for epoch in range(TRAIN_CONFIG['NUM_EPOCHS']):
        if distributed and sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Pack data into tensors for distributed_clip compatibility
            # Concatenate input_ids and attention_mask
            x_packed = torch.cat([
                batch['input_ids_x'].to(device),
                batch['attention_mask_x'].to(device)
            ], dim=1)

            y_packed = torch.cat([
                batch['input_ids_y'].to(device),
                batch['attention_mask_y'].to(device)
            ], dim=1)

            if distributed:
                # Use distributed training step
                # Update config to match actual batch size
                actual_batch_size = x_packed.shape[0] * dist.get_world_size()
                config_copy = TRAIN_CONFIG.copy()
                config_copy['GLOBAL_BATCH_SIZE'] = actual_batch_size
                loss = distributed_train_step(
                    model, optimizer, x_packed, y_packed, config_copy
                )
            else:
                # Use trivial implementation for single GPU
                z_x, z_y = model(x_packed, y_packed)

                # Compute similarity matrix
                S = torch.matmul(z_x, z_y.T) / TRAIN_CONFIG['TAU']

                # Compute loss
                labels = torch.arange(z_x.shape[0], device=device)
                loss_x = F.cross_entropy(S, labels, reduction='sum')
                loss_y = F.cross_entropy(S.T, labels, reduction='sum')
                loss = (loss_x + loss_y) / (2 * z_x.shape[0])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = loss.item()

            total_loss += loss
            num_batches += 1

            # Log progress
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']}], "
                      f"Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss:.4f}")

        # Log epoch summary
        if rank == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']}] completed. "
                  f"Average Loss: {avg_loss:.4f}")

    # Save model (only from rank 0)
    if rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), 'theorem_contrastive_model.pt')
        print("Model saved to theorem_contrastive_model.pt")

# ===================================================================
# Main Entry Point
# ===================================================================

if __name__ == "__main__":
    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed mode
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(f"Running in distributed mode. Rank: {rank}, World Size: {world_size}")
        train(rank, world_size, distributed=True)

        dist.destroy_process_group()
    else:
        # Single GPU or CPU mode
        print("Running in single device mode")
        train(distributed=False)
