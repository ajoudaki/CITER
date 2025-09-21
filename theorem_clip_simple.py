#!/usr/bin/env python3
"""
Simple CLIP-style contrastive learning for mathematical theorems and lemmas.

This implementation:
1. Loads pairs of theorems/lemmas from same papers
2. Uses BERT to encode them (CLS token embedding)
3. Applies contrastive learning to align embeddings from same papers
4. Can run distributed or single GPU using the distributed_clip framework
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import random
from pathlib import Path
from typing import Optional
import os

# Import distributed training utilities
from distributed_clip import distributed_train_step


class MathTextDataset(Dataset):
    """Dataset of mathematical theorem/lemma pairs from same papers."""

    def __init__(self, jsonl_path: str, max_length: int = 256):
        self.max_length = max_length
        self.pairs = []

        with open(jsonl_path) as f:
            for line in f:
                paper = json.loads(line)
                statements = paper.get('lemmas', []) + paper.get('theorems', [])
                if len(statements) >= 2:
                    # Store all possible pairs from this paper
                    self.pairs.append(statements)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        statements = self.pairs[idx]
        # Sample two different statements
        x, y = random.sample(statements, 2) if len(statements) >= 2 else (statements[0], statements[0])
        return x, y


class TextEncoder(nn.Module):
    """BERT-based text encoder that outputs normalized embeddings."""

    def __init__(self, model_name: str = 'bert-base-uncased', output_dim: int = 256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # CLS token
        projected = self.projection(cls_embedding)
        return F.normalize(projected, p=2, dim=-1)


class ContrastiveModel(nn.Module):
    """Wrapper model compatible with distributed_clip."""

    def __init__(self, encoder: TextEncoder):
        super().__init__()
        # Share the same encoder for both sides
        self.encoder_x = encoder
        self.encoder_y = encoder

    def forward(self, x, y):
        # x and y are tuples of (input_ids, attention_mask)
        z_x = self.encoder_x(x[0], x[1])
        z_y = self.encoder_y(y[0], y[1])
        return z_x, z_y


def collate_fn(batch, tokenizer, max_length):
    """Custom collate function to tokenize text pairs."""
    texts_x, texts_y = zip(*batch)

    # Tokenize all texts
    tokens_x = tokenizer(
        list(texts_x),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    tokens_y = tokenizer(
        list(texts_y),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return (tokens_x['input_ids'], tokens_x['attention_mask']), \
           (tokens_y['input_ids'], tokens_y['attention_mask'])


def train_simple(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tau: float = 0.07
) -> float:
    """Simple single-GPU training step."""
    model.train()
    total_loss = 0

    for (x, y) in dataloader:
        x = (x[0].to(device), x[1].to(device))
        y = (y[0].to(device), y[1].to(device))

        optimizer.zero_grad()

        # Forward pass
        z_x, z_y = model(x, y)

        # Compute contrastive loss
        logits = torch.matmul(z_x, z_y.T) / tau
        labels = torch.arange(len(z_x), device=device)

        loss_x = F.cross_entropy(logits, labels)
        loss_y = F.cross_entropy(logits.T, labels)
        loss = (loss_x + loss_y) / 2

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Configuration
    config = {
        'data_path': 'data/lemmas_theorems_toy.jsonl',
        'batch_size': 8,
        'learning_rate': 5e-5,
        'epochs': 20,
        'max_length': 256,
        'output_dim': 256,
        'tau': 0.07,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = MathTextDataset(config['data_path'], config['max_length'])
    print(f"Loaded {len(dataset)} paper statement collections")

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, config['max_length'])
    )

    # Initialize model
    encoder = TextEncoder(output_dim=config['output_dim'])
    model = ContrastiveModel(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Training loop
    print("Starting training...")
    for epoch in range(config['epochs']):
        avg_loss = train_simple(model, dataloader, optimizer, device, config['tau'])
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), 'theorem_model_simple.pt')
    print("Model saved to theorem_model_simple.pt")


if __name__ == "__main__":
    main()