from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1.5e-4
    temperature: float = 0.1
    num_workers: int = 4
    gradient_clip_value: float = 1.0
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    eval_k_values: List[int] = None

    def __post_init__(self):
        if self.eval_k_values is None:
            self.eval_k_values = [1, 3, 5, 10, 50]

@dataclass
class TrainingMetrics:
    """Stores training and validation metrics."""
    train_loss: float
    val_loss: float
    top_k_accuracy: Dict[int, float]
    mrr: float
    median_rank: float
    mean_rank: float
    val_size: int

class Trainer:
    """Unified trainer class handling both training and validation."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, save_dir: Path, device: torch.device):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.device = device
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True
        )

    def _get_embeddings(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for source and target texts."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        with torch.set_grad_enabled(self.model.training):
            # Get source embeddings
            source_out = self.model.model(
                input_ids=batch['source_input_ids'],
                attention_mask=batch['source_attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
            source_emb = self.model._extract_token_embedding(
                source_out.last_hidden_state,
                batch['source_input_ids'],
                self.model.tokenizer.convert_tokens_to_ids(self.model.config.cite_token)
            )
            
            # Get target embeddings
            target_out = self.model.model(
                input_ids=batch['target_input_ids'],
                attention_mask=batch['target_attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
            target_emb = self.model._extract_token_embedding(
                target_out.last_hidden_state,
                batch['target_input_ids'],
                self.model.tokenizer.convert_tokens_to_ids(self.model.config.ref_token)
            )
            
        return source_emb, target_emb

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            source_emb, target_emb = self._get_embeddings(batch)
            
            # Compute loss
            similarity = torch.matmul(source_emb, target_emb.transpose(0, 1)) / self.config.temperature
            labels = torch.arange(similarity.size(0)).to(self.device)
            loss = self.criterion(similarity, labels)
            
            # Optimize
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_value
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches

    def validate(self, val_loader: DataLoader, epoch: int) -> TrainingMetrics:
        """Validate model and compute metrics."""
        self.model.eval()
        
        # Collect all embeddings
        all_source_embs = []
        all_target_embs = []
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc='Validation'):
            source_emb, target_emb = self._get_embeddings(batch)
            
            # Compute validation loss
            similarity = torch.matmul(source_emb, target_emb.transpose(0, 1)) / self.config.temperature
            labels = torch.arange(similarity.size(0)).to(self.device)
            loss = self.criterion(similarity, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_source_embs.append(source_emb.cpu())
            all_target_embs.append(target_emb.cpu())
        
        # Concatenate all embeddings
        source_embeddings = torch.cat(all_source_embs, dim=0)
        target_embeddings = torch.cat(all_target_embs, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(source_embeddings, target_embeddings)
        metrics.val_loss = total_loss / num_batches
        
        # Update scheduler and save model
        self.scheduler.step(metrics.val_loss)
        self._save_checkpoint(metrics, epoch)
        
        return metrics

    def _compute_metrics(self, source_embs: torch.Tensor, target_embs: torch.Tensor) -> TrainingMetrics:
        """Compute all validation metrics."""
        total_samples = source_embs.size(0)
        chunk_size = 512
        all_rankings = []
        
        # Compute rankings in chunks to avoid OOM
        for i in range(0, total_samples, chunk_size):
            chunk_end = min(i + chunk_size, total_samples)
            source_chunk = source_embs[i:chunk_end].to(self.device)
            
            similarity = torch.matmul(source_chunk, target_embs.to(self.device).t()) / self.config.temperature
            rankings = torch.argsort(similarity, dim=-1, descending=True)
            all_rankings.append(rankings.cpu())
            
            del similarity, source_chunk
            torch.cuda.empty_cache()
        
        rankings = torch.cat(all_rankings, dim=0)
        
        # Calculate metrics
        correct_at_k = {k: 0 for k in self.config.eval_k_values}
        reciprocal_ranks = []
        ranks = []
        
        for i in range(total_samples):
            rank = (rankings[i] == i).nonzero().item() + 1
            ranks.append(rank)
            reciprocal_ranks.append(1.0 / rank)
            
            for k in self.config.eval_k_values:
                if rank <= k:
                    correct_at_k[k] += 1
        
        return TrainingMetrics(
            train_loss=0.0,  # Set by train_model
            val_loss=0.0,    # Set by validate
            top_k_accuracy={k: count / total_samples for k, count in correct_at_k.items()},
            mrr=float(np.mean(reciprocal_ranks)),
            median_rank=float(np.median(ranks)),
            mean_rank=float(np.mean(ranks)),
            val_size=total_samples
        )

    def _save_checkpoint(self, metrics: TrainingMetrics, epoch: int):
        """Save model checkpoint and metrics."""
        torch.save(
            self.model.state_dict(),
            self.save_dir / f'model_epoch_{epoch}.pt'
        )
        torch.save(
            metrics.__dict__,
            self.save_dir / f'metrics_epoch_{epoch}.pt'
        )

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainingConfig,
    save_dir: Path,
    device: torch.device
) -> List[TrainingMetrics]:
    """Main training loop."""
    trainer = Trainer(model, config, save_dir, device)
    metrics_history = []
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Training phase
        train_loss = trainer.train_epoch(train_loader)
        
        # Validation phase
        if val_loader:
            metrics = trainer.validate(val_loader, epoch)
            metrics.train_loss = train_loss
            metrics_history.append(metrics)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {metrics.val_loss:.4f}")
            print(f"Best Top-1 Accuracy: {metrics.top_k_accuracy[1]:.4f}")
            print(f"Mean Reciprocal Rank: {metrics.mrr:.4f}")
            # top k accuracies
            [print(f"Top-{k} Accuracy: {v:.3f}") for k, v in metrics.top_k_accuracy.items()]
    
    return metrics_history