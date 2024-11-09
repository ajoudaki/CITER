from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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

class ModelTrainer:
    """Handles model training and validation."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        save_dir: Path,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.device = device
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            loss = self._process_batch(batch)
            
            # Optimization step
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_value
            )
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })
            
        return total_loss / num_batches

    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> TrainingMetrics:
        """Performs validation and computes metrics."""
        self.model.eval()
        validation_evaluator = ValidationEvaluator(
            self.model,
            self.config,
            self.device
        )
        metrics = validation_evaluator.evaluate(val_loader)
        
        # Update learning rate scheduler
        self.scheduler.step(metrics.val_loss)
        
        # Save model if it's the best so far
        self._save_model_if_best(metrics, epoch)
        
        return metrics

    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Processes a single batch and computes loss."""
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        # Get model outputs
        source_outputs = self.model.model(
            input_ids=batch['source_input_ids'],
            attention_mask=batch['source_attention_mask'],
            output_hidden_states=False,
            return_dict=True
        )
        
        target_outputs = self.model.model(
            input_ids=batch['target_input_ids'],
            attention_mask=batch['target_attention_mask'],
            output_hidden_states=False,
            return_dict=True
        )
        
        # Extract embeddings
        source_embeddings = self.model.text_encoder.embedding_extractor.get_token_embedding(
            source_outputs.last_hidden_state,
            batch['source_input_ids'],
            self.model.tokenizer.convert_tokens_to_ids(self.model.config.cite_token)
        )
        
        target_embeddings = self.model.text_encoder.embedding_extractor.get_token_embedding(
            target_outputs.last_hidden_state,
            batch['target_input_ids'],
            self.model.tokenizer.convert_tokens_to_ids(self.model.config.ref_token)
        )
        
        # Normalize embeddings
        source_embeddings = torch.nn.functional.normalize(source_embeddings, dim=-1)
        target_embeddings = torch.nn.functional.normalize(target_embeddings, dim=-1)
        
        # Compute similarity and loss
        similarity = torch.matmul(source_embeddings, target_embeddings.transpose(0, 1)) / self.config.temperature
        labels = torch.arange(similarity.size(0)).to(self.device)
        
        return self.criterion(similarity, labels)

    def _save_model_if_best(self, metrics: TrainingMetrics, epoch: int):
        """Saves model if it achieves best validation loss."""
        checkpoint_path = self.save_dir / f'model_epoch_{epoch}.pt'
        metrics_path = self.save_dir / f'metrics_epoch_{epoch}.pt'
        
        torch.save(self.model.state_dict(), checkpoint_path)
        torch.save(metrics.__dict__, metrics_path)

class ValidationEvaluator:
    """Handles model validation and metric computation."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, val_loader: DataLoader) -> TrainingMetrics:
        """Evaluates model performance on validation set."""
        print("\nRunning validation...")
        
        # Collect embeddings and compute validation loss
        embeddings = self._collect_embeddings(val_loader)
        val_loss = embeddings['total_loss'] / embeddings['num_batches']
        
        # Compute ranking metrics
        ranking_metrics = self._compute_ranking_metrics(
            embeddings['source_embeddings'],
            embeddings['target_embeddings']
        )
        
        return TrainingMetrics(
            train_loss=0.0,  # Set by trainer
            val_loss=val_loss,
            **ranking_metrics
        )

    def _collect_embeddings(self, val_loader: DataLoader) -> Dict:
        """Collects embeddings and computes validation loss."""
        total_loss = 0
        num_batches = 0
        all_source_embeddings = []
        all_target_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Computing validation embeddings'):
                # Process batch
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                try:
                    # Get embeddings
                    source_emb, target_emb = self._get_batch_embeddings(batch)
                    
                    # Compute loss
                    similarity = torch.matmul(source_emb, target_emb.transpose(0, 1)) / self.config.temperature
                    labels = torch.arange(similarity.size(0)).to(self.device)
                    loss = self.criterion(similarity, labels)
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Store embeddings
                    all_source_embeddings.append(source_emb.cpu())
                    all_target_embeddings.append(target_emb.cpu())
                    
                except ValueError as e:
                    print(f"Skipping batch due to error: {e}")
                    continue
        
        return {
            'source_embeddings': torch.cat(all_source_embeddings, dim=0),
            'target_embeddings': torch.cat(all_target_embeddings, dim=0),
            'total_loss': total_loss,
            'num_batches': num_batches
        }

    def _get_batch_embeddings(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts embeddings for a batch of inputs."""
        # Get model outputs
        source_outputs = self.model.model(
            input_ids=batch['source_input_ids'],
            attention_mask=batch['source_attention_mask'],
            output_hidden_states=False,
            return_dict=True
        )
        
        target_outputs = self.model.model(
            input_ids=batch['target_input_ids'],
            attention_mask=batch['target_attention_mask'],
            output_hidden_states=False,
            return_dict=True
        )
        
        # Extract and normalize embeddings
        source_emb = self.model.text_encoder.embedding_extractor.get_token_embedding(
            source_outputs.last_hidden_state,
            batch['source_input_ids'],
            self.model.tokenizer.convert_tokens_to_ids(self.model.config.cite_token)
        )
        
        target_emb = self.model.text_encoder.embedding_extractor.get_token_embedding(
            target_outputs.last_hidden_state,
            batch['target_input_ids'],
            self.model.tokenizer.convert_tokens_to_ids(self.model.config.ref_token)
        )
        
        return (
            torch.nn.functional.normalize(source_emb, dim=-1),
            torch.nn.functional.normalize(target_emb, dim=-1)
        )

    def _compute_ranking_metrics(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> Dict:
        """Computes ranking metrics for validation set."""
        total_samples = source_embeddings.size(0)
        print(f"\nComputing rankings for {total_samples} samples...")
        
        # Process in chunks to avoid OOM
        chunk_size = 512
        all_rankings = []
        
        for i in range(0, total_samples, chunk_size):
            chunk_end = min(i + chunk_size, total_samples)
            source_chunk = source_embeddings[i:chunk_end].to(self.device)
            
            # Calculate similarity and rankings
            similarity = torch.matmul(source_chunk, target_embeddings.to(self.device).t()) / self.config.temperature
            rankings = torch.argsort(similarity, dim=-1, descending=True)
            all_rankings.append(rankings.cpu())
            
            # Clean up GPU memory
            del similarity, source_chunk
            torch.cuda.empty_cache()
        
        # Concatenate all rankings
        all_rankings = torch.cat(all_rankings, dim=0)
        
        # Compute metrics
        metrics = self._calculate_metrics(all_rankings, total_samples)
        print("\nValidation Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        return metrics

    def _calculate_metrics(
        self,
        rankings: torch.Tensor,
        total_samples: int
    ) -> Dict:
        """Calculates various ranking metrics."""
        correct_at_k = {k: 0 for k in self.config.eval_k_values}
        reciprocal_ranks = []
        all_ranks = []
        
        for i in range(total_samples):
            # Find rank of correct match
            rank = (rankings[i] == i).nonzero().item() + 1
            
            # Update top-k accuracy
            for k in self.config.eval_k_values:
                if rank <= k:
                    correct_at_k[k] += 1
            
            reciprocal_ranks.append(1.0 / rank)
            all_ranks.append(rank)
        
        metrics = {
            'top_k_accuracy': {
                k: correct_at_k[k] / total_samples
                for k in self.config.eval_k_values
            },
            'mrr': float(np.mean(reciprocal_ranks)),
            'median_rank': float(np.median(all_ranks)),
            'mean_rank': float(np.mean(all_ranks)),
            'val_size': total_samples
        }
        
        return metrics

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainingConfig,
    save_dir: Path,
    device: torch.device
) -> List[TrainingMetrics]:
    """Main training loop."""
    trainer = ModelTrainer(model, config, save_dir, device)
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
    
    return metrics_history