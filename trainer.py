# Standard library imports
import random 
from pathlib import Path
from typing import List, Optional, Union

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import Subset
from torch.optim import AdamW
from torch.utils.data import DataLoader 
from transformers import (
    AutoTokenizer,
)
import tqdm 

from wiki_processor import CitationExtractor
from config import TrainingConfig
from model import CitationModel
from data_processing import prepare_training_data, create_training_batches, citation_collate_fn, CitationDataset



def validate_batch_structure(batch, config: TrainingConfig):
    c1 = (batch['source_ids']==config.cite_token_id).sum()==batch['cited_art_ids'].shape[0]  # special cite tokens correspond to the cited article ids
    c2 =  (batch['cited_art_ids'].shape[0]==batch['labels'].shape[0])  # each cited article id has a corresponding target label
    c3 = (batch['target_ids']==config.ref_token_id).sum()==batch['target_art_ids'].shape[0]  # special ref tokens correspond to the target article ids
    return c1 and c2 and c3 

def compute_retrieval_metrics(logits, labels, ks=[1, 5, 10, 50, 100, 1000]):
    # Get rankings of correct targets
    correct_scores = logits[torch.arange(logits.size(0)), labels]
    rankings = (logits >= correct_scores.unsqueeze(1)).sum(1)
    
    # Compute MRR
    mrr = (1.0 / rankings).mean().item()
    
    # Compute top-k accuracy for different k values
    metrics = {'mrr': mrr}
    for k in ks:
        if k <= logits.size(1):  # Only compute if k is not larger than number of targets
            top_k_acc = (rankings <= k).float().mean().item()
            metrics[f'top_{k}_accuracy'] = top_k_acc
    
    return metrics
    


def validate_epoch(
    model,
    val_dataloader,
    global_step: int,
    epoch: int,
    wandb,
    config: TrainingConfig
) -> dict:
    import torch
    import tqdm
    from torch.cuda import empty_cache
    
    print("\nStarting validation...")
    model.eval()
    empty_cache()
    
    # Initialize tracking variables
    total_val_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Validation loop
    with torch.no_grad():
        progress_bar = tqdm.tqdm(val_dataloader, desc="Validating")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Validate batch structure
            if not validate_batch_structure(batch, config):
                continue
                
            
            # Forward pass
            outputs = model.forward_backward(**batch, compute_backward=False)
            loss = outputs.loss
            logits = outputs.logits
            
            # Update metrics
            total_val_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == batch['labels']).sum().item()
            total_samples += len(batch['labels'])
            
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log batch-level metrics
            wandb.log({
                "val/batch_loss": loss.item(),
                "val/batch_accuracy": (predictions == batch['labels']).float().mean().item(),
                "val/batch_size": len(batch['labels']),
                "epoch": epoch
            }, step=global_step )
            
            # Clear memory
            del outputs, loss, logits, predictions, batch
            empty_cache()
    
    
    # Calculate average metrics
    avg_val_loss = total_val_loss / len(val_dataloader)
    accuracy = total_correct / total_samples
    
    
    # Combine all metrics
    val_metrics = {
        'loss': avg_val_loss,
        'accuracy': accuracy,
        'num_samples': total_samples / len(val_dataloader),
        'mrr': 0
    }

    # Log epoch-level validation metrics
    wandb_val_metrics = {f'val/{k}': v for k, v in val_metrics.items()}
    wandb.log({
        **wandb_val_metrics,
        "epoch": epoch
    }, step=global_step)
    
    # Print validation metrics
    print("\nValidation metrics:")
    for metric, value in val_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    # Clear memory
    empty_cache()
    
    return val_metrics

class TrainingManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [config.cite_token, config.ref_token]
        })
        
        # Update config with tokenizer-dependent values
        config.cite_token_id = self.tokenizer.convert_tokens_to_ids(config.cite_token)
        config.ref_token_id = self.tokenizer.convert_tokens_to_ids(config.ref_token)
        config.vocab_size = len(self.tokenizer)
        
        # Initialize model
        self.model = CitationModel(config)
        
        # Load checkpoint if specified
        if config.resume_from:
            self.checkpoint = self.load_checkpoint(config.resume_from)
    
    def get_checkpoint_path(self, step: Optional[int] = None, epoch: Optional[int] = None, is_best: bool = False) -> Path:
        checkpoint_dir = self.config.get_checkpoint_dir()
        
        if is_best:
            return checkpoint_dir / "best_model.pt"
        elif step is not None:
            return checkpoint_dir / f"checkpoint-step-{step}.pt"
        elif epoch is not None:
            return checkpoint_dir / f"checkpoint-epoch-{epoch}.pt"
        else:
            raise ValueError("Must specify either step, epoch, or is_best=True")
        
    def save_checkpoint(self, 
                    path: Path, 
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    epoch: Optional[int] = None,
                    batch_in_epoch: Optional[int] = None,
                    global_step: Optional[int] = None,
                    val_metrics: Optional[dict] = None,
                    best_val_metrics: Optional[dict] = None,
                    wandb_run_id: Optional[str] = None,
                    is_best: bool = False):
        
        # Save RNG states as numpy arrays
        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().cpu().numpy(),
            'cuda': torch.cuda.get_rng_state().cpu().numpy() if torch.cuda.is_available() else None
        }
        
        # Prepare save dictionary
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'rng_state': rng_state,
        }
        
        # Add optional states
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            save_dict['epoch'] = epoch
        if batch_in_epoch is not None:
            save_dict['batch_in_epoch'] = batch_in_epoch
        if global_step is not None:
            save_dict['global_step'] = global_step
        if val_metrics is not None:
            save_dict['validation_metrics'] = val_metrics
        if best_val_metrics is not None:
            save_dict['best_val_metrics'] = best_val_metrics
        if wandb_run_id is not None:
            save_dict['wandb_run_id'] = wandb_run_id
        
        # Save checkpoint and config
        torch.save(save_dict, path)
        self.config.save(path.parent / 'config.yaml')
        
        if is_best:
            print(f"\nSaved new best model to {path}")
            if val_metrics:
                print("Best validation metrics:")
                for metric in ['loss', 'accuracy', 'mrr']:
                    if metric in val_metrics:
                        print(f"  {metric}: {val_metrics[metric]:.4f}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> dict:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load config if present
        if 'config' in checkpoint:
            resume_from = self.config.resume_from
            self.config = checkpoint['config']
            self.config.resume_from = resume_from
        
        # Restore RNG states
        if 'rng_state' in checkpoint:
            random.setstate(checkpoint['rng_state']['python'])
            np.random.set_state(checkpoint['rng_state']['numpy'])
            torch.set_rng_state(torch.tensor(checkpoint['rng_state']['torch'], dtype=torch.uint8))
            if torch.cuda.is_available() and checkpoint['rng_state']['cuda'] is not None:
                torch.cuda.set_rng_state(torch.tensor(checkpoint['rng_state']['cuda'], dtype=torch.uint8))

        # Initialize missing fields with defaults if not present
        default_fields = {
            'optimizer_state_dict': None,
            'scaler_state_dict': None,
            'epoch': 0,
            'batch_in_epoch': 0,
            'global_step': 0,
            'validation_metrics': None,
            'best_val_metrics': {'loss': float('inf')},
            'wandb_run_id': None
        }
        
        for field, default_value in default_fields.items():
            if field not in checkpoint:
                checkpoint[field] = default_value
        
        return checkpoint
    
    def get_model(self) -> CitationModel:
        return self.model
    
    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
    
    def get_tokenized_data(self, cache_path=None):
        if cache_path:
            tokenized_data = prepare_training_data(cache_path=cache_path)
        else:
            preprocessor = CitationExtractor()
            sources, citation_data = preprocessor.find_source_citations()
            tokenized_data = prepare_training_data(sources, citation_data, self.tokenizer, cache_dir="cache")
        return tokenized_data

    def train_citation_matcher(
            self, 
            results: List[dict],
        ) -> CitationModel:
            """
            Training function without grad scaler.
            """
            import wandb
            import gc
            
            config = self.config
            model = self.model
            tokenizer = self.tokenizer
            
            # Set random seeds
            config.set_seed()
            
            # Initialize or resume wandb run
            if config.resume_from:
                checkpoint = self.checkpoint
                wandb_run_id = checkpoint['wandb_run_id']
                print(f"Resuming wandb run: {wandb_run_id}")
                wandb.init(
                    project=config.project_name,
                    name=config.run_name,
                    id=wandb_run_id,
                    resume="must"
                )
            else:
                wandb.init(
                    project=config.project_name,
                    name=config.run_name,
                    config=config,
                )
                
                # Update run name in config if not set
                if not config.run_name:
                    config.run_name = wandb.run.name
            
            # Initialize training state
            global_step = 0
            start_epoch = 0
            batch_in_epoch = 0
            best_val_metrics = {'loss': float('inf')}
            
            # Move model to device
            model = model.to(config.device)
            
            # Initialize optimizer
            optimizer = AdamW([
                {
                    'params': [p for n, p in model.named_parameters() if n != 'logit_scale'],
                    'lr': config.learning_rate,
                    'weight_decay': config.weight_decay,
                    'eps': config.Adam_eps
                },
                {
                    'params': [model.logit_scale],
                    'lr': config.logits_learning_rate,
                    'weight_decay': 0
                }
            ])
            
            # Load checkpoint state if resuming
            if config.resume_from:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                global_step = checkpoint['global_step']
                start_epoch = checkpoint['epoch']
                batch_in_epoch = checkpoint['batch_in_epoch']
                best_val_metrics = checkpoint['best_val_metrics']
                print(f"Resumed from checkpoint at epoch {start_epoch}, batch {batch_in_epoch}, step {global_step}")
            
            for epoch in range(start_epoch, config.num_epochs):
                print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
                
                # Log current scale
                current_scale = model.logit_scale.exp().item()
                print(f"Current logit scale: {current_scale:.4f}")
                wandb.log({"logit_scale": current_scale}, step=global_step)
                
                # Training data preparation
                print("Collating training data with new random masks...")
                batches = create_training_batches(results, tokenizer, config)
                dataset = CitationDataset(batches)
                
                # Create train/val split
                indices = np.arange(len(dataset))
                train_size = int(len(dataset) * config.train_ratio)
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
        
                
                train_dataset = Subset(dataset, train_indices)
                val_dataset = Subset(dataset, val_indices)
                
                # Create dataloaders
                generator = torch.Generator()
                generator.manual_seed(config.seed + epoch)
                
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=citation_collate_fn,
                    generator=generator
                )
                
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=config.val_batch_size, 
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=citation_collate_fn
                )
                
                # Clear memory
                del batches, dataset
                gc.collect()
                torch.cuda.empty_cache()
                
                # Training phase
                model.train()
                # model.transformer.gradient_checkpointing_enable()
                total_train_loss = 0
                train_steps = 0
                
                progress_bar = tqdm.tqdm(train_dataloader, desc="Training")
                
                for batch_idx, batch in enumerate(progress_bar):     
                    if not validate_batch_structure(batch, config):
                        continue
                    # Skip previously processed batches if resuming
                    if epoch == start_epoch and batch_idx < batch_in_epoch:
                        continue
                    
                    optimizer.zero_grad()
                    
                    # Forward-Backward pass with micro-batches
                    outputs = model.forward_backward(**batch, compute_backward=True)
                    loss = outputs.loss
                    
                    # Backward pass
                    # loss.backward()
                    
                    if config.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    optimizer.step()
                    
                    # Update tracking
                    total_train_loss += loss.item()
                    train_steps += 1
                    
                    # Log metrics
                    wandb.log({
                        "train/batch_loss": loss,
                        'logit_scale': model.logit_scale.item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/batch_in_epoch": batch_idx,
                        "train/num_cited_art_ids": batch['cited_art_ids'].shape[0],
                        "train/num_target_art_ids": batch['target_art_ids'].shape[0],
                        "epoch": epoch
                    }, step=global_step)
                    
                    progress_bar.set_postfix({'loss': loss})
                    
                    # Save checkpoint periodically
                    if global_step > 0 and global_step % config.checkpoint_every == 0:
                        checkpoint_path = self.get_checkpoint_path(step=global_step)
                        self.save_checkpoint(
                            checkpoint_path,
                            optimizer=optimizer,
                            epoch=epoch,
                            batch_in_epoch=batch_idx,
                            global_step=global_step,
                            wandb_run_id=wandb.run.id
                        )
                        print(f"\nSaved checkpoint at step {global_step} to {checkpoint_path}")
                    
                    global_step += 1
                    
                    # Clear memory
                    del outputs, loss, batch
                    torch.cuda.empty_cache()
                
                # Log epoch-level training metrics
                avg_train_loss = total_train_loss / train_steps
                print(f"\nAverage training loss: {avg_train_loss:.4f}")
                wandb.log({
                    "train/epoch_loss": avg_train_loss,
                    "epoch": epoch
                }, step=global_step)
                
                # Validation phase
                if len(val_dataloader)==0:
                    print("\nValidation set is empty, so continuing with training ... ")
                    continue
                print("\nRunning validation...")
                torch.cuda.empty_cache()
                val_metrics = validate_epoch(
                    model=model,
                    val_dataloader=val_dataloader,
                    global_step=global_step,
                    epoch=epoch,
                    wandb=wandb,
                    config=config
                )
                
                # Save best model if validation loss improved
                if val_metrics['loss'] < best_val_metrics['loss']:
                    best_val_metrics = val_metrics
                    best_model_path = self.get_checkpoint_path(is_best=True)
                    self.save_checkpoint(
                        best_model_path,
                        optimizer=optimizer,
                        epoch=epoch,
                        batch_in_epoch=batch_idx,
                        global_step=global_step,
                        val_metrics=val_metrics,
                        best_val_metrics=best_val_metrics,
                        wandb_run_id=wandb.run.id,
                        is_best=True
                    )
                    
                    # Update wandb summary with best metrics
                    wandb.run.summary.update({
                        "best_val_loss": val_metrics['loss'],
                        "best_val_accuracy": val_metrics['accuracy'],
                        "best_val_mrr": val_metrics['mrr'],
                        "best_model_epoch": epoch,
                        "best_model_step": global_step
                    })
                
                # Save epoch checkpoint
                epoch_checkpoint_path = self.get_checkpoint_path(epoch=epoch)
                self.save_checkpoint(
                    epoch_checkpoint_path,
                    optimizer=optimizer,
                    epoch=epoch,
                    batch_in_epoch=batch_idx,
                    global_step=global_step,
                    val_metrics=val_metrics,
                    best_val_metrics=best_val_metrics,
                    wandb_run_id=wandb.run.id
                )
                
                # Clear memory after each epoch
                del val_metrics, train_dataloader, val_dataloader
                gc.collect()
                torch.cuda.empty_cache()
            
            wandb.finish()
            return model