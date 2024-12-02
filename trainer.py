# Standard library imports
import random 
from pathlib import Path
from typing import List, Optional, Union

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
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
    
def validate_citation_matcher(
    model,
    val_dataloader,
    return_embeddings: bool = False,
    k_values: List[int] = [1, 5, 10, 50, 100, 1000],
    similarity_batch_size: int = 512,
    config: TrainingConfig = None
):
    device = config.device
    
    model.eval()
    
    # Lists to store accumulated embeddings and IDs
    all_cite_embeds = []
    all_ref_embeds = []
    all_cited_art_ids = []
    all_target_art_ids = []
    
    # Accumulate embeddings and IDs
    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader, desc="Computing embeddings"):
            if not validate_batch_structure(batch, config):
                continue
            # Move batch to device and convert to FP16
            batch = {k: (v.to(device, dtype=torch.float16) if isinstance(v, torch.FloatTensor) 
                        else v.to(device)) for k, v in batch.items()}
            
            # Process source text
            source_outputs = model.transformer(
                input_ids=batch['source_ids'],
                attention_mask=batch['attention_mask'],
                return_dict=True
            )
            
            # Process target text
            target_outputs = model.transformer(
                input_ids=batch['target_ids'],
                attention_mask=batch['target_attention_mask'],
                return_dict=True
            )
            
            # Extract embeddings with masks
            cite_mask = model.get_citation_masks(batch['source_ids'])
            cite_embeds = source_outputs.last_hidden_state[cite_mask]
            ref_mask = model.get_reference_masks(batch['target_ids'])
            ref_embeds = target_outputs.last_hidden_state[ref_mask]
            
            # Normalize and move to CPU immediately
            cite_embeds = F.normalize(cite_embeds, p=2, dim=-1).cpu()
            ref_embeds = F.normalize(ref_embeds, p=2, dim=-1).cpu()
            
            # Store embeddings and IDs on CPU
            all_cite_embeds.append(cite_embeds)
            all_ref_embeds.append(ref_embeds)
            all_cited_art_ids.append(batch['cited_art_ids'].cpu())
            all_target_art_ids.append(batch['target_art_ids'].cpu())
            
            # Clear GPU cache after each batch
            del source_outputs, target_outputs, cite_embeds, ref_embeds
            torch.cuda.empty_cache()
    
    # Concatenate all accumulated tensors
    cite_embeds = torch.cat(all_cite_embeds)
    ref_embeds = torch.cat(all_ref_embeds)
    cited_art_ids = torch.cat(all_cited_art_ids)
    target_art_ids = torch.cat(all_target_art_ids)
    
    # Get unique target art IDs and create mapping
    target_art_ids_unique, unique_indices = np.unique(target_art_ids.numpy(), return_index=True)
    target_art_ids_unique = torch.tensor(target_art_ids_unique)
    ref_embeds_unique = ref_embeds[torch.tensor(unique_indices)]
    
    # Create ID to index mapping
    id2i = {id.item(): i for i, id in enumerate(target_art_ids_unique)}
    labels = torch.tensor([id2i[id.item()] for id in cited_art_ids], dtype=torch.long)
    
    # Process in smaller batches for similarity computation
    total_loss = 0
    total_correct = 0
    all_predictions = []
    logits_list = []  # Store logits temporarily for metrics computation
    labels_list = []  # Store labels temporarily for metrics computation
    
    num_batches = (len(cite_embeds) + similarity_batch_size - 1) // similarity_batch_size
    logit_scale = torch.clamp(model.logit_scale, 0, torch.log(torch.tensor(20.0)))
    
    # Move ref_embeds to GPU once
    ref_embeds_unique = ref_embeds_unique.to(device)
    
    for i in tqdm.tqdm(range(num_batches), desc="Computing similarities"):
        start_idx = i * similarity_batch_size
        end_idx = min((i + 1) * similarity_batch_size, len(cite_embeds))
        
        # Process batch
        cite_embeds_batch = cite_embeds[start_idx:end_idx].to(device)
        labels_batch = labels[start_idx:end_idx].to(device)
        
        # Compute similarities and loss
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits_batch = torch.matmul(cite_embeds_batch, ref_embeds_unique.t()) * logit_scale.exp()
            loss_batch = F.cross_entropy(logits_batch, labels_batch)
        
        total_loss += loss_batch.item() * len(labels_batch)
        predictions_batch = torch.argmax(logits_batch, dim=-1)
        total_correct += (predictions_batch == labels_batch).sum().item()
        
        # Store predictions and move to CPU
        all_predictions.append(predictions_batch.cpu())
        logits_list.append(logits_batch.cpu())
        labels_list.append(labels_batch.cpu())
        
        # Clear GPU memory
        del logits_batch, cite_embeds_batch, labels_batch, predictions_batch
        torch.cuda.empty_cache()
    
    # Compute final metrics
    num_citations = len(cite_embeds)
    accuracy = total_correct / num_citations
    avg_loss = total_loss / num_citations
    
    # Compute retrieval metrics
    all_logits = torch.cat(logits_list)
    all_labels = torch.cat(labels_list)
    retrieval_metrics = compute_retrieval_metrics(all_logits, all_labels, ks=k_values)
    
    # Clear temporary lists
    del logits_list, labels_list
    torch.cuda.empty_cache()
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'num_citations': num_citations,
        'num_unique_targets': len(target_art_ids_unique),
        'mrr': retrieval_metrics['mrr']
    }
    
    # Add top-k accuracies
    for k in k_values:
        if f'top_{k}_accuracy' in retrieval_metrics:
            results[f'top_{k}_accuracy'] = retrieval_metrics[f'top_{k}_accuracy']
    
    if return_embeddings:
        results.update({
            'cite_embeds': cite_embeds,
            'ref_embeds': ref_embeds_unique.cpu(),
            'cited_art_ids': cited_art_ids,
            'target_art_ids': target_art_ids_unique,
            'logits': all_logits,
            'labels': labels
        })
    
    # Final cleanup
    del cite_embeds, ref_embeds, ref_embeds_unique
    torch.cuda.empty_cache()
    
    return results



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
            self.load_checkpoint(config.resume_from)
    
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
                       scaler: Optional[GradScaler] = None,
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
        if scaler is not None:
            save_dict['scaler_state_dict'] = scaler.state_dict()
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
        self.config.save(path.parent)
        
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
    
    def get_results(self, cache_path=None):
        if cache_path:
            results = prepare_training_data(cache_path=cache_path)
        else:
            preprocessor = CitationExtractor()
            sources, citation_data = preprocessor.find_source_citations()
            results = prepare_training_data(sources, citation_data, self.tokenizer, cache_dir="cache")
        return results

    
    def train_citation_matcher(
        self, 
        results: List[dict],
    ) -> CitationModel:
        """
        Memory-optimized training function with enhanced checkpoint management.
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
            checkpoint = self.load_checkpoint(config.resume_from)
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
        scaler = GradScaler()
        
        # Move model to device and enable memory efficient training
        model = model.to(config.device)
        model.transformer.gradient_checkpointing_enable()
        
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
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
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
    
            from torch.utils.data import Subset
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
                batch_size=int(config.batch_size * 1.8),
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
            total_train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm.tqdm(train_dataloader, desc="Training")
            
            for batch_idx, batch in enumerate(progress_bar):     
                if not validate_batch_structure(batch, config):
                    continue
                # Skip previously processed batches if resuming
                if epoch == start_epoch and batch_idx < batch_in_epoch:
                    continue
                
                batch = {k: v.to(config.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**batch)
                    loss = outputs.loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if config.max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                
                # Update tracking
                total_train_loss += loss.item()
                train_steps += 1
                
                # Log metrics
                wandb.log({
                    "train/batch_loss": loss.item(),
                    'logit_scale': model.logit_scale.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/batch_in_epoch": batch_idx,
                    "epoch": epoch
                }, step=global_step)
                
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Save checkpoint periodically
                if global_step > 0 and global_step % config.checkpoint_every == 0:
                    checkpoint_path = self.get_checkpoint_path(step=global_step)
                    self.save_checkpoint(
                        checkpoint_path,
                        optimizer=optimizer,
                        scaler=scaler,
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
                print("\nValidation set is emptpy, so continuing with training ... ")
                continue
            print("\nRunning validation...")
            torch.cuda.empty_cache()
            model.eval()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                val_metrics = validate_citation_matcher(
                    model=model,
                    val_dataloader=val_dataloader,
                    k_values=config.k_values,
                    config=config,
                )
            
            # Log validation metrics
            wandb_val_metrics = {
                "val/loss": val_metrics['loss'],
                "val/accuracy": val_metrics['accuracy'],
                "val/mrr": val_metrics['mrr']
            }
            
            for k in config.k_values:
                if f'top_{k}_accuracy' in val_metrics:
                    wandb_val_metrics[f"val/top_{k}_accuracy"] = val_metrics[f'top_{k}_accuracy']
            
            wandb.log(wandb_val_metrics, step=global_step)
            
            # Print validation metrics
            print(f"\nValidation metrics:")
            for metric, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
            
            # Save best model if validation loss improved
            if val_metrics['loss'] < best_val_metrics['loss']:
                best_val_metrics = val_metrics
                best_model_path = self.get_checkpoint_path(is_best=True)
                self.save_checkpoint(
                    best_model_path,
                    optimizer=optimizer,
                    scaler=scaler,
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
                scaler=scaler,
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

