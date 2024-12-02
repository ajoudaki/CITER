# Standard library imports
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict

# Third-party imports
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
)

from config import TrainingConfig

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

@dataclass
class CitationModelOutput:
    """Custom output class for the citation model."""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cite_embeds: Optional[torch.FloatTensor] = None
    ref_embeds: Optional[torch.FloatTensor] = None

class CitationModel(nn.Module):
    """Custom model for citation matching using transformer embeddings."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        # Load base model configuration
        base_config = AutoConfig.from_pretrained(config.model_name)
        
        # Store configuration
        self.config = config
        
        # Load base transformer model
        self.transformer = AutoModel.from_pretrained(config.model_name)
        self.transformer.to(config.device)
        
        # Resize token embeddings if needed
        if config.vocab_size != self.transformer.config.vocab_size:
            self.transformer.resize_token_embeddings(config.vocab_size)

        # Add learnable logit scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * config.initial_logit_scale)
        self.logit_scale.to(config.device)

    def get_citation_masks(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for citation token positions."""
        return input_ids == self.config.cite_token_id
    
    def get_reference_masks(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for reference token positions."""
        return input_ids == self.config.ref_token_id
      
    def forward_backward_microbatches(
        self,
        input_ids: torch.Tensor,
        mask_token_id: int,
        attention_mask: Optional[torch.Tensor] = None,
        embedding_gradients: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.FloatTensor]], torch.Tensor]:
        micro_batch_size = self.config.micro_batch_size
        
        # Create dataset from input tensors
        dataset = torch.utils.data.TensorDataset(
            input_ids, 
            attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            pin_memory=True
        )
        
        all_embeddings = []
        accumulated_loss = torch.tensor(0.0, device=self.config.device) if embedding_gradients is not None else None
        grad_idx = 0
        
        for curr_input_ids, curr_attention_mask in dataloader:
            curr_input_ids = curr_input_ids.to(self.config.device)
            curr_attention_mask = curr_attention_mask.to(self.config.device)
            
            with torch.set_grad_enabled(embedding_gradients is not None):
                outputs = self.transformer(
                    input_ids=curr_input_ids,
                    attention_mask=curr_attention_mask,
                    return_dict=True,
                    output_hidden_states=False
                )
                
                curr_mask = (curr_input_ids == mask_token_id)
                curr_embeds = F.normalize(outputs.last_hidden_state[curr_mask], p=2, dim=-1)
                
                if embedding_gradients is not None:
                    num_curr_embeds = curr_embeds.size(0)
                    curr_grads = embedding_gradients[grad_idx:grad_idx + num_curr_embeds]
                    loss = torch.sum(curr_embeds * curr_grads)
                    loss.backward()
                    accumulated_loss += loss.item()
                    grad_idx += num_curr_embeds
                else:
                    all_embeddings.append(curr_embeds.detach().cpu())
                    
        return accumulated_loss if embedding_gradients is not None else torch.cat(all_embeddings, dim=0).to(self.config.device)

    def forward_backward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        cited_art_ids: Optional[torch.Tensor] = None,
        target_art_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        compute_backward: bool = True,
    ) -> Union[Tuple, CitationModelOutput]:
        """Forward pass with gradient accumulation using micro-batches."""
        
        # Move labels to device
        labels = labels.to(self.config.device)
        
        # First pass: get embeddings and compute initial loss
        all_cite_embeds = self.forward_backward_microbatches(
            input_ids=source_ids,
            mask_token_id=self.config.cite_token_id,
            attention_mask=attention_mask,
        )
        
        all_ref_embeds = self.forward_backward_microbatches(
            input_ids=target_ids,
            mask_token_id=self.config.ref_token_id,
            attention_mask=target_attention_mask,
        )
        
        # Create copies that require gradients
        cite_embeds_grad = all_cite_embeds.detach().requires_grad_(True)
        ref_embeds_grad = all_ref_embeds.detach().requires_grad_(True)
        
        # Compute similarity and loss
        logit_scale = torch.clamp(self.logit_scale, 0, torch.log(torch.tensor(20.0, device=self.config.device)))
        logits = torch.matmul(cite_embeds_grad, ref_embeds_grad.t()) * logit_scale.exp()
        loss = F.cross_entropy(logits, labels)
        
        # Compute gradients
        loss.backward()
        
        if compute_backward:        
            # Get gradient
            cite_grads = cite_embeds_grad.grad.clone()
            ref_grads = ref_embeds_grad.grad.clone()
            
            # Second pass: compute loss using gradients
            self.forward_backward_microbatches(
                input_ids=source_ids,
                mask_token_id=self.config.cite_token_id,
                attention_mask=attention_mask,
                embedding_gradients=cite_grads,
            )
            
            self.forward_backward_microbatches(
                input_ids=target_ids,
                mask_token_id=self.config.ref_token_id,
                attention_mask=target_attention_mask,
                embedding_gradients=ref_grads,
            )
            
            # Clean up intermediate tensors
            del cite_embeds_grad, ref_embeds_grad, cite_grads, ref_grads
            torch.cuda.empty_cache()
            
        if return_dict:
            return CitationModelOutput(
                loss=loss,
                logits=logits,
                cite_embeds=all_cite_embeds,
                ref_embeds=all_ref_embeds
            )
        
        return (loss, logits, all_cite_embeds, all_ref_embeds)
    
    def validate(
        self,
        val_dataloader,
        return_embeddings: bool = False,
    ) -> Dict[str, Union[float, int, torch.Tensor]]:
        """
        Validation method using forward_backward implementation.
        """
        self.eval()
        device = self.config.device
        k_values = self.config.k_values
        
        all_logits = []
        all_labels = []
        total_loss = 0
        total_citations = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_dataloader, desc="Validating"):
                # Process batch
                outputs = self.forward_backward(
                    source_ids=batch['source_ids'].to(device),
                    target_ids=batch['target_ids'].to(device),
                    labels=batch['labels'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    target_attention_mask=batch['target_attention_mask'].to(device),
                    cited_art_ids=batch['cited_art_ids'].to(device),
                    target_art_ids=batch['target_art_ids'].to(device),
                    compute_backward=False
                )
                
                # Accumulate results
                total_loss += outputs.loss.item() * len(batch['labels'])
                total_citations += len(batch['labels'])
                all_logits.append(outputs.logits.cpu())
                all_labels.append(batch['labels'].cpu())
                
                # Clear GPU memory
                torch.cuda.empty_cache()
        
        # Concatenate results
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        # Calculate metrics
        avg_loss = total_loss / total_citations
        accuracy = (all_logits.argmax(dim=-1) == all_labels).float().mean().item()
        
        # Compute retrieval metrics
        retrieval_metrics = compute_retrieval_metrics(all_logits, all_labels, ks=k_values)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_citations': total_citations,
            'mrr': retrieval_metrics['mrr']
        }
        
        # Add top-k accuracies
        for k in k_values:
            if f'top_{k}_accuracy' in retrieval_metrics:
                results[f'top_{k}_accuracy'] = retrieval_metrics[f'top_{k}_accuracy']
        
        if return_embeddings:
            results.update({
                'logits': all_logits,
                'labels': all_labels
            })
        
        return results
    