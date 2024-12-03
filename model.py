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
        # self.transformer = torch.compile(self.transformer)
        # self.transformer.to(config.device)
        
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
                
                if embedding_gradients is None:
                    all_embeddings.append(curr_embeds)
                else:
                    num_curr_embeds = curr_embeds.size(0)
                    curr_grads = embedding_gradients[grad_idx:grad_idx + num_curr_embeds]
                    loss = torch.sum(curr_embeds * curr_grads)
                    loss.backward()
                    accumulated_loss += loss.item()
                    grad_idx += num_curr_embeds


        torch.cuda.empty_cache()
        
        if embedding_gradients is None:
            return torch.cat(all_embeddings, dim=0)
        else:
            return accumulated_loss
                    

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
        compute_backward: bool = False,
    ) -> Union[Tuple, CitationModelOutput]:
        """Forward pass with gradient accumulation using micro-batches."""
        
        # Move labels to device
        labels = labels.to(self.config.device)
    
        with torch.no_grad():
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

        with torch.amp.autocast('cuda', dtype=torch.float16):
            with torch.enable_grad():
                # Create copies that require gradients
                all_cite_embeds.requires_grad_(True)
                all_ref_embeds.requires_grad_(True)
                
                # Compute similarity and loss
                logit_scale = torch.clamp(self.logit_scale, 0, torch.log(torch.tensor(20.0, device=self.config.device)))
                logits = torch.matmul(all_cite_embeds, all_ref_embeds.t()) * logit_scale.exp()
                loss = F.cross_entropy(logits, labels)
                
                # Compute gradients
                loss.backward()
        
                if compute_backward:        
                    
                    # Second pass: compute loss using gradients
                    self.forward_backward_microbatches(
                        input_ids=source_ids,
                        mask_token_id=self.config.cite_token_id,
                        attention_mask=attention_mask,
                        embedding_gradients=all_cite_embeds.grad,
                    )
                    
                    self.forward_backward_microbatches(
                        input_ids=target_ids,
                        mask_token_id=self.config.ref_token_id,
                        attention_mask=target_attention_mask,
                        embedding_gradients=all_ref_embeds.grad,
                    )
                    
                    # Clean up intermediate tensors
                    # del all_cite_embeds, all_ref_embeds, cite_grads, ref_grads

        all_ref_embeds.requires_grad_(False)
        all_cite_embeds.requires_grad_(False)
        if return_dict:
            return CitationModelOutput(
                loss=loss.detach().cpu(),
                logits=logits.detach().cpu(),
                cite_embeds=all_cite_embeds.cpu(),
                ref_embeds=all_ref_embeds.cpu()
            )
        
        return (loss, logits, all_cite_embeds, all_ref_embeds)
    
