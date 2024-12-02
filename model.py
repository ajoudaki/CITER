# Standard library imports
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
)

from config import TrainingConfig


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

    def forward_microbatches(
        self,
        input_ids: torch.Tensor,
        mask_token_id: int,
        attention_mask: Optional[torch.Tensor] = None,
        embedding_gradients: Optional[torch.Tensor] = None,
        micro_batch_size: int = 32,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.FloatTensor]], torch.Tensor]:
        micro_batch_size = self.config.micro_batch_size
        total_samples = input_ids.size(0)
        all_embeddings = []
        accumulated_loss = torch.tensor(0.0, device=self.config.device) if embedding_gradients is not None else None
        
        # Track current position in the gradient tensor if provided
        grad_idx = 0
        
        # Process in micro-batches
        for i in range(0, total_samples, micro_batch_size):
            # Get current micro batch
            batch_slice = slice(i, min(i + micro_batch_size, total_samples))
            curr_input_ids = input_ids[batch_slice].to(self.config.device)
            curr_attention_mask = attention_mask[batch_slice].to(self.config.device) if attention_mask is not None else None
            
            # Determine whether to compute gradients
            if embedding_gradients is not None:
                context = torch.enable_grad() 
                self.transformer.train()
                self.transformer.gradient_checkpointing_enable()
            else:
                self.transformer.eval()
                self.transformer.gradient_checkpointing_disable()
                context = torch.no_grad()
            
            with context:
                # Extract embeddings with proper memory management
                outputs = self.transformer(
                    input_ids=curr_input_ids,
                    attention_mask=curr_attention_mask,
                    return_dict=True
                )
                
                # Get mask and extract embeddings
                curr_mask = (curr_input_ids == mask_token_id)
                curr_embeds = outputs.last_hidden_state[curr_mask]
                
                # Clear transformer outputs immediately
                del outputs.last_hidden_state
                del outputs
                
                # Normalize embeddings
                curr_embeds = F.normalize(curr_embeds, p=2, dim=-1)
                
                # Handle embeddings based on gradient presence
                if embedding_gradients is None:
                    detached_embeds = curr_embeds.detach().cpu() 
                    all_embeddings.append(detached_embeds)
                    del curr_embeds
                else:
                    num_curr_embeds = curr_embeds.size(0)
                    curr_grads = embedding_gradients[grad_idx:grad_idx + num_curr_embeds].to(self.config.device)
                    
                    # Compute loss and clean up immediately
                    curr_loss = torch.sum(curr_embeds * curr_grads)
                    curr_loss.backward()
                    accumulated_loss += curr_loss.item()
                    
                    del curr_embeds
                    del curr_grads
                    grad_idx += num_curr_embeds
            
            # Clean up batch tensors
            if curr_attention_mask is not None:
                del curr_attention_mask
            del curr_input_ids, curr_mask
            torch.cuda.empty_cache()
        
        if embedding_gradients is not None:
            return accumulated_loss
        else:
            # Concatenate all embeddings on CPU then move to GPU
            concatenated_embeddings = torch.cat(all_embeddings, dim=0).to(self.config.device)
            return concatenated_embeddings
        
    # def forward_microbatches(
    #     self,
    #     input_ids: torch.Tensor,
    #     mask_token_id: int,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     embedding_gradients: Optional[torch.Tensor] = None,
    # ) -> Union[Tuple[torch.Tensor, Optional[torch.FloatTensor]], torch.Tensor]:
    #     micro_batch_size = self.config.micro_batch_size
        
    #     # Create dataset from input tensors
    #     dataset = torch.utils.data.TensorDataset(
    #         input_ids, 
    #         attention_mask if attention_mask is not None else torch.ones_like(input_ids)
    #     )
        
    #     # Create dataloader
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=micro_batch_size,
    #         shuffle=False,
    #         pin_memory=True
    #     )
        
    #     all_embeddings = []
    #     accumulated_loss = torch.tensor(0.0, device=self.config.device) if embedding_gradients is not None else None
    #     grad_idx = 0
        
    #     for curr_input_ids, curr_attention_mask in dataloader:
    #         curr_input_ids = curr_input_ids.to(self.config.device)
    #         curr_attention_mask = curr_attention_mask.to(self.config.device)
            
    #         with torch.set_grad_enabled(embedding_gradients is not None):
    #             outputs = self.transformer(
    #                 input_ids=curr_input_ids,
    #                 attention_mask=curr_attention_mask,
    #                 return_dict=True,
    #                 output_hidden_states=False
    #             )
                
    #             curr_mask = (curr_input_ids == mask_token_id)
    #             curr_embeds = F.normalize(outputs.last_hidden_state[curr_mask], p=2, dim=-1)
                
    #             if embedding_gradients is not None:
    #                 num_curr_embeds = curr_embeds.size(0)
    #                 curr_grads = embedding_gradients[grad_idx:grad_idx + num_curr_embeds]
    #                 loss = torch.sum(curr_embeds * curr_grads)
    #                 loss.backward()
    #                 accumulated_loss += loss.item()
    #                 grad_idx += num_curr_embeds
    #             else:
    #                 all_embeddings.append(curr_embeds.detach().cpu())
                    
    #     return accumulated_loss if embedding_gradients is not None else torch.cat(all_embeddings, dim=0).to(self.config.device)

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
    ) -> Union[Tuple, CitationModelOutput]:
        """Forward pass with gradient accumulation using micro-batches."""
        
        # Move labels to device
        labels = labels.to(self.config.device)
        
        # First pass: get embeddings and compute initial loss
        all_cite_embeds = self.forward_microbatches(
            input_ids=source_ids,
            mask_token_id=self.config.cite_token_id,
            attention_mask=attention_mask,
        )
        
        all_ref_embeds = self.forward_microbatches(
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
        
        # Get gradient
        cite_grads = cite_embeds_grad.grad.clone()
        ref_grads = ref_embeds_grad.grad.clone()
        
        # Second pass: compute loss using gradients
        self.forward_microbatches(
            input_ids=source_ids,
            mask_token_id=self.config.cite_token_id,
            attention_mask=attention_mask,
            embedding_gradients=cite_grads,
        )
        
        self.forward_microbatches(
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
    