# Standard library imports
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional, Tuple, Union

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
        
        # Resize token embeddings if needed
        if config.vocab_size != self.transformer.config.vocab_size:
            self.transformer.resize_token_embeddings(config.vocab_size)

        # Add learnable logit scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * config.initial_logit_scale)

    
    def get_citation_masks(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for citation token positions."""
        return input_ids == self.config.cite_token_id
    
    def get_reference_masks(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for reference token positions."""
        return input_ids == self.config.ref_token_id
    
    def forward(
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
        """Forward pass of the model."""
        
        # Process source text
        source_outputs = self.transformer(
            input_ids=source_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Process target text
        target_outputs = self.transformer(
            input_ids=target_ids,
            attention_mask=target_attention_mask,
            return_dict=True
        )
        
        # Get citation mask and extract citation embeddings
        cite_mask = self.get_citation_masks(source_ids)
        cite_embeds = source_outputs.last_hidden_state[cite_mask]
        
        # Get reference mask and extract reference embeddings
        ref_mask = self.get_reference_masks(target_ids)
        ref_embeds = target_outputs.last_hidden_state[ref_mask]
        
        # Normalize embeddings
        cite_embeds = F.normalize(cite_embeds, p=2, dim=-1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=-1)
        
        # Clamp logit scale to prevent numerical instability
        logit_scale = torch.clamp(self.logit_scale, 0, torch.log(torch.tensor(20.0)))
        
        # Compute similarity scores with learned scale
        logits = torch.matmul(cite_embeds, ref_embeds.t()) * logit_scale.exp()

        # compute the loss 
        loss = F.cross_entropy(logits, labels)
        
        if return_dict:
            return CitationModelOutput(
                loss=loss,
                logits=logits,
                cite_embeds=cite_embeds,
                ref_embeds=ref_embeds
            )
        
        return (loss, logits, cite_embeds, ref_embeds)