from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

@dataclass
class ModelConfig:
    """Configuration for the citation matching model."""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    cite_token: str = "<CITE>"
    ref_token: str = "<REF>"
    temperature: float = 0.07
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CitationMatcher(nn.Module):
    """Main citation matching model with integrated text encoding."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
            add_prefix_space=True
        )
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [config.cite_token, config.ref_token]
        })
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = AutoModel.from_pretrained(config.model_name).to(config.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _extract_token_embedding(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int
    ) -> torch.Tensor:
        """Extracts and normalizes embeddings for a specific token."""
        batch_size = input_ids.size(0)
        embeddings = []
        
        for batch_idx in range(batch_size):
            token_positions = (input_ids[batch_idx] == token_id).nonzero()
            if len(token_positions) == 0:
                raise ValueError(f"Token ID {token_id} not found in sequence {batch_idx}")
            position = token_positions[-1].item()
            embeddings.append(hidden_states[batch_idx, position, :])
        
        embeddings = torch.stack(embeddings)
        return nn.functional.normalize(embeddings, dim=-1)

    def _encode_text(self, text: str, token_id: int) -> torch.Tensor:
        """Encodes text and extracts normalized token embedding."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.config.device)
        
        outputs = self.model(**inputs, return_dict=True)
        return self._extract_token_embedding(
            outputs.last_hidden_state,
            inputs['input_ids'],
            token_id
        )

    def forward(
        self,
        source_contexts: List[str],
        target_pages: List[str]
    ) -> torch.Tensor:
        """Compute similarity between source and target pages."""
        cite_token_id = self.tokenizer.convert_tokens_to_ids(self.config.cite_token)
        ref_token_id = self.tokenizer.convert_tokens_to_ids(self.config.ref_token)
        
        # Encode all texts
        source_embeddings = torch.cat([
            self._encode_text(ctx, cite_token_id) for ctx in source_contexts
        ], dim=0)
        
        target_embeddings = torch.cat([
            self._encode_text(page, ref_token_id) for page in target_pages
        ], dim=0)
        
        # Compute similarity matrix
        return torch.matmul(
            source_embeddings,
            target_embeddings.transpose(0, 1)
        ) / self.config.temperature

class CitationDataset(Dataset):
    """Dataset for citation matching with optimized batch processing."""
    
    def __init__(
        self,
        sources: List[str],
        targets: List[str],
        tokenizer: AutoTokenizer,
        config: ModelConfig,
        batch_size: int = 1024,
        verbose: bool = True
    ):
        assert len(sources) == len(targets), "Sources and targets must have same length"
        self.processed_samples = self._preprocess_samples(
            sources, targets, tokenizer, config, batch_size, verbose
        )

    def _ensure_ref_token_batch(
        self,
        tokens: torch.Tensor,
        ref_token_id: int,
        pad_token_id: int
    ) -> torch.Tensor:
        """Ensure reference token is present in batch of sequences."""
        # Find sequences missing ref token
        has_ref = (tokens == ref_token_id).any(dim=1)
        missing_ref = ~has_ref

        if missing_ref.any():
            # Find last non-pad position for sequences missing ref token
            pad_mask = (tokens != pad_token_id)
            last_nonpad = pad_mask.long().argmax(dim=1)
            
            # Add ref token at last non-pad position
            missing_indices = missing_ref.nonzero(as_tuple=True)[0]
            tokens[missing_indices, last_nonpad[missing_indices]] = ref_token_id
        
        return tokens

    def _preprocess_samples(
        self,
        sources: List[str],
        targets: List[str],
        tokenizer: AutoTokenizer,
        config: ModelConfig,
        batch_size: int,
        verbose: bool
    ) -> List[Dict[str, torch.Tensor]]:
        """Preprocess samples with optimized batch processing."""
        total_samples = len(sources)
        processed_samples = []
        total_processed = 0
        total_skipped = 0
        
        # Process in batches
        iterator = range(0, total_samples, batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Processing samples", unit="batch")
        
        cite_token_id = tokenizer.convert_tokens_to_ids(config.cite_token)
        ref_token_id = tokenizer.convert_tokens_to_ids(config.ref_token)
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, total_samples)
            batch_sources = sources[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            # Batch tokenization
            source_tokens = tokenizer(
                batch_sources,
                padding='max_length',
                truncation=True,
                max_length=config.max_length,
                return_tensors='pt'
            )
            
            target_tokens = tokenizer(
                batch_targets,
                padding='max_length',
                truncation=True,
                max_length=config.max_length,
                return_tensors='pt'
            )
            
            # Process source tokens
            has_citation = (source_tokens['input_ids'] == cite_token_id).any(dim=1)
            valid_indices = has_citation.nonzero(as_tuple=True)[0]
            
            # Update statistics
            total_processed += len(valid_indices)
            total_skipped += len(batch_sources) - len(valid_indices)
            
            if len(valid_indices) == 0:
                continue
            
            # Ensure ref token in target sequences
            target_tokens['input_ids'] = self._ensure_ref_token_batch(
                target_tokens['input_ids'],
                ref_token_id,
                tokenizer.pad_token_id
            )
            
            # Create samples for valid sequences
            for idx in valid_indices:
                processed_samples.append({
                    'source_input_ids': source_tokens['input_ids'][idx],
                    'source_attention_mask': source_tokens['attention_mask'][idx],
                    'target_input_ids': target_tokens['input_ids'][idx],
                    'target_attention_mask': target_tokens['attention_mask'][idx]
                })
        
        if verbose:
            print(f"\nProcessed {total_processed} samples")
            print(f"Skipped {total_skipped} samples without citation")
        
        return processed_samples

    def __len__(self) -> int:
        return len(self.processed_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_samples[idx]

def create_dataloader(
    dataset: CitationDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Creates a DataLoader for the citation dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0].keys()
        },
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )