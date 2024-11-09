from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
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

class TokenEmbeddingExtractor:
    """Handles extraction of token embeddings from model outputs."""
    
    @staticmethod
    def get_token_embedding(
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int
    ) -> torch.Tensor:
        """
        Extracts embeddings for a specific token from each sequence in the batch.
        
        Args:
            hidden_states: Model output hidden states [batch_size, seq_len, hidden_dim]
            input_ids: Input token IDs [batch_size, seq_len]
            token_id: ID of the token to extract
            
        Returns:
            torch.Tensor: Token embeddings [batch_size, hidden_dim]
        """
        batch_size = input_ids.size(0)
        embeddings = []
        
        for batch_idx in range(batch_size):
            token_positions = (input_ids[batch_idx] == token_id).nonzero()
            
            if len(token_positions) == 0:
                raise ValueError(f"Token ID {token_id} not found in sequence {batch_idx}")
            
            # Use the last occurrence of the token if multiple exist
            position = token_positions[-1].item()
            embedding = hidden_states[batch_idx, position, :]
            embeddings.append(embedding)
        
        return torch.stack(embeddings)

class TextEncoder:
    """Handles text encoding and embedding extraction."""
    
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.embedding_extractor = TokenEmbeddingExtractor()

    def encode_text(
        self,
        text: str,
        target_token_id: int
    ) -> torch.Tensor:
        """
        Encodes text and extracts the embedding for a specific token.
        
        Args:
            text: Input text to encode
            target_token_id: ID of the token to extract embedding for
            
        Returns:
            torch.Tensor: Normalized token embedding
        """
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.config.device)
        
        # Get model outputs
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract and normalize token embedding
        token_embedding = self.embedding_extractor.get_token_embedding(
            outputs.last_hidden_state,
            inputs['input_ids'],
            target_token_id
        )
        
        return nn.functional.normalize(token_embedding, dim=-1)

class CitationMatcher(nn.Module):
    """Main citation matching model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
            add_prefix_space=True
        )
        
        # Add special tokens
        special_tokens = {
            'additional_special_tokens': [config.cite_token, config.ref_token]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Set pad token if not present (needed for some models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize encoder model
        self.model = AutoModel.from_pretrained(
            config.model_name,
        ).to(config.device)
        
        # Resize token embeddings to account for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize text encoder
        self.text_encoder = TextEncoder(self.model, self.tokenizer, config)

    def forward(
        self,
        source_contexts: List[str],
        target_pages: List[str]
    ) -> torch.Tensor:
        """
        Compute similarity between source and target pages.
        
        Args:
            source_contexts: List of source text contexts
            target_pages: List of target pages
            
        Returns:
            torch.Tensor: Similarity matrix between sources and targets
        """
        # Get token IDs
        cite_token_id = self.tokenizer.convert_tokens_to_ids(self.config.cite_token)
        ref_token_id = self.tokenizer.convert_tokens_to_ids(self.config.ref_token)
        
        # Encode all texts
        source_embeddings = []
        target_embeddings = []
        
        for context in source_contexts:
            source_emb = self.text_encoder.encode_text(context, cite_token_id)
            source_embeddings.append(source_emb)
        
        for page in target_pages:
            target_emb = self.text_encoder.encode_text(page, ref_token_id)
            target_embeddings.append(target_emb)
        
        # Stack all embeddings
        source_embeddings = torch.cat(source_embeddings, dim=0)
        target_embeddings = torch.cat(target_embeddings, dim=0)
        
        # Compute similarity matrix
        similarity = torch.matmul(
            source_embeddings,
            target_embeddings.transpose(0, 1)
        ) / self.config.temperature
        
        return similarity

@dataclass
class DatasetStats:
    """Statistics for dataset preprocessing."""
    total_samples: int = 0
    processed: int = 0
    skipped_no_cite: int = 0
    skipped_errors: int = 0
    
    def update(self, processed: int = 0, skipped_no_cite: int = 0, skipped_errors: int = 0):
        self.processed += processed
        self.skipped_no_cite += skipped_no_cite
        self.skipped_errors += skipped_errors
    
    def __str__(self) -> str:
        return (f"Total samples: {self.total_samples}\n"
                f"Processed: {self.processed}\n"
                f"Skipped (no citation): {self.skipped_no_cite}\n"
                f"Skipped (errors): {self.skipped_errors}")

class BatchProcessor:
    """Handles batch processing of text samples."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        cite_token: str,
        ref_token: str,
        batch_size: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cite_token_id = tokenizer.convert_tokens_to_ids(cite_token)
        self.ref_token_id = tokenizer.convert_tokens_to_ids(ref_token)
        self.batch_size = batch_size

    def process_batch(
        self,
        sources: List[str],
        targets: List[str]
    ) -> Tuple[List[Dict[str, torch.Tensor]], DatasetStats]:
        """Process a batch of samples efficiently."""
        # Batch tokenization
        source_tokens = self.tokenizer(
            sources,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        target_tokens = self.tokenizer(
            targets,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process all target tokens at once
        target_tokens['input_ids'] = self._ensure_ref_token_batch(target_tokens['input_ids'])
        
        # Check citation tokens for all sources at once
        has_citation = (source_tokens['input_ids'] == self.cite_token_id).any(dim=1)
        
        processed_samples = []
        stats = DatasetStats(total_samples=len(sources))
        
        for idx in range(len(sources)):
            if has_citation[idx]:
                sample = {
                    'source_input_ids': source_tokens['input_ids'][idx],
                    'source_attention_mask': source_tokens['attention_mask'][idx],
                    'target_input_ids': target_tokens['input_ids'][idx],
                    'target_attention_mask': target_tokens['attention_mask'][idx]
                }
                processed_samples.append(sample)
                stats.update(processed=1)
            else:
                stats.update(skipped_no_cite=1)
        
        return processed_samples, stats

    def _ensure_ref_token_batch(self, tokens: torch.Tensor) -> torch.Tensor:
        """Ensure reference token is present in batch of sequences."""
        batch_size = tokens.size(0)
        
        # Find all ref token positions in batch
        ref_positions = (tokens == self.ref_token_id).nonzero(as_tuple=True)
        batch_indices = ref_positions[0].unique()
        
        # Handle sequences without ref token
        missing_ref = torch.ones(batch_size, dtype=torch.bool)
        missing_ref[batch_indices] = False
        
        if missing_ref.any():
            # Find last non-pad position for sequences missing ref token
            pad_mask = (tokens != self.tokenizer.pad_token_id)
            last_nonpad = pad_mask.long().argmax(dim=1)
            
            # Add ref token at last non-pad position
            missing_indices = missing_ref.nonzero(as_tuple=True)[0]
            tokens[missing_indices, last_nonpad[missing_indices]] = self.ref_token_id
        
        # Handle multiple ref tokens
        ref_counts = (tokens == self.ref_token_id).sum(dim=1)
        multiple_refs = ref_counts > 1
        
        if multiple_refs.any():
            for idx in multiple_refs.nonzero(as_tuple=True)[0]:
                ref_pos = (tokens[idx] == self.ref_token_id).nonzero(as_tuple=True)[0]
                # Keep only the last ref token
                tokens[idx, ref_pos[:-1]] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        
        return tokens

class CitationDataset(Dataset):
    """Dataset for citation matching with optimized batch processing."""
    
    def __init__(
        self,
        sources: List[str],
        targets: List[str],
        tokenizer: AutoTokenizer,
        config: "ModelConfig",
        batch_size: int = 1024,
        verbose: bool = True
    ):
        assert len(sources) == len(targets), "Sources and targets must have same length"
        self.config = config
        self.processor = BatchProcessor(
            tokenizer=tokenizer,
            max_length=config.max_length,
            cite_token=config.cite_token,
            ref_token=config.ref_token,
            batch_size=batch_size
        )
        self.processed_samples = self._preprocess_samples(sources, targets, verbose)

    def _preprocess_samples(
        self,
        sources: List[str],
        targets: List[str],
        verbose: bool
    ) -> List[Dict[str, torch.Tensor]]:
        """Preprocess samples in batches."""
        total_samples = len(sources)
        batch_size = self.processor.batch_size
        processed_samples = []
        total_stats = DatasetStats(total_samples=total_samples)
        
        # Process in batches with progress bar
        iterator = range(0, total_samples, batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Processing samples", unit="batch")
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, total_samples)
            batch_sources = sources[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            try:
                batch_samples, batch_stats = self.processor.process_batch(
                    batch_sources,
                    batch_targets
                )
                processed_samples.extend(batch_samples)
                total_stats.update(
                    processed=batch_stats.processed,
                    skipped_no_cite=batch_stats.skipped_no_cite,
                    skipped_errors=batch_stats.skipped_errors
                )
                
            except Exception as e:
                if verbose:
                    print(f"Error processing batch {start_idx}-{end_idx}: {str(e)}")
                total_stats.update(skipped_errors=end_idx - start_idx)
        
        if verbose:
            print("\nPreprocessing Statistics:")
            print(str(total_stats))
        
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
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates batch of samples into training format."""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }