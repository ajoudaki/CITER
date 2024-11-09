from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

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

class CitationDataset(Dataset):
    """Dataset for citation matching."""
    
    def __init__(
        self,
        sources: List[str],
        targets: List[str],
        tokenizer: AutoTokenizer,
        config: ModelConfig,
        verbose: bool = True
    ):
        assert len(sources) == len(targets), "Sources and targets must have same length"
        self.config = config
        self.tokenizer = tokenizer
        self.processed_samples = self._preprocess_samples(sources, targets, verbose)

    def _preprocess_samples(
        self,
        sources: List[str],
        targets: List[str],
        verbose: bool
    ) -> List[Dict[str, torch.Tensor]]:
        """Preprocess and validate all samples."""
        processed_samples = []
        stats = {'skipped_no_cite': 0, 'skipped_errors': 0, 'processed': 0}
        
        for idx, (source, target) in enumerate(zip(sources, targets)):
            try:
                sample = self._process_single_sample(source, target)
                if sample:
                    processed_samples.append(sample)
                    stats['processed'] += 1
                else:
                    stats['skipped_no_cite'] += 1
            except Exception as e:
                stats['skipped_errors'] += 1
                if verbose:
                    print(f"Error processing sample {idx}: {str(e)}")
            
            if verbose and (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1} samples...")
        
        if verbose:
            print(f"Preprocessing stats: {stats}")
        
        return processed_samples

    def _process_single_sample(
        self,
        source: str,
        target: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single source-target pair."""
        # Tokenize inputs
        source_tokens = self.tokenizer(
            source,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        target_tokens = self.tokenizer(
            target,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Ensure reference token in target
        target_tokens['input_ids'] = self._ensure_ref_token(target_tokens['input_ids'])
        
        # Check if citation token exists in source
        cite_token_id = self.tokenizer.convert_tokens_to_ids(self.config.cite_token)
        if cite_token_id not in source_tokens['input_ids'][0]:
            return None
        
        return {
            'source_input_ids': source_tokens['input_ids'].squeeze(0),
            'source_attention_mask': source_tokens['attention_mask'].squeeze(0),
            'target_input_ids': target_tokens['input_ids'].squeeze(0),
            'target_attention_mask': target_tokens['attention_mask'].squeeze(0)
        }

    def _ensure_ref_token(self, tokens: torch.Tensor) -> torch.Tensor:
        """Ensure reference token is present in the sequence."""
        ref_token_id = self.tokenizer.convert_tokens_to_ids(self.config.ref_token)
        pad_token_id = self.tokenizer.pad_token_id
        
        for i in range(tokens.size(0)):
            sequence = tokens[i]
            ref_positions = (sequence == ref_token_id).nonzero()
            
            if len(ref_positions) == 0:
                # Add ref token at last non-pad position
                non_pad_positions = (sequence != pad_token_id).nonzero()
                if len(non_pad_positions) > 0:
                    last_non_pad_pos = non_pad_positions[-1]
                    sequence[last_non_pad_pos] = ref_token_id
            elif len(ref_positions) > 1:
                # Keep only the last ref token
                for pos in ref_positions[:-1]:
                    sequence[pos] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        
        return tokens

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