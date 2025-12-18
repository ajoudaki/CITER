"""
Abstract interfaces for the graph-contrastive learning pipeline.

These are the "Never-Change" contracts that the rest of the codebase relies on.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class AbstractEmbeddingModel(nn.Module, ABC):
    """
    The immutable contract for any embedding model (BERT, RoBERTa, Qwen, etc.).

    All embedding models must implement this interface to be compatible with
    the training pipeline.
    """

    @abstractmethod
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized embeddings for input sequences.

        Args:
            input_ids: Token IDs [Batch, SeqLen]
            attention_mask: Attention mask [Batch, SeqLen]

        Returns:
            torch.Tensor: L2-normalized embeddings [Batch, HiddenDim]
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Returns metadata about the model for logging and reproducibility.

        Returns:
            Dict containing at minimum:
                - hidden_dim: int
                - model_name: str
                - model_type: str
        """
        pass


class AbstractGraphDataset(ABC):
    """
    The immutable contract for the graph dataset.

    __getitem__ must return a dictionary with exact keys:
    {
       'input_ids_x': tensor,     # [SeqLen]
       'attention_mask_x': tensor, # [SeqLen]
       'input_ids_y': tensor,     # [SeqLen]
       'attention_mask_y': tensor, # [SeqLen]
       'weight': float            # Edge weight for loss weighting
    }

    This interface ensures that any data source (ArXiv, StackExchange, ProofWiki, etc.)
    can be used with the training pipeline without modification.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of edges (training pairs)."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single training pair by edge index.

        Args:
            idx: Edge index

        Returns:
            Dict with keys: input_ids_x, attention_mask_x,
                           input_ids_y, attention_mask_y, weight
        """
        pass


class AbstractClusterSampler(ABC):
    """
    The immutable contract for cluster-aware sampling.

    This sampler ensures that distributed batches contain semantically
    related items (e.g., theorems from the same paper, QA pairs with same tag),
    maximizing the gradient signal from hard negatives.
    """

    @abstractmethod
    def __iter__(self):
        """Yield edge indices in cluster-aware order."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""
        pass

    @abstractmethod
    def set_epoch(self, epoch: int):
        """Set the epoch for shuffling reproducibility."""
        pass
