"""
Graph-Contrastive Learning Module.

This module provides a scalable architecture for contrastive learning on graph data,
supporting multiple data sources (StackExchange, ArXiv, etc.) through a unified interface.

Main Components:
    - models.py: Abstract interfaces for models and datasets
    - graph_config.py: Edge type registry and configuration
    - preprocess_graph.py: Data preprocessing script
    - dataset.py: UniversalGraphDataset and HierarchicalClusterSampler
"""

from .models import AbstractEmbeddingModel, AbstractGraphDataset, AbstractClusterSampler
from .graph_config import EdgeType, SourceType, EDGE_REGISTRY, get_edge_config, get_prompts
from .dataset import UniversalGraphDataset, HierarchicalClusterSampler, AllNodesDataset, create_dataloader

__all__ = [
    # Abstract interfaces
    'AbstractEmbeddingModel',
    'AbstractGraphDataset',
    'AbstractClusterSampler',
    # Configuration
    'EdgeType',
    'SourceType',
    'EDGE_REGISTRY',
    'get_edge_config',
    'get_prompts',
    # Dataset classes
    'UniversalGraphDataset',
    'HierarchicalClusterSampler',
    'AllNodesDataset',
    'create_dataloader',
]
