"""
UniversalGraphDataset and HierarchicalClusterSampler.

This module provides the data loading infrastructure for graph-contrastive learning,
using memory-mapped Arrow and NumPy files for efficient random access.
"""

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import torch
from torch.utils.data import Dataset, Sampler
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .models import AbstractGraphDataset, AbstractClusterSampler
from .graph_config import get_prompts, format_text_with_prompt


class UniversalGraphDataset(AbstractGraphDataset, Dataset):
    """
    Universal dataset for graph-contrastive learning.

    Loads preprocessed nodes (Arrow) and edges (NumPy) with zero-copy access.
    Each __getitem__ returns a tokenized edge (source, target) pair.

    Args:
        data_dir: Directory containing nodes.arrow and edges.npy
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length for tokenization
        use_prompts: Whether to prepend edge-type prompts to text
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 256,
        use_prompts: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_prompts = use_prompts

        # Load nodes (memory-mapped Arrow)
        nodes_path = self.data_dir / "nodes.arrow"
        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")

        print(f"Loading nodes from: {nodes_path}")
        self.nodes_table = feather.read_table(str(nodes_path), memory_map=True)
        self.num_nodes = len(self.nodes_table)

        # Create lookup arrays for fast access
        self.global_ids = self.nodes_table.column('global_id').to_pylist()
        self.texts = self.nodes_table.column('text')  # Keep as Arrow array for zero-copy
        self.cluster_ids = self.nodes_table.column('cluster_id').to_numpy()

        # Load edges (memory-mapped NumPy)
        edges_path = self.data_dir / "edges.npy"
        if not edges_path.exists():
            raise FileNotFoundError(f"Edges file not found: {edges_path}")

        print(f"Loading edges from: {edges_path}")
        self.edges = np.load(str(edges_path), mmap_mode='r')
        self.num_edges = len(self.edges)

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        print(f"Dataset loaded: {self.num_nodes:,} nodes, {self.num_edges:,} edges")

    def __len__(self) -> int:
        """Return the total number of edges (training pairs)."""
        return self.num_edges

    def get_text(self, global_id: int) -> str:
        """Get text content for a node by global ID (zero-copy from Arrow)."""
        # Arrow array indexing returns a scalar
        return self.texts[global_id].as_py()

    def get_cluster_id(self, global_id: int) -> int:
        """Get cluster ID for a node."""
        return self.cluster_ids[global_id]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single training pair by edge index.

        Returns:
            Dict with keys: input_ids_x, attention_mask_x,
                           input_ids_y, attention_mask_y, weight
        """
        # Read edge: [src_id, dst_id, edge_type_id, weight]
        edge = self.edges[idx]
        src_id = int(edge[0])
        dst_id = int(edge[1])
        edge_type_id = int(edge[2])
        weight = float(edge[3])

        # Get text content
        src_text = self.get_text(src_id)
        dst_text = self.get_text(dst_id)

        # Apply prompts if enabled
        if self.use_prompts:
            src_prompt, dst_prompt = get_prompts(edge_type_id)
            src_text = format_text_with_prompt(src_text, src_prompt)
            dst_text = format_text_with_prompt(dst_text, dst_prompt)

        # Tokenize
        tokens_x = self.tokenizer(
            src_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        tokens_y = self.tokenizer(
            dst_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids_x': tokens_x['input_ids'].squeeze(0),
            'attention_mask_x': tokens_x['attention_mask'].squeeze(0),
            'input_ids_y': tokens_y['input_ids'].squeeze(0),
            'attention_mask_y': tokens_y['attention_mask'].squeeze(0),
            'weight': torch.tensor(weight, dtype=torch.float32),
        }


class HierarchicalClusterSampler(AbstractClusterSampler, Sampler):
    """
    Cluster-aware sampler for hard negative mining in distributed training.

    This sampler ensures that batches contain edges from the same clusters
    (papers, question threads), maximizing the gradient signal from hard negatives.

    Algorithm:
        1. Load cluster_ids for all source nodes of edges
        2. Sort edge indices by cluster_id
        3. Partition clusters (not edges) across world_size
        4. Shuffle within each partition

    Args:
        dataset: UniversalGraphDataset instance
        num_replicas: Number of distributed processes (world_size)
        rank: Current process rank
        shuffle: Whether to shuffle within partitions
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dataset: UniversalGraphDataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Build cluster -> edge mapping
        print(f"Building cluster index for {len(dataset):,} edges...")
        self._build_cluster_index()

    def _build_cluster_index(self):
        """Build mapping from clusters to edge indices."""
        # Get source node cluster for each edge
        edge_clusters = []
        for i in range(len(self.dataset)):
            src_id = int(self.dataset.edges[i, 0])
            cluster_id = self.dataset.get_cluster_id(src_id)
            edge_clusters.append((cluster_id, i))

        # Sort by cluster_id
        edge_clusters.sort(key=lambda x: x[0])

        # Group edges by cluster
        self.cluster_to_edges: Dict[int, List[int]] = {}
        for cluster_id, edge_idx in edge_clusters:
            if cluster_id not in self.cluster_to_edges:
                self.cluster_to_edges[cluster_id] = []
            self.cluster_to_edges[cluster_id].append(edge_idx)

        # Get sorted list of cluster IDs
        self.cluster_ids = sorted(self.cluster_to_edges.keys())
        print(f"Found {len(self.cluster_ids):,} unique clusters")

        # Partition clusters across ranks
        clusters_per_rank = len(self.cluster_ids) // self.num_replicas
        remainder = len(self.cluster_ids) % self.num_replicas

        # Distribute clusters: first 'remainder' ranks get one extra cluster
        if self.rank < remainder:
            start_cluster = self.rank * (clusters_per_rank + 1)
            end_cluster = start_cluster + clusters_per_rank + 1
        else:
            start_cluster = remainder * (clusters_per_rank + 1) + (self.rank - remainder) * clusters_per_rank
            end_cluster = start_cluster + clusters_per_rank

        self.my_cluster_ids = self.cluster_ids[start_cluster:end_cluster]

        # Collect all edges for this rank's clusters
        self.my_edge_indices = []
        for cluster_id in self.my_cluster_ids:
            self.my_edge_indices.extend(self.cluster_to_edges[cluster_id])

        print(f"Rank {self.rank}: {len(self.my_cluster_ids):,} clusters, "
              f"{len(self.my_edge_indices):,} edges")

    def __iter__(self):
        """Yield edge indices in cluster-aware order."""
        # Create local copy for shuffling
        indices = self.my_edge_indices.copy()

        if self.shuffle:
            # Shuffle within partitions while keeping cluster structure
            rng = np.random.RandomState(self.seed + self.epoch)

            # Shuffle clusters
            shuffled_clusters = self.my_cluster_ids.copy()
            rng.shuffle(shuffled_clusters)

            # Rebuild indices with shuffled cluster order
            indices = []
            for cluster_id in shuffled_clusters:
                cluster_edges = self.cluster_to_edges[cluster_id].copy()
                rng.shuffle(cluster_edges)
                indices.extend(cluster_edges)

        return iter(indices)

    def __len__(self) -> int:
        """Return total number of samples for this rank."""
        return len(self.my_edge_indices)

    def set_epoch(self, epoch: int):
        """Set the epoch for shuffling reproducibility."""
        self.epoch = epoch


class AllNodesDataset(Dataset):
    """
    Dataset that returns all nodes for embedding computation.

    Used for computing embeddings of all nodes in the graph for
    retrieval/evaluation purposes.

    Args:
        data_dir: Directory containing nodes.arrow
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 256,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load nodes
        nodes_path = self.data_dir / "nodes.arrow"
        self.nodes_table = feather.read_table(str(nodes_path), memory_map=True)
        self.num_nodes = len(self.nodes_table)

        # Keep columns as Arrow arrays for efficiency
        self.texts = self.nodes_table.column('text')
        self.cluster_ids = self.nodes_table.column('cluster_id').to_numpy()
        self.source_types = self.nodes_table.column('source_type').to_numpy()

        print(f"AllNodesDataset: {self.num_nodes:,} nodes")

    def __len__(self) -> int:
        return self.num_nodes

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return tokenized node and its metadata."""
        text = self.texts[idx].as_py()
        cluster_id = self.cluster_ids[idx]

        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'cluster_id': torch.tensor(cluster_id, dtype=torch.long),
            'global_id': torch.tensor(idx, dtype=torch.long),
        }


class PreprocessedGraphDataset(Dataset):
    """
    Dataset wrapper for preprocessed graph data that matches StratifiedTheoremDataset interface.

    This class provides a drop-in replacement for StratifiedTheoremDataset,
    allowing the training script to use preprocessed Arrow/NumPy data without modification.

    Args:
        data_dir: Directory containing nodes.arrow and edges.npy
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length for tokenization
        split: 'train' or 'eval' - determines which portion of edges to use
        train_ratio: Fraction of edges to use for training (default 0.9)
        seed: Random seed for train/eval split
        use_prompts: Whether to prepend edge-type prompts to text
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 512,
        split: str = 'train',
        train_ratio: float = 0.9,
        seed: int = 42,
        use_prompts: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed
        self.use_prompts = use_prompts
        self.epoch = 0

        # Load nodes (memory-mapped Arrow)
        nodes_path = self.data_dir / "nodes.arrow"
        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")

        self.nodes_table = feather.read_table(str(nodes_path), memory_map=True)
        self.num_nodes = len(self.nodes_table)
        self.texts = self.nodes_table.column('text')
        self.cluster_ids = self.nodes_table.column('cluster_id').to_numpy()

        # Load edges (memory-mapped NumPy)
        edges_path = self.data_dir / "edges.npy"
        if not edges_path.exists():
            raise FileNotFoundError(f"Edges file not found: {edges_path}")

        self.all_edges = np.load(str(edges_path), mmap_mode='r')
        num_total_edges = len(self.all_edges)

        # Split edges into train/eval
        rng = np.random.default_rng(seed)
        indices = np.arange(num_total_edges)
        rng.shuffle(indices)

        n_train = int(num_total_edges * train_ratio)
        if split == 'train':
            self.edge_indices = indices[:n_train]
        else:
            self.edge_indices = indices[n_train:]

        # For epoch shuffling (train only)
        self.shuffled_indices = self.edge_indices.copy()

        print(f"PreprocessedGraphDataset ({split}): {len(self.edge_indices):,} edges "
              f"from {self.num_nodes:,} nodes")

    def reset_epoch(self):
        """Shuffle edge indices for new epoch (compatibility with StratifiedTheoremDataset)."""
        if self.split == 'train':
            rng = np.random.default_rng(self.seed + self.epoch)
            self.shuffled_indices = self.edge_indices.copy()
            rng.shuffle(self.shuffled_indices)
        self.epoch += 1

    def __len__(self) -> int:
        return len(self.edge_indices)

    def get_text(self, global_id: int) -> str:
        """Get text content for a node by global ID."""
        return self.texts[global_id].as_py()

    def get_cluster_id(self, global_id: int) -> int:
        """Get cluster ID for a node."""
        return self.cluster_ids[global_id]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a training pair by index.

        Returns dict with keys matching StratifiedTheoremDataset:
            input_ids_x, attention_mask_x, input_ids_y, attention_mask_y
        """
        # Get the actual edge index (shuffled for train, fixed for eval)
        if self.split == 'train':
            edge_idx = self.shuffled_indices[idx]
        else:
            edge_idx = self.edge_indices[idx]

        # Read edge: [src_id, dst_id, edge_type_id, weight]
        edge = self.all_edges[edge_idx]
        src_id = int(edge[0])
        dst_id = int(edge[1])
        edge_type_id = int(edge[2])

        # Get text content
        src_text = self.get_text(src_id)
        dst_text = self.get_text(dst_id)

        # Apply prompts if enabled
        if self.use_prompts:
            src_prompt, dst_prompt = get_prompts(edge_type_id)
            src_text = format_text_with_prompt(src_text, src_prompt)
            dst_text = format_text_with_prompt(dst_text, dst_prompt)

        # Tokenize
        tokens_x = self.tokenizer(
            src_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        tokens_y = self.tokenizer(
            dst_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids_x': tokens_x['input_ids'].squeeze(0),
            'attention_mask_x': tokens_x['attention_mask'].squeeze(0),
            'input_ids_y': tokens_y['input_ids'].squeeze(0),
            'attention_mask_y': tokens_y['attention_mask'].squeeze(0),
        }


def create_dataloader(
    data_dir: str,
    tokenizer,
    batch_size: int,
    max_length: int = 256,
    num_workers: int = 4,
    use_prompts: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Create a DataLoader with cluster-aware sampling.

    Args:
        data_dir: Directory containing processed data
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size per GPU
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        use_prompts: Whether to use edge-type prompts
        distributed: Whether to use distributed sampling
        rank: Current process rank
        world_size: Total number of processes
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        DataLoader with cluster-aware sampling
    """
    from torch.utils.data import DataLoader

    dataset = UniversalGraphDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        use_prompts=use_prompts,
    )

    if distributed:
        sampler = HierarchicalClusterSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and not distributed),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, dataset
