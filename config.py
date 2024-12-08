# Standard library imports
import os
import subprocess
import random 
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# Third-party imports
import numpy as np
import torch
import yaml


@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "bert-base-uncased"
    initial_logit_scale: float = 3
    
    # Random seed configuration
    seed: int = 42
    
    # Token configuration
    cite_token: str = "[[CITE]]"
    ref_token: str = "[[REF]]"
    cite_token_id: Optional[int] = None
    ref_token_id: Optional[int] = None
    vocab_size: Optional[int] = None

    
    # Text processing configuration
    max_length: int = 512
    source_len: int = 512
    target_len: int = 128
    max_targets: int = 5
    overlap: float = 0.5
    
    # Training configuration
    batch_size: int = 512
    val_batch_size: int = 512
    retrieval_batch_size: int = 1024
    micro_batch_size: int = 32
    
    num_epochs: int = 100
    learning_rate: float = 1.5e-4
    logits_learning_rate: float = 0
    max_grad_norm: float = 1.0
    Adam_eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_steps: int = 0
    train_ratio: float = 0.5
    collate_sample_size: int = None
    
    # Evaluation configuration
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 50, 100, 1000])
    
    # Checkpoint configuration
    root_dir: Path = "."  # Default to current directory
    project_name: str = "citation-matching"

    checkpoint_every: int = None
    run_name: Optional[str] = None
    resume_from: Optional[str] = None
    
    # Hardware configuration
    device: Optional[torch.device] = None
    
    @property
    def data_dir(self) -> Path:
        return Path(self.root_dir) / "data"
    
    @property
    def output_dir(self) -> Path:
        return Path(self.root_dir) / "outputs"
    
    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints" / self.project_name
        
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"
        
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"
        
    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    def get_gpu_with_most_memory(self) -> Optional[torch.device]:
        """
        Returns the GPU device with the most available memory.
        If no GPUs are available, returns None.
        """
        if not torch.cuda.is_available():
            return None
            
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            return None
    
        try:
            # Get free memory for all GPUs using nvidia-smi
            result = subprocess.check_output([
                'nvidia-smi', 
                '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'
            ])
            free_memories = [int(x) for x in result.decode('utf-8').strip().split('\n')]
            
            # Find GPU with most free memory
            max_free_memory = 0
            selected_gpu = 0
            
            for gpu_id, free_memory in enumerate(free_memories):
                print(f"GPU {gpu_id} free memory = {free_memory} MB")
                
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    selected_gpu = gpu_id

            print(f"Selected GPU {selected_gpu}")
            return torch.device(f'cuda:{selected_gpu}')
            
        except subprocess.CalledProcessError:
            return torch.device('cuda:0')  # Fallback to first GPU if nvidia-smi fails


    def __post_init__(self):
        if self.device is None:
            # Try to get GPU with most memory first
            self.device = self.get_gpu_with_most_memory() or torch.device("cpu")
            
        # checkpoint after every 50,000 samples have been trained on
        if self.checkpoint_every is None:
            self.checkpoint_every = 50000 // self.batch_size
            
        # Override root_dir with environment variable if set
        if "CITER_ROOT" in os.environ:
            self.root_dir = Path(os.environ["CITER_ROOT"])          
    
    def get_checkpoint_dir(self) -> Path:
        if self.project_name and self.run_name:
            checkpoint_path = Path(self.checkpoint_dir) / self.run_name
        elif self.project_name:
            checkpoint_path = Path(self.checkpoint_dir)
        else:
            checkpoint_path = Path(self.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        return checkpoint_path
    
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
