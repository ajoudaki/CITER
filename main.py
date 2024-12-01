# Standard library imports
import random 
import bz2
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import hashlib

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
import tqdm 
import yaml

from config import TrainingConfig
from train_validate import TrainingManager

config = TrainingConfig(
    project_name="citation-matching",
    root_dir=Path("/mnt/HDD/amir/paperGPT"),
    checkpoint_every=1000,
    seed=42,
    collate_sample_size=2000,
    batch_size=280,
    initial_logit_scale=np.log(1/0.05),
    train_ratio=.5,
    # learning_rate=1.5e-4,
    learning_rate=2e-5,
    logits_learning_rate=0,
    max_grad_norm=0.5,
    device="cuda:1"
)


if __name__ == "__main__":
    experiment = TrainingManager(config)
    # results = experiment.get_results(cache_path='./cache/tokenized_1caf5def_eb27a5477eaa3d549aebc4886f3717d1.pt')
    results = experiment.get_results(cache_path=config.cache_dir / "tokenized.pt")
    
    # Train from scratch
    trained_model = experiment.train_citation_matcher(results)
    
    # # Or resume from checkpoint
    # config.resume_from = './checkpoints/citation-matching/icy-water-83/checkpoint-step-2000.pt'
    # experiment = TrainingManager(config)
    # trained_model = experiment.train_citation_matcher(results)