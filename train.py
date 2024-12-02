# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np

from config import TrainingConfig
from trainer import TrainingManager

config = TrainingConfig(
    project_name="citation-matching",
    root_dir=Path("/mnt/HDD/amir/paperGPT"),
    checkpoint_every=1000,
    # collate_sample_size=2000,
    seed=42,
    batch_size=270, # roberta needs smaller batches
    initial_logit_scale=np.log(1/0.05),
    train_ratio=.5,
    # learning_rate=1.5e-4,
    model_name = "FacebookAI/roberta-base",
    learning_rate=2e-5, # adjusted learning rate for roberta 
    logits_learning_rate=0,
    max_grad_norm=0.5,
    device="cuda:1"
)


if __name__ == "__main__":
    experiment = TrainingManager(config)
    results = experiment.get_results(cache_path=config.cache_dir / "tokenized.pt")
    
    # Train from scratch
    trained_model = experiment.train_citation_matcher(results)
    
    # # Or resume from checkpoint
    # config.resume_from = './checkpoints/citation-matching/icy-water-83/checkpoint-step-2000.pt'
    # experiment = TrainingManager(config)
    # trained_model = experiment.train_citation_matcher(results)