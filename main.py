# Standard library imports

# Third-party imports
from config import TrainingConfig
from trainer import TrainingManager


# config = TrainingConfig.load('configs/bert-base-small.yaml')
config = TrainingConfig.load('configs/bert-base-tiny.yaml')
# config = TrainingConfig.load('configs/stella-500m.yaml')

if __name__ == "__main__":
    trainer = TrainingManager(config)
    tokenized_data = trainer.get_tokenized_data(cache_path=config.cache_dir / "tokenized.pt")
    
    # Train from scratch
    trained_model = trainer.train_citation_matcher(tokenized_data)
    
    # # Or resume from checkpoint
    # config.resume_from = config.checkpoint_dir / 'daily-pine-230/checkpoint-step-100.pt'
    # trainer = TrainingManager(config)
    # trained_model = trainer.train_citation_matcher(tokenized_data)