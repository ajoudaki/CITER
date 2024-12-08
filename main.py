from config import TrainingConfig
from trainer import TrainingManager
import argparse
from typing import get_type_hints
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to YAML config file (default: configs/default.yaml)')
    
    hints = get_type_hints(TrainingConfig)
    for attr, type_hint in hints.items():
        arg_type = Path if attr.endswith('_dir') or attr == 'resume_from' else type_hint
        parser.add_argument(f'--{attr}', type=arg_type, help=f'Override {attr} from config')
    
    return parser.parse_args()

def update_config_from_args(config, args):
    return config if not args else config.__dict__.update(
        {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    ) or config

if __name__ == "__main__":
    args = parse_args()
    config = update_config_from_args(TrainingConfig.load(args.config), args)
    
    trainer = TrainingManager(config)
    tokenized_data = trainer.get_tokenized_data(cache_path=config.cache_dir / "tokenized.pt")
    trained_model = trainer.train_citation_matcher(tokenized_data)