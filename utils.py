from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterator, Union, Optional
import xml.etree.ElementTree as ET
import bz2
import json
import sqlite3
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import os
import logging
from datetime import datetime
import torch.cuda.amp  # For automatic mixed precision
import yaml

@dataclass
class WikiArticle:
    """Represents a Wikipedia article with its metadata."""
    title: str
    text: str
    timestamp: str
    is_redirect: bool

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

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1.5e-4
    temperature: float = 0.1
    num_workers: int = 4
    gradient_clip_value: float = 1.0
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    eval_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 50])

@dataclass
class DataConfig:
    """Configuration for data preparation and loading."""
    data_path: str = "./data/wiki_articles.jsonl"
    train_sample_size: int = 1000
    val_sample_size: int = 100
    val_samples_per_article: int = 10
    train_samples_per_article: int = 1

@dataclass
class ExperimentConfig:
    """Unified configuration for the entire experiment."""
    experiment_name: str
    output_dir: str = "experiments"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create nested dataclass instances
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))

        return cls(
            experiment_name=config_dict['experiment_name'],
            output_dir=config_dict.get('output_dir', 'experiments'),
            model=model_config,
            training=training_config,
            data=data_config
        )

    def save_yaml(self, output_path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = {
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'model': {k: v for k, v in self.model.__dict__.items() if not k.startswith('_')},
            'training': self.training.__dict__,
            'data': self.data.__dict__
        }
        
        # Remove device from model config as it can't be serialized
        if 'device' in config_dict['model']:
            del config_dict['model']['device']

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class WikiDumpProcessor:
    """Processes Wikipedia XML dumps and extracts articles."""
    
    def __init__(self, dump_path: str):
        self.dump_path = dump_path
        self._ns = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}
        self._skip_prefixes = {
            'Wikipedia:', 'Template:', 'Category:', 'Portal:', 'File:', 
            'MediaWiki:', 'Help:', 'Book:', 'Draft:', 'TimedText:', 
            'Module:', 'Special:'
        }

    def iter_articles(self, skip_redirects: bool = True) -> Iterator[WikiArticle]:
        """Iterates through valid articles in the dump."""
        dump_file = bz2.BZ2File(self.dump_path) if self.dump_path.endswith('.bz2') else open(self.dump_path, 'rb')
        
        for _, elem in ET.iterparse(dump_file, events=('end',)):
            if not elem.tag.endswith('page'):
                continue

            # Extract basic article data
            title = elem.find('.//mw:title', self._ns).text
            if any(title.startswith(prefix) for prefix in self._skip_prefixes):
                elem.clear()
                continue

            # Get revision data
            rev = elem.find('.//mw:revision', self._ns)
            text = rev.find('mw:text', self._ns).text if rev is not None else ''
            timestamp = rev.find('mw:timestamp', self._ns).text if rev is not None else ''
            is_redirect = bool(re.match(r'#REDIRECT', text or '', re.IGNORECASE))

            if skip_redirects and is_redirect:
                elem.clear()
                continue

            yield WikiArticle(title=title, text=text, timestamp=timestamp, is_redirect=is_redirect)
            elem.clear()

class ArticleStorage:
    """Handles storage and retrieval of Wikipedia articles."""
    
    def __init__(self, processor: WikiDumpProcessor):
        self.processor = processor

    def save_to_jsonl(self, output_path: Union[str, Path], sample_size: Optional[int] = None) -> int:
        """Saves articles to a JSONL file."""
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, article in enumerate(self.processor.iter_articles()):
                if sample_size is not None and i >= sample_size:
                    break
                json.dump(article.__dict__, f, ensure_ascii=False)
                f.write('\n')
                count += 1
        return count

    def save_to_sqlite(self, db_path: Union[str, Path], sample_size: Optional[int] = None,
                      batch_size: int = 1000) -> int:
        """Saves articles to a SQLite database."""
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS articles
                    (title TEXT PRIMARY KEY, text TEXT, timestamp TEXT, is_redirect INTEGER)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_title ON articles(title)')
        
        count = 0
        batch = []
        
        try:
            for i, article in enumerate(self.processor.iter_articles()):
                if sample_size is not None and i >= sample_size:
                    break
                    
                batch.append((article.title, article.text, article.timestamp, 
                            1 if article.is_redirect else 0))
                
                if len(batch) >= batch_size:
                    c.executemany('INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?)', batch)
                    conn.commit()
                    count += len(batch)
                    batch = []
            
            if batch:
                c.executemany('INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?)', batch)
                conn.commit()
                count += len(batch)
                
        finally:
            conn.close()
            
        return count

class WikiProcessor:
    """Prepares citation data for model training."""
    
    def __init__(self, articles_dict: Dict[str, str]):
        self.articles_dict = articles_dict

    def create_citation_pairs(self, sample_size: int = 1000, cite_samples_per_article: int = 1) -> Tuple[List[str], List[str]]:
        """Creates source-target pairs for citation matching."""
        articles = np.random.permutation(list(self.articles_dict.keys()))[:sample_size]
        sources, targets = [], []
        
        for title in articles:
            text = self.articles_dict[title]
            citations = [(cit.split('|')[0] if '|' in cit else cit) for cit in re.findall(r'\[\[(.*?)\]\]', text)]
            valid_citations = [c for c in citations if c.lower() in self.articles_dict]
            
            if not valid_citations:
                continue
                
            for citation in np.random.choice(valid_citations, 
                                           min(cite_samples_per_article, len(valid_citations)), 
                                           replace=False):
                try:
                    # Clean and prepare content
                    source_text = self._clean_wiki_text(text)
                    source_text = source_text.replace(f"[[{citation}]]", "<CITE>")
                    target_text = f"{self._clean_wiki_text(self.articles_dict[citation.lower()])} <REF>"
                    
                    sources.append(source_text)
                    targets.append(target_text)
                except Exception:
                    continue
        
        return sources, targets

    @staticmethod
    def _clean_wiki_text(text: str) -> str:
        """Cleans wiki content by removing metadata and formatting."""
        # Find main content starting from first bold title
        match = re.search(r"'''([^']+?)'''", text)
        if match:
            text = text[match.start():]
        
        # Remove wiki elements and clean up
        text = re.sub(r'\[\[Category:.*?\]\]|\[\[File:.*?\]\]|\{\{stub\}\}', '', text)
        return '\n'.join(line for line in text.split('\n') if line.strip())
    

class CitationMatcher(nn.Module):
    """Main citation matching model with integrated text encoding."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
            add_prefix_space=True
        )
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [config.cite_token, config.ref_token]
        })
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = AutoModel.from_pretrained(config.model_name).to(config.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _extract_token_embedding(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int
    ) -> torch.Tensor:
        """Extracts and normalizes embeddings for a specific token."""
        batch_size = input_ids.size(0)
        embeddings = []
        
        for batch_idx in range(batch_size):
            token_positions = (input_ids[batch_idx] == token_id).nonzero()
            if len(token_positions) == 0:
                raise ValueError(f"Token ID {token_id} not found in sequence {batch_idx}")
            position = token_positions[-1].item()
            embeddings.append(hidden_states[batch_idx, position, :])
        
        embeddings = torch.stack(embeddings)
        return nn.functional.normalize(embeddings, dim=-1)

    def _encode_text(self, text: str, token_id: int) -> torch.Tensor:
        """Encodes text and extracts normalized token embedding."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.config.device)
        
        outputs = self.model(**inputs, return_dict=True)
        return self._extract_token_embedding(
            outputs.last_hidden_state,
            inputs['input_ids'],
            token_id
        )

    def forward(
        self,
        source_contexts: List[str],
        target_pages: List[str]
    ) -> torch.Tensor:
        """Compute similarity between source and target pages."""
        cite_token_id = self.tokenizer.convert_tokens_to_ids(self.config.cite_token)
        ref_token_id = self.tokenizer.convert_tokens_to_ids(self.config.ref_token)
        
        # Encode all texts
        source_embeddings = torch.cat([
            self._encode_text(ctx, cite_token_id) for ctx in source_contexts
        ], dim=0)
        
        target_embeddings = torch.cat([
            self._encode_text(page, ref_token_id) for page in target_pages
        ], dim=0)
        
        # Compute similarity matrix
        return torch.matmul(
            source_embeddings,
            target_embeddings.transpose(0, 1)
        ) / self.config.temperature

class CitationDataset(Dataset):
    """Dataset for citation matching with optimized batch processing."""
    
    def __init__(
        self,
        sources: List[str],
        targets: List[str],
        tokenizer: AutoTokenizer,
        config: ModelConfig,
        batch_size: int = 1024,
        verbose: bool = True
    ):
        assert len(sources) == len(targets), "Sources and targets must have same length"
        self.processed_samples = self._preprocess_samples(
            sources, targets, tokenizer, config, batch_size, verbose
        )

    def _ensure_ref_token_batch(
        self,
        tokens: torch.Tensor,
        ref_token_id: int,
        pad_token_id: int
    ) -> torch.Tensor:
        """Ensure reference token is present in batch of sequences."""
        # Find sequences missing ref token
        has_ref = (tokens == ref_token_id).any(dim=1)
        missing_ref = ~has_ref

        if missing_ref.any():
            # Find last non-pad position for sequences missing ref token
            pad_mask = (tokens != pad_token_id)
            last_nonpad = pad_mask.long().argmax(dim=1)
            
            # Add ref token at last non-pad position
            missing_indices = missing_ref.nonzero(as_tuple=True)[0]
            tokens[missing_indices, last_nonpad[missing_indices]] = ref_token_id
        
        return tokens

    def _extract_citation_context(self, text: str, cite_token: str, window_tokens: int, tokenizer) -> str:
        """Extract context window around citation token, handling token lengths properly."""
        # Find citation token position
        cite_pos = text.find(cite_token)
        if cite_pos == -1:
            return text
            
        # Split text into before and after citation
        before_citation = text[:cite_pos]
        after_citation = text[cite_pos + len(cite_token):]
        
        # Tokenize both parts
        before_tokens = tokenizer.encode(before_citation, add_special_tokens=False)
        after_tokens = tokenizer.encode(after_citation, add_special_tokens=False)
        
        # Calculate how many tokens to keep on each side
        tokens_per_side = window_tokens // 2
        
        # Take tokens from both sides
        before_context = before_tokens[-tokens_per_side:] if len(before_tokens) > tokens_per_side else before_tokens
        after_context = after_tokens[:tokens_per_side] if len(after_tokens) > tokens_per_side else after_tokens
        
        # Decode back to text
        before_text = tokenizer.decode(before_context)
        after_text = tokenizer.decode(after_context)
        
        return before_text + cite_token + after_text

    def _preprocess_samples(
        self,
        sources: List[str],
        targets: List[str],
        tokenizer: AutoTokenizer,
        config: ModelConfig,
        batch_size: int,
        verbose: bool
    ) -> List[Dict[str, torch.Tensor]]:
        """Preprocess samples with proper token-based context window."""
        total_samples = len(sources)
        processed_samples = []
        total_processed = 0
        total_skipped = 0
        
        # Process in batches
        iterator = range(0, total_samples, batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Processing samples", unit="batch")
        
        cite_token_id = tokenizer.convert_tokens_to_ids(config.cite_token)
        ref_token_id = tokenizer.convert_tokens_to_ids(config.ref_token)
        
        # Calculate context window in tokens, leaving room for special tokens
        context_window_size = config.max_length - 2  # Account for [CLS] and [SEP]
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, total_samples)
            batch_sources = sources[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            # Extract context for each source text
            contextualized_sources = [
                self._extract_citation_context(text, config.cite_token, context_window_size, tokenizer)
                for text in batch_sources
            ]
            
            # Batch tokenization
            source_tokens = tokenizer(
                contextualized_sources,
                padding='max_length',
                truncation=True,
                max_length=config.max_length,
                return_tensors='pt'
            )
            
            target_tokens = tokenizer(
                batch_targets,
                padding='max_length',
                truncation=True,
                max_length=config.max_length,
                return_tensors='pt'
            )
            
            # Rest of processing remains the same...
            has_citation = (source_tokens['input_ids'] == cite_token_id).any(dim=1)
            valid_indices = has_citation.nonzero(as_tuple=True)[0]
            
            total_processed += len(valid_indices)
            total_skipped += len(batch_sources) - len(valid_indices)
            
            if len(valid_indices) == 0:
                continue
                
            target_tokens['input_ids'] = self._ensure_ref_token_batch(
                target_tokens['input_ids'],
                ref_token_id,
                tokenizer.pad_token_id
            )
            
            for idx in valid_indices:
                processed_samples.append({
                    'source_input_ids': source_tokens['input_ids'][idx],
                    'source_attention_mask': source_tokens['attention_mask'][idx],
                    'target_input_ids': target_tokens['input_ids'][idx],
                    'target_attention_mask': target_tokens['attention_mask'][idx]
                })
        
        if verbose:
            print(f"\nProcessed {total_processed} samples")
            print(f"Skipped {total_skipped} samples without citation")
        
        return processed_samples

    # def _preprocess_samples(
    #     self,
    #     sources: List[str],
    #     targets: List[str],
    #     tokenizer: AutoTokenizer,
    #     config: ModelConfig,
    #     batch_size: int,
    #     verbose: bool
    # ) -> List[Dict[str, torch.Tensor]]:
    #     """Preprocess samples with optimized batch processing."""
    #     total_samples = len(sources)
    #     processed_samples = []
    #     total_processed = 0
    #     total_skipped = 0
        
    #     # Process in batches
    #     iterator = range(0, total_samples, batch_size)
    #     if verbose:
    #         iterator = tqdm(iterator, desc="Processing samples", unit="batch")
        
    #     cite_token_id = tokenizer.convert_tokens_to_ids(config.cite_token)
    #     ref_token_id = tokenizer.convert_tokens_to_ids(config.ref_token)
        
    #     for start_idx in iterator:
    #         end_idx = min(start_idx + batch_size, total_samples)
    #         batch_sources = sources[start_idx:end_idx]
    #         batch_targets = targets[start_idx:end_idx]
            
    #         # Batch tokenization
    #         source_tokens = tokenizer(
    #             batch_sources,
    #             padding='max_length',
    #             truncation=True,
    #             max_length=config.max_length,
    #             return_tensors='pt'
    #         )
            
    #         target_tokens = tokenizer(
    #             batch_targets,
    #             padding='max_length',
    #             truncation=True,
    #             max_length=config.max_length,
    #             return_tensors='pt'
    #         )
            
    #         # Process source tokens
    #         has_citation = (source_tokens['input_ids'] == cite_token_id).any(dim=1)
    #         valid_indices = has_citation.nonzero(as_tuple=True)[0]
            
    #         # Update statistics
    #         total_processed += len(valid_indices)
    #         total_skipped += len(batch_sources) - len(valid_indices)
            
    #         if len(valid_indices) == 0:
    #             continue
            
    #         # Ensure ref token in target sequences
    #         target_tokens['input_ids'] = self._ensure_ref_token_batch(
    #             target_tokens['input_ids'],
    #             ref_token_id,
    #             tokenizer.pad_token_id
    #         )
            
    #         # Create samples for valid sequences
    #         for idx in valid_indices:
    #             processed_samples.append({
    #                 'source_input_ids': source_tokens['input_ids'][idx],
    #                 'source_attention_mask': source_tokens['attention_mask'][idx],
    #                 'target_input_ids': target_tokens['input_ids'][idx],
    #                 'target_attention_mask': target_tokens['attention_mask'][idx]
    #             })
        
    #     if verbose:
    #         print(f"\nProcessed {total_processed} samples")
    #         print(f"Skipped {total_skipped} samples without citation")
        
    #     return processed_samples

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
        collate_fn=lambda batch: {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0].keys()
        },
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )


@dataclass
class TrainingMetrics:
    """Stores training and validation metrics."""
    train_loss: float
    val_loss: float
    top_k_accuracy: Dict[int, float]
    mrr: float
    median_rank: float
    mean_rank: float
    val_size: int

def setup_logging(output_dir: Path) -> None:
    """Configure logging for the training process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def create_output_directory() -> Path:
    """Create and return output directory for this training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"training_runs/run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_and_prepare_data(
    jsonl_path: str,
    train_sample_size: int,
    val_sample_size: int,
    train_samples_per_article: int = 1,
    val_samples_per_article: int = 10
) -> tuple:
    """Load and prepare training and validation data."""
    logging.info("Loading articles from JSONL file...")
    articles_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            articles_dict[article['title'].lower()] = article['text']
    
    logging.info(f"Loaded {len(articles_dict)} articles")
    
    # Prepare citation data
    preprocessor = WikiProcessor(articles_dict)
    
    logging.info("Preparing training data...")
    train_sources, train_targets = preprocessor.create_citation_pairs(
        sample_size=train_sample_size,
        cite_samples_per_article=train_samples_per_article
    )

    S = set(train_sources)
    T = set(train_targets)
    
    logging.info("Preparing validation data...")
    val_sources, val_targets = preprocessor.create_citation_pairs(
        sample_size=val_sample_size,
        cite_samples_per_article=val_samples_per_article
    )

    # Remove any validation samples that are also in the training set
    val_sources, val_targets = zip(*[
        (source, target) for source, target in zip(val_sources, val_targets)
        if source not in S and target not in T
    ])
    
    return train_sources, train_targets, val_sources, val_targets


# New imports needed at the top of the file
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from typing import Optional, Dict, List, Tuple, Union
import torch.distributed as dist
from pathlib import Path
# Add this at the top of the file with other imports
from functools import lru_cache
# Add at the top of your file
from dataclasses import dataclass, field
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedDataParallelKwargs

@dataclass
class AcceleratorConfig:
    """Configuration for Accelerator initialization."""
    mixed_precision: str = 'fp16'
    gradient_accumulation_steps: int = 1
    device_placement: bool = True
    kwargs_handlers: list = field(default_factory=lambda: [
        DistributedDataParallelKwargs(find_unused_parameters=True)
    ])

def initialize_accelerator(config: AcceleratorConfig = None) -> Accelerator:
    """Initialize accelerator with given configuration."""
    if AcceleratorState._shared_state:
        # If accelerator is already initialized, return current instance
        return Accelerator()
    
    if config is None:
        config = AcceleratorConfig()
    
    return Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        device_placement=config.device_placement,
        kwargs_handlers=config.kwargs_handlers
    )
class Trainer:
    """Updated trainer class with consistent Accelerator initialization."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, save_dir: Path):
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get existing or create new accelerator
        self.accelerator = initialize_accelerator()
        
        # Prepare model and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True
        )
        
        # Prepare for distributed training
        self.model, self.optimizer, self.criterion = self.accelerator.prepare(
            model, self.optimizer, self.criterion
        )

    def _get_embeddings(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for source and target texts."""
        # Move batch to device using accelerator
        batch = {k: v for k, v in batch.items()}
        
        with torch.set_grad_enabled(self.model.training):
            # Get source embeddings
            source_out = self.model.model(
                input_ids=batch['source_input_ids'],
                attention_mask=batch['source_attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
            source_emb = self.model._extract_token_embedding(
                source_out.last_hidden_state,
                batch['source_input_ids'],
                self.model.tokenizer.convert_tokens_to_ids(self.model.config.cite_token)
            )
            
            # Get target embeddings
            target_out = self.model.model(
                input_ids=batch['target_input_ids'],
                attention_mask=batch['target_attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
            target_emb = self.model._extract_token_embedding(
                target_out.last_hidden_state,
                batch['target_input_ids'],
                self.model.tokenizer.convert_tokens_to_ids(self.model.config.ref_token)
            )
            
        return source_emb, target_emb

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch with Accelerator."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Wrap dataloader with accelerator
        train_loader = self.accelerator.prepare(train_loader)
        
        for batch in tqdm(train_loader, desc='Training'):
            source_emb, target_emb = self._get_embeddings(batch)
            
            # Compute loss
            similarity = torch.matmul(source_emb, target_emb.transpose(0, 1)) / self.config.temperature
            labels = torch.arange(similarity.size(0)).to(similarity.device)
            loss = self.criterion(similarity, labels)
            
            # Optimize with accelerator
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
        # Gather losses from all processes
        if dist.is_initialized():
            total_loss = torch.tensor(total_loss).to(self.accelerator.device)
            num_batches = torch.tensor(num_batches).to(self.accelerator.device)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item()
            num_batches = num_batches.item()
            
        return total_loss / num_batches

    def validate(self, val_loader: DataLoader, epoch: int) -> Optional[TrainingMetrics]:
        """Validate model with Accelerate support."""
        self.model.eval()
        
        # Prepare validation dataloader
        val_loader = self.accelerator.prepare(val_loader)
        
        # Collect all embeddings
        all_source_embs = []
        all_target_embs = []
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc='Validation'):
            source_emb, target_emb = self._get_embeddings(batch)
            
            # Compute validation loss
            similarity = torch.matmul(source_emb, target_emb.transpose(0, 1)) / self.config.temperature
            labels = torch.arange(similarity.size(0)).to(similarity.device)
            loss = self.criterion(similarity, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Gather embeddings from all processes
            source_emb = self.accelerator.gather(source_emb)
            target_emb = self.accelerator.gather(target_emb)
            
            all_source_embs.append(source_emb.cpu())
            all_target_embs.append(target_emb.cpu())
        
        # Gather losses from all processes
        if dist.is_initialized():
            total_loss = torch.tensor(total_loss).to(self.accelerator.device)
            num_batches = torch.tensor(num_batches).to(self.accelerator.device)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item()
            num_batches = num_batches.item()
        
        # Concatenate all embeddings
        source_embeddings = torch.cat(all_source_embs, dim=0)
        target_embeddings = torch.cat(all_target_embs, dim=0)
        
        # Compute metrics only on main process
        if self.accelerator.is_main_process:
            metrics = self._compute_metrics(source_embeddings, target_embeddings)
            metrics.val_loss = total_loss / num_batches
            
            # Update scheduler and save model
            self.scheduler.step(metrics.val_loss)
            if self.save_dir:
                self._save_checkpoint(metrics, epoch)
            
            return metrics
        return None

    def _compute_metrics(self, source_embs: torch.Tensor, target_embs: torch.Tensor) -> TrainingMetrics:
        """Compute all validation metrics."""
        total_samples = source_embs.size(0)
        chunk_size = 512
        all_rankings = []
        
        # Compute rankings in chunks to avoid OOM
        for i in range(0, total_samples, chunk_size):
            chunk_end = min(i + chunk_size, total_samples)
            source_chunk = source_embs[i:chunk_end].to(self.accelerator.device)
            
            similarity = torch.matmul(source_chunk, target_embs.to(self.accelerator.device).t()) / self.config.temperature
            rankings = torch.argsort(similarity, dim=-1, descending=True)
            all_rankings.append(rankings.cpu())
            
            del similarity, source_chunk
            torch.cuda.empty_cache()
        
        rankings = torch.cat(all_rankings, dim=0)
        
        # Calculate metrics
        correct_at_k = {k: 0 for k in self.config.eval_k_values}
        reciprocal_ranks = []
        ranks = []
        
        for i in range(total_samples):
            rank = (rankings[i] == i).nonzero().item() + 1
            ranks.append(rank)
            reciprocal_ranks.append(1.0 / rank)
            
            for k in self.config.eval_k_values:
                if rank <= k:
                    correct_at_k[k] += 1
        
        return TrainingMetrics(
            train_loss=0.0,  # Set by train_model
            val_loss=0.0,    # Set by validate
            top_k_accuracy={k: count / total_samples for k, count in correct_at_k.items()},
            mrr=float(np.mean(reciprocal_ranks)),
            median_rank=float(np.median(ranks)),
            mean_rank=float(np.mean(ranks)),
            val_size=total_samples
        )

    def _save_checkpoint(self, metrics: TrainingMetrics, epoch: int):
        """Save model checkpoint and metrics."""
        if self.accelerator.is_main_process and self.save_dir:
            # Get unwrapped model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(
                unwrapped_model.state_dict(),
                self.save_dir / f'model_epoch_{epoch}.pt'
            )
            torch.save(
                metrics.__dict__,
                self.save_dir / f'metrics_epoch_{epoch}.pt'
            )

def setup_environment() -> None:
    """Configure training environment."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize accelerator
    accelerator = initialize_accelerator()
    if accelerator.is_main_process:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainingConfig,
    save_dir: Path
) -> List[TrainingMetrics]:
    """Main training loop with Accelerate integration."""
    trainer = Trainer(model, config, save_dir)
    metrics_history = []
    
    for epoch in range(config.num_epochs):
        if trainer.accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Training phase
        train_loss = trainer.train_epoch(train_loader)
        
        # Validation phase
        if val_loader:
            metrics = trainer.validate(val_loader, epoch)
            
            # Only log metrics on main process
            if trainer.accelerator.is_main_process and metrics is not None:
                metrics.train_loss = train_loss
                metrics_history.append(metrics)
                
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"Training Loss: {train_loss:.4f}")
                print(f"Validation Loss: {metrics.val_loss:.4f}")
                print(f"Best Top-1 Accuracy: {metrics.top_k_accuracy[1]:.4f}")
                print(f"Mean Reciprocal Rank: {metrics.mrr:.4f}")
                print(f"Validation Size: {metrics.val_size}")
                [print(f"Top-{k} Accuracy: {v:.3f}") for k, v in metrics.top_k_accuracy.items()]
    
    return metrics_history

def main(config_path: Optional[str] = None) -> List[TrainingMetrics]:
    """Main training pipeline with consistent Accelerator initialization."""
    # Initialize accelerator first
    accelerator = initialize_accelerator()
    
    # Load configuration
    if config_path:
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = ExperimentConfig(experiment_name="configs/default_experiment")
    
    # Create output directory - only on main process
    if accelerator.is_main_process:
        output_dir = Path(config.output_dir) / config.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save configuration
        config.save_yaml(output_dir / "config.yaml")
        # Setup logging
        setup_logging(output_dir)
    
    # Setup environment
    setup_environment()
    
    # Rest of the main function remains the same...
    train_sources, train_targets, val_sources, val_targets = load_and_prepare_data(
        jsonl_path=config.data.data_path,
        train_sample_size=config.data.train_sample_size,
        val_sample_size=config.data.val_sample_size,
        train_samples_per_article=config.data.train_samples_per_article,
        val_samples_per_article=config.data.val_samples_per_article
    )
    
    model = CitationMatcher(config.model)
    
    train_dataset = CitationDataset(
        sources=train_sources,
        targets=train_targets,
        tokenizer=model.tokenizer,
        config=config.model,
        verbose=accelerator.is_main_process
    )
    
    val_dataset = CitationDataset(
        sources=val_sources,
        targets=val_targets,
        tokenizer=model.tokenizer,
        config=config.model,
        verbose=accelerator.is_main_process
    )
    
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    if accelerator.is_main_process:
        logging.info("Starting training...")
    
    metrics_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        save_dir=output_dir / 'checkpoints' if accelerator.is_main_process else None
    )
    
    if accelerator.is_main_process:
        torch.save(
            {
                'metrics_history': [metric.__dict__ for metric in metrics_history],
                'model_config': config.model.__dict__,
                'training_config': config.training.__dict__,
                'data_config': config.data.__dict__
            },
            output_dir / 'training_history.pt'
        )
        logging.info(f"Training completed. Results saved to {output_dir}")
    
    return metrics_history
