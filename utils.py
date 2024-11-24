# Standard library imports
import bz2
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments
)
import tqdm 
import yaml

@dataclass
class WikiArticle:
    """Represents a Wikipedia article with its metadata."""
    title: str
    text: str
    timestamp: str
    is_redirect: bool

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

    def __init__(self, jsonl_path: str = "data/wiki_articles.jsonl"):
        
        # Load articles
        logging.info("Loading articles from JSONL file...")
        self.articles_dict = {}
        self.id2ref = {}
        self.ref2id = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                article = json.loads(line)
                ref = article['title'].lower()
                id = len(self.articles_dict) + 1
                self.articles_dict[ref] = self.clean_wiki_text(article['text'])
                self.ref2id[ref] = id 
                self.id2ref[id] = ref
        logging.info(f"Loaded {len(self.articles_dict)} articles.")

    def _find_citations(self,text):
        citations = []
        for match in re.finditer(r'\[\[(.*?)\]\]', text):
            match_text = match.group(1)
            citation = match_text.split('|') if '|' in match_text else [match_text]
            citation = [(c.split('#')[0] if '#' in c else c) for c in citation]
            ref = None
            for cit in citation:
                if cit.lower() in self.articles_dict:
                    ref = cit.lower()
                    break
            if ref:
                citations.append((match.start(), match.end(), self.ref2id[ref]))
        return citations

    @staticmethod
    def clean_wiki_text(text: str) -> str:
        """Cleans wiki content by removing metadata and formatting."""
        # Find main content starting from first bold title
        match = re.search(r"'''([^']+?)'''", text)
        if match:
            text = text[match.start():]

        # Remove wiki elements and clean up
        text = re.sub(r'\[\[File:.*\]\]|\[\[Category:.*\]\]|\{\{stub.*\}\}', '', text)
        return '\n'.join(line for line in text.split('\n') if line.strip())

    def find_source_citations(self) -> Tuple[List[str], List[Tuple[List[str], int, int]]]:
        """Creates source-target pairs for citation matching."""

        articles = list(self.articles_dict.keys())
        sources = []
        citation_data = []

        for title in articles:
            text = self.articles_dict[title]
            source_text = self.clean_wiki_text(text)
            citations = self._find_citations(source_text)            
            sources.append(source_text)
            citation_data.append(citations)

        return sources, citation_data

def get_cache_path(sources, model_name: str, cache_dir: str) -> str:
    """Generate a unique cache path based on input data and model name."""
    # Create a hash of the sources and model name
    content_hash = hashlib.md5(str(sources).encode()).hexdigest()
    model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"tokenized_{model_hash}_{content_hash}.pt")

def tokenize_sources(sources=None, citation_data=None, tokenizer=None, batch_size=1000, cache_dir="cache", cache_path=None):
    # Generate cache path
    if cache_path is None:
        cache_path = get_cache_path(sources, tokenizer.name_or_path, cache_dir)
    
    # Check if cached results exist
    if os.path.exists(cache_path):
        logging.info(f"Loading cached tokenized results from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    logging.info("Tokenizing sources...")
    # Process in batches
    all_results = []
    for batch_start in tqdm.tqdm(range(0, len(sources), batch_size), total=len(sources)//batch_size):
        batch_end = min(batch_start + batch_size, len(sources))
        batch_sources = sources[batch_start:batch_end]
        batch_citations = citation_data[batch_start:batch_end]
        
        # Batch encode
        batch_encoded = tokenizer.batch_encode_plus(
            batch_sources,
            add_special_tokens=False,
            return_offsets_mapping=True,
            padding=False,
            return_tensors=None
        )
        
        # Process each item in the batch
        for idx in range(len(batch_sources)):
            offset_mapping = batch_encoded["offset_mapping"][idx]
            input_ids = batch_encoded["input_ids"][idx]
            
            # Create offset to index mapping
            off2i = {s:i for i, (s,_) in enumerate(offset_mapping)}
            off2i.update({e:i+1 for i, (_,e) in enumerate(offset_mapping)})
            
            # Create citation tokens array
            mask_tokens = np.zeros(len(input_ids), dtype=int)
            cite_tokens = np.zeros(len(input_ids), dtype=int)
            
            # Fill in citations
            for i, j, art_id in batch_citations[idx]:
                s, e = off2i[i], off2i[j]
                cite_tokens[s] = art_id
                mask_tokens[s:e] = art_id
            
            # Store results
            all_results.append({
                'input_ids': np.array(input_ids),
                'cite_tokens': cite_tokens,
                'mask_tokens': mask_tokens,
                'attention_mask': batch_encoded["attention_mask"][idx] if "attention_mask" in batch_encoded else None
            })

    # Cache the results
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(all_results, cache_path)
    logging.info(f"Cached tokenized results to {cache_path}")
    
    return all_results

def collate(results, tokenizer, config):
    cite_token = tokenizer.convert_tokens_to_ids(config.cite_token)
    ref_token = tokenizer.convert_tokens_to_ids(config.ref_token)
    bracket_tokens = tokenizer.convert_tokens_to_ids(['[',']'])
    pad_token = tokenizer.pad_token_id

    collated_data = []
    # id_to_tokenized = {i: result for i, result in enumerate(results)}
    
    for i in tqdm.tqdm(range(len(results))):
        result = results[i]
        if config.collate_sample_size and len(collated_data)>config.collate_sample_size:
            break
        
        # Process each source segment
        for s in range(0, len(result['input_ids']), int((1-config.overlap)*config.source_len)):
            e = s + config.source_len
            
            # Get source segment
            input_ids = result['input_ids'][s:e].copy()
            cite_tokens = result['cite_tokens'][s:e]
            mask_tokens = result['mask_tokens'][s:e]
            
            # Skip if segment is too short
            if len(input_ids) < config.source_len // 2:
                continue
                
            # Get all citations from this segment
            present_citations = np.unique(cite_tokens[cite_tokens > 0])
            if len(present_citations) > config.max_targets:
                present_citations = np.random.choice(present_citations, config.max_targets, replace=False)
            max_targets = min(config.max_targets, len(present_citations))

            # Skip if segment is too short
            if len(input_ids) < config.source_len // 2:
                continue
            # Skip if no citations
            if max_targets == 0:
                continue
            
            # Initialize target arrays
            target_ids = np.full((max_targets, config.target_len), pad_token, dtype=np.int64)
            target_attention_mask = np.zeros((max_targets, config.target_len), dtype=np.int64)
            
            
            # Prepare source: 
            # only keep citation tokens that are sampled to be masked 
            cite_tokens_mask = np.isin(cite_tokens, present_citations)
            # don't mask citations that are not sampled 
            mask_tokens = np.where(np.isin(mask_tokens, present_citations), mask_tokens, 0)
            # remove brackets from the rest of the text 
            mask_tokens = np.where(np.isin(input_ids,bracket_tokens),1, mask_tokens)
            # don't mask the citation tokens 
            mask_tokens[cite_tokens_mask] = 0
            # set the citation tokens (first token of a citation range) as special token <CITE> 
            input_ids[cite_tokens_mask] = cite_token
            # mask all tokens in a citation, except for the first (special) token 
            source_ids = input_ids[mask_tokens == 0]

            # keep the cited article ids in the text in the order they appear (with repeats)
            # & keep the unique cited artile ids 
            # this will enable us to link each special cite token to a target via the article id
            target_art_ids = present_citations
            cited_art_ids = cite_tokens[cite_tokens_mask]
            
            # Pad or truncate source
            if len(source_ids) > config.source_len:
                source_ids = source_ids[:config.source_len]
            elif len(source_ids) < config.source_len:
                source_ids = np.pad(source_ids, 
                                  (0, config.source_len - len(source_ids)),
                                  'constant', 
                                  constant_values=pad_token)
            
            # Create source attention mask
            attention_mask = (source_ids != pad_token).astype(np.int64)
            
            # Process each target
            for idx, citation_id in enumerate(present_citations):
                # Get pre-tokenized target content
                # ids are 1-indexed 
                target_data = results[citation_id - 1]
                target_tokens = target_data['input_ids']
                
                # Truncate if needed and add ref_token
                if len(target_tokens) >= config.target_len - 1:
                    target_tokens = target_tokens[:config.target_len-1]
                target_tokens = np.append(target_tokens, ref_token)
                
                # Pad to target_len
                if len(target_tokens) < config.target_len:
                    target_tokens = np.pad(target_tokens,
                                         (0, config.target_len - len(target_tokens)),
                                         'constant',
                                         constant_values=pad_token)
                
                # Store in target arrays
                target_ids[idx] = target_tokens
                target_attention_mask[idx] = (target_tokens != pad_token)
                # citation_ids[idx] = citation_id


            # Store the collected data
            collated_data.append({
                'source_art_id': i+1,
                'source_ids': torch.tensor(source_ids, dtype=torch.long),
                'cited_art_ids': torch.tensor(cited_art_ids, dtype=torch.long),
                'target_art_ids': torch.tensor(target_art_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'target_attention_mask': torch.tensor(target_attention_mask, dtype=torch.long),
            })
    
    return collated_data

class CitationDataset(torch.utils.data.Dataset):
    """Dataset for citation data with stacked targets."""
    
    def __init__(self, collated_data):
        self.data = collated_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def citation_collate_fn(batch):
    # Stack sources normally
    source_ids = torch.stack([item['source_ids'] for item in batch])
    cited_art_ids = torch.cat([item['cited_art_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Concatenate targets
    target_art_ids_all = torch.cat([item['target_art_ids'] for item in batch])
    target_ids = torch.cat([item['target_ids'] for item in batch])
    target_attention_mask = torch.cat([item['target_attention_mask'] for item in batch])

    # Get unique indices and inverse indices
    target_art_ids, unique_indices = np.unique(target_art_ids_all.numpy(), return_index=True)
    target_art_ids = torch.tensor(target_art_ids)
    unique_indices = torch.tensor(unique_indices)
    
    # Use unique indices to get corresponding targets
    target_ids = target_ids[unique_indices]
    target_attention_mask = target_attention_mask[unique_indices]

    id2i = {id.item():i for i,id in enumerate(target_art_ids)}
    labels = torch.tensor([id2i[id.item()] for id in cited_art_ids],dtype=torch.long)

      
    return {
        'source_ids': source_ids,
        'cited_art_ids': cited_art_ids,
        'target_art_ids': target_art_ids,
        'target_ids': target_ids,
        'attention_mask': attention_mask,
        'target_attention_mask': target_attention_mask,
        'labels': labels,
    }



class CitationConfig(PretrainedConfig):
    """Configuration class for CitationModel."""
    model_type = "citation"
    
    def __init__(
        self,
        base_model_name="bert-base-uncased",
        vocab_size=30522,
        cite_token_id=None,
        ref_token_id=None,
        temperature=0.07,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.vocab_size = vocab_size
        self.cite_token_id = cite_token_id
        self.ref_token_id = ref_token_id
        self.temperature = temperature

@dataclass
class CitationModelOutput:
    """Custom output class for the citation model."""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cite_embeds: Optional[torch.FloatTensor] = None
    ref_embeds: Optional[torch.FloatTensor] = None

class CitationModel(nn.Module):
    """Custom model for citation matching using transformer embeddings."""
    
    def __init__(self, config: CitationConfig):
        super().__init__()
        
        # Load base model configuration
        base_config = AutoConfig.from_pretrained(config.base_model_name)
        
        # Store configuration
        self.config = config
        
        # Load base transformer model
        self.transformer = AutoModel.from_pretrained(config.base_model_name)
        
        # Resize token embeddings if needed
        if config.vocab_size != self.transformer.config.vocab_size:
            self.transformer.resize_token_embeddings(config.vocab_size)
    
    def get_citation_masks(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for citation token positions."""
        return input_ids == self.config.cite_token_id
    
    def get_reference_masks(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for reference token positions."""
        return input_ids == self.config.ref_token_id
    
    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        cited_art_ids: Optional[torch.Tensor] = None,
        target_art_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple, CitationModelOutput]:
        """Forward pass of the model."""
        
        # Process source text
        source_outputs = self.transformer(
            input_ids=source_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Process target text
        target_outputs = self.transformer(
            input_ids=target_ids,
            attention_mask=target_attention_mask,
            return_dict=True
        )
        
        # Get citation mask and extract citation embeddings
        cite_mask = self.get_citation_masks(source_ids)
        cite_embeds = source_outputs.last_hidden_state[cite_mask]
        
        # Get reference mask and extract reference embeddings
        ref_mask = self.get_reference_masks(target_ids)
        ref_embeds = target_outputs.last_hidden_state[ref_mask]
        
        # Normalize embeddings
        cite_embeds = F.normalize(cite_embeds, p=2, dim=-1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=-1)
        
        # Compute similarity scores
        logits = torch.matmul(cite_embeds, ref_embeds.t()) / self.config.temperature

        # compute the loss 
        loss = F.cross_entropy(logits, labels)
        
        if return_dict:
            return CitationModelOutput(
                loss=loss,
                logits=logits,
                cite_embeds=cite_embeds,
                ref_embeds=ref_embeds
            )
        
        return (loss, logits, cite_embeds, ref_embeds)


def compute_retrieval_metrics(logits, labels, ks=[1, 5, 10, 50, 100, 1000]):
    # Get rankings of correct targets
    correct_scores = logits[torch.arange(logits.size(0)), labels]
    rankings = (logits >= correct_scores.unsqueeze(1)).sum(1)
    
    # Compute MRR
    mrr = (1.0 / rankings).mean().item()
    
    # Compute top-k accuracy for different k values
    metrics = {'mrr': mrr}
    for k in ks:
        if k <= logits.size(1):  # Only compute if k is not larger than number of targets
            top_k_acc = (rankings <= k).float().mean().item()
            metrics[f'top_{k}_accuracy'] = top_k_acc
    
    return metrics

def validate_citation_model(
    model,
    val_dataloader,
    device: str = None,
    return_embeddings: bool = False,
    k_values: List[int] = [1, 5, 10, 50, 100, 1000],
    similarity_batch_size: int = 4096
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Lists to store accumulated embeddings and IDs
    all_cite_embeds = []
    all_ref_embeds = []
    all_cited_art_ids = []
    all_target_art_ids = []
    
    # Accumulate embeddings and IDs
    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader, desc="Computing embeddings"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Process source text
            source_outputs = model.transformer(
                input_ids=batch['source_ids'],
                attention_mask=batch['attention_mask'],
                return_dict=True
            )
            
            # Process target text
            target_outputs = model.transformer(
                input_ids=batch['target_ids'],
                attention_mask=batch['target_attention_mask'],
                return_dict=True
            )
            
            # Get citation mask and extract citation embeddings
            cite_mask = model.get_citation_masks(batch['source_ids'])
            cite_embeds = source_outputs.last_hidden_state[cite_mask]
            
            # Get reference mask and extract reference embeddings
            ref_mask = model.get_reference_masks(batch['target_ids'])
            ref_embeds = target_outputs.last_hidden_state[ref_mask]
            
            # Normalize embeddings
            cite_embeds = F.normalize(cite_embeds, p=2, dim=-1)
            ref_embeds = F.normalize(ref_embeds, p=2, dim=-1)
            
            # Store embeddings and IDs
            all_cite_embeds.append(cite_embeds.cpu())
            all_ref_embeds.append(ref_embeds.cpu())
            all_cited_art_ids.append(batch['cited_art_ids'].cpu())
            all_target_art_ids.append(batch['target_art_ids'].cpu())
    
    # Concatenate all accumulated tensors
    cite_embeds = torch.cat(all_cite_embeds)
    ref_embeds = torch.cat(all_ref_embeds)
    cited_art_ids = torch.cat(all_cited_art_ids)
    target_art_ids = torch.cat(all_target_art_ids)
    
    # Get unique target art IDs and create mapping
    target_art_ids_unique, unique_indices = np.unique(target_art_ids.numpy(), return_index=True)
    target_art_ids_unique = torch.tensor(target_art_ids_unique)
    ref_embeds_unique = ref_embeds[torch.tensor(unique_indices)]
    
    # Create ID to index mapping
    id2i = {id.item(): i for i, id in enumerate(target_art_ids_unique)}
    labels = torch.tensor([id2i[id.item()] for id in cited_art_ids], dtype=torch.long)
    
    # Move reference embeddings to device once
    ref_embeds_unique = ref_embeds_unique.to(device)
    
    # Initialize arrays to store results
    total_loss = 0
    total_correct = 0
    all_predictions = []
    all_logits_list = []
    all_labels_list = []
    
    # Process citation embeddings in batches
    num_batches = (len(cite_embeds) + similarity_batch_size - 1) // similarity_batch_size
    
    for i in tqdm.tqdm(range(num_batches), desc="Computing similarities"):
        start_idx = i * similarity_batch_size
        end_idx = min((i + 1) * similarity_batch_size, len(cite_embeds))
        
        # Move batch of citation embeddings to device
        cite_embeds_batch = cite_embeds[start_idx:end_idx].to(device)
        labels_batch = labels[start_idx:end_idx].to(device)
        
        # Compute similarity scores for this batch
        logits_batch = torch.matmul(cite_embeds_batch, ref_embeds_unique.t()) / model.config.temperature
        
        # Compute loss for this batch
        loss_batch = F.cross_entropy(logits_batch, labels_batch)
        total_loss += loss_batch.item() * len(labels_batch)
        
        # Compute predictions and accuracy for this batch
        predictions_batch = torch.argmax(logits_batch, dim=-1)
        total_correct += (predictions_batch == labels_batch).sum().item()
        
        # Store predictions and logits for later computation
        all_predictions.append(predictions_batch.cpu())
        all_logits_list.append(logits_batch.cpu())
        all_labels_list.append(labels_batch.cpu())
        
        # Clear GPU memory
        del logits_batch, cite_embeds_batch, labels_batch, predictions_batch
        torch.cuda.empty_cache()
    
    # Concatenate all predictions and compute accuracy
    all_predictions = torch.cat(all_predictions)
    num_citations = len(cite_embeds)
    accuracy = total_correct / num_citations
    avg_loss = total_loss / num_citations
    
    # Compute retrieval metrics using concatenated logits
    all_logits = torch.cat(all_logits_list)
    all_labels = torch.cat(all_labels_list)
    retrieval_metrics = compute_retrieval_metrics(all_logits, all_labels, ks=k_values)
    
    # Prepare results dictionary (matching original function's output)
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'num_citations': num_citations,
        'num_unique_targets': len(target_art_ids_unique),
        'mrr': retrieval_metrics['mrr']
    }
    
    # Add top-k accuracies to results
    for k in k_values:
        if f'top_{k}_accuracy' in retrieval_metrics:
            results[f'top_{k}_accuracy'] = retrieval_metrics[f'top_{k}_accuracy']
    
    if return_embeddings:
        results.update({
            'cite_embeds': cite_embeds.cpu(),
            'ref_embeds': ref_embeds_unique.cpu(),
            'cited_art_ids': cited_art_ids,
            'target_art_ids': target_art_ids_unique,
            'logits': torch.cat(all_logits_list),  # Full logits matrix
            'labels': labels.cpu()
        })
    
    return results
    

def train_citation_model(
    model,
    results,
    tokenizer,
    config,
    train_ratio: float = 0.8,
    num_epochs: int = 5,
    learning_rate: float = 1.5e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    device: str = None,
    save_path: str = "citation_model.pt",
    batch_size: int = 128,
    temperatures = [],
    k_values = [1, 5, 10, 50, 100, 1000]
):
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Move model to device
    model = model.to(device)
    
    # Enable memory efficient training
    model.transformer.gradient_checkpointing_enable()
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        eps=1e-8  # Increased epsilon for FP16 training stability
    )
    
    # Training loop
    best_val_metrics = {'loss': float('inf')}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        if epoch < len(temperatures):
            model.config.temperature = temperatures[epoch]
            print(f"temperature changed to {temperatures[epoch]}")
        
        # Create new collated data for this epoch
        print("Collating training data with new random masks...")
        collated = collate(results, tokenizer, config)
        dataset = CitationDataset(collated)
        train_size = int(len(dataset) * train_ratio)
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:]
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Pin memory for faster GPU transfer
            collate_fn=citation_collate_fn
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2, # use larger batch for validation (no gradients) 
            shuffle=False,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Pin memory for faster GPU transfer
            collate_fn=citation_collate_fn
        )
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm.tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(**batch)
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update tracking variables
            total_train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / train_steps
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        print("\nRunning validation...")
        torch.cuda.empty_cache()
        model.eval()
        
        val_metrics = validate_citation_model(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            k_values=k_values
        )
        
        print(f"\nValidation metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Accuracy (top-1): {val_metrics['accuracy']:.4f}")
        print(f"  Mean Reciprocal Rank: {val_metrics['mrr']:.4f}")
        print(f"  Number of citations: {val_metrics['num_citations']}")
        print(f"  Number of unique targets: {val_metrics['num_unique_targets']}")
        print("\nTop-k accuracy:")
        for k in k_values:
            if f'top_{k}_accuracy' in val_metrics:
                print(f"  k={k}: {val_metrics[f'top_{k}_accuracy']:.4f}")
        
        # Save best model based on validation loss
        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'validation_metrics': val_metrics,
                'temperature': model.config.temperature
            }, save_path)
            print(f"\nSaved new best model to {save_path}")
            print(f"Best validation metrics so far:")
            print(f"  Loss: {best_val_metrics['loss']:.4f}")
            print(f"  Accuracy (top-1): {best_val_metrics['accuracy']:.4f}")
            print(f"  MRR: {best_val_metrics['mrr']:.4f}")
    
    return model


@dataclass
class ExperimentConfig:
    """Configuration   the citation matching model."""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    source_len: int = 512
    target_len: int = 128
    max_targets: int = 5
    overlap: float = 0.5
    cite_token: str = "<CITE>"
    ref_token: str = "<REF>"
    temperature: float = 0.07
    collate_sample_size: int = None,
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
