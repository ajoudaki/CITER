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



class CitationExtractor:
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
                self.articles_dict[ref] = self.sanitize_wiki_content(article['text'])
                self.ref2id[ref] = id 
                self.id2ref[id] = ref
        logging.info(f"Loaded {len(self.articles_dict)} articles.")

    def extract_citation_spans(self,text):
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
    def sanitize_wiki_content(text: str) -> str:
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
            source_text = self.sanitize_wiki_content(text)
            citations = self.extract_citation_spans(source_text)            
            sources.append(source_text)
            citation_data.append(citations)

        return sources, citation_data


# experiment related 

@dataclass
class ExperimentConfig:
    pass

class Experiment:
    pass


def generate_cache_key(sources, model_name: str, cache_dir: str) -> str:
    """Generate a unique cache path based on input data and model name."""
    # Create a hash of the sources and model name
    content_hash = hashlib.md5(str(sources).encode()).hexdigest()
    model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"tokenized_{model_hash}_{content_hash}.pt")

def prepare_training_data(sources=None, citation_data=None, tokenizer=None, batch_size=1000, cache_dir="cache", cache_path=None):
    # Generate cache path
    if cache_path is None:
        cache_path = generate_cache_key(sources, tokenizer.name_or_path, cache_dir)
    
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

def create_training_batches(results, tokenizer, config):
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
            # set the citation tokens (first token of a citation range) as special cite token  
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

class CitationDataset(Dataset):
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



@dataclass
class CitationModelOutput:
    """Custom output class for the citation model."""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cite_embeds: Optional[torch.FloatTensor] = None
    ref_embeds: Optional[torch.FloatTensor] = None

class CitationModel(nn.Module):
    """Custom model for citation matching using transformer embeddings."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        
        # Load base model configuration
        base_config = AutoConfig.from_pretrained(config.model_name)
        
        # Store configuration
        self.config = config
        
        # Load base transformer model
        self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Resize token embeddings if needed
        if config.vocab_size != self.transformer.config.vocab_size:
            self.transformer.resize_token_embeddings(config.vocab_size)

        # Add learnable logit scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * config.initial_logit_scale)

    
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
        
        # Clamp logit scale to prevent numerical instability
        logit_scale = torch.clamp(self.logit_scale, 0, torch.log(torch.tensor(20.0)))
        
        # Compute similarity scores with learned scale
        logits = torch.matmul(cite_embeds, ref_embeds.t()) * logit_scale.exp()

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

def validate_batch_structure(batch, config: ExperimentConfig):
    c1 = (batch['source_ids']==config.cite_token_id).sum()==batch['cited_art_ids'].shape[0]  # special cite tokens correspond to the cited article ids
    c2 =  (batch['cited_art_ids'].shape[0]==batch['labels'].shape[0])  # each cited article id has a corresponding target label
    c3 = (batch['target_ids']==config.ref_token_id).sum()==batch['target_art_ids'].shape[0]  # special ref tokens correspond to the target article ids
    return c1 and c2 and c3 
    
def validate_citation_matcher(
    model,
    val_dataloader,
    return_embeddings: bool = False,
    k_values: List[int] = [1, 5, 10, 50, 100, 1000],
    similarity_batch_size: int = 512,
    config: ExperimentConfig = None
):
    device = config.device
    
    model.eval()
    
    # Lists to store accumulated embeddings and IDs
    all_cite_embeds = []
    all_ref_embeds = []
    all_cited_art_ids = []
    all_target_art_ids = []
    
    # Accumulate embeddings and IDs
    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader, desc="Computing embeddings"):
            if not validate_batch_structure(batch, config):
                continue
            # Move batch to device and convert to FP16
            batch = {k: (v.to(device, dtype=torch.float16) if isinstance(v, torch.FloatTensor) 
                        else v.to(device)) for k, v in batch.items()}
            
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
            
            # Extract embeddings with masks
            cite_mask = model.get_citation_masks(batch['source_ids'])
            cite_embeds = source_outputs.last_hidden_state[cite_mask]
            ref_mask = model.get_reference_masks(batch['target_ids'])
            ref_embeds = target_outputs.last_hidden_state[ref_mask]
            
            # Normalize and move to CPU immediately
            cite_embeds = F.normalize(cite_embeds, p=2, dim=-1).cpu()
            ref_embeds = F.normalize(ref_embeds, p=2, dim=-1).cpu()
            
            # Store embeddings and IDs on CPU
            all_cite_embeds.append(cite_embeds)
            all_ref_embeds.append(ref_embeds)
            all_cited_art_ids.append(batch['cited_art_ids'].cpu())
            all_target_art_ids.append(batch['target_art_ids'].cpu())
            
            # Clear GPU cache after each batch
            del source_outputs, target_outputs, cite_embeds, ref_embeds
            torch.cuda.empty_cache()
    
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
    
    # Process in smaller batches for similarity computation
    total_loss = 0
    total_correct = 0
    all_predictions = []
    logits_list = []  # Store logits temporarily for metrics computation
    labels_list = []  # Store labels temporarily for metrics computation
    
    num_batches = (len(cite_embeds) + similarity_batch_size - 1) // similarity_batch_size
    logit_scale = torch.clamp(model.logit_scale, 0, torch.log(torch.tensor(20.0)))
    
    # Move ref_embeds to GPU once
    ref_embeds_unique = ref_embeds_unique.to(device)
    
    for i in tqdm.tqdm(range(num_batches), desc="Computing similarities"):
        start_idx = i * similarity_batch_size
        end_idx = min((i + 1) * similarity_batch_size, len(cite_embeds))
        
        # Process batch
        cite_embeds_batch = cite_embeds[start_idx:end_idx].to(device)
        labels_batch = labels[start_idx:end_idx].to(device)
        
        # Compute similarities and loss
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits_batch = torch.matmul(cite_embeds_batch, ref_embeds_unique.t()) * logit_scale.exp()
            loss_batch = F.cross_entropy(logits_batch, labels_batch)
        
        total_loss += loss_batch.item() * len(labels_batch)
        predictions_batch = torch.argmax(logits_batch, dim=-1)
        total_correct += (predictions_batch == labels_batch).sum().item()
        
        # Store predictions and move to CPU
        all_predictions.append(predictions_batch.cpu())
        logits_list.append(logits_batch.cpu())
        labels_list.append(labels_batch.cpu())
        
        # Clear GPU memory
        del logits_batch, cite_embeds_batch, labels_batch, predictions_batch
        torch.cuda.empty_cache()
    
    # Compute final metrics
    num_citations = len(cite_embeds)
    accuracy = total_correct / num_citations
    avg_loss = total_loss / num_citations
    
    # Compute retrieval metrics
    all_logits = torch.cat(logits_list)
    all_labels = torch.cat(labels_list)
    retrieval_metrics = compute_retrieval_metrics(all_logits, all_labels, ks=k_values)
    
    # Clear temporary lists
    del logits_list, labels_list
    torch.cuda.empty_cache()
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'num_citations': num_citations,
        'num_unique_targets': len(target_art_ids_unique),
        'mrr': retrieval_metrics['mrr']
    }
    
    # Add top-k accuracies
    for k in k_values:
        if f'top_{k}_accuracy' in retrieval_metrics:
            results[f'top_{k}_accuracy'] = retrieval_metrics[f'top_{k}_accuracy']
    
    if return_embeddings:
        results.update({
            'cite_embeds': cite_embeds,
            'ref_embeds': ref_embeds_unique.cpu(),
            'cited_art_ids': cited_art_ids,
            'target_art_ids': target_art_ids_unique,
            'logits': all_logits,
            'labels': labels
        })
    
    # Final cleanup
    del cite_embeds, ref_embeds, ref_embeds_unique
    torch.cuda.empty_cache()
    
    return results


@dataclass
class ExperimentConfig:
    # Model configuration
    model_name: str = "bert-base-uncased"
    vocab_size: Optional[int] = None
    initial_logit_scale: float = np.log(1/0.07)
    
    # Random seed configuration
    seed: int = 42
    
    # Token configuration
    cite_token: str = "[[CITE]]"
    ref_token: str = "[[REF]]"
    cite_token_id: Optional[int] = None
    ref_token_id: Optional[int] = None
    
    # Text processing configuration
    max_length: int = 512
    source_len: int = 512
    target_len: int = 128
    max_targets: int = 5
    overlap: float = 0.5
    
    # Training configuration
    num_epochs: int = 100
    learning_rate: float = 1.5e-4
    logits_learning_rate: float = 1.5e-2
    max_grad_norm: float = 1.0
    Adam_eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_steps: int = 0
    batch_size: int = 200
    train_ratio: float = 0.5
    collate_sample_size: Optional[int] = None
    
    # Evaluation configuration
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 50, 100, 1000])
    
    # Checkpoint configuration
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 1000
    project_name: str = "citation-matching"
    run_name: Optional[str] = None
    resume_from: Optional[str] = None
    
    # Hardware configuration
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def get_checkpoint_dir(self) -> Path:
        if self.project_name and self.run_name:
            checkpoint_path = Path(self.checkpoint_dir) / self.project_name / self.run_name
        elif self.project_name:
            checkpoint_path = Path(self.checkpoint_dir) / self.project_name
        else:
            checkpoint_path = Path(self.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        return checkpoint_path
    
    def save(self, path: Path):
        with open(path / "config.yaml", 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        with open(path / "config.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [config.cite_token, config.ref_token]
        })
        
        # Update config with tokenizer-dependent values
        config.cite_token_id = self.tokenizer.convert_tokens_to_ids(config.cite_token)
        config.ref_token_id = self.tokenizer.convert_tokens_to_ids(config.ref_token)
        config.vocab_size = len(self.tokenizer)
        
        # Initialize model
        self.model = CitationModel(config)
        
        # Load checkpoint if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
    
    def get_checkpoint_path(self, step: Optional[int] = None, epoch: Optional[int] = None, is_best: bool = False) -> Path:
        checkpoint_dir = self.config.get_checkpoint_dir()
        
        if is_best:
            return checkpoint_dir / "best_model.pt"
        elif step is not None:
            return checkpoint_dir / f"checkpoint-step-{step}.pt"
        elif epoch is not None:
            return checkpoint_dir / f"checkpoint-epoch-{epoch}.pt"
        else:
            raise ValueError("Must specify either step, epoch, or is_best=True")
    
    def save_checkpoint(self, 
                       path: Path, 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scaler: Optional[GradScaler] = None,
                       epoch: Optional[int] = None,
                       batch_in_epoch: Optional[int] = None,
                       global_step: Optional[int] = None,
                       val_metrics: Optional[dict] = None,
                       best_val_metrics: Optional[dict] = None,
                       wandb_run_id: Optional[str] = None,
                       is_best: bool = False):
        
        # Save RNG states as numpy arrays
        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().cpu().numpy(),
            'cuda': torch.cuda.get_rng_state().cpu().numpy() if torch.cuda.is_available() else None
        }
        
        # Prepare save dictionary
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'rng_state': rng_state,
        }
        
        # Add optional states
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if scaler is not None:
            save_dict['scaler_state_dict'] = scaler.state_dict()
        if epoch is not None:
            save_dict['epoch'] = epoch
        if batch_in_epoch is not None:
            save_dict['batch_in_epoch'] = batch_in_epoch
        if global_step is not None:
            save_dict['global_step'] = global_step
        if val_metrics is not None:
            save_dict['validation_metrics'] = val_metrics
        if best_val_metrics is not None:
            save_dict['best_val_metrics'] = best_val_metrics
        if wandb_run_id is not None:
            save_dict['wandb_run_id'] = wandb_run_id
        
        # Save checkpoint and config
        torch.save(save_dict, path)
        self.config.save(path.parent)
        
        if is_best:
            print(f"\nSaved new best model to {path}")
            if val_metrics:
                print("Best validation metrics:")
                for metric in ['loss', 'accuracy', 'mrr']:
                    if metric in val_metrics:
                        print(f"  {metric}: {val_metrics[metric]:.4f}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> dict:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load config if present
        if 'config' in checkpoint:
            resume_from = self.config.resume_from
            self.config = checkpoint['config']
            self.config.resume_from = resume_from
        
        # Restore RNG states
        if 'rng_state' in checkpoint:
            random.setstate(checkpoint['rng_state']['python'])
            np.random.set_state(checkpoint['rng_state']['numpy'])
            torch.set_rng_state(torch.tensor(checkpoint['rng_state']['torch'], dtype=torch.uint8))
            if torch.cuda.is_available() and checkpoint['rng_state']['cuda'] is not None:
                torch.cuda.set_rng_state(torch.tensor(checkpoint['rng_state']['cuda'], dtype=torch.uint8))

        # Initialize missing fields with defaults if not present
        default_fields = {
            'optimizer_state_dict': None,
            'scaler_state_dict': None,
            'epoch': 0,
            'batch_in_epoch': 0,
            'global_step': 0,
            'validation_metrics': None,
            'best_val_metrics': {'loss': float('inf')},
            'wandb_run_id': None
        }
        
        for field, default_value in default_fields.items():
            if field not in checkpoint:
                checkpoint[field] = default_value
        
        return checkpoint
    
    def get_model(self) -> CitationModel:
        return self.model
    
    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
    
    def get_results(self, cache_path=None):
        if cache_path:
            results = prepare_training_data(cache_path=cache_path)
        else:
            preprocessor = CitationExtractor()
            sources, citation_data = preprocessor.find_source_citations()
            results = prepare_training_data(sources, citation_data, self.tokenizer, cache_dir="cache")
        return results



def train_citation_matcher(
    experiment: Experiment,
    results: List[dict],
) -> CitationModel:
    """
    Memory-optimized training function with enhanced checkpoint management.
    """
    import wandb
    import gc
    
    config = experiment.config
    model = experiment.model
    tokenizer = experiment.tokenizer
    
    # Set random seeds
    config.set_seed()
    
    # Initialize or resume wandb run
    if config.resume_from:
        checkpoint = experiment.load_checkpoint(config.resume_from)
        wandb_run_id = checkpoint['wandb_run_id']
        print(f"Resuming wandb run: {wandb_run_id}")
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            id=wandb_run_id,
            resume="must"
        )
    else:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config,
        )
        
        # Update run name in config if not set
        if not config.run_name:
            config.run_name = wandb.run.name
    
    # Initialize training state
    global_step = 0
    start_epoch = 0
    batch_in_epoch = 0
    best_val_metrics = {'loss': float('inf')}
    scaler = GradScaler()
    
    # Move model to device and enable memory efficient training
    model = model.to(config.device)
    model.transformer.gradient_checkpointing_enable()
    
    # Initialize optimizer
    optimizer = AdamW([
        {
            'params': [p for n, p in model.named_parameters() if n != 'logit_scale'],
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay,
            'eps': config.Adam_eps
        },
        {
            'params': [model.logit_scale],
            'lr': config.logits_learning_rate,
            'weight_decay': 0
        }
    ])
    
    # Load checkpoint state if resuming
    if config.resume_from:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        global_step = checkpoint['global_step']
        start_epoch = checkpoint['epoch']
        batch_in_epoch = checkpoint['batch_in_epoch']
        best_val_metrics = checkpoint['best_val_metrics']
        print(f"Resumed from checkpoint at epoch {start_epoch}, batch {batch_in_epoch}, step {global_step}")
    
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Log current scale
        current_scale = model.logit_scale.exp().item()
        print(f"Current logit scale: {current_scale:.4f}")
        wandb.log({"logit_scale": current_scale}, step=global_step)
        
        # Training data preparation
        print("Collating training data with new random masks...")
        training_batches = create_training_batches(results, tokenizer, config)
        dataset = CitationDataset(training_batches)
        
        # Create train/val split
        indices = np.arange(len(dataset))
        train_size = int(len(dataset) * config.train_ratio)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        from torch.utils.data import Subset
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        # Create dataloaders
        generator = torch.Generator()
        generator.manual_seed(config.seed + epoch)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=citation_collate_fn,
            generator=generator
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=int(config.batch_size * 1.8),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=citation_collate_fn
        )
        
        # Clear memory
        del collated, dataset
        gc.collect()
        torch.cuda.empty_cache()
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm.tqdm(train_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):     
            if not validate_batch_structure(batch, config):
                continue
            # Skip previously processed batches if resuming
            if epoch == start_epoch and batch_idx < batch_in_epoch:
                continue
            
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            if config.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update tracking
            total_train_loss += loss.item()
            train_steps += 1
            
            # Log metrics
            wandb.log({
                "train/batch_loss": loss.item(),
                'logit_scale': model.logit_scale.item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/batch_in_epoch": batch_idx,
                "epoch": epoch
            }, step=global_step)
            
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Save checkpoint periodically
            if global_step > 0 and global_step % config.checkpoint_every == 0:
                checkpoint_path = experiment.get_checkpoint_path(step=global_step)
                experiment.save_checkpoint(
                    checkpoint_path,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    batch_in_epoch=batch_idx,
                    global_step=global_step,
                    wandb_run_id=wandb.run.id
                )
                print(f"\nSaved checkpoint at step {global_step} to {checkpoint_path}")
            
            global_step += 1
            
            # Clear memory
            del outputs, loss, batch
            torch.cuda.empty_cache()
        
        # Log epoch-level training metrics
        avg_train_loss = total_train_loss / train_steps
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "epoch": epoch
        }, step=global_step)
        
        # Validation phase
        print("\nRunning validation...")
        torch.cuda.empty_cache()
        model.eval()
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            val_metrics = validate_citation_matcher(
                model=model,
                val_dataloader=val_dataloader,
                k_values=config.k_values,
                config=config,
            )
        
        # Log validation metrics
        wandb_val_metrics = {
            "val/loss": val_metrics['loss'],
            "val/accuracy": val_metrics['accuracy'],
            "val/mrr": val_metrics['mrr']
        }
        
        for k in config.k_values:
            if f'top_{k}_accuracy' in val_metrics:
                wandb_val_metrics[f"val/top_{k}_accuracy"] = val_metrics[f'top_{k}_accuracy']
        
        wandb.log(wandb_val_metrics, step=global_step)
        
        # Print validation metrics
        print(f"\nValidation metrics:")
        for metric, value in val_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        # Save best model if validation loss improved
        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            best_model_path = experiment.get_checkpoint_path(is_best=True)
            experiment.save_checkpoint(
                best_model_path,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                batch_in_epoch=batch_idx,
                global_step=global_step,
                val_metrics=val_metrics,
                best_val_metrics=best_val_metrics,
                wandb_run_id=wandb.run.id,
                is_best=True
            )
            
            # Update wandb summary with best metrics
            wandb.run.summary.update({
                "best_val_loss": val_metrics['loss'],
                "best_val_accuracy": val_metrics['accuracy'],
                "best_val_mrr": val_metrics['mrr'],
                "best_model_epoch": epoch,
                "best_model_step": global_step
            })
        
        # Save epoch checkpoint
        epoch_checkpoint_path = experiment.get_checkpoint_path(epoch=epoch)
        experiment.save_checkpoint(
            epoch_checkpoint_path,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            batch_in_epoch=batch_idx,
            global_step=global_step,
            val_metrics=val_metrics,
            best_val_metrics=best_val_metrics,
            wandb_run_id=wandb.run.id
        )
        
        # Clear memory after each epoch
        del val_metrics, train_dataloader, val_dataloader
        gc.collect()
        torch.cuda.empty_cache()
    
    wandb.finish()
    return model


config = ExperimentConfig(
    project_name="citation-matching",
    # model_name="google-bert/bert-large-uncased",
    # model_name="FacebookAI/roberta-base",
    # run_name=None,
    checkpoint_dir="./checkpoints",
    checkpoint_every=500,
    seed=42,
    collate_sample_size=2000,
    batch_size=290,
    initial_logit_scale=np.log(1/0.05),
    train_ratio=.95,
    learning_rate=1.5e-4,
    # learning_rate=2e-5,
    logits_learning_rate=0,
    max_grad_norm=0.5,
    device="cuda:0"
)


if __name__ == "__main__":
    experiment = Experiment(config)
    results = experiment.get_results(cache_path='./cache/tokenized_1caf5def_eb27a5477eaa3d549aebc4886f3717d1.pt')
    
    # Train from scratch
    trained_model = train_citation_matcher(experiment, results)
    
    # # Or resume from checkpoint
    # config.resume_from = './checkpoints/citation-matching/icy-water-83/checkpoint-step-2000.pt'
    # experiment = Experiment(config)
    # trained_model = train_citation_matcher(experiment, results)