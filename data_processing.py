# Standard library imports
import logging
import os
import hashlib

# Third-party imports
import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm 


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

    data = []
    # id_to_tokenized = {i: result for i, result in enumerate(results)}
    
    for i in tqdm.tqdm(range(len(results))):
        result = results[i]
        if config.collate_sample_size and len(data)>config.collate_sample_size:
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
            data.append({
                'source_art_id': i+1,
                'source_ids': torch.tensor(source_ids, dtype=torch.long),
                'cited_art_ids': torch.tensor(cited_art_ids, dtype=torch.long),
                'target_art_ids': torch.tensor(target_art_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'target_attention_mask': torch.tensor(target_attention_mask, dtype=torch.long),
            })
    
    return data

class CitationDataset(Dataset):
    """Dataset for citation data with stacked targets."""
    
    def __init__(self, data):
        self.data = data
    
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
