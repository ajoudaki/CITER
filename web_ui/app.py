#!/usr/bin/env python3
"""
Flask web application for browsing mathematical theorems and lemmas with LaTeX rendering.
"""

from flask import Flask, render_template, jsonify, request
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from tqdm import tqdm

app = Flask(__name__)

# Global storage for loaded dataset
dataset_cache = {}
current_dataset = None

# Global storage for loaded models
loaded_models = {}
current_model = None
model_embeddings_cache = {}

def load_dataset(dataset_name: str = 'toy') -> List[Dict]:
    """Load dataset from JSONL file."""
    global dataset_cache, current_dataset

    if dataset_name in dataset_cache:
        current_dataset = dataset_cache[dataset_name]
        return current_dataset

    dataset_path = Path(f'../data/lemmas_theorems/{dataset_name}.jsonl')

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")

    papers = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            paper = json.loads(line.strip())
            papers.append(paper)

    dataset_cache[dataset_name] = papers
    current_dataset = papers
    return papers

def get_paper_summary(paper: Dict) -> Dict:
    """Extract summary information from a paper."""
    # Handle both data formats
    if 'statements' in paper:
        # New format with statements array
        return {
            'arxiv_id': paper.get('arxiv_id', 'Unknown'),
            'title': paper.get('title', 'Untitled'),
            'num_theorems': len([s for s in paper.get('statements', []) if s.get('type') == 'theorem']),
            'num_lemmas': len([s for s in paper.get('statements', []) if s.get('type') == 'lemma']),
            'total_statements': len(paper.get('statements', []))
        }
    else:
        # Old format with direct theorems/lemmas arrays
        return {
            'arxiv_id': paper.get('arxiv_id', f"Paper_{id(paper)%10000:04d}"),
            'title': paper.get('title', f"Paper (T:{len(paper.get('theorems', []))}, L:{len(paper.get('lemmas', []))})"),
            'num_theorems': len(paper.get('theorems', [])),
            'num_lemmas': len(paper.get('lemmas', [])),
            'total_statements': len(paper.get('theorems', [])) + len(paper.get('lemmas', []))
        }

@app.route('/')
def index():
    """Main page with paper browser."""
    return render_template('index.html')

@app.route('/api/datasets')
def list_datasets():
    """List available datasets."""
    datasets_dir = Path('../data/lemmas_theorems')
    datasets = []

    if datasets_dir.exists():
        for path in datasets_dir.glob('*.jsonl'):
            name = path.stem
            size_mb = path.stat().st_size / (1024 * 1024)
            datasets.append({
                'name': name,
                'size': f"{size_mb:.1f}MB",
                'path': str(path)
            })

    return jsonify(datasets)

@app.route('/api/load_dataset/<dataset_name>')
def load_dataset_api(dataset_name: str):
    """Load a specific dataset and return paper list."""
    try:
        papers = load_dataset(dataset_name)
        summaries = [get_paper_summary(paper) for paper in papers]
        return jsonify({
            'success': True,
            'num_papers': len(papers),
            'papers': summaries
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/paper/<int:paper_idx>')
def get_paper(paper_idx: int):
    """Get detailed information for a specific paper."""
    if current_dataset is None:
        return jsonify({'success': False, 'error': 'No dataset loaded'}), 400

    if paper_idx < 0 or paper_idx >= len(current_dataset):
        return jsonify({'success': False, 'error': 'Invalid paper index'}), 400

    paper = current_dataset[paper_idx]

    # Handle both data formats
    if 'statements' in paper:
        # New format with statements array
        theorems = []
        lemmas = []
        others = []

        for stmt in paper.get('statements', []):
            stmt_data = {
                'text': stmt.get('text', ''),
                'type': stmt.get('type', 'unknown')
            }

            if stmt['type'] == 'theorem':
                theorems.append(stmt_data)
            elif stmt['type'] == 'lemma':
                lemmas.append(stmt_data)
            else:
                others.append(stmt_data)
    else:
        # Old format with direct theorems/lemmas arrays
        theorems = [{'text': t, 'type': 'theorem'} for t in paper.get('theorems', [])]
        lemmas = [{'text': l, 'type': 'lemma'} for l in paper.get('lemmas', [])]
        others = []

    return jsonify({
        'success': True,
        'arxiv_id': paper.get('arxiv_id', f"Paper_{paper_idx:04d}"),
        'title': paper.get('title', f"Paper {paper_idx + 1} (T:{len(theorems)}, L:{len(lemmas)})"),
        'theorems': theorems,
        'lemmas': lemmas,
        'others': others
    })

def load_model(model_name: str):
    """Load a trained model for similarity search."""
    global loaded_models, current_model

    if model_name in loaded_models:
        current_model = model_name
        return loaded_models[model_name]

    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = torch.cuda.is_available()  # Use fp16 if CUDA is available
    print(f"Loading model on {device} (fp16: {use_fp16})")

    # Simple encoder class matching training code
    class SimpleEncoder(nn.Module):
        def __init__(self, base_model, projection):
            super().__init__()
            self.base_model = base_model
            self.projection = projection

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token for BERT, last token for Qwen
            if 'bert' in model_name.lower():
                embedding = outputs.last_hidden_state[:, 0, :]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(len(sequence_lengths), device=sequence_lengths.device)
                embedding = outputs.last_hidden_state[batch_indices, sequence_lengths, :]
            return F.normalize(self.projection(embedding), p=2, dim=-1)

    # Load the model
    if model_name == 'bert-base':
        base_model_name = 'bert-base-uncased'
        hidden_dim = 768
    elif model_name == 'qwen-1.5b':
        base_model_name = 'Qwen/Qwen2.5-1.5B'
        hidden_dim = 1536
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load base model and LoRA adapters
    base_model = AutoModel.from_pretrained(base_model_name)
    lora_path = Path(f'../outputs/{model_name}_lora_adapters')
    if lora_path.exists():
        base_model = PeftModel.from_pretrained(base_model, str(lora_path))

    # Load projection layer - output dim is 2048 based on saved models
    projection_path = Path(f'../outputs/{model_name}_projection.pt')
    projection = nn.Linear(hidden_dim, 2048)  # Changed from 768 to 2048
    if projection_path.exists():
        projection.load_state_dict(torch.load(projection_path, map_location='cpu'))

    # Create encoder and move to device
    encoder = SimpleEncoder(base_model, projection)
    encoder = encoder.to(device)

    # Convert to fp16 if using CUDA
    if use_fp16:
        encoder = encoder.half()

    encoder.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    loaded_models[model_name] = {
        'encoder': encoder,
        'tokenizer': tokenizer,
        'device': device,
        'use_fp16': use_fp16
    }
    current_model = model_name
    return loaded_models[model_name]

def compute_embeddings_for_dataset():
    """Compute embeddings for all statements in the current dataset."""
    global model_embeddings_cache, current_dataset, current_model

    if not current_model or not current_dataset:
        return None

    cache_key = f"{current_model}_{id(current_dataset)}"
    if cache_key in model_embeddings_cache:
        return model_embeddings_cache[cache_key]

    model_data = loaded_models[current_model]
    encoder = model_data['encoder']
    tokenizer = model_data['tokenizer']
    device = model_data['device']
    use_fp16 = model_data.get('use_fp16', False)

    all_embeddings = []
    all_metadata = []

    # Collect all texts first
    all_texts = []
    for paper_idx, paper in enumerate(current_dataset):
        # Get statements
        if 'statements' in paper:
            statements = paper['statements']
        else:
            statements = []
            for theorem in paper.get('theorems', []):
                statements.append({'text': theorem, 'type': 'theorem'})
            for lemma in paper.get('lemmas', []):
                statements.append({'text': lemma, 'type': 'lemma'})

        for stmt in statements:
            all_texts.append(stmt.get('text', ''))
            all_metadata.append({
                'paper_idx': paper_idx,
                'paper_title': paper.get('title', 'Untitled'),
                'type': stmt.get('type', 'unknown'),
                'text': stmt.get('text', '')
            })

    # Process in batches for efficiency - use larger batch size with fp16
    batch_size = 32 if use_fp16 else 8  # Larger batch size with fp16
    print(f"Computing embeddings for {len(all_texts)} statements (batch_size={batch_size}, fp16={use_fp16})...")
    with torch.no_grad():
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Processing batches"):
            batch_texts = all_texts[i:i+batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )

            # Move inputs to device and get embeddings
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Use autocast for mixed precision
            with autocast(enabled=use_fp16):
                embeddings = encoder(input_ids, attention_mask)

            # Move embeddings back to CPU for storage
            all_embeddings.append(embeddings.float().cpu())

            # Clear GPU memory after each batch
            del embeddings, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
    else:
        all_embeddings = torch.zeros(0, 2048)  # Changed from 768 to 2048

    model_embeddings_cache[cache_key] = {
        'embeddings': all_embeddings,
        'metadata': all_metadata
    }

    return model_embeddings_cache[cache_key]

@app.route('/api/models')
def list_models():
    """List available models."""
    models = []
    outputs_dir = Path('../outputs')

    # Check for bert model
    if (outputs_dir / 'bert-base_projection.pt').exists():
        models.append({'name': 'bert-base', 'display': 'BERT Base'})

    # Check for qwen model
    if (outputs_dir / 'qwen-1.5b_projection.pt').exists():
        models.append({'name': 'qwen-1.5b', 'display': 'Qwen 1.5B'})

    return jsonify(models)

@app.route('/api/load_model/<model_name>')
def load_model_api(model_name: str):
    """Load a specific model."""
    try:
        load_model(model_name)
        # Precompute embeddings for current dataset
        if current_dataset:
            compute_embeddings_for_dataset()
        return jsonify({'success': True, 'model': model_name})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/find_similar', methods=['POST'])
def find_similar():
    """Find similar statements to a query statement."""
    if not current_model:
        return jsonify({'success': False, 'error': 'No model loaded'}), 400

    if not current_dataset:
        return jsonify({'success': False, 'error': 'No dataset loaded'}), 400

    paper_idx = request.json.get('paper_idx')
    stmt_idx = request.json.get('stmt_idx')
    stmt_type = request.json.get('stmt_type')

    # Get embeddings
    cache_data = compute_embeddings_for_dataset()
    if not cache_data:
        return jsonify({'success': False, 'error': 'Failed to compute embeddings'}), 400

    embeddings = cache_data['embeddings']
    metadata = cache_data['metadata']

    # Find the query embedding
    query_idx = None
    for i, meta in enumerate(metadata):
        if meta['paper_idx'] == paper_idx:
            # Count statements of the same type to find the right one
            if meta['type'] == stmt_type:
                if stmt_idx == 0:
                    query_idx = i
                    break
                stmt_idx -= 1

    if query_idx is None:
        return jsonify({'success': False, 'error': 'Statement not found'}), 400

    # Compute similarities
    query_embedding = embeddings[query_idx:query_idx+1]
    similarities = torch.matmul(query_embedding, embeddings.t()).squeeze()

    # Sort and get top results (exclude self)
    top_k = min(1000, len(similarities) - 1)
    values, indices = torch.topk(similarities, top_k + 1)

    # Filter out self and prepare results
    results = []
    for i, idx in enumerate(indices):
        if idx != query_idx:
            meta = metadata[idx]
            results.append({
                'similarity': float(values[i]),
                'paper_idx': meta['paper_idx'],
                'paper_title': meta['paper_title'],
                'type': meta['type'],
                'text': meta['text']  # Show full text, no truncation
            })

    return jsonify({
        'success': True,
        'results': results[:top_k],
        'query': metadata[query_idx]
    })

@app.route('/api/search', methods=['POST'])
def search_statements():
    """Search for statements containing specific text."""
    if current_dataset is None:
        return jsonify({'success': False, 'error': 'No dataset loaded'}), 400

    query = request.json.get('query', '').lower()
    if not query:
        return jsonify({'success': False, 'error': 'Empty query'}), 400

    results = []
    for paper_idx, paper in enumerate(current_dataset):
        for stmt in paper.get('statements', []):
            if query in stmt.get('text', '').lower():
                results.append({
                    'paper_idx': paper_idx,
                    'paper_title': paper.get('title', 'Untitled'),
                    'arxiv_id': paper.get('arxiv_id', 'Unknown'),
                    'type': stmt.get('type', 'unknown'),
                    'text': stmt.get('text', '')  # Show full text, no truncation
                })

    return jsonify({
        'success': True,
        'num_results': len(results),
        'results': results[:100]  # Limit to 100 results
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)