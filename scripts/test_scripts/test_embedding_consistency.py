#!/usr/bin/env python3
"""Test if query embedding matches precomputed embedding for same text."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel


class SimpleEncoder(nn.Module):
    def __init__(self, base_model, projection, model_type: str):
        super().__init__()
        self.base_model = base_model
        self.projection = projection
        self.model_type = model_type

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        if 'bert' in self.model_type.lower():
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(sequence_lengths), device=sequence_lengths.device)
            embedding = outputs.last_hidden_state[batch_indices, sequence_lengths, :]
        return F.normalize(self.projection(embedding), p=2, dim=-1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = Path('outputs/demo/big_run_qwen-7b')

    # Load model exactly like compute_embeddings.py
    base_model_name = 'Qwen/Qwen2.5-Math-7B-Instruct'
    hidden_dim = 3584
    model_type = 'qwen-7b'

    print(f"Loading {base_model_name} with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModel.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map={'': 0}
    )

    # Load LoRA
    lora_path = model_dir / 'qwen-2.5-math-7b_lora_adapters'
    print(f"Loading LoRA from {lora_path}")
    base_model = PeftModel.from_pretrained(base_model, str(lora_path))

    # Load projection
    projection = nn.Linear(hidden_dim, 2048)
    proj_path = model_dir / 'qwen-2.5-math-7b_projection.pt'
    print(f"Loading projection from {proj_path}")
    projection.load_state_dict(torch.load(proj_path, map_location='cpu'))

    # Create encoder
    encoder = SimpleEncoder(base_model, projection, model_type)
    encoder.projection = encoder.projection.to(device).half()
    encoder.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # The exact query text
    query_text = """Let $f_P(p)$ denote the pdf of the random variable $P$ from which the $p_i$ are drawn for a VBSC. Let $f_P(p)$ be known at encoder and decoder and let the realizations of the $p_i$ be unknown at encoder and decoder. Then, the capacity of the VBSC is

\\begin{equation*} C_{VBSC} = C_{BSC}\\left(\\Exp\\big[P\\big]\\right) \\enspace . \\end{equation*}"""

    # Compute embedding
    print("\nComputing query embedding...")
    with torch.no_grad():
        inputs = tokenizer(
            [query_text],
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with autocast(device_type='cuda', enabled=True):
            query_embedding = encoder(input_ids, attention_mask)

        query_embedding = query_embedding.float().cpu()

    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Query embedding norm: {torch.norm(query_embedding).item():.6f}")
    print(f"Query embedding first 10: {query_embedding[0, :10]}")

    # Load precomputed embeddings
    print("\nLoading precomputed embeddings...")
    emb_path = model_dir / 'embeddings/toy/embeddings.pt'
    precomputed_emb = torch.load(emb_path, map_location='cpu', weights_only=False)

    print(f"Precomputed embedding[0] shape: {precomputed_emb[0].shape}")
    print(f"Precomputed embedding[0] norm: {torch.norm(precomputed_emb[0]).item():.6f}")
    print(f"Precomputed embedding[0] first 10: {precomputed_emb[0, :10]}")

    # Compare
    similarity = torch.matmul(query_embedding, precomputed_emb[0]).item()
    print(f"\n{'='*60}")
    print(f"Similarity between query and precomputed[0]: {similarity:.8f} ({similarity*100:.2f}%)")
    print(f"{'='*60}")

    if similarity > 0.99:
        print("✅✅✅ PERFECT MATCH - Embeddings are consistent!")
    elif similarity > 0.95:
        print("✅ VERY HIGH - Embeddings are mostly consistent")
    else:
        print("❌ LOW SIMILARITY - Embeddings are INCONSISTENT!")
        print("\nDifference analysis:")
        diff = (query_embedding - precomputed_emb[0]).abs()
        print(f"  Max difference: {diff.max().item():.6f}")
        print(f"  Mean difference: {diff.mean().item():.6f}")


if __name__ == '__main__':
    main()
