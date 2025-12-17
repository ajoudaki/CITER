#!/usr/bin/env python3
"""
Bag-of-Tokens Retrieval Baseline (GPU-accelerated scoring) for JSONL corpus of papers.

Each line in the input JSONL is a paper/record with fields:
  - "lemmas":   list[str]
  - "theorems": list[str]

Process:
  - Tokenize every lemma/theorem with a Hugging Face tokenizer.
  - Merge lemmas+theorems into one pool per paper.
  - Query schedule: first one query per eligible paper (>=2 items) before repeats.
  - Per query: candidate set = 1 positive (same paper, different item) + K negatives (other papers).
  - Score on GPU:
        * cosine  : dense count BoW + cosine
        * jaccard : dense binary BoW + Jaccard
        * bm25    : dense count BoW + BM25 with corpus IDF + avg doc length      # CHANGE
  - Report Top-1 / Top-10 / MRR.

Usage example:
  python bag_token_retrieval_gpu.py \
      --input path/to/data.jsonl \
      --hf_model bert-base-uncased \
      --negatives 50 \
      --max_queries 5000 \
      --similarity bm25 \
      --bm25_k1 1.2 \
      --bm25_b 0.75 \
      --device cuda \
      --gpu_batch_size 2048 \
      --random_seed 42

Requirements: pip install transformers torch tqdm
"""

import argparse
import json
import os
import random
import sys
import math  # CHANGE: for BM25 idf
from typing import Dict, List, Tuple, Any, Iterator, Optional

# ---- Dependencies ----
try:
    import torch  # GPU scoring
except ImportError:
    print("This script requires 'torch'. Install with: pip install torch", file=sys.stderr)
    raise

try:
    from transformers import AutoTokenizer
except ImportError:
    print("This script requires 'transformers'. Install with: pip install transformers", file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ---------------------------
# Data loading & tokenization
# ---------------------------

def verify_input_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")


def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    verify_input_file_exists(path)
    records = []
    print(f"[INFO] Loading papers from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading papers", unit="line"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on a line: {e}")
    print(f"[INFO] Loaded {len(records)} papers.")
    return records


def extract_items(records: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Returns a list of papers; each is a list of strings (lemmas + theorems).
    """
    papers: List[List[str]] = []
    total_lemmas = 0
    total_theorems = 0
    empty_before_merge = 0

    print("[INFO] Extracting and merging lemmas + theorems per paper ...")
    for rec in tqdm(records, desc="Extracting items", unit="paper"):
        lemmas = rec.get("lemmas", [])
        theorems = rec.get("theorems", [])
        if not isinstance(lemmas, list) or not isinstance(theorems, list):
            raise ValueError("Both 'lemmas' and 'theorems' must be lists if present.")
        total_lemmas += sum(1 for x in lemmas if isinstance(x, str) and x.strip() != "")
        total_theorems += sum(1 for x in theorems if isinstance(x, str) and x.strip() != "")

        items = []
        items.extend([x for x in lemmas if isinstance(x, str) and x.strip() != ""])
        items.extend([x for x in theorems if isinstance(x, str) and x.strip() != ""])
        if len(items) == 0:
            empty_before_merge += 1
        papers.append(items)

    print(f"[INFO] Total lemmas: {total_lemmas}, total theorems: {total_theorems}.")
    print(f"[INFO] Papers with zero usable items before merge: {empty_before_merge}.")
    return papers


def get_vocab_size(tokenizer) -> int:
    if hasattr(tokenizer, "vocab_size") and isinstance(tokenizer.vocab_size, int):
        return int(tokenizer.vocab_size)
    vocab = tokenizer.get_vocab()
    return int(len(vocab))


def tokenize_items(papers: List[List[str]], hf_model: str, batch_size: int = 64) -> Tuple[List[List[List[int]]], int]:
    """
    Tokenize each item across papers into lists of token IDs.
    Returns: (tokenized_papers, vocab_size)
    """
    print(f"[INFO] Loading tokenizer: {hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
    vocab_size = get_vocab_size(tokenizer)

    flat_texts: List[str] = []
    idx_map: List[Tuple[int, int]] = []
    for p_idx, items in enumerate(papers):
        for i_idx, text in enumerate(items):
            flat_texts.append(text)
            idx_map.append((p_idx, i_idx))

    print(f"[INFO] Tokenizing {len(flat_texts)} items ...")
    tokenized_flat: List[List[int]] = [None] * len(flat_texts)  # type: ignore

    for start in tqdm(range(0, len(flat_texts), batch_size), desc="Tokenizing", unit="batch"):
        batch = flat_texts[start:start + batch_size]
        enc = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
        for j, ids in enumerate(enc["input_ids"]):
            tokenized_flat[start + j] = ids

    tokenized_papers: List[List[List[int]]] = [[[] for _ in items] for items in papers]
    for (p_idx, i_idx), ids in zip(idx_map, tokenized_flat):
        tokenized_papers[p_idx][i_idx] = ids

    print("[INFO] Tokenization complete.")
    return tokenized_papers, vocab_size


# ---------------------------
# GPU BoT utilities
# ---------------------------

def ensure_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Install CUDA-enabled PyTorch or use --device cpu.")
        return torch.device("cuda")
    elif device_arg.lower() in ("cpu", "auto"):
        if device_arg.lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")
    else:
        if device_arg.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"{device_arg} requested but CUDA not available.")
            return torch.device(device_arg)
        raise ValueError(f"Unrecognized device: {device_arg}")


def ids_to_dense_bow(ids: List[int], vocab_size: int, device: torch.device, binary: bool) -> torch.Tensor:
    """
    Build a dense bag-of-token vector using torch.bincount on the target device.
    Returns float32 tensor of shape [vocab_size].
    """
    if not ids:
        return torch.zeros(vocab_size, dtype=torch.float32, device=device)
    t = torch.tensor(ids, dtype=torch.long, device=device)
    counts = torch.bincount(t, minlength=vocab_size).to(torch.float32)
    if binary:
        counts = (counts > 0).to(torch.float32)
    return counts


def batch_ids_to_dense_bow(batch_ids: List[List[int]], vocab_size: int, device: torch.device, binary: bool) -> torch.Tensor:
    """
    Stack multiple items into a matrix [batch, vocab_size] on device.
    """
    mats = []
    for ids in batch_ids:
        mats.append(ids_to_dense_bow(ids, vocab_size, device, binary))
    return torch.stack(mats, dim=0) if mats else torch.zeros((0, vocab_size), dtype=torch.float32, device=device)


# ---------------------------
# Sampling & evaluation
# ---------------------------

def build_index(tokenized_papers: List[List[List[int]]]) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    all_items: List[Tuple[int, int]] = []
    all_tokens: List[List[int]] = []
    for p_idx, items in enumerate(tokenized_papers):
        for i_idx, ids in enumerate(items):
            all_items.append((p_idx, i_idx))
            all_tokens.append(ids)
    return all_items, all_tokens


def papers_with_at_least_k_items(papers: List[List[Any]], k: int) -> List[int]:
    return [i for i, items in enumerate(papers) if len(items) >= k]


def sample_negatives(candidate_pool: List[Tuple[int, int]],
                     query_paper_idx: int,
                     k: int,
                     rng: random.Random) -> List[int]:
    """
    Choose up to k indices from papers != query_paper_idx.
    If k >= available_negatives, return ALL eligible negatives (no replacement).  # CHANGE (cap)
    """
    eligible = [idx for idx, (pidx, _) in enumerate(candidate_pool) if pidx != query_paper_idx]
    if k >= len(eligible):
        return list(eligible)
    return rng.sample(eligible, k)


# CHANGE: streaming query iterator to avoid constructing all queries up front
def query_iterator(paper_to_indices: Dict[int, List[int]],
                   eligible_papers: List[int],
                   rng: random.Random,
                   max_queries: int) -> Iterator[int]:
    """
    Yields query indices:
      - First pass: exactly one random item per eligible paper (order randomized)
      - Second pass: remaining items (order randomized per paper and across papers)
    Stops as soon as 'max_queries' is reached (if > 0).
    """
    yielded = 0
    chosen_once: Dict[int, int] = {}

    first_order = list(eligible_papers)
    rng.shuffle(first_order)
    for p in first_order:
        q_idx = rng.choice(paper_to_indices[p])
        chosen_once[p] = q_idx
        yield q_idx
        yielded += 1
        if max_queries > 0 and yielded >= max_queries:
            return

    second_order = list(eligible_papers)
    rng.shuffle(second_order)
    for p in second_order:
        remaining = list(paper_to_indices[p])
        rng.shuffle(remaining)
        for idx in remaining:
            if idx == chosen_once.get(p):
                continue
            yield idx
            yielded += 1
            if max_queries > 0 and yielded >= max_queries:
                return


# ---------- BM25 corpus stats (IDF & avg doc length) ----------

def compute_bm25_corpus_stats(all_tokens: List[List[int]], vocab_size: int) -> Tuple[torch.Tensor, float]:
    """
    Compute BM25 IDF vector (length vocab_size) and average document length.
    Returns (idf_cpu_float32_tensor, avgdl_float).                              # CHANGE
    """
    print("[INFO] Computing BM25 corpus statistics (document frequencies & avg doc length) ...")  # CHANGE
    df = [0] * vocab_size
    total_len = 0
    for ids in tqdm(all_tokens, desc="BM25: scanning docs", unit="doc"):  # CHANGE
        total_len += len(ids)
        # unique terms per document for df
        for t in set(ids):
            if 0 <= t < vocab_size:
                df[t] += 1
    N = len(all_tokens)
    avgdl = (total_len / max(N, 1)) if N > 0 else 0.0
    idf = [0.0] * vocab_size
    for t in range(vocab_size):
        n = df[t]
        # Robertson-Sparck Jones with +1 variant (Lucene-style)                 # CHANGE
        idf[t] = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
    idf_tensor_cpu = torch.tensor(idf, dtype=torch.float32)
    nonzero_df = sum(1 for x in df if x > 0)
    print(f"[INFO] BM25: N={N}, avgdl={avgdl:.3f}, terms with df>0: {nonzero_df}/{vocab_size}")  # CHANGE
    return idf_tensor_cpu, avgdl


def evaluate_retrieval_gpu(tokenized_papers: List[List[List[int]]],
                           all_tokens: List[List[int]],
                           vocab_size: int,
                           negatives: int,
                           max_queries: int,
                           similarity: str,
                           random_seed: int,
                           device: torch.device,
                           gpu_batch_size: int,
                           bm25_idf_cpu: Optional[torch.Tensor] = None,  # CHANGE
                           bm25_avgdl: Optional[float] = None,            # CHANGE
                           bm25_k1: float = 1.2,                          # CHANGE
                           bm25_b: float = 0.75                           # CHANGE
                           ) -> Dict[str, float]:
    rng = random.Random(random_seed)
    print("[INFO] Building global index of items ...")
    all_items, _ = build_index(tokenized_papers)
    print(f"[INFO] Total items in index: {len(all_items)}")

    # Eligible papers (>=2 items)
    eligible = papers_with_at_least_k_items(tokenized_papers, 2)
    eligible_papers_set = set(eligible)
    if not eligible_papers_set:
        raise ValueError("No papers have at least 2 items to form (query, positive) pairs.")
    print(f"[INFO] Papers eligible as query sources (≥2 items): {len(eligible_papers_set)}")

    # Map paper -> list of global item indices
    paper_to_indices: Dict[int, List[int]] = {}
    for idx, (p_idx, _) in enumerate(all_items):
        paper_to_indices.setdefault(p_idx, []).append(idx)

    # Precompute the expected total for tqdm without materializing queries
    total_queries = max_queries if max_queries > 0 else sum(len(paper_to_indices[p]) for p in eligible_papers_set)

    # Prepare BM25 stats on demand                                           # CHANGE
    if similarity == "bm25":
        if bm25_idf_cpu is None or bm25_avgdl is None:
            bm25_idf_cpu, bm25_avgdl = compute_bm25_corpus_stats(all_tokens, vocab_size)
        idf_device = bm25_idf_cpu.to(device)
        print(f"[INFO] BM25 params: k1={bm25_k1}, b={bm25_b}, avgdl={bm25_avgdl:.3f}")  # CHANGE

    print(f"[INFO] Using device: {device.type.upper()}{'' if device.index is None else f':{device.index}'}")
    print(f"[INFO] Evaluating up to {total_queries} queries "
          f"(negatives per query = {negatives}, gpu_batch_size = {gpu_batch_size}, mode = {similarity}) ...")

    mrr_total = 0.0
    top1 = 0
    top10 = 0
    n_evaluated = 0

    use_binary = (similarity == "jaccard")

    q_iter = query_iterator(
        paper_to_indices=paper_to_indices,
        eligible_papers=list(eligible_papers_set),
        rng=rng,
        max_queries=max_queries
    )

    for q_idx in tqdm(q_iter, desc="Evaluating queries", unit="query", total=total_queries):
        q_paper, _ = all_items[q_idx]
        q_ids = all_tokens[q_idx]

        same_paper_indices = [idx for idx in paper_to_indices[q_paper] if idx != q_idx]
        if not same_paper_indices:
            continue
        pos_idx = rng.choice(same_paper_indices)

        neg_indices = sample_negatives(all_items, q_paper, negatives, rng)
        # Safety: ensure no same-paper items in negatives and query not included
        neg_indices = [i for i in neg_indices if all_items[i][0] != q_paper and i != q_idx]

        candidate_indices = [pos_idx] + neg_indices
        rng.shuffle(candidate_indices)

        # ---- Scoring paths ----
        if similarity in ("cosine", "jaccard"):
            # Build query vector
            q_vec = ids_to_dense_bow(q_ids, vocab_size, device, binary=use_binary)  # [V]

            sims_parts: List[torch.Tensor] = []
            for start in range(0, len(candidate_indices), gpu_batch_size):
                chunk_inds = candidate_indices[start:start + gpu_batch_size]
                cand_ids_batch = [all_tokens[i] for i in chunk_inds]
                C = batch_ids_to_dense_bow(cand_ids_batch, vocab_size, device, binary=use_binary)  # [B,V]

                if similarity == "cosine":
                    q_norm = torch.linalg.vector_norm(q_vec) + 1e-12
                    c_norms = torch.linalg.vector_norm(C, dim=1) + 1e-12
                    sims_chunk = (C @ q_vec) / (c_norms * q_norm)
                else:
                    # Jaccard on binary vectors: inter / union
                    inter = (C * q_vec).sum(dim=1)
                    union = C.sum(dim=1) + q_vec.sum() - inter
                    sims_chunk = inter / (union + 1e-12)
                sims_parts.append(sims_chunk)

            sims = torch.cat(sims_parts, dim=0)  # [num_candidates]

        elif similarity == "bm25":  # CHANGE: BM25 path
            # Unique query tokens (IDs) on device
            if len(q_ids) == 0:
                # Degenerate (shouldn't happen given earlier filtering)
                sims = torch.zeros(len(candidate_indices), dtype=torch.float32, device=device)
            else:
                q_terms = torch.unique(torch.tensor(q_ids, dtype=torch.long, device=device))
                idf_q = idf_device[q_terms]  # [Tq]

                sims_parts: List[torch.Tensor] = []
                for start in range(0, len(candidate_indices), gpu_batch_size):
                    chunk_inds = candidate_indices[start:start + gpu_batch_size]
                    cand_ids_batch = [all_tokens[i] for i in chunk_inds]
                    # BM25 needs **counts** (not binary)
                    C = batch_ids_to_dense_bow(cand_ids_batch, vocab_size, device, binary=False)  # [B,V]

                    # Document lengths from counts
                    dl = C.sum(dim=1)  # [B]
                    norm = (1.0 - bm25_b) + bm25_b * (dl / (bm25_avgdl + 1e-12))  # [B]

                    tf = C[:, q_terms]  # [B,Tq]
                    denom = tf + bm25_k1 * norm.view(-1, 1)  # [B,Tq]
                    numer = tf * (bm25_k1 + 1.0)             # [B,Tq]
                    per_term = (numer / (denom + 1e-12)) * idf_q.view(1, -1)  # [B,Tq]
                    sims_chunk = per_term.sum(dim=1)  # [B]
                    sims_parts.append(sims_chunk)

                sims = torch.cat(sims_parts, dim=0)

        else:
            raise ValueError("similarity must be one of: cosine, jaccard, bm25")

        # Rank (higher is better)
        sims_cpu = sims.detach().to("cpu")
        sorted_idx = torch.argsort(sims_cpu, descending=True)

        pos_loc = candidate_indices.index(pos_idx)
        where = (sorted_idx == pos_loc).nonzero(as_tuple=False)
        if where.numel() == 0:
            # Fallback linear search
            rank_pos = None
            for r_i in range(sorted_idx.numel()):
                if sorted_idx[r_i].item() == pos_loc:
                    rank_pos = r_i
                    break
            if rank_pos is None:
                continue
            rank = rank_pos + 1
        else:
            rank = int(where[0, 0].item()) + 1

        n_evaluated += 1
        mrr_total += 1.0 / rank
        if rank == 1:
            top1 += 1
        if rank <= 10:
            top10 += 1

    if n_evaluated == 0:
        raise ValueError("No queries were evaluated; check your data and parameters.")

    return {
        "evaluated_queries": float(n_evaluated),
        "MRR": mrr_total / n_evaluated,
        "Top1": top1 / n_evaluated,
        "Top10": top10 / n_evaluated
    }


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bag-of-Tokens Retrieval Baseline (lemmas+theorems merged, GPU scoring + BM25)")
    parser.add_argument("--input", required=True, help="Path to input JSONL file with 'lemmas' and/or 'theorems'.")
    parser.add_argument("--hf_model", default="bert-base-uncased",
                        help="HF tokenizer model name or path (default: bert-base-uncased).")
    parser.add_argument("--negatives", type=int, default=50, help="Number of negatives per query (default: 50).")
    parser.add_argument("--max_queries", type=int, default=5000,
                        help="Max queries to evaluate (0 = use all eligible; default: 5000).")
    parser.add_argument("--similarity", choices=["cosine", "jaccard", "bm25"], default="cosine",  # CHANGE
                        help="Similarity: cosine (TF), jaccard (binary), or bm25 (corpus-weighted).")  # CHANGE
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--batch_size", type=int, default=64, help="Tokenizer batch size (default: 64).")
    parser.add_argument("--device", default="cuda", help="Device: 'cuda', 'cuda:0', 'cpu', or 'auto' (default: cuda).")
    parser.add_argument("--gpu_batch_size", type=int, default=4096,
                        help="Max number of candidates scored per GPU chunk (default: 4096).")
    # BM25 hyperparameters                                                    # CHANGE
    parser.add_argument("--bm25_k1", type=float, default=1.2, help="BM25 k1 (default: 1.2).")  # CHANGE
    parser.add_argument("--bm25_b", type=float, default=0.75, help="BM25 b (default: 0.75).")  # CHANGE
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Data
    records = load_jsonl_records(args.input)
    papers_text = extract_items(records)

    # Drop completely empty papers
    non_empty = [(idx, items) for idx, items in enumerate(papers_text) if len(items) > 0]
    removed = len(papers_text) - len(non_empty)
    if not non_empty:
        raise ValueError("No non-empty papers (lemmas+theorems).")
    print(f"[INFO] Removing {removed} empty papers; keeping {len(non_empty)}.")

    papers_text_compact = [items for _, items in non_empty]

    # Tokenize
    tokenized_papers, vocab_size = tokenize_items(papers_text_compact, args.hf_model, batch_size=args.batch_size)

    # Flatten for quick access to token lists (same order as build_index)
    print("[INFO] Preparing flat token access list ...")
    all_items, all_tokens = build_index(tokenized_papers)
    print(f"[INFO] Vocab size: {vocab_size}")

    # Device
    device = ensure_device(args.device)

    # Precompute BM25 stats if needed                                          # CHANGE
    bm25_idf_cpu = None
    bm25_avgdl = None
    if args.similarity == "bm25":                                              # CHANGE
        bm25_idf_cpu, bm25_avgdl = compute_bm25_corpus_stats(all_tokens, vocab_size)

    # Evaluate (GPU)
    metrics = evaluate_retrieval_gpu(
        tokenized_papers=tokenized_papers,
        all_tokens=all_tokens,
        vocab_size=vocab_size,
        negatives=args.negatives,
        max_queries=args.max_queries,
        similarity=args.similarity,
        random_seed=args.random_seed,
        device=device,
        gpu_batch_size=args.gpu_batch_size,
        bm25_idf_cpu=bm25_idf_cpu,               # CHANGE
        bm25_avgdl=bm25_avgdl,                   # CHANGE
        bm25_k1=args.bm25_k1,                    # CHANGE
        bm25_b=args.bm25_b                       # CHANGE
    )

    print("Unit: lemmas+theorems (merged)")
    print(f"HF tokenizer: {args.hf_model}")
    print(f"Negatives per query: {args.negatives}")
    print(f"Similarity: {args.similarity}")
    if args.similarity == "bm25":
        print(f"BM25 k1={args.bm25_k1}, b={args.bm25_b}")
    print(f"Evaluated queries: {int(metrics['evaluated_queries'])}")
    print(f"Top-1 Accuracy: {metrics['Top1']:.4f}")
    print(f"Top-10 Accuracy: {metrics['Top10']:.4f}")
    print(f"MRR: {metrics['MRR']:.4f}")


if __name__ == "__main__":
    main()
