#!/usr/bin/env python3
"""
Embedding-based Retrieval (cosine similarity) for lemma/theorem items.

This script reads precomputed embedding shards (.pt files) that each contain:
  - 'ids': Python list (or similar) of string IDs in the form
           "{paper_idx}-lemma-{i}" or "{paper_idx}-theorem-{i}"
  - 'embeddings': torch.FloatTensor of shape [num_items, dim]

It reconstructs the paper→items mapping from the ID strings and evaluates
retrieval where the positive is from the same paper as the query, and
negatives are from other papers. Scoring uses cosine similarity.

Usage example:
  python embedding_cosine_retrieval.py \
      --embeddings_dir path/to/pt_shards \
      --negatives 50 \
      --max_queries 5000 \
      --device cuda \
      --gpu_batch_size 4096 \
      --random_seed 42

Requirements: pip install torch tqdm
"""

import argparse
import os
import re
import sys
import random
from typing import Dict, List, Tuple, Any, Iterator, Optional

try:
    import torch
except ImportError:
    print("This script requires 'torch'. Install with: pip install torch", file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ---------------------------
# Utilities
# ---------------------------

ID_PATTERN = re.compile(r"^(?P<paper>\d+)-(?:lemma|theorem)-(?P<idx>\d+)$")


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


def list_pt_files(embeddings_dir: str) -> List[str]:
    if not os.path.isdir(embeddings_dir):
        raise FileNotFoundError(f"Embeddings dir not found: {embeddings_dir}")
    files = [os.path.join(embeddings_dir, f) for f in os.listdir(embeddings_dir) if f.endswith(".pt")]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No .pt files found in {embeddings_dir}")
    return files


def coerce_ids_to_list(obj: Any) -> List[str]:
    """
    Convert various container types to a Python list[str]. Supported:
      - list[str]
      - list[bytes] (decoded as utf-8)
      - tuple[str]
      - numpy arrays (object/str) if present (optional dependency)
    """
    # Already a list of strings
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], str)):
        return obj
    # List of bytes -> decode
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], (bytes, bytearray))):
        return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in obj]
    # Tuple[str]
    if isinstance(obj, tuple) and (len(obj) == 0 or isinstance(obj[0], str)):
        return list(obj)
    # Numpy array
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind in ("U", "S", "O"):
                return [str(x) for x in obj.tolist()]
    except Exception:
        pass
    # Torch tensor of strings is not supported in PyTorch; guard explicitly
    if torch.is_tensor(obj):
        raise TypeError("'ids' appears to be a torch.Tensor; expected list-like of strings.")
    # Fallback
    raise TypeError(f"Unsupported 'ids' container type: {type(obj)}")


def parse_paper_index(id_str: str) -> int:
    m = ID_PATTERN.match(id_str)
    if not m:
        raise ValueError(f"ID does not match expected pattern '<paper>-(lemma|theorem)-<idx>': {id_str}")
    return int(m.group("paper"))


def load_embedding_shards(pt_files: List[str]) -> Tuple[List[str], torch.Tensor, List[int]]:
    """
    Load multiple .pt shards and return:
      - all_ids: list[str]
      - all_embeds: torch.FloatTensor [N, D]
      - item_to_paper: list[int] (paper index per item)
    """
    all_ids: List[str] = []
    all_embeds: List[torch.Tensor] = []
    item_to_paper: List[int] = []

    for path in tqdm(pt_files, desc="Loading shards", unit="file"):
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or ("ids" not in obj or "embeddings" not in obj):
            raise ValueError(f"Shard {path} must be a dict with keys 'ids' and 'embeddings'.")

        ids = coerce_ids_to_list(obj["ids"])  # list[str]
        emb = obj["embeddings"]
        if not torch.is_tensor(emb):
            raise TypeError(f"Shard {path}: 'embeddings' must be a torch.Tensor.")
        if emb.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError(f"Shard {path}: 'embeddings' must be a floating tensor, got {emb.dtype}.")
        emb = emb.to(dtype=torch.float32, device="cpu")

        if len(ids) != emb.shape[0]:
            raise ValueError(f"Shard {path}: len(ids)={len(ids)} != embeddings.shape[0]={emb.shape[0]}.")

        all_ids.extend(ids)
        all_embeds.append(emb)
        # Map each id to its paper index
        for s in ids:
            item_to_paper.append(parse_paper_index(s))

    all_embeds_tensor = torch.cat(all_embeds, dim=0) if all_embeds else torch.zeros((0, 0), dtype=torch.float32)
    return all_ids, all_embeds_tensor, item_to_paper


def normalize_embeddings(emb: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize embeddings along dim=1. Zeros remain zeros.
    Input: [N, D]. Output: [N, D] float32 on CPU.
    """
    if emb.numel() == 0:
        return emb
    # Use stable row-wise normalization with eps; zero rows remain zero.
    return torch.nn.functional.normalize(emb, p=2, dim=1, eps=1e-12)


# ---------------------------
# Sampling & evaluation (mirrors bag_of_tokens_retrieval scheduling)
# ---------------------------

def papers_with_at_least_k_items(paper_to_indices: Dict[int, List[int]], k: int) -> List[int]:
    return [p for p, idxs in paper_to_indices.items() if len(idxs) >= k]


def sample_negatives(global_indices: List[int],
                     item_to_paper: List[int],
                     query_paper_idx: int,
                     k: int,
                     rng: random.Random) -> List[int]:
    eligible = [idx for idx in global_indices if item_to_paper[idx] != query_paper_idx]
    if k >= len(eligible):
        return list(eligible)
    return rng.sample(eligible, k)


def query_iterator(paper_to_indices: Dict[int, List[int]],
                   eligible_papers: List[int],
                   rng: random.Random,
                   max_queries: int) -> Iterator[int]:
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


def evaluate_retrieval_cosine(all_ids: List[str],
                              all_embeds_norm_cpu: torch.Tensor,
                              item_to_paper: List[int],
                              negatives: int,
                              max_queries: int,
                              random_seed: int,
                              device: torch.device,
                              gpu_batch_size: int) -> Dict[str, float]:
    rng = random.Random(random_seed)

    N = all_embeds_norm_cpu.shape[0]
    if N == 0:
        raise ValueError("No embeddings loaded.")
    dim = all_embeds_norm_cpu.shape[1]
    print(f"[INFO] Loaded {N} items with embedding dim {dim}.")

    # Build paper→indices mapping and list of global indices
    paper_to_indices: Dict[int, List[int]] = {}
    for idx, p in enumerate(item_to_paper):
        paper_to_indices.setdefault(p, []).append(idx)
    all_global_indices = list(range(N))

    eligible_papers = papers_with_at_least_k_items(paper_to_indices, 2)
    if not eligible_papers:
        raise ValueError("No papers have at least 2 items to form (query, positive) pairs.")
    print(f"[INFO] Papers eligible as query sources (≥2 items): {len(eligible_papers)}")

    total_queries = max_queries if max_queries > 0 else sum(len(paper_to_indices[p]) for p in eligible_papers)
    print(f"[INFO] Using device: {device.type.upper()}{'' if device.index is None else f':{device.index}'}")
    print(f"[INFO] Evaluating up to {total_queries} queries (negatives per query = {negatives}, gpu_batch_size = {gpu_batch_size}) ...")

    mrr_total = 0.0
    top1 = 0
    top10 = 0
    n_evaluated = 0

    # Iterate queries with the same schedule as the bag-of-tokens baseline
    q_iter = query_iterator(
        paper_to_indices=paper_to_indices,
        eligible_papers=list(eligible_papers),
        rng=rng,
        max_queries=max_queries,
    )

    for q_idx in tqdm(q_iter, desc="Evaluating queries", unit="query", total=total_queries):
        q_paper = item_to_paper[q_idx]
        same_paper_indices = [idx for idx in paper_to_indices[q_paper] if idx != q_idx]
        if not same_paper_indices:
            continue
        pos_idx = rng.choice(same_paper_indices)

        neg_indices = sample_negatives(all_global_indices, item_to_paper, q_paper, negatives, rng)
        neg_indices = [i for i in neg_indices if item_to_paper[i] != q_paper and i != q_idx]

        candidate_indices = [pos_idx] + neg_indices
        rng.shuffle(candidate_indices)

        # Cosine scoring via dot product on unit vectors
        q_vec = all_embeds_norm_cpu[q_idx].to(device)

        sims_parts: List[torch.Tensor] = []
        for start in range(0, len(candidate_indices), gpu_batch_size):
            chunk_inds = candidate_indices[start:start + gpu_batch_size]
            C = all_embeds_norm_cpu[chunk_inds].to(device)  # [B, D]
            sims_chunk = C @ q_vec  # [B]
            sims_parts.append(sims_chunk)

        sims = torch.cat(sims_parts, dim=0) if sims_parts else torch.zeros(0, device=device)

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
        "Top10": top10 / n_evaluated,
    }


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embedding-based Retrieval (cosine) for lemma/theorem items")
    p.add_argument("--embeddings_dir", required=True, help="Directory containing .pt files with 'ids' and 'embeddings'.")
    p.add_argument("--negatives", type=int, default=50, help="Number of negatives per query (default: 50).")
    p.add_argument("--max_queries", type=int, default=5000, help="Max queries to evaluate (0 = use all eligible; default: 5000).")
    p.add_argument("--random_seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--device", default="cuda", help="Device: 'cuda', 'cuda:0', 'cpu', or 'auto' (default: cuda).")
    p.add_argument("--gpu_batch_size", type=int, default=4096, help="Max candidates per device chunk (default: 4096).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pt_files = list_pt_files(args.embeddings_dir)
    print(f"[INFO] Found {len(pt_files)} shard(s) in {args.embeddings_dir}.")

    all_ids, all_embeds_cpu, item_to_paper = load_embedding_shards(pt_files)

    print("[INFO] Normalizing embeddings (L2) ...")
    all_embeds_norm_cpu = normalize_embeddings(all_embeds_cpu)

    device = ensure_device(args.device)

    metrics = evaluate_retrieval_cosine(
        all_ids=all_ids,
        all_embeds_norm_cpu=all_embeds_norm_cpu,
        item_to_paper=item_to_paper,
        negatives=args.negatives,
        max_queries=args.max_queries,
        random_seed=args.random_seed,
        device=device,
        gpu_batch_size=args.gpu_batch_size,
    )

    print("Unit: lemmas+theorems (merged)")
    print(f"Negatives per query: {args.negatives}")
    print("Similarity: cosine (embeddings)")
    print(f"Evaluated queries: {int(metrics['evaluated_queries'])}")
    print(f"Top-1 Accuracy: {metrics['Top1']:.4f}")
    print(f"Top-10 Accuracy: {metrics['Top10']:.4f}")
    print(f"MRR: {metrics['MRR']:.4f}")


if __name__ == "__main__":
    main()
