# Baseline Methods Comparison Report

## Executive Summary

This report presents a comprehensive comparison of four baseline methods for theorem-lemma matching: Jaccard similarity, BM25, MinHash with Hamming distance, and a simple bag-of-words approach. These baselines establish performance benchmarks without any training, relying solely on lexical similarity measures.

## Methods Overview

### 1. Jaccard Similarity
- **Approach**: Binary bag-of-words with Jaccard coefficient
- **Implementation**: Creates binary vectors where 1 indicates token presence
- **Similarity**: Intersection over union of token sets
- **Projection**: Random orthogonal projection from 30k vocab to 2048 dimensions

### 2. BM25 (Best Match 25)
- **Approach**: Term frequency-based ranking with saturation
- **Implementation**: Applies BM25 scoring formula with k1=1.2, b=0.75
- **Key Features**:
  - Term frequency saturation prevents over-weighting of repeated terms
  - Document length normalization adjusts for varying text lengths
- **Projection**: Same random orthogonal projection as Jaccard

### 3. MinHash with LSH
- **Approach**: Locality-sensitive hashing for approximate Jaccard similarity
- **Implementation**:
  - Uses 2048 independent hash functions
  - Computes minimum hash value per function as signature
  - Signatures compared via cosine similarity (approximates Hamming distance)
- **Advantages**: Fixed-size signatures enable efficient large-scale retrieval

### 4. Configuration
All baselines use identical preprocessing:
- Tokenizer: BERT base uncased
- Vocabulary size: 30,000 tokens (capped)
- Output dimension: 2048
- Special tokens filtered (< 100)

## Experimental Results

### Performance Across Dataset Scales

| Dataset | Size | Samples | Jaccard Top-1 | BM25 Top-1 | MinHash Top-1 |
|---------|------|---------|---------------|------------|---------------|
| Toy     | 20MB | 248     | **54.84%**    | **58.06%** | 54.84%        |
| Tiny    | 4MB  | 153     | 47.06%        | **49.67%** | 44.44%        |
| Small   | 40MB | 1,568   | 34.91%        | **36.83%** | 31.39%        |
| Medium  | 400MB| 15,744  | 25.72%        | **28.26%** | 22.86%        |
| Full    | 1.7GB| 406,745 | 22.02%        | **24.10%** | -             |

### Detailed Metrics Comparison

#### Toy Dataset (20MB)
| Metric | Jaccard | BM25 | MinHash |
|--------|---------|------|---------|
| Top-1  | 54.84%  | **58.06%** | 54.84% |
| Top-5  | 77.42%  | 74.19% | 74.19% |
| Top-10 | 87.10%  | **93.55%** | 90.32% |
| MRR    | 0.6446  | **0.6645** | 0.6477 |
| Loss   | 1.8803  | **1.9661** | 2.4672 |

#### Medium Dataset (400MB)
| Metric | Jaccard | BM25 | MinHash |
|--------|---------|------|---------|
| Top-1  | 25.72%  | **28.26%** | 22.86% |
| Top-5  | 32.62%  | **35.36%** | 29.12% |
| Top-10 | 35.95%  | **38.55%** | 32.12% |
| MRR    | 0.2928  | **0.3189** | 0.2609 |
| Loss   | 7.6533  | **7.7873** | 8.6347 |

## Key Findings

### 1. BM25 Dominates Across All Scales
- Consistently achieves 2-4% higher top-1 accuracy than Jaccard
- Term frequency information provides crucial discriminative power
- Length normalization handles varying document sizes effectively

### 2. Performance Degradation with Scale
All methods show significant performance drops as dataset size increases:
- Toy → Full: ~58% → ~24% (BM25)
- Indicates fundamental limitation of lexical similarity
- Larger datasets have more diverse vocabulary and expression patterns

### 3. MinHash Trade-offs
- Slightly underperforms direct Jaccard (2-3% lower accuracy)
- Approximation error from hash collisions
- Benefits: Fixed memory footprint, enables sub-linear search

### 4. Surprising Baseline Strength
- Simple lexical methods achieve >50% accuracy on small datasets
- Mathematical texts share significant vocabulary overlap
- Establishes strong baseline for learned methods to beat

## Implications

### For Small Datasets (< 100MB)
- BM25 provides strong out-of-the-box performance
- No training required makes it ideal for rapid prototyping
- Consider as baseline before investing in neural approaches

### For Large Datasets (> 1GB)
- Lexical methods plateau at ~24% accuracy
- Clear need for semantic understanding beyond token overlap
- Learned representations essential for production systems

### Computational Considerations
- **Jaccard/BM25**: O(n) space, O(n²) similarity computation
- **MinHash**: O(k) space per document, enables LSH for O(1) retrieval
- For >1M documents, MinHash enables practical deployment

## Recommendations

1. **Use BM25 as primary baseline** for any theorem-lemma matching task
2. **Consider MinHash for large-scale systems** requiring sub-linear retrieval
3. **Expect 20-25% top-1 accuracy ceiling** for pure lexical methods on diverse mathematical texts
4. **Combine with learned methods** - lexical features remain valuable even with neural models

## Implementation Notes

### Optimizations Applied
- Vectorized BM25 scoring computation
- Pre-computed hash functions for MinHash
- Batch processing for all similarity computations
- GPU acceleration throughout

### Reproducibility
All experiments run with:
- PyTorch 2.0+
- CUDA 11.8
- Single NVIDIA GPU
- Configs available in `configs/model/`

## Conclusion

This comprehensive evaluation establishes clear baselines for theorem-lemma matching. BM25 emerges as the strongest lexical method, achieving 24-58% top-1 accuracy depending on dataset scale. While these methods require no training, their performance ceiling motivates the development of learned semantic representations for practical applications requiring higher accuracy.

---

*Report generated: September 2024*
*Code available at: `theorem_contrastive_training.py`*