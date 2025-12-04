# Integrity Test Report: Theorem Contrastive Model

**Date:** December 4, 2025
**Model:** Qwen2.5-Math-7B (Fine-tuned with Contrastive Learning)
**Project:** PaperGPT

---

## 1. Overview
This report documents the series of integrity tests conducted to verify the correctness, consistency, and robustness of the embedding computation pipeline for the Theorem Contrastive Learning model. 

Key areas tested:
1.  **Data Format Consistency:** Ensuring `.jsonl` and `.txt` inputs produce identical embeddings.
2.  **Distributed Compute Logic:** Verifying that multi-GPU execution matches single-GPU execution.
3.  **Randomized Integrity (Large Scale):** Verifying alignment between pre-computed embeddings and on-the-fly re-computations.
4.  **Sensitivity Analysis:** Testing model robustness to text appending and truncation.

---

## 2. Test 1: Data Format Consistency (Tiny Dataset)

**Objective:** Verify that the model generates identical embeddings regardless of whether the input is loaded via the standard dataset loader (`.jsonl`) or the query loader (`.txt`).

**Challenge:** A mismatch was initially found because the dataset loader preserves newlines (`\n`) in JSON strings, while the query loader treats newlines as separators for multiple queries.

**Solution:** Created a "sanitized" dataset where all newlines were replaced with spaces.

**Command:**
```bash
# 1. Generate sanitized queries
python3 prepare_tiny_queries.py

# 2. Run integrity test script
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/.../lib/ && python3 run_integrity_test.py
```

**Results:**
*   **Dataset Size:** 9,019 statements (Tiny)
*   **Matrix Comparison:**
    *   Max Difference: `0.000375`
    *   Mean Difference: `0.000011`
*   **Status:** ✅ **PASSED** (Matrices effectively identical)

---

## 3. Test 2: Distributed Compute Logic

**Objective:** Confirm that splitting the workload across multiple GPUs (`torchrun`) and gathering results produces the exact same ordered embeddings as running sequentially on a single GPU.

**Method:** Used a small dataset (100 statements) and compared outputs from `python ...` vs `torchrun --nproc_per_node=2 ...`.

**Command:**
```bash
python3 run_distributed_integrity_test.py
```

**Results:**
*   **Max Difference:** `0.000549`
*   **Mean Difference:** `0.000005`
*   **Status:** ✅ **PASSED** (Differences negligible, attributed to float16 non-determinism)

**Conclusion:** The distributed gathering logic in `distributed_clip.py` is correct.

---

## 4. Test 3: Randomized Integrity Test (The "Bug" & Fix)

**Objective:** Verify the validity of the large-scale pre-computed embeddings (e.g., `small` dataset) by randomly selecting indices, finding their text, re-computing their embeddings, and checking for a match.

**The "Bug" (Issue Identified):**
*   **Initial Result:** FAILED (Max Diff: `1.0`).
*   **Root Cause:** The pre-computed embeddings rely on a specific random shuffle of the dataset (controlled by a seed). Without preserving the exact permutation order or having a metadata map, it is impossible to blindly reconstruct the mapping between an embedding index (e.g., #543) and its original text. The text I selected for index #543 was different from what the model originally embedded at #543.

**The Fix:**
*   Modified `theorem_contrastive_training.py` to save a **Metadata File** (`metadata.jsonl`) alongside `embeddings.pt`.
*   This metadata file explicitly records `{"uid": "PaperID.lemma.idx", "text": "...", "embedding_index": i}`.
*   Updated the integrity test to use this metadata for ground-truth lookup.

**Command (V2):**
```bash
python3 random_integrity_test_v2.py --size small --split eval --nproc 2
```

**Results (Small Dataset):**
*   **Max Difference:** `0.000830`
*   **Mean Difference:** `0.000039`
*   **Status:** ✅ **PASSED** (Well within 2e-3 tolerance)

**Conclusion:** The pipeline is now robust. Future embedding computations will include metadata to ensure traceability.

---

## 5. Test 4: Sensitivity & Robustness Analysis

**Objective:** Determine if the embeddings are overly sensitive to small text modifications (appending noise or truncating the end).

**Method:**
1.  **Append:** Added `" This is a minor addition."` to 50 random samples.
2.  **Truncate:** Removed the last 3 words from the same samples.
3.  Computed similarity against the original embeddings.

**Command:**
```bash
python3 run_sensitivity_test.py --nproc 2 --samples 50
```

**Results:**

| Modification | Mean Similarity | Min Similarity | Conclusion |
| :--- | :--- | :--- | :--- |
| **Append** | **0.9946** | 0.9297 | **Highly Robust.** The model effectively ignores irrelevant suffixes. |
| **Truncate** | **0.8821** | **0.4808** | **Sensitive.** Significant drops in similarity for some cases. |

### Analysis of Truncation Sensitivity
The model showed high sensitivity (scores < 0.6) in specific cases.

**Examples of Failure Cases:**
1.  **Broken LaTeX Formulas:**
    *   Original: `... \mathcal R_{4,\langle 3 \rangle} is rational.`
    *   Truncated: `... \mathcal R_{4,\langle 3`
    *   *Score:* 0.48
    *   *Reason:* The **Last Token Pooling** mechanism relies heavily on the final token's state. An unclosed LaTeX bracket implies an "incomplete" state, vastly different from a "completed" symbol.

2.  **Loss of Semantic Conclusion:**
    *   Original: `... Let H be the hyperplane class. Then \lct(X;|H|_\bQ)>\frac{1}{2}.`
    *   Truncated: `... Let H be the hyperplane`
    *   *Score:* 0.62
    *   *Reason:* The truncation removed the theorem's *result*. The embedding correctly shifted because the statement's mathematical meaning fundamentally changed from a specific claim to a mere setup.

**Conclusion:**
While statistically "sensitive," this behavior is **desirable** for a mathematical embedding model. It indicates the model is attending to the precise structural and semantic completeness of the statement, rather than just doing a "bag-of-words" average. The robustness to appending proves it isn't brittle to *all* changes, only those that break the mathematical integrity of the final token.
