# Length Mismatch Analysis: Training vs Inference

## Scenario: Train max_len=256, Infer max_len=512

### Summary of Issues

| Component | Issue Severity | Description |
|-----------|---------------|-------------|
| **Position Embeddings** | 🟡 Medium | RoPE can extrapolate but quality degrades |
| **Attention Patterns** | 🔴 High | Model never learned long-range dependencies |
| **Projection Layer** | 🟡 Medium | Trained on different input distributions |
| **Last Token Selection** | 🟢 None | Dynamically adapts to any length |
| **Memory/Speed** | 🟡 Medium | 4x memory/compute for 2x length increase |

---

## Detailed Analysis

### 1. **Position Embeddings: Extrapolation**

```python
# Training: Model sees positions [0, 1, 2, ..., 255]
# Inference: Model needs positions [0, 1, 2, ..., 511]
```

**Qwen uses RoPE (Rotary Position Embeddings):**
- Can extrapolate to unseen positions
- BUT: Quality degrades beyond training length
- Positions 256-511 will have **untested** encoding

**Impact:** Moderate quality degradation for very long sequences.

---

### 2. **Attention Pattern Distribution Shift** ⚠️ CRITICAL

This is the **most serious issue**:

#### Training Distribution:
```python
# Last token at position k (where k ≤ 256)
# Attends to: [0, 1, ..., k]
# Average attention context: ~128 tokens
# Long-range attention: max 256 tokens apart
```

**Model learns:**
- How to aggregate info from ≤256 tokens
- Attention weights optimized for this range
- Projection layer expects this input distribution

#### Inference with len=512:
```python
# Last token at position 512
# Attends to: [0, 1, ..., 512]
# Average context: ~256 tokens (DOUBLED!)
# Long-range attention: 512 tokens apart (NEVER SEEN!)
```

**Consequences:**
1. **Attention weight distribution shift**
   - Training: weights learned for max distance of 256
   - Inference: needs to attend 512 tokens back
   - Weights may be poorly calibrated

2. **Information dilution**
   - More tokens to attend to → each token gets less attention
   - Early tokens (position 0-100) may be ignored
   - Model may focus too much on recent tokens (position 400-512)

3. **Hidden state distribution shift**
   ```python
   # Training last token hidden state statistics:
   mean_train = E[h_last | len ≤ 256]
   var_train = Var[h_last | len ≤ 256]

   # Inference:
   mean_infer = E[h_last | len = 512]  # Different!
   var_infer = Var[h_last | len = 512]  # Different!

   # Projection layer was optimized for (mean_train, var_train)
   # Now gets (mean_infer, var_infer) → suboptimal!
   ```

---

### 3. **Projection Layer: Implicit Assumptions**

```python
self.projection = nn.Linear(3584, 2048)
```

This layer was trained with:
```python
# Input distribution during training
X_train ~ h_last(len ≤ 256)

# Learned weights
W = argmin E[||W · X_train - target||²]
```

At inference with len=512:
```python
# New input distribution
X_infer ~ h_last(len = 512)

# If X_infer has different statistics:
# - Different mean → bias mismatch
# - Different variance → scale mismatch
# - Different correlation structure → feature mismatch
```

**Result:** Embeddings may be **miscalibrated**
- Similarity scores may be systematically biased
- Ranking quality degraded

---

### 4. **Practical Impact on Your Use Case**

#### Citation Discovery Task:
```python
# Query: Long theorem (400 tokens)
"Let X be a sequence of random matrices with ..." [truncated to 256]

# Database: Mix of short and long theorems
Short: "For all x, f(x) > 0"  # Fully represented
Long: "Consider the spectral distribution..." [truncated to 256]

# Both query and database are truncated → CONSISTENT
# But: Model never sees complete long theorems
```

**Issue:** Long theorems are **artificially shortened**
- May lose critical information in the tail
- Theorems 1 and 2 might be identical after truncation
- Similarity scores become less meaningful

---

### 5. **Memory and Computational Issues**

```python
# Attention complexity: O(L²)
# Training: 256² = 65,536 operations per head
# Inference (512): 512² = 262,144 operations (4x increase!)

# Memory for attention: O(L²)
# Training: 256² = 64KB per head (for float16)
# Inference: 512² = 256KB per head (4x memory!)
```

With multi-head attention (32 heads for Qwen):
- Training: 64KB × 32 = 2MB per sample
- Inference (512): 256KB × 32 = 8MB per sample

**Batch size impact:**
- Batch=8 at training: 16MB
- Batch=8 at inference (512): 64MB
- **Risk of OOM (Out of Memory)**

---

## Recommendations

### If You Must Use Different Lengths:

#### Option 1: **Stay at 256** (Current - Safest)
- ✅ Consistent with training
- ✅ No distribution shift
- ❌ Long theorems truncated

#### Option 2: **Truncate at Inference Too**
```python
# Always use max_length=256 at inference
tokens = tokenizer(query, max_length=256, truncation=True)
```
- ✅ Matches training distribution
- ✅ Fast and memory efficient
- ❌ Loses information from long theorems

#### Option 3: **Increase Training Length**
```python
# Retrain with max_length=512
cfg.training.max_length = 512
```
- ✅ Handles long theorems
- ✅ No distribution mismatch
- ❌ Requires full retraining
- ❌ 4x more expensive training

#### Option 4: **Use Sliding Window** (Advanced)
```python
# For theorems > 256 tokens:
# 1. Split into overlapping windows of 256
# 2. Embed each window
# 3. Average or max-pool the embeddings
```
- ✅ Handles arbitrary length
- ✅ No distribution shift
- ❌ More complex implementation
- ❌ Slower inference

---

## Test: Does Your Data Exceed 256 Tokens?

Check if this is even a problem:

```python
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")

# Load your dataset
with open('data/lemmas_theorems/small.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Check theorem lengths
lengths = []
for paper in data:
    for stmt in paper.get('theorems', []) + paper.get('lemmas', []):
        tokens = tokenizer(stmt, return_tensors='pt')['input_ids']
        lengths.append(tokens.shape[1])

# Statistics
print(f"Mean length: {np.mean(lengths):.1f}")
print(f"Median length: {np.median(lengths):.1f}")
print(f"95th percentile: {np.percentile(lengths, 95):.1f}")
print(f"Max length: {max(lengths)}")
print(f"% truncated at 256: {100 * sum(l > 256 for l in lengths) / len(lengths):.1f}%")
```

**If <5% are truncated:** Not a major issue
**If >20% truncated:** Consider retraining with longer max_length

---

## Conclusion

### Your Current Setup (256/256): ✅ **SAFE**

### Hypothetical Mismatch (256/512): ⚠️ **RISKY**

**Main risks:**
1. Attention pattern distribution shift (HIGH RISK)
2. Projection layer calibration mismatch (MEDIUM RISK)
3. Position embedding extrapolation (LOW-MEDIUM RISK)
4. Memory/compute constraints (MEDIUM RISK)

**Recommendation:** Keep inference at max_length=256 unless you retrain.
