# Guide: Notebook 1 — Transformer Fundamentals

> **Read this before opening the notebook.** It tells you what concepts to know, what code you need to write from memory, and what to expect in the interview.

---

## What This Notebook Covers

The transformer architecture is the foundation of every modern LLM including all Mistral models. Mistral interviewers specifically ask you to implement pieces of this live. You need to be able to write these functions in under 10 minutes each, with correct types and no bugs.

---

## Concepts to Understand (No Code Needed — Just Know the Explanation)

### 1. Why Self-Attention?
- RNNs process tokens sequentially — slow to train, hard to parallelize
- Attention processes all tokens in parallel and captures long-range dependencies in one step
- The word "bank" in "river bank" vs "bank account" gets different representations depending on surrounding tokens — that's attention working

### 2. Q, K, V — What They Are
- **Q (Query):** "What am I looking for?" — comes from the current token
- **K (Key):** "What do I have to offer?" — comes from each token in context
- **V (Value):** "What information do I actually send?" — comes from each token
- The dot product `Q · K` measures relevance. High score = pay more attention to that token's V

### 3. Why Scale by sqrt(d_k)?
- Dot products of high-dimensional vectors have large variance
- Without scaling, softmax gets very peaked (near-zero gradients → vanishing gradient problem)
- Dividing by `sqrt(d_k)` keeps the variance ≈ 1 regardless of dimension size

### 4. Why Multiple Heads?
- Each head can learn to attend to different relationships simultaneously
- Head 1 might track syntactic structure, Head 2 might track coreference ("she" → "Maria")
- Outputs of all heads are concatenated and projected back to d_model

### 5. Pre-LN vs Post-LN
- **Post-LN** (original 2017 paper): `output = LayerNorm(x + sublayer(x))` → unstable at large scale
- **Pre-LN** (GPT-2, Mistral): `output = x + sublayer(LayerNorm(x))` → more stable, easier to train deep models
- Mistral and all modern LLMs use Pre-LN

### 6. Residual Connections
- `x = x + sublayer(x)` — adds skip connection around each block
- Gradients can flow directly through the addition without going through the nonlinearity
- Without residuals, very deep networks don't train

### 7. Causal Masking
- During training, we predict all next tokens in parallel
- Token at position 3 must NOT see positions 4, 5, 6... (that's the future)
- We mask future positions by setting their scores to `-inf` before softmax → they become 0 after softmax
- Why `-inf` not `0`? Because `softmax(0) > 0`, but `softmax(-inf) = 0` exactly

---

## Code You Must Write From Memory

Practice these until you can write them without hesitation:

### 1. Softmax (Numerically Stable)
```python
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```
**Key:** subtract max before exp. That's it.

### 2. Scaled Dot-Product Attention
```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, V), weights
```
**Key shapes:** Q, K, V all `(batch, seq, d_k)`, output `(batch, seq, d_v)`.

### 3. Causal Mask
```python
def make_causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
```
`k=1` means "above the main diagonal" = future positions.

### 4. Multi-Head Attention Split/Merge
```python
# Split: (batch, seq, d_model) → (batch, heads, seq, d_k)
def split_heads(x, num_heads):
    batch, seq, d = x.shape
    return x.reshape(batch, seq, num_heads, d // num_heads).transpose(0, 2, 1, 3)

# Merge: (batch, heads, seq, d_k) → (batch, seq, d_model)
def merge_heads(x):
    batch, heads, seq, d_k = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, seq, heads * d_k)
```

### 5. Layer Normalization
```python
def layer_norm(x, gamma, beta, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta
```

### 6. Full Transformer Block (Pre-LN)
```python
def block(x, attn, ffn, ln1, ln2, mask=None):
    x = x + attn(ln1(x), mask=mask)   # attention sublayer
    x = x + ffn(ln2(x))               # FFN sublayer
    return x
```

---

## What the Interviewer Will Ask Verbally

Be ready to explain these without code:

1. "Walk me through what happens when the word 'bank' is processed in a transformer"
2. "Why is attention O(n²)? How does Mistral's sliding window attention improve this?"
3. "What does Multi-Head Attention let us learn that single-head doesn't?"
4. "What's the difference between encoder and decoder attention?"

---

## Numbers to Know

| Thing | Value |
|---|---|
| Mistral 7B d_model | 4096 |
| Mistral 7B num_layers | 32 |
| Mistral 7B num_q_heads | 32 |
| Mistral 7B num_kv_heads | 8 (GQA) |
| Mistral 7B head_dim | 128 |
| Attention complexity | O(n²·d) |
| SWA complexity | O(n·W·d) where W=4096 |

---

## Common Mistakes to Avoid

1. **Transposing wrong dimension:** `K.T` works for 2D but not batched. Use `K.transpose(0, 2, 1)`.
2. **Forgetting keepdims=True** in softmax max subtraction — broadcasting will fail silently.
3. **Forgetting to scale:** Dividing by `sqrt(d_k)` BEFORE softmax, not after.
4. **Post-LN vs Pre-LN:** Mistral uses Pre-LN. Don't get this wrong.
5. **Causal mask direction:** `np.triu(ones, k=1)` = upper triangle WITHOUT diagonal = future tokens.
