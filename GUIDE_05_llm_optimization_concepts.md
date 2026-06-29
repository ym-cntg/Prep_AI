# Guide: Notebook 5 — LLM Optimization & Serving

> **This is the system design prep.** Know how to calculate memory requirements, explain KV cache, and design an inference cluster. These are the exact prompts Mistral's system design round uses.

---

## Memory Math — Know These Formulas Cold

### Model Weight Memory
```
memory_GB = num_params × bytes_per_param / 1e9
```

| Dtype | Bytes per param | Mistral 7B | Mixtral 8x7B (47B) |
|---|---|---|---|
| FP32 | 4 | 29 GB | 188 GB |
| BF16 / FP16 | 2 | **14 GB** | **94 GB** |
| INT8 | 1 | 7 GB | 47 GB |
| INT4 | 0.5 | 3.5 GB | 23 GB |

**Say in interview:** "A 7B model in BF16 is about 14GB. INT4 halves that to 3.5GB. A 40GB GPU can run it comfortably in INT4 or tight in BF16."

### KV Cache Memory
```
kv_memory = 2 × num_layers × num_kv_heads × head_dim × batch_size × seq_len × bytes
```

For Mistral 7B (32 layers, 8 KV heads, 128 head_dim, BF16 = 2 bytes):
```
per_token_per_request = 2 × 32 × 8 × 128 × 1 × 1 × 2 = 131,072 bytes ≈ 128 KB
```

| Seq Length | KV Cache (1 request) |
|---|---|
| 1,024 | 128 MB |
| 4,096 | 512 MB |
| 16,384 | 2 GB |
| 32,768 | 4 GB |

**The insight:** At long contexts, KV cache dominates memory — it can exceed model weights for large batches.

---

## KV Cache — What It Is and Why It Matters

### The Problem Without KV Cache
Every time you generate a new token, naive attention recomputes K and V for ALL previous tokens:
- Token 1: compute K,V for [t1]
- Token 2: compute K,V for [t1, t2]
- Token 3: compute K,V for [t1, t2, t3]
- ...
- Token N: compute K,V for [t1, ..., tN]

Total compute: O(N²) — quadratic! For 1000 tokens, you compute KV 1000+999+...+1 ≈ 500K times.

### The Solution
**Cache K and V** for all previous tokens. At each new step, only compute K, V for the new token and append to cache.

- New compute: O(N) — linear!
- New memory: O(N) per sequence in the cache (shown above)

**Why not cache Q?** Q is the current token's "query" — it's what changes every step, not what we're looking up.

### Prefill vs Decode Phase
- **Prefill:** Process the entire prompt in parallel (batched matrix multiply = compute bound)
- **Decode:** Generate one token at a time using KV cache (memory bandwidth bound)

This is why generation feels different from processing — different compute bottlenecks.

---

## Quantization — The Basics

### What It Is
Represent weights in lower precision. Each weight normally takes 16 bits (BF16). Quantize to 8 bits (INT8) or 4 bits (INT4).

### The Trade-Off
- More compression → more memory savings → can fit bigger batch or longer context
- More compression → more quantization error → slight quality loss
- INT8: ~5% quality loss, usually acceptable
- INT4: ~10-15% quality loss, sometimes noticeable

### Absmax INT8 (Know the Formula)
```python
scale = max(|W|) / 127
W_quantized = round(W / scale).clip(-128, 127)

# To use: W_approx = W_quantized * scale
```

### Methods to Know Names Of
- **GPTQ:** Post-training, uses Hessian approximation to minimize error
- **AWQ:** Activation-aware — preserves weights that correspond to large activations
- **bitsandbytes (bnb):** Library by Tim Dettmers, used by HuggingFace. INT8 and NF4 (4-bit NormalFloat)
- **GGUF:** Format used by llama.cpp for CPU inference

---

## Continuous Batching

### Static Batching (Old Way)
1. Collect N requests
2. Process all N until EVERY one finishes
3. Start next batch

**Problem:** If one request wants 2000 tokens and others want 20, the 20-token requests wait for the 2000-token one. GPU idle for long sequences.

### Continuous Batching (vLLM, PagedAttention)
1. Fill N GPU slots
2. When ANY request finishes, immediately fill that slot with the next waiting request
3. Never have empty slots

**Result:** 3-4x better GPU utilization. This is what vLLM implements.

**PagedAttention:** Manages KV cache like virtual memory pages — no more contiguous allocation needed, no memory waste.

---

## Speculative Decoding

**Problem:** Large model generates one token per forward pass. Can we go faster?

**Idea:**
1. Small "draft" model generates K tokens quickly (cheap)
2. Large "target" model verifies all K tokens in ONE parallel forward pass
3. Accept/reject each based on probability ratio
4. If rejected, resample from target distribution

**Win condition:** When draft model is accurate (~70%+ acceptance), you get K tokens for the cost of ~1 large model forward pass.

**No quality loss:** The rejection sampling mathematically guarantees the same distribution as the target model alone.

---

## Parallelism Strategies

### Tensor Parallelism (TP)
- Split weight matrices across GPUs (column-parallel and row-parallel)
- All-reduce between GPUs after each operation
- **Use when:** Model is too large for one GPU but you need low latency
- Mistral 7B: 2 GPUs (TP=2) fits easily

### Pipeline Parallelism (PP)
- Split layers across GPUs (GPU 0 = layers 1-8, GPU 1 = layers 9-16, ...)
- One batch flows through "pipeline" stages
- **Use when:** Model is very deep, many GPUs
- **Problem:** "Pipeline bubble" = GPUs idle waiting for previous stage

### Data Parallelism (DP)
- Same model on each GPU, different batch per GPU
- All-reduce gradients in training
- **Use for:** Training throughput scaling

### Expert Parallelism (EP)
- Different experts live on different GPUs
- Tokens are routed to appropriate GPU
- **Use for:** MoE models (Mixtral)

---

## System Design: Serving Mixtral 8x7B

**Prompt:** "Design inference serving for Mixtral 8x7B with p95 < 5s latency at 1000 concurrent users"

**Your answer structure:**

1. **Memory analysis:**
   - 47B params × 2 bytes = 94GB in BF16 → needs 2 × 80GB A100s minimum
   - KV cache for 1000 concurrent × 2048 tokens = ~8GB per replica

2. **Per-replica:**
   - 2 A100-80GB with Tensor Parallelism (TP=2)
   - vLLM for continuous batching + PagedAttention
   - 64 concurrent requests per replica

3. **Scale out:**
   - 1000 concurrent / 64 per replica = 16 replicas
   - 16 × 2 GPUs = 32 A100s total

4. **Architecture:**
   ```
   Users → Load Balancer → [16 replicas × 2 GPUs]
                          ↓
                     Request Queue → Continuous Batching Engine
                          ↓
                     KV Cache (PagedAttention) → Response
   ```

5. **Optimizations:**
   - Prefix caching for shared system prompts
   - Request prioritization (short requests first)
   - INT4 quantization (halves to 47GB → fits on 1 GPU, doubles throughput)
   - Speculative decoding with a small draft model

---

## Flash Attention — What It Solves

**Standard attention problem:**
- Computing attention score matrix `QKᵀ` creates a (N × N) matrix
- For N=32k: 32k × 32k × 2 bytes = **2GB just for attention scores per head per batch**
- Must read/write this to GPU RAM (HBM) → bandwidth bottleneck

**Flash Attention solution:**
- Compute attention in tiles that fit in fast on-chip SRAM (~20MB)
- Never materialize the full N × N matrix in HBM
- Same numerical result, ~3-4x faster, O(1) extra memory

**Key concept:** Attention is memory-bandwidth bound, not compute bound. Reducing HBM reads/writes is the win.

---

## What the Interviewer Will Ask

1. "How much GPU memory does a 7B model need in production?"
   Answer: 14GB in BF16 (weights only) + KV cache. With INT4, ~3.5GB for weights.

2. "At seq_len=32k, what's the KV cache memory per request for Mistral 7B?"
   Answer: 4GB (calculation: 2 × 32 × 8 × 128 × 32768 × 2 = 4GB)

3. "Why does vLLM outperform naive PyTorch serving?"
   Answer: Continuous batching (no padding waste), PagedAttention (no KV cache fragmentation), CUDA kernel optimizations.

4. "Your inference server shows p99 latency of 60s but p50 is 2s. What's happening?"
   Answer: Long-tail requests blocking short ones (head-of-line blocking). Fix: preemption, priority queue, max_new_tokens limits.
