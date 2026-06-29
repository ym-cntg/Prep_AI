# Guide: Notebook 6 — MoE, LoRA, and Alignment

> **This is Mistral-specific knowledge.** Interviewers will definitely ask about Mixtral's MoE architecture and fine-tuning methods since these are central to Mistral's product.

---

## Mixture of Experts (MoE) — Mistral's Secret Weapon

### The Core Idea
Instead of one big FFN layer, use N smaller expert FFN layers. A router picks which K experts handle each token.

```
Token → Router → [scores for 8 experts] → pick top-2 → weighted sum of 2 expert outputs
```

### Why This Is Smart
- **Parameters:** 47B total (8 experts × ~6B per expert)
- **Active compute:** Only 2 experts fire per token → ~13B active params worth of compute
- **Result:** Capability of a 47B model at compute cost of a ~13B model

### The Router
```python
# Router is just a linear layer: d_model → num_experts
router_logits = token @ W_router  # (d_model,) @ (d_model, 8) → (8,)
probs = softmax(router_logits)     # probability for each expert
top2_experts = argsort(probs)[-2:]  # pick top 2
```

### Why Exactly Top-2 (Not Top-1 or Top-4)?
- Top-1: less expressive, training instability
- Top-4: compute cost increases, marginal quality gain
- Top-2 is the empirical sweet spot in the Mixtral paper

### Load Balancing Problem
Without a balancing loss, the router collapses to always picking expert 0 and 1:
- They get more gradient updates → they get better
- Better experts get picked more → feedback loop
- Eventually only 1-2 experts train, 6 are wasted

**Fix:** Add auxiliary loss that penalizes imbalance:
```
L_aux = num_experts × Σᵢ (fraction_dispatched_to_i × mean_router_prob_for_i)
```
Perfect balance → loss = 1.0. Higher loss = more imbalanced.

### How to Explain Mixtral in an Interview
"Mixtral 8x7B has 32 transformer layers. Each layer replaces the single FFN with 8 expert FFNs. A learned router selects 2 experts per token. Only those 2 compute, so the effective FLOP count is similar to a 13B dense model, but the model has 47B parameters — more capacity without proportionally more compute."

---

## LoRA — Low-Rank Adaptation

### The Problem LoRA Solves
Full fine-tuning: update all 7B parameters. Need:
- 14GB for model weights
- 14GB for gradients
- 56GB for Adam optimizer states (fp32 m and v)
= **84GB** for 7B model. Doesn't fit on a single GPU.

### LoRA's Solution
Freeze all original weights W₀. Add a tiny trainable add-on:
```
W_effective = W₀ + B @ A
```
Where:
- W₀ ∈ ℝ^(d×k) — frozen, untouched
- A ∈ ℝ^(d×r) — trainable, randomly initialized
- B ∈ ℝ^(r×k) — trainable, **initialized to zero**

**Why B=0 at init?** So that `ΔW = B@A = 0` initially. The model starts identical to the base model and LoRA adapters start learning from that baseline.

### Parameter Count
For d=k=4096 (one attention projection in Mistral 7B), r=8:
- Full params: 4096 × 4096 = 16.7M
- LoRA params: 4096×8 + 8×4096 = 65,536 (0.4% of full)
- With 32 layers, 4 modules (Q,K,V,O): 65,536 × 4 × 32 = 8.4M trainable vs 7B frozen

### Memory During Fine-Tuning
- Base model: 14GB (frozen, no gradients needed)
- LoRA params: 8.4M × 4 bytes (float32) = 34MB
- LoRA gradients: 34MB
- Optimizer states: 68MB
- **Total:** ~14.5GB — fits on a single 24GB GPU!

### The Scale Factor (Alpha)
```python
scale = alpha / rank  # typically alpha = rank * 2
output = W0(x) + scale * (x @ A @ B)
```
Higher alpha → LoRA has more influence. Think of it as a learning rate multiplier.

### Where to Inject LoRA
- Minimum: Q, V projections (original paper)
- Better: Q, K, V, O (all attention projections)
- Maximum: all attention + FFN projections
- Mistral fine-tuning best practice: at least Q, K, V, O

### QLoRA
LoRA + quantized base model:
- Load base model in INT4 (3.5GB instead of 14GB)
- Keep LoRA adapters in BF16
- Total memory: ~4GB for 7B model fine-tuning
- Enables fine-tuning on a consumer GPU (RTX 3090, 24GB)

---

## DPO — Direct Preference Optimization

### The Problem with RLHF
Standard alignment (RLHF = Reinforcement Learning from Human Feedback) requires:
1. Train a reward model on human preferences
2. Use PPO (RL algorithm) to optimize policy against reward model
3. This is unstable, slow, and complex

### DPO's Insight
The optimal RLHF policy has a closed form. You can directly train on preference pairs without a reward model.

### DPO Loss
Given a preference pair (y_w = winner, y_l = loser) for prompt x:
```python
reward_w = log π(y_w|x) - log π_ref(y_w|x)
reward_l = log π(y_l|x) - log π_ref(y_l|x)
loss = -log(sigmoid(β × (reward_w - reward_l)))
```

**Intuition:** Increase probability of y_w relative to reference model, decrease probability of y_l. β controls how much the policy is allowed to deviate from the reference.

### When to Use DPO vs RLHF
- DPO: simpler, no reward model needed, works well for style/format alignment
- RLHF: better for complex tasks where reward model can explore diverse feedback signals

---

## Grouped Query Attention (GQA) — Mistral's Memory Trick

### Standard Multi-Head Attention (MHA)
- 32 Q heads, 32 K heads, 32 V heads
- KV cache = 32 heads × 2 (K+V) × seq_len × head_dim

### Grouped Query Attention (GQA) — What Mistral 7B Uses
- 32 Q heads but only 8 KV heads
- Each group of 4 Q heads shares the same K and V head
- KV cache reduced by 4× (32/8)

```python
# Expand K and V to match Q heads
K_expanded = np.repeat(K, groups, axis=1)  # 8 heads → 32 heads
V_expanded = np.repeat(V, groups, axis=1)
# Then standard attention
```

### Multi-Query Attention (MQA)
Extreme case: 1 KV head for ALL Q heads. Even more memory savings, slight quality loss.

### The Numbers
| Approach | KV Heads | KV Cache for 7B, seq=32k |
|---|---|---|
| MHA (standard) | 32 | 16 GB |
| GQA (Mistral 7B) | 8 | 4 GB |
| MQA | 1 | 0.5 GB |

---

## Sliding Window Attention (SWA)

### Why Not Full Attention?
Full attention: O(N²). For N=32k: 1 billion operations per layer. At 32 layers: 32 billion.

### SWA
Each token attends to only the last W tokens:
- O(N × W) instead of O(N²)
- Mistral uses W = 4096

### But Wait — Doesn't This Lose Long-Range Context?
No! Through multiple layers, information propagates:
- Layer 1: token can see 4096 tokens back
- Layer 2: each of those can see 4096 tokens back from their position
- After k layers: effective range = k × W

32 layers × 4096 window = effectively 131k tokens of range even for earlier layers.

---

## RoPE — Rotary Position Embedding

### Why Not Sinusoidal PE?
- Sinusoidal PE: additive, interacts poorly with attention at long range
- Doesn't extrapolate well beyond training length

### RoPE Key Idea
Rotate Q and K vectors by an angle that depends on position:
- `RoPE(q, pos) = q × cos(θ×pos) + rotate_half(q) × sin(θ×pos)`
- The dot product `RoPE(q,pos1) · RoPE(k,pos2)` naturally encodes the relative position `pos1-pos2`

### Why RoPE is Better
- Relative position encoded directly in attention scores
- No extra parameters
- Better extrapolation (with RoPE variants like YaRN) to longer sequences than seen in training

---

## Perplexity — The Standard LLM Metric

**Formula:** `PPL = exp(-1/N × Σ log P(wᵢ | w_{<i}))`

**Interpretation:**
- PPL = 1: perfect model (assigns prob 1 to every correct token — impossible)
- PPL = vocab_size: random guessing
- Good LLMs: PPL 5-30 on standard benchmarks

**Limitation:** Can't compare PPL across models with different tokenizers. Longer tokenizations inflate PPL.

---

## Key Answers for Common Interview Questions

**Q: How does Mixtral compare to a 70B dense model?**
A: Mixtral 8x7B has 47B total params but only 13B active per token. A 70B dense model has 70B active per token — 5x more compute. But Mixtral has 47B total capacity. In practice, Mixtral 8x7B performs comparably to Llama 2 70B at ~5x lower inference compute.

**Q: LoRA rank choice?**
A: r=8 or r=16 for most tasks. r=64 for large-scale fine-tuning where you can afford the extra params. Higher rank = more expressiveness but more compute and potential overfitting.

**Q: Why not just fine-tune the whole model?**
A: Memory, cost, and catastrophic forgetting. Fine-tuning all 7B parameters requires 4-5x the model size in GPU memory for gradients + optimizer states. LoRA trains 0.1% of params with minimal quality loss.

**Q: DPO vs RLHF in production?**
A: Mistral's Instruct models use DPO because it's simpler and works well for instruction following. RLHF is used where more complex reward modeling is needed (like Anthropic's Constitutional AI).
