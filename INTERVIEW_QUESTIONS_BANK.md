# Mistral AI — Interview Questions Bank

> **Use this at the END of prep as a mock interview.** Cover each question with your hand and answer out loud before revealing. Time yourself — you should be able to answer each Q&A question in under 2 minutes and each coding question in under 30 minutes.

*Source: Glassdoor, Taro, JobMentis, Interview Coder, norahq, techinterview.org — aggregated from real candidate reports 2024-2026*

---

## TYPE 1 — LLM Knowledge Q&A

### Transformer Architecture

- Explain self-attention. What are Q, K, V — what does each one represent?
- Why do we scale dot products by `sqrt(d_k)`? What breaks without it?
- What is causal masking? Why do we use `-inf` rather than `0`?
- What is the difference between Pre-LN and Post-LN? Which does Mistral use and why?
- What is Multi-Head Attention? Why multiple heads instead of one big one?
- What is Grouped Query Attention (GQA)? How does Mistral 7B use it and what does it save?
- What is Sliding Window Attention? What complexity does it reduce and how?
- What is RoPE? How is it different from sinusoidal positional encoding?
- What is SwiGLU? Why do Mistral models use it over ReLU?
- What is Flash Attention? Why is attention memory-bound, not compute-bound?

### KV Cache

- What is the KV cache? Why does it exist?
- We cache K and V but not Q. Why not Q?
- How much memory does the KV cache use for Mistral 7B at seq_len=32k? (walk through the math)
- What is PagedAttention? What problem does it solve over naive KV cache allocation?
- What is the difference between the prefill phase and the decode phase? Why do they have different compute bottlenecks?
- What is prefix caching and when does it help?
- At what point does KV cache become the bottleneck compared to model weights?

### RAG

- Walk me through a full RAG pipeline from raw documents to final answer.
- What happens when your chunk size is too large? Too small?
- Dense retrieval vs BM25 — what does each capture that the other misses?
- What is hybrid search? How do you combine dense and sparse scores?
- What is a reranker? When is the added latency worth it?
- RAG vs fine-tuning — a customer wants their model to "know" their product catalog. Which do you choose?
- Name 3 RAG failure modes and how you'd fix each.
- How many chunks can you fit in Mistral 7B's 32k context at 250 tokens per chunk?

### MoE / Mixtral-Specific

- How does Mixtral 8x7B work? What is a "sparse" mixture of experts?
- Why does Mixtral use exactly 2 of 8 experts per token — not 1, not 4?
- What is the router in MoE? What does it output?
- What is expert collapse? Why does it happen and how do you prevent it?
- What is the load balancing auxiliary loss? What does a loss value of 1.0 mean?
- Mixtral has 47B total parameters but only ~13B active per token. Explain that.
- How does expert parallelism work when serving Mixtral across multiple GPUs?

### Fine-Tuning & Alignment

- Explain LoRA. What is the rank decomposition `W = W₀ + BA`?
- Why does B initialize to zero in LoRA? What would happen if both A and B were random?
- For Mistral 7B with rank=8, how many trainable parameters does LoRA add? What % of total?
- What is QLoRA? How much memory does it require vs full fine-tuning?
- Which modules do you inject LoRA into? Why Q and V (minimum) vs all 4 projections?
- What is RLHF? What are its three stages?
- What is DPO? Why is it simpler than RLHF?
- What is catastrophic forgetting? How do you mitigate it?
- When would you choose fine-tuning over RAG? Give a concrete example.

### Inference Optimization

- What is quantization? Walk through absmax INT8.
- INT8 vs INT4 — what's the memory saving and quality tradeoff?
- Name 3 quantization methods (GPTQ, AWQ, bitsandbytes) and what differentiates them.
- What is static batching? What is continuous batching? Why is continuous batching 3-4x better?
- What is speculative decoding? When does it fail to help?
- What is tensor parallelism vs pipeline parallelism? When would you choose each?
- How much GPU memory does Mistral 7B need in BF16? INT4?

### Evaluation

- What is perplexity? Write the formula. What are its limitations?
- Can you compare perplexity across models with different tokenizers? Why not?
- What is LLM-as-judge evaluation? What biases does it introduce?
- What benchmarks would you use to evaluate a code model like Codestral?
- What is MMLU? What is HumanEval? What does each measure?

---

## TYPE 2 — Python Live Coding

> Write these from scratch. No importing sampling functions. Type hints required. Test cases required.

### Must Know Cold

- Implement numerically stable softmax
- Implement `scaled_dot_product_attention(Q, K, V, mask=None)`
- Implement `make_causal_mask(seq_len)` — returns upper-triangular boolean mask
- Implement `top_k_sample(logits, k, temperature=1.0)` — no library for core logic
- Implement `top_p_sample(logits, p, temperature=1.0)` — no library for core logic
- Implement `apply_temperature(logits, temperature)`
- Implement `apply_repetition_penalty(logits, generated_tokens, penalty)`
- Implement `layer_norm(x, gamma, beta, eps=1e-6)`

### Also Asked

- Implement a BPE tokenizer in ~50 lines (`get_pairs`, `merge_pair`, `train_bpe`)
- Implement multi-head attention with `split_heads` and `merge_heads`
- Implement a sliding window attention mask
- Implement `cosine_similarity_batch(query, matrix)` — vectorized, no loops
- Parallelize an embedding lookup across 4 workers using `concurrent.futures`
- Implement `absmax_quantize_int8(weights)` and dequantize

### Mistral API Coding (Live in Colab)

- Build a minimal RAG agent using the Mistral API — chunking, embedding, retrieval, generation
- Audit a pull request that uses the Mistral API — find bugs in async code and naming
- Use the Mistral API + a third-party API together in a tool-calling agent
- Implement streaming chat with token-by-token output
- Implement multi-turn conversation with history tracking
- Implement retry logic with exponential backoff

---

## TYPE 3 — System Design

> Structure every answer: clarify → components → deep dive on bottleneck → tradeoffs → failure modes

- Design an inference serving system for Mixtral 8x7B (47B params) with p95 latency < 500ms
- Design a training cluster for an 8B dense model with 2,000 GPUs
- Design La Plateforme's rate-limiting and metering system with fair multi-tenant allocation
- Design a RAG system for 1M enterprise documents with sub-500ms p95 latency
- Design a fine-tuning pipeline for customer-specific models with data privacy guarantees
- A customer is running Mistral 7B on a single 24GB GPU — they want to double their concurrent users without adding hardware. What do you do?
- Design the prefix caching layer for a high-volume chatbot where 80% of requests share the same system prompt

---

## TYPE 4 — Debugging & Opinions

> Think out loud. They want to see your reasoning, not just the answer.

- Your training loss spikes at step 40k after being stable for 30k steps. Walk me through your diagnosis.
- What would your eval suite look like for a Codestral-class code generation model?
- A customer's chatbot "forgets" earlier messages in long conversations. What's the bug?
- Your model keeps repeating the same phrase over and over. What sampling parameters do you change?
- You deployed a model and p99 latency is 45s but p50 is 2s. What's causing the long tail?
- A candidate shows you this attention code: `scores = Q @ K.T`. What's the bug?
- Your RAG system retrieves the right chunks but the model still halluccinates. What do you investigate?
- After fine-tuning, the model performs well on the target task but worse on general Q&A. What happened and how do you fix it?

---

## TYPE 5 — Mistral Company & Strategy

> They specifically don't want generic AI enthusiasm. Ground every answer in Mistral's actual position.

- Why Mistral specifically — not OpenAI, not Anthropic?
- What is La Plateforme and how does it compare to OpenAI's API?
- Why does Mistral release open-weight models? What is the strategic rationale?
- What is the EU AI Act? Which risk tier do most Mistral models fall into?
- How does Mistral's European AI sovereignty angle differentiate it from US labs?
- What is Codestral and who is it for?
- What is the difference between Mistral Small, Medium, and Large — when would you recommend each to a customer?

---

## Quick-Fire Numbers (Say Without Thinking)

| Question | Answer |
|---|---|
| Mistral 7B in BF16 | 14 GB |
| Mistral 7B in INT4 | 3.5 GB |
| Mixtral 8x7B total params | 47B |
| Mixtral 8x7B active params per token | ~13B |
| Mistral 7B num layers | 32 |
| Mistral 7B Q heads / KV heads | 32 Q / 8 KV |
| Mistral 7B head dim | 128 |
| Mistral 7B context length | 32k |
| Sliding window size | 4096 |
| Attention complexity (naive) | O(n²) |
| Chinchilla: tokens per param | ~20 |
| LoRA r=8 on one 4096×4096 proj | 65,536 params (0.4%) |
| KV cache at seq=32k, batch=1 | ~4 GB |
