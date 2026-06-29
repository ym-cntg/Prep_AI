# Mistral AI — Technical Deep Dive Prep Checklist

**Role:** Technical Deep Dive (60 min, live Q&A + Python coding + AI knowledge + system design)
**Source:** Lauren's email + Glassdoor + interview research

---

## How the 60-Min Round Is Split (estimated)

| Segment | Time | Weight |
|---|---|---|
| LLM / AI Knowledge Q&A | ~20 min | High |
| Python Live Coding | ~25 min | High |
| System Design | ~15 min | Medium |

---

## SECTION 1 — Transformer Architecture (Must Know Cold)

### Attention Mechanism
- [ ] Explain self-attention: Q, K, V matrices — what they represent
- [ ] Scaled dot-product attention formula: `softmax(QK^T / sqrt(d_k)) * V`
- [ ] Why do we scale by `sqrt(d_k)`? What happens without it?
- [ ] Multi-Head Attention — why multiple heads? How are outputs combined?
- [ ] Cross-attention vs self-attention (encoder-decoder models)
- [ ] Causal / masked attention — why GPT-style models use it
- [ ] Flash Attention — what problem does it solve? (memory IO vs compute)

### Positional Encoding
- [ ] Sinusoidal positional encoding (original Transformer)
- [ ] Learned positional embeddings vs sinusoidal
- [ ] RoPE (Rotary Position Embedding) — Mistral uses this; understand the core idea
- [ ] ALiBi — linear bias on attention scores, how it enables length generalization

### Transformer Block
- [ ] LayerNorm: Pre-LN vs Post-LN — why Pre-LN is more stable at scale
- [ ] FFN: shape of the MLP layers, SwiGLU activation (used in Mistral models)
- [ ] Residual connections — why they matter for gradient flow
- [ ] Embedding and unembedding (lm_head) weight tying

### Mistral Architecture Specifics
- [ ] Sliding Window Attention (SWA) — what it is, memory complexity reduction
- [ ] Grouped Query Attention (GQA) — fewer KV heads than Q heads, why?
- [ ] Mistral 7B architectural choices vs Llama 2 7B
- [ ] Mixtral 8x7B — Sparse Mixture of Experts (MoE)
  - [ ] How many experts are active per token? (2 of 8)
  - [ ] Router mechanism (top-2 gating)
  - [ ] Load balancing loss
  - [ ] Why MoE: same compute cost, more parameters

---

## SECTION 2 — Training & Pre-Training

- [ ] Pre-training objective: next-token prediction, cross-entropy loss
- [ ] BFloat16 vs Float32 vs Float16 — when each is used
- [ ] Gradient clipping, warmup schedules, cosine LR decay
- [ ] Gradient checkpointing — what it trades (compute for memory)
- [ ] Data parallelism vs tensor parallelism vs pipeline parallelism
- [ ] Mixed precision training — FP16 forward, FP32 optimizer states
- [ ] Loss spikes — common causes, how to diagnose (step 40k spike scenario)
- [ ] Chinchilla scaling laws — compute-optimal token-to-param ratio (~20 tokens/param)
- [ ] Emergent capabilities — what they are, whether they're real

---

## SECTION 3 — Tokenization

- [ ] What is a token? Why not characters or words?
- [ ] Byte-Pair Encoding (BPE) algorithm — merging most frequent pairs
- [ ] SentencePiece vs tiktoken vs HuggingFace tokenizers
- [ ] Vocabulary size tradeoffs (32k vs 64k vs 128k)
- [ ] Why tokenization affects arithmetic, code, non-Latin languages
- [ ] `<unk>`, `<bos>`, `<eos>` special tokens
- [ ] Tokenizer fertility — tokens per word across languages

---

## SECTION 4 — Inference & Sampling

- [ ] Greedy decoding vs beam search vs sampling
- [ ] Temperature — how it sharpens/flattens the distribution
- [ ] Top-k sampling — truncate to top k logits then sample
- [ ] Top-p (nucleus) sampling — accumulate until cumulative prob >= p
- [ ] Min-p sampling — relative threshold to max prob token
- [ ] Repetition penalty — how it works
- [ ] KV Cache — what is cached, memory cost, why it enables fast autoregressive generation
- [ ] Prefill vs decode phase — different compute profiles (memory bound vs compute bound)
- [ ] Batching for inference: static batching vs continuous batching (PagedAttention)
- [ ] Speculative decoding — draft model + verification, when it wins

---

## SECTION 5 — Fine-Tuning & Alignment

- [ ] Full fine-tuning vs PEFT
- [ ] LoRA (Low-Rank Adaptation) — rank decomposition, where matrices are injected
- [ ] QLoRA — quantized base model + LoRA adapters
- [ ] Instruction tuning — supervised fine-tuning (SFT) on (prompt, response) pairs
- [ ] RLHF pipeline: SFT → reward model → PPO
- [ ] DPO (Direct Preference Optimization) — simpler alternative to RLHF
- [ ] Catastrophic forgetting — mitigation strategies
- [ ] When NOT to fine-tune (RAG is usually better for knowledge injection)

---

## SECTION 6 — RAG (Retrieval-Augmented Generation)

- [ ] RAG architecture: retriever + reader
- [ ] Embedding models — how text becomes vectors (dense retrieval)
- [ ] Chunking strategies: fixed-size, sentence, semantic
- [ ] Vector databases: FAISS, Chroma, Pinecone, Weaviate — tradeoffs
- [ ] Similarity metrics: cosine similarity, dot product, L2
- [ ] Hybrid search: dense + BM25 sparse retrieval
- [ ] Reranking: cross-encoder reranker after retrieval
- [ ] Context window packing — how many chunks to retrieve
- [ ] RAG failure modes: retrieval misses, context too long, hallucination on retrieved content
- [ ] When to use RAG vs fine-tuning vs long context

---

## SECTION 7 — Evaluation

- [ ] Perplexity — what it measures, formula, limitations
- [ ] BLEU, ROUGE — when used, what they miss
- [ ] LLM-as-judge — using a stronger model to score outputs
- [ ] MT-Bench, MMLU, HumanEval, GSM8K — what each benchmarks
- [ ] Why benchmarks saturate and what comes next
- [ ] Evals for safety: red-teaming, adversarial prompts
- [ ] Vibe evals vs structured evals — know the distinction

---

## SECTION 8 — Optimization & Serving at Scale

- [ ] Quantization: INT8, INT4 (GPTQ, AWQ, bitsandbytes)
- [ ] Pruning: structured vs unstructured
- [ ] Knowledge distillation — teacher/student
- [ ] vLLM — PagedAttention, why it improves GPU memory utilization
- [ ] Tensor parallelism for inference (split model across GPUs)
- [ ] Memory math: 7B model in FP16 = ~14GB, INT4 = ~3.5GB
- [ ] Throughput vs latency tradeoffs in serving
- [ ] Batching: larger batches = better GPU utilization but worse p99 latency
- [ ] SLA targets: p50/p95/p99 latency, TTFT (time to first token), TPS

---

## SECTION 9 — Mistral Product Knowledge

- [ ] Mistral 7B — flagship small dense model
- [ ] Mixtral 8x7B / 8x22B — MoE variants
- [ ] Mistral Small, Medium, Large — product tiers
- [ ] Codestral — code-specialized model
- [ ] Mistral Embed — embedding model
- [ ] La Plateforme — Mistral's API platform (like OpenAI API)
- [ ] Open weights vs closed models — Mistral's dual strategy
- [ ] European AI sovereignty angle — why it matters to Mistral
- [ ] EU AI Act — high-risk AI systems, transparency requirements
- [ ] Function calling / Tool use in Mistral API
- [ ] JSON mode / structured output

---

## SECTION 10 — Python Coding (Live, 30 min)

### Must implement from scratch (no library for core logic)
- [ ] **Softmax** — numerically stable version
- [ ] **Scaled dot-product attention** — single head
- [ ] **Multi-head attention** — full implementation
- [ ] **Transformer block** — attention + FFN + LayerNorm
- [ ] **BPE tokenizer** — simplified 50-line version
- [ ] **Top-k sampling** — from logits tensor
- [ ] **Top-p (nucleus) sampling** — from logits tensor
- [ ] **Temperature scaling** — divide logits before softmax
- [ ] **Cosine similarity** — for RAG retrieval
- [ ] **Simple RAG pipeline** — embed, retrieve, generate

### API / Practical Coding
- [ ] Mistral API: chat completions with `mistralai` SDK
- [ ] Streaming responses
- [ ] Tool/Function calling
- [ ] Embeddings API
- [ ] Async calls with `asyncio`
- [ ] Retry logic with exponential backoff
- [ ] Pydantic models for structured output parsing

### Python Fluency
- [ ] Type hints everywhere (PEP 484 / PEP 526)
- [ ] Dataclasses and Pydantic BaseModel
- [ ] List/dict comprehensions
- [ ] Generators and `yield`
- [ ] Context managers (`__enter__`/`__exit__`)
- [ ] `asyncio` basics: `async def`, `await`, `asyncio.gather`
- [ ] NumPy vectorized operations (no explicit Python loops)

---

## SECTION 11 — System Design (15 min)

### Likely Prompts
- Design an LLM inference server for Mixtral 70B with p95 < 500ms
- Design a RAG system for enterprise document Q&A at 1M docs
- Design La Plateforme's rate-limiting and usage metering
- Design a fine-tuning pipeline for customer-specific models

### Framework to Answer
1. **Clarify** — scale, latency SLA, consistency needs
2. **High-level components** — model serving, retrieval, cache, queue
3. **Deep dive on bottleneck** — usually GPU memory or latency
4. **Tradeoffs** — throughput vs latency, cost vs quality
5. **Failure modes** — what breaks first under load

### Key Concepts
- [ ] Load balancing across GPU replicas
- [ ] Request batching / continuous batching
- [ ] Model sharding (tensor + pipeline parallelism)
- [ ] Caching: prompt cache, KV cache, response cache
- [ ] Queuing: priority queues, rate limiting per tenant
- [ ] Monitoring: TTFT, TPS, GPU utilization, error rate
- [ ] Cold start — model loading latency mitigation

---

## SECTION 12 — Coding Practice Notebooks (Do These In Order)

- [ ] `01_transformer_fundamentals.ipynb` — attention from scratch
- [ ] `02_tokenization_and_sampling.ipynb` — BPE, top-k/p
- [ ] `03_mistral_api_basics.ipynb` — SDK, chat, streaming, tools
- [ ] `04_rag_pipeline.ipynb` — embeddings, retrieval, generation
- [ ] `05_llm_optimization_concepts.ipynb` — quantization, KV cache, batching
- [ ] `06_advanced_moe_and_finetuning.ipynb` — MoE, LoRA concepts

---

## Interview Day Checklist

- [ ] Working camera + microphone
- [ ] Google Colab open and tested in browser
- [ ] Screen sharing tested
- [ ] `mistralai` Python package installed / available in Colab
- [ ] Have a Mistral API key ready (get from console.mistral.ai)
- [ ] Stable internet connection
- [ ] Quiet environment

---

## Quick-Reference: Numbers to Know

| Thing | Number |
|---|---|
| Mistral 7B params | 7.3B |
| Mixtral 8x7B active params per token | ~13B (2 of 8 experts) |
| 7B model FP16 memory | ~14 GB |
| 7B model INT4 memory | ~3.5 GB |
| Chinchilla optimal tokens per param | ~20 |
| Attention complexity (naive) | O(n²) |
| Sliding window size (Mistral 7B) | 4096 |
| Context length (Mistral 7B) | 32k (with SWA) |
| Typical vocab size | 32k–128k |
| LoRA rank r typical values | 4, 8, 16, 64 |
