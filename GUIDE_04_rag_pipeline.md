# Guide: Notebook 4 — RAG Pipeline

> **RAG (Retrieval-Augmented Generation) is one of the most common LLM system design topics.** Know the full pipeline, the failure modes, and when to use RAG vs alternatives.

---

## The Big Picture — Why RAG?

LLMs have a knowledge cutoff and can't be updated cheaply. But your users need answers about:
- Your company's internal docs
- Recent events
- Customer-specific data

**Three options:**
1. **Fine-tuning:** Train the model on your data. Expensive, needs labeled data, doesn't update easily.
2. **Long context:** Stuff everything in the prompt. Expensive per call, "needle in haystack" problem.
3. **RAG:** Store docs in a database, retrieve relevant ones at query time, inject into prompt. **Usually the right choice.**

**Rule of thumb Mistral will love:** "RAG is for knowledge injection, fine-tuning is for behavior change."

---

## The RAG Pipeline — Step by Step

```
Documents → Chunk → Embed → Store → [User Query] → Embed → Retrieve → Prompt → Generate → Answer
```

### Step 1: Chunking
Split documents into pieces small enough to be meaningful but large enough to contain a full thought.

**Fixed-size chunking:**
```python
def chunk_fixed(text, chunk_size=512, overlap=64):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap  # overlap for context continuity
    return chunks
```

**Why overlap?** A sentence might be split across chunk boundaries. Overlap ensures it appears complete in at least one chunk.

**Sentence chunking:** More semantically coherent. Use `re.split(r'(?<=[.!?])\s+', text)` to split on sentence boundaries.

**Good chunk size:** 200-500 tokens. Too small = lacks context. Too large = retrieval is too broad.

### Step 2: Embed
Convert each chunk into a dense vector (embedding). Similar meaning = similar vector direction.

```python
# Mistral embed API
response = client.embeddings.create(model='mistral-embed', inputs=[chunk_text])
embedding = response.data[0].embedding  # list of 1024 floats
```

### Step 3: Store in Vector Database
Store `(id, text, metadata, embedding)` tuples. For production: FAISS, Pinecone, Weaviate, Chroma.

### Step 4: Retrieve
At query time:
1. Embed the user's question
2. Find chunks with most similar embeddings (cosine similarity)
3. Return top-k chunks

### Step 5: Generate
Build a prompt with retrieved context + the question. Call the LLM.

---

## Code You Must Write From Memory

### Cosine Similarity
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
```

### Batch Cosine Similarity (for fast retrieval)
```python
def cosine_similarity_batch(query, matrix):
    # query: (d,), matrix: (N, d)
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norm @ query_norm  # (N,)
```
**Key:** Normalize both, then matrix multiply. This is how FAISS-style search works.

### Simple Vector Store
```python
class VectorStore:
    def __init__(self):
        self.docs = []
        self._matrix = None
    
    def add(self, doc):  # doc has .embedding and .text
        self.docs.append(doc)
        self._matrix = None  # invalidate cache
    
    def search(self, query_embedding, top_k=5):
        if self._matrix is None:
            self._matrix = np.stack([d.embedding for d in self.docs])
        scores = cosine_similarity_batch(query_embedding, self._matrix)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], float(scores[i])) for i in top_idx]
```

### RAG Prompt Template
```python
def build_rag_prompt(question, contexts):
    context_str = '\n'.join(f'[{i+1}] {c}' for i, c in enumerate(contexts))
    return f"""Use the following context to answer the question. 
If the answer isn't in the context, say "I don't know."

Context:
{context_str}

Question: {question}

Answer:"""
```

---

## BM25 — Sparse Retrieval

Dense embeddings capture meaning. BM25 captures exact keyword matches. Neither is perfect alone.

**BM25 score formula:**
```
score(D, Q) = Σ IDF(t) × TF(t,D) × (k1+1) / (TF(t,D) + k1 × (1-b + b×|D|/avgdl))
```

**Key intuition:**
- IDF (Inverse Document Frequency): rare words matter more than common words ("the" vs "perplexity")
- TF (Term Frequency): the more times the query term appears in the doc, the better — but with diminishing returns
- b=0.75: normalizes for document length (long docs unfairly get more term matches)

---

## Hybrid Search

Combine dense (meaning) + sparse (keywords):

```python
# Normalize scores to [0,1], then weighted sum
combined = 0.7 * normalize(dense_scores) + 0.3 * normalize(bm25_scores)
```

**When sparse retrieval wins:**
- Exact entity names: "GPT-4o" shouldn't be retrieved by searching "GPT"
- Serial numbers, codes, IDs
- Rare technical terms

**When dense retrieval wins:**
- Semantic meaning: "how do I fix this" = "troubleshooting guide"
- Paraphrases and synonyms

---

## RAG Failure Modes (Know All 6)

| Failure | Symptom | Fix |
|---|---|---|
| **Wrong chunks retrieved** | Answer ignores context, completely off | Better chunking, hybrid search |
| **Context too long** | Model ignores context beyond first few chunks | Fewer chunks, reranking, summarization |
| **Chunk boundary** | Answer is split across two chunks | Overlap, sentence-aware chunking |
| **Semantic gap** | Query uses different words than documents | Dense retrieval with good embeddings |
| **Stale knowledge** | Old document retrieved | Add timestamps, freshness reranking |
| **Hallucination on context** | Model makes up facts even with good retrieval | Stricter prompt: "only use the context" |

---

## System Design: RAG at Scale

For the Mistral system design round on RAG, structure your answer like this:

**Q: Design a RAG system for 1M enterprise documents**

1. **Ingestion pipeline:**
   - Parse documents (PDF, Word, web) → text
   - Chunk (sentence-aware, 300-500 tokens)
   - Embed in batches (async calls to mistral-embed)
   - Store in vector DB (Pinecone/Weaviate for managed, FAISS for self-hosted)

2. **Query pipeline:**
   - Embed user query
   - Retrieve top-20 chunks (dense + BM25 hybrid)
   - Rerank to top-5 with cross-encoder
   - Build prompt (max 3000 tokens context)
   - Call mistral-large for final answer

3. **Infrastructure:**
   - Cache: cache embeddings (never re-embed same text), cache responses for identical questions
   - Monitoring: retrieval precision@k, answer accuracy, latency
   - Freshness: re-embed documents on update, timestamp metadata

4. **Tradeoffs to mention:**
   - More chunks retrieved = better recall but worse precision and higher latency
   - Larger chunks = fewer retrieval calls but harder for model to extract relevant part
   - Reranking adds 100-200ms but significantly improves quality

---

## Numbers to Know

| Thing | Value |
|---|---|
| mistral-embed output dimension | 1024 |
| Typical chunk size | 200-500 tokens |
| Typical top-k for retrieval | 5-20 |
| Max chunks to fit in 7B context | ~128 (at 250 tokens each in 32k context) |
| FAISS search time for 1M vectors | < 10ms on CPU |
