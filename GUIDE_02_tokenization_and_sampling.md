# Guide: Notebook 2 — Tokenization & Sampling

> **Mistral specifically tests top-k and top-p sampling in their live coding round.** These are the two functions most likely to be asked. Know them cold.

---

## What This Notebook Covers

Tokenization is how text becomes numbers the model can process. Sampling is how the model's output probabilities become actual text. Both are tested.

---

## Tokenization — Concepts to Know

### What is a Token?
- A token is the basic unit of text a model processes
- Roughly 3/4 of a word on average in English (varies a lot by language)
- "tokenization" might be ["token", "ization"] = 2 tokens
- Code and numbers are especially "expensive" — `12345` might be 3 tokens

### Byte-Pair Encoding (BPE) Algorithm
This is the algorithm behind most modern tokenizers (GPT, Mistral, Llama, etc.).

**Steps:**
1. Start with each character as a separate token
2. Count every consecutive pair of tokens in the corpus
3. Find the most frequent pair
4. Merge that pair into a new single token
5. Repeat N times (N = how many merges = vocabulary size roughly)

**Example:**
- Start: `["l", "o", "w", "</w>"]`
- After merging `("l","o")`: `["lo", "w", "</w>"]`
- After merging `("lo","w")`: `["low", "</w>"]`

**Interview question:** "Why doesn't BPE split on spaces?"
Answer: It adds a special marker (`</w>` or `Ġ`) to indicate word boundaries. Words are pre-tokenized on whitespace first.

### Vocabulary Size Tradeoffs
- **Larger vocab (128k):** fewer tokens per sentence → faster inference, but larger embedding table
- **Smaller vocab (32k):** more tokens per sentence → slower inference, but smaller embedding table
- Non-English languages get more tokens per word with English-trained tokenizers → "fertility" problem

---

## Sampling — Concepts to Know

### The Flow
```
Model output → logits (raw scores) → temperature scaling → probability distribution → sample
```

### Temperature
- Divides logits BEFORE softmax: `softmax(logits / T)`
- T < 1: peaks the distribution (model more confident, less creative)
- T = 1: original distribution
- T > 1: flattens the distribution (more random)
- T → 0: greedy (argmax)
- T → ∞: uniform random

### Greedy
Just take `argmax(probs)`. Fast, deterministic, often repetitive.

### Top-k
- Keep only the k highest-probability tokens
- Set everything else to -inf before softmax
- Sample from just those k tokens
- Problem: k=50 might include garbage tokens when distribution is peaked, or miss good tokens when it's flat

### Top-p (Nucleus) — MOST IMPORTANT
- Sort tokens by probability, highest first
- Keep adding tokens until cumulative probability reaches p
- Sample from that "nucleus"
- Adaptive: peaked distribution → small nucleus; flat distribution → large nucleus

### Why top-p beats top-k
- k is fixed regardless of confidence
- With top-k=50: if model is very confident, you're still sampling from 50 tokens (too many)
- With top-p=0.9: if model assigns 95% probability to one token, nucleus = 1 token (correctly tight)

---

## Code You Must Write From Memory

### Temperature
```python
def apply_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError('temperature must be > 0')
    return logits / temperature
```

### Top-k Sampling
```python
def top_k_sample(logits, k, temperature=1.0):
    logits = logits / temperature
    
    # Get top-k indices
    top_k_idx = np.argpartition(logits, -k)[-k:]
    
    # Zero out everything else
    filtered = np.full_like(logits, -np.inf)
    filtered[top_k_idx] = logits[top_k_idx]
    
    # Softmax and sample
    probs = softmax(filtered)
    return int(np.random.choice(len(probs), p=probs))
```

**Key:** `np.argpartition(arr, -k)[-k:]` gets top-k indices WITHOUT full sort. Faster than `argsort`.

### Top-p Sampling
```python
def top_p_sample(logits, p, temperature=1.0):
    logits = logits / temperature
    probs = softmax(logits)
    
    # Sort descending
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    
    # Find cutoff
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, p, side='left') + 1
    cutoff = max(1, min(cutoff, len(probs)))
    
    # Nucleus
    nucleus_idx = sorted_idx[:cutoff]
    nucleus_probs = probs[nucleus_idx]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    
    return int(np.random.choice(nucleus_idx, p=nucleus_probs))
```

**Key things:**
- Sort DESCENDING (`[::-1]`)
- `searchsorted` + 1 (include the token that pushed over threshold)
- Always at least 1 token (`max(1, ...)`)
- Renormalize before sampling

### BPE Core (simplified version to know)
```python
def get_pairs(word_freqs):
    pairs = Counter()
    for word_tuple, freq in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            pairs[(word_tuple[i], word_tuple[i+1])] += freq
    return pairs

def merge_pair(pair, word_freqs):
    merged = pair[0] + pair[1]
    new_freqs = {}
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word)-1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_freqs[tuple(new_word)] = freq
    return new_freqs
```

---

## What the Interviewer Will Ask Verbally

1. "Compare top-k and top-p sampling. When would you use each?"
2. "What happens to generation quality at very low temperature?"
3. "Why does BPE work? What's the alternative and why is BPE better?"
4. "If a model has a vocabulary of 128k, what does that mean for memory?"

---

## Key Interview Answers

**Q: What temperature should you use for code generation?**
A: Low temperature (0.2-0.4). Code has strict syntax — you don't want randomness. High temperature for creative writing.

**Q: What does top-p=1.0 mean?**
A: Include ALL tokens in the nucleus (no filtering). Same as sampling from the raw distribution.

**Q: What does top-p=0.0 mean?**
A: Invalid. p must be > 0. Use greedy (argmax) if you want deterministic output.

**Q: Token fertility — what is it?**
A: Number of tokens per word. English: ~1.3 tokens/word. Japanese/Chinese: much higher (worse tokenizer coverage). Affects model efficiency and cost for non-English languages.
