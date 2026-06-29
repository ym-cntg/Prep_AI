# Guide: Notebook 3 — Mistral API Basics

> **You need to be able to use the Mistral API fluently during a live interview with Google Colab open.** This guide teaches you every pattern you'll need.

---

## Setup — Do This Before the Interview

```bash
pip install mistralai pydantic
```

Get an API key: https://console.mistral.ai/ (free tier available)

Set it as an environment variable:
```bash
export MISTRAL_API_KEY="your-key-here"
```

---

## Pattern 1: Basic Chat Completion

This is the most fundamental thing. Know it cold.

```python
from mistralai import Mistral

client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))

response = client.chat.complete(
    model='mistral-small-latest',
    messages=[
        {'role': 'user', 'content': 'Explain attention in one sentence.'}
    ]
)

print(response.choices[0].message.content)
```

**What each part does:**
- `Mistral(api_key=...)` — creates the client. One per program.
- `client.chat.complete(...)` — synchronous call
- `model=` — which model to use (see model list below)
- `messages=` — list of dicts with `role` and `content`
- `response.choices[0].message.content` — the text response

---

## Pattern 2: Multi-Turn Conversation

Keep a list of messages and append to it:

```python
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'}
]

def chat(user_msg):
    messages.append({'role': 'user', 'content': user_msg})
    response = client.chat.complete(model='mistral-small-latest', messages=messages)
    assistant_msg = response.choices[0].message.content
    messages.append({'role': 'assistant', 'content': assistant_msg})
    return assistant_msg

print(chat('What is KV caching?'))
print(chat('Why does it save compute?'))  # model remembers context
```

**Key:** Always pass the FULL message history. The API is stateless — it doesn't remember previous calls.

---

## Pattern 3: Streaming

```python
with client.chat.stream(
    model='mistral-small-latest',
    messages=[{'role': 'user', 'content': 'Count to 5.'}],
) as stream:
    for event in stream:
        chunk = event.data.choices[0].delta.content
        if chunk:
            print(chunk, end='', flush=True)
print()  # newline
```

**Why streaming matters:** Users see text immediately instead of waiting for the full response. Always use streaming in production chat interfaces.

---

## Pattern 4: Tool / Function Calling

```python
tools = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get weather for a city',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {'type': 'string', 'description': 'City name'}
            },
            'required': ['city']
        }
    }
}]

# Step 1: Ask model (may return tool call)
response = client.chat.complete(
    model='mistral-small-latest',
    messages=[{'role': 'user', 'content': 'Weather in Paris?'}],
    tools=tools,
    tool_choice='auto',
)

msg = response.choices[0].message

# Step 2: Check if model wants to call a tool
if msg.tool_calls:
    for tool_call in msg.tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        result = my_functions[fn_name](**fn_args)  # actually call it
        
        messages.append({'role': 'tool', 'name': fn_name,
                        'content': result, 'tool_call_id': tool_call.id})

# Step 3: Get final response
final = client.chat.complete(model='mistral-small-latest', messages=messages)
```

**The pattern:** 2 API calls. First call may produce tool calls. Execute them. Second call produces the final answer.

---

## Pattern 5: Embeddings

```python
response = client.embeddings.create(
    model='mistral-embed',
    inputs=['First text', 'Second text'],
)

# response.data is a list of embedding objects
embeddings = [item.embedding for item in response.data]
# Each embedding is a list of floats (1024 dimensions for mistral-embed)
```

---

## Pattern 6: JSON Mode (Structured Output)

```python
response = client.chat.complete(
    model='mistral-small-latest',
    messages=[{'role': 'user', 'content': 'Return JSON with keys name, age'}],
    response_format={'type': 'json_object'},  # THIS FORCES JSON
)

import json
data = json.loads(response.choices[0].message.content)
```

**With Pydantic parsing:**
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

response = client.chat.complete(
    model='mistral-small-latest',
    messages=[{'role': 'user', 'content': f'Return JSON for a person named John who is 30. Schema: {Person.model_json_schema()}'}],
    response_format={'type': 'json_object'},
)

person = Person.model_validate_json(response.choices[0].message.content)
print(person.name, person.age)
```

---

## Pattern 7: Async Calls

```python
from mistralai import AsyncMistral
import asyncio

async_client = AsyncMistral(api_key=os.getenv('MISTRAL_API_KEY'))

async def async_chat(prompt):
    response = await async_client.chat.complete_async(
        model='mistral-small-latest',
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.choices[0].message.content

async def parallel():
    results = await asyncio.gather(
        async_chat('Question 1'),
        async_chat('Question 2'),
        async_chat('Question 3'),
    )
    return results

results = asyncio.run(parallel())
```

**Why async?** Send 10 API requests simultaneously instead of sequentially → 10x faster when doing batch processing.

---

## Pattern 8: Retry Logic

```python
import time, random

def chat_with_retry(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.chat.complete(
                model='mistral-small-latest',
                messages=[{'role': 'user', 'content': prompt}],
            ).choices[0].message.content
        except SDKError as e:
            if e.status_code == 429 and attempt < max_retries - 1:
                # Rate limit: exponential backoff + jitter
                delay = (2 ** attempt) + random.random()
                print(f'Rate limited. Waiting {delay:.1f}s...')
                time.sleep(delay)
            else:
                raise
```

**Exponential backoff:** `1s → 2s → 4s → 8s` + random jitter to prevent thundering herd.

---

## Model Reference Card

| Model ID | Use Case | Speed | Cost |
|---|---|---|---|
| `mistral-small-latest` | Quick tasks, testing | Fast | Cheap |
| `mistral-medium-latest` | General use | Medium | Medium |
| `mistral-large-latest` | Complex reasoning | Slow | Expensive |
| `open-mistral-7b` | Open source, self-host | Fastest | Free (self-host) |
| `open-mixtral-8x7b` | Open source, good quality | Fast | Free (self-host) |
| `codestral-latest` | Code generation | Fast | Medium |
| `mistral-embed` | Embeddings | Fast | Very cheap |

---

## API Parameters to Know

```python
client.chat.complete(
    model='mistral-small-latest',
    messages=[...],
    temperature=0.7,        # 0.0 to 1.0 (default 0.7)
    max_tokens=1000,        # max output tokens
    top_p=0.9,              # nucleus sampling
    stream=False,           # use client.chat.stream() instead for streaming
    tools=[...],            # function definitions
    tool_choice='auto',     # 'auto', 'any', 'none', or specific function
    response_format={'type': 'json_object'},  # force JSON output
)
```

---

## What the Interviewer Will Ask Verbally

1. "You need to process 10,000 documents overnight. How do you structure the API calls?"
   Answer: Async + concurrency, with rate limit handling, batched where possible

2. "A customer reports their chatbot 'forgets' earlier messages. What's the bug?"
   Answer: Not passing the full message history to each API call

3. "How would you reduce cost on a chatbot that's spending $5k/month?"
   Answer: Use smaller model for simple queries, cache common responses, use max_tokens limit, stream to avoid timeout retries

4. "What's the difference between tool_choice='auto' and 'any'?"
   Answer: 'auto' lets model decide whether to call a tool. 'any' forces it to call SOME tool.
