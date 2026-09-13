# Converting Google Gemini SDK code

Like Anthropic, Gemini's SDK has its own call shape — swap it for the `openai` client pointed at
OpenRouter, don't just change the base URL.

## Before (`google.generativeai`, the older SDK)

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro")
response = model.generate_content("hello")
print(response.text)
```

## Before (`google-genai`, the newer SDK)

```python
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="hello",
)
print(response.text)
```

## After (either case)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

response = client.chat.completions.create(
    model="google/gemini-2.0-flash",
    messages=[{"role": "user", "content": "hello"}],
)
print(response.choices[0].message.content)
```

## What changes

- A bare string prompt (`generate_content("hello")`) becomes a `messages=[{"role": "user",
  "content": "hello"}]` list.
- Multi-turn/chat sessions (`model.start_chat(history=...)`) become a growing `messages` list you
  pass in full on each call, same as any OpenAI-style chat loop.
- System instructions (`GenerativeModel(..., system_instruction=...)`) become a
  `{"role": "system", ...}` entry at the front of `messages`.
- Reading the reply moves from `response.text` to `response.choices[0].message.content`.
- Streaming: `generate_content(..., stream=True)` becomes `chat.completions.create(stream=True,
  ...)`, chunks read via `chunk.choices[0].delta.content`.

## Model name mapping

Gemini model names get a `google/` prefix on OpenRouter: `gemini-1.5-pro` →
`google/gemini-1.5-pro`, `gemini-2.0-flash` → `google/gemini-2.0-flash`, `gemini-2.5-pro` →
`google/gemini-2.5-pro`. Keep the user's original model choice — just re-prefix it. If unsure of
the exact current slug, use `google/<original-model-name>` mechanically rather than guessing.
