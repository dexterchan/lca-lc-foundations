# Converting OpenAI SDK code

This is the easy case — OpenRouter *is* an OpenAI-compatible endpoint, so the call shape stays
identical. Only the client construction changes.

## Before

```python
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hello"}],
)
```

## After

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "hello"}],
)
```

## What changes

- `base_url="https://openrouter.ai/api/v1"` added to the client constructor.
- `api_key` now reads `OPENROUTER_KEY` instead of `OPENAI_API_KEY`.
- Model name gets an `openai/` prefix (OpenRouter routes by `provider/model`).
- Everything else — streaming (`stream=True`), tool calls, `response_format`, async client
  (`AsyncOpenAI`) — works unchanged, since the wire format is identical.

## Model name mapping

`gpt-4o` → `openai/gpt-4o`, `gpt-4o-mini` → `openai/gpt-4o-mini`, `gpt-4-turbo` →
`openai/gpt-4-turbo`, `o1` → `openai/o1`, etc. Keep whatever specific model the user already had;
just add the `openai/` prefix. If unsure of the exact current OpenRouter slug, keep the mapping
mechanical (`openai/<original-model-name>`) rather than guessing at a different model.
