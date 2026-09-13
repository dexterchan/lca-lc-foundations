# Converting Anthropic (Claude) SDK code

OpenRouter does not expose Anthropic's native `messages` API shape, so this isn't a base_url swap
— replace the `anthropic` client with the `openai` client and translate the call shape.

## Before

```python
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "hello"}],
)
print(message.content[0].text)
```

## After

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.5",
    max_tokens=1024,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ],
)
print(response.choices[0].message.content)
```

## What changes

- `system=` becomes a `{"role": "system", ...}` entry at the front of `messages`, instead of a
  separate top-level argument.
- Reading the reply moves from `message.content[0].text` to
  `response.choices[0].message.content`.
- Streaming: `client.messages.stream(...)` becomes `client.chat.completions.create(stream=True,
  ...)`, and chunks are read via `chunk.choices[0].delta.content` instead of
  `event.delta.text`.
- Tool use: Anthropic's `tools=[{"name": ..., "input_schema": ...}]` becomes OpenAI-style
  `tools=[{"type": "function", "function": {"name": ..., "parameters": ...}}]`. The tool call
  shows up in `response.choices[0].message.tool_calls` instead of a `tool_use` content block.

## Model name mapping

Anthropic model names get a `anthropic/` prefix on OpenRouter: `claude-sonnet-4-5` →
`anthropic/claude-sonnet-4.5`, `claude-opus-4-1` → `anthropic/claude-opus-4.1`,
`claude-haiku-4-5` → `anthropic/claude-haiku-4.5`. Keep the user's original model tier — don't
substitute a different one. If unsure of the exact current slug, use
`anthropic/<original-model-name>` mechanically rather than guessing.
