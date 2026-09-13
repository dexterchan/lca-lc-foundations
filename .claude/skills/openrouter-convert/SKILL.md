---
name: openrouter-convert
description: Converts Python code that calls the OpenAI SDK, Anthropic (Claude) SDK, or Google Gemini SDK so it instead sends requests to OpenRouter, using the OPENROUTER_KEY env var for auth. Use this whenever the user asks to "switch to OpenRouter", "use OpenRouter as the endpoint", "route my OpenAI/Claude/Gemini calls through OpenRouter", or wants one unified endpoint for multiple LLM providers. Scans the whole project for SDK client setup code (not just one file the user names) and rewrites it.
---

# Convert LLM SDK calls to OpenRouter

## Why this works this way

OpenRouter only speaks one API shape: the OpenAI "chat completions" format. It does not have
separate native endpoints for Anthropic's or Google's SDKs. So the correct fix is **not** "point
the Claude SDK at a different URL" — it's "replace the Claude/Gemini SDK calls with the `openai`
Python client, pointed at OpenRouter, with an OpenRouter-style model name."

For code already using the `openai` package, the fix is small: change `base_url` and `api_key`.
For code using `anthropic` or `google-genai`, the fix is bigger: swap the client and call shape
to the OpenAI one, then translate the model name.

## Step 1: Find what needs converting

Search the project (not just one file) for SDK usage:

```bash
grep -rlE "import openai|from openai|OpenAI\(|import anthropic|from anthropic|Anthropic\(|google\.generativeai|google\.genai|GenerativeModel\(" --include="*.py" .
```

Also check for API keys already loaded (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` /
`GEMINI_API_KEY`) so you know which client constructions to target.

List the matching files and confirm scope with the user only if the list looks larger than
expected (e.g. touches vendored/third-party code) — otherwise proceed.

## Step 2: The target shape

Every converted call site should end up using the `openai` package like this:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

response = client.chat.completions.create(
    model="openai/gpt-4o",  # see model name mapping below
    messages=[{"role": "user", "content": "..."}],
)
print(response.choices[0].message.content)
```

Async code uses `AsyncOpenAI` the same way.

Never hardcode the key — always read `os.environ["OPENROUTER_KEY"]` (or `os.getenv(...)`, matching
whatever style the surrounding file already uses for other env vars). If the file loads secrets
via `python-dotenv`, keep that pattern and just change the key name.

## Step 3: Convert each SDK

Read the relevant reference file for the SDK you're converting — each has the exact before/after
patterns and the streaming/tool-call/vision variants:

- `references/openai_sdk.md` — smallest change, just swap `base_url`/`api_key`
- `references/anthropic_sdk.md` — rewrite `client.messages.create(...)` call shape to chat completions
- `references/gemini_sdk.md` — rewrite `GenerativeModel`/`generate_content` call shape to chat completions

Each reference also lists the OpenRouter model-name prefix for that provider (e.g. Anthropic
models become `anthropic/claude-...`, Gemini models become `google/gemini-...`, OpenAI models
become `openai/gpt-...`). Keep the user's original model choice, just re-prefixed — don't silently
swap them to a different model tier.

## Step 4: Environment and dependencies

- Make sure `OPENROUTER_KEY` is referenced in `.env.example` (or wherever the project documents
  required env vars) if one exists — this repo already has it there.
- Add `openai` to the project's dependency file (`requirements.txt`, `pyproject.toml`, etc.) if a
  converted file needs it and it isn't already a dependency.
- Don't remove the original `anthropic` / `google-genai` package from dependencies unless the user
  confirms nothing else in the project still needs it — leave that decision to them if unsure.

## Step 5: Verify

After converting, run a quick syntax/import check on touched files (e.g. `python -m py_compile
<file>` or run the project's existing test suite if one covers these call sites). Report which
files changed and which SDK each one came from.
