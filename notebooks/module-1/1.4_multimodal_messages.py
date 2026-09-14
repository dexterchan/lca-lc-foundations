# %% [markdown]
# ## Text input

# %% [markdown]
# https://platform.openai.com/docs/models

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# This notebook is a multimodal demo (image + audio cells below need OpenAI
# anyway), so keep the text cell on OpenAI too for consistency, via OpenRouter.
text_model = init_chat_model(
    model="openai/gpt-5-nano",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

agent = create_agent(
    model=text_model,
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
)

# %%
from langchain.messages import HumanMessage

question = HumanMessage(content=[
    {"type": "text", "text": "What is the capital of The Moon?"}
])

response = agent.invoke(
    {"messages": [question]}
)

print(response['messages'][-1].content)

# %% [markdown]
# ## Image input

# %%
from ipywidgets import FileUpload
from IPython.display import display

uploader = FileUpload(accept='.png', multiple=False)
display(uploader)

# %%
print(uploader.value)

# %%
import base64

# Get the first (and only) uploaded file dict
uploaded_file = uploader.value[0]

# This is a memoryview
content_mv = uploaded_file["content"]

# Convert memoryview -> bytes
img_bytes = bytes(content_mv)  # or content_mv.tobytes()

# Now base64 encode
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# %%
# Image input needs a vision-capable model — DeepSeek doesn't do images,
# so this cell uses OpenAI via OpenRouter instead of the deepseek text_model above.
vision_model = init_chat_model(
    model="openai/gpt-5-nano",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

vision_agent = create_agent(
    model=vision_model,
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
)

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Tell me about this capital"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

response = vision_agent.invoke(
    {"messages": [multimodal_question]}
)

print(response['messages'][-1].content)

# %% [markdown]
# ## Audio input

# %%
import sounddevice as sd
from scipy.io.wavfile import write
import base64
import io
import time
from tqdm import tqdm

# Recording settings
duration = 5  # seconds
sample_rate = 44100

print("Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
# Progress bar for the duration
for _ in tqdm(range(duration * 10)):   # update 10× per second
    time.sleep(0.1)
sd.wait()
print("Done.")

# Write WAV to an in-memory buffer
buf = io.BytesIO()
write(buf, sample_rate, audio)
wav_bytes = buf.getvalue()

aud_b64 = base64.b64encode(wav_bytes).decode("utf-8")

# %%
# Audio input also needs an OpenAI audio-capable model — no DeepSeek equivalent.
audio_model = init_chat_model(
    model="openai/gpt-audio",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

audio_agent = create_agent(
    model=audio_model,
)

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Tell me about this audio file"},
    {"type": "audio", "base64": aud_b64, "mime_type": "audio/wav"}
])

response = audio_agent.invoke(
    {"messages": [multimodal_question]}
)

print(response['messages'][-1].content)
