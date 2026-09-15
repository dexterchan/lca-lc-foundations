# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest


@dataclass
class LanguageContext:
    user_language: str = "English"


@dynamic_prompt
def user_language_prompt(request: ModelRequest) -> str:
    """Generate a system prompt based on the user's language."""
    user_language = request.runtime.context.user_language
    base_prompt = "You are a helpful assistant."

    if user_language != "English":
        return f"{base_prompt} only respond in {user_language}."
    return base_prompt


# %%
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model(
    model="deepseek/deepseek-v4-flash-0731",
    model_provider="deepseek",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

agent = create_agent(
    model=model,
    context_schema=LanguageContext,
    middleware=[user_language_prompt],
)

# %%
from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="Irish"),
)

print(response["messages"][-1].content)

# %%
response = agent.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="Spanish"),
)

print(response["messages"][-1].content)

# %%
response = agent.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="French"),
)

print(response["messages"][-1].content)
