# %% [markdown]
# ## Without web search

# %%
from dotenv import load_dotenv

load_dotenv()

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
    model=model
)

# %%
from langchain.messages import HumanMessage

question = HumanMessage(content="How up to date is your training knowledge?")

response = agent.invoke(
    {"messages": [question]}
)

# %%
print(response['messages'][-1].content)

# %% [markdown]
# ## Add web search tool

# %%
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

web_search.invoke("Who is the current mayor of San Francisco?")

# %%
agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt="You are a helpful assistant that can search the web for information."
)

question = HumanMessage(content="Who is the current mayor of San Francisco?")

response = agent.invoke(
    {"messages": [question]}
)

# %%
from pprint import pprint

pprint(response['messages'])

# %% [markdown]
# trace: https://smith.langchain.com/public/59432173-0dd6-49e8-9964-b16be6048426/r
