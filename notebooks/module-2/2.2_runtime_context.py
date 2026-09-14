# %%

from dotenv import load_dotenv

load_dotenv()

# %%

from dataclasses import dataclass

@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"

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
    context_schema=ColourContext  
)

# %%

from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext()
)

# %%

from pprint import pprint

pprint(response)

# %% [markdown]

# ## Accessing Context

# %%

from langchain.tools import tool, ToolRuntime

@tool
def get_favourite_colour(runtime: ToolRuntime[ColourContext]) -> str:
    """Get the favourite colour of the user"""
    return runtime.context.favourite_colour

@tool
def get_least_favourite_colour(runtime: ToolRuntime[ColourContext]) -> str:
    """Get the least favourite colour of the user"""
    return runtime.context.least_favourite_colour

# %%

agent = create_agent(
    model=model,
    tools=[get_favourite_colour, get_least_favourite_colour],
    context_schema=ColourContext
)

# %%

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext()
)

pprint(response)

# %%

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext(favourite_colour="green")
)

pprint(response)
