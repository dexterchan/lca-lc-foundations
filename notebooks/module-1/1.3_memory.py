# %%
from dotenv import load_dotenv

load_dotenv()

# %% [markdown]
# ## No memory

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
    model
)

# %%
from langchain.messages import HumanMessage

question = HumanMessage(content="Hello my name is Seán and my favourite colour is green")

response = agent.invoke(
    {"messages": [question]}
)

# %%
from pprint import pprint

pprint(response)

# %%
question = HumanMessage(content="What's my favourite colour?")

response = agent.invoke(
    {"messages": [question]}
)

pprint(response)

# %% [markdown]
# ## Memory

# %%
from langgraph.checkpoint.memory import InMemorySaver


agent = create_agent(
    model,
    checkpointer=InMemorySaver(),
)

# %%
from langchain.messages import HumanMessage

question = HumanMessage(content="Hello my name is Seán and my favourite colour is green")
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [question]},
    config,
)

# %%
pprint(response)

# %%
question = HumanMessage(content="What's my favourite colour?")

response = agent.invoke(
    {"messages": [question]},
    config,
)

pprint(response)
