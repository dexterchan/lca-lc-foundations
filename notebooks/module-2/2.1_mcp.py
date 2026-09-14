# %%

from dotenv import load_dotenv

load_dotenv()

# %%

import sys
import asyncio

# Fix for Windows issues in Jupyter notebooks
if sys.platform == "win32":
    # 1. Use ProactorEventLoop for subprocess support
    if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 2. Redirect stderr to avoid fileno() error when launching MCP servers
    if "ipykernel" in sys.modules:
        sys.stderr = sys.__stderr__

# %% [markdown]

# ## Local MCP server

# %%

from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "local_server": {
                "transport": "stdio",
                "command": "uv",
                "args": ["resources/2.1_mcp_server.py"],
                "args": [
                                "run",
                                "python",
                                "resources/2.1_mcp_server.py"
                            ]
            }
    }
)

# %%

# get tools
tools = await client.get_tools()

# get resources
resources = await client.get_resources("local_server")

# get prompts
prompt = await client.get_prompt("local_server", "prompt")
prompt = prompt[0].content

# %%

import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model(
    model="openai/gpt-5-nano",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=prompt
)

# %%

from langchain.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}

response = await agent.ainvoke(
    {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
    config=config
)

# %%

from pprint import pprint

pprint(response)

# %% [markdown]

# ## Online MCP

# %%

client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": "uv",
            "args": [
                "run",
                "python",
                "-m",
                "mcp_server_time",
                "--local-timezone=America/New_York"
            ]
        }
    }
)

tools = await client.get_tools()

# %%

agent = create_agent(
    model=model,
    tools=tools,
)

# %%

question = HumanMessage(content="What time is it?")

response = await agent.ainvoke(
    {"messages": [question]}
)

pprint(response)
