# %%

from dotenv import load_dotenv

load_dotenv()

# %%

from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
    }
)

tools = await client.get_tools()

# %%

import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

model = init_chat_model(
    model="deepseek/deepseek-v4-flash-0731",
    model_provider="deepseek",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

system_prompt = """
                You are a travel agent. No follow up questions.
                """

agent = create_agent(
    model,
    tools=tools,
    checkpointer=InMemorySaver(),
    system_prompt=system_prompt
)

# %%

from langchain.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}

response = await agent.ainvoke(
    {"messages": [HumanMessage(content="Get me a direct flight from San Francisco to Tokyo on March 31st")]},
    config
    )

# %%

from pprint import pprint

pprint(response)

# %%

print(response["messages"][-1].content)

# %%
