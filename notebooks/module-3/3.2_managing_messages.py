# %%
from dotenv import load_dotenv

load_dotenv()

# %% [markdown]
# ## Summarize messages

# %%
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

model = init_chat_model(
    model="deepseek/deepseek-v4-flash-0731",
    model_provider="deepseek",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

summarize_model = init_chat_model(
    model="deepseek/deepseek-v3.2",
    model_provider="deepseek",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=summarize_model,
            trigger=("tokens", 100),
            keep=("messages", 1),
        )
    ],
)

# %%
from langchain.messages import HumanMessage, AIMessage
from pprint import pprint

response = agent.invoke(
    {"messages": [
        HumanMessage(content="What is the capital of the moon?"),
        AIMessage(content="The capital of the moon is Lunapolis."),
        HumanMessage(content="What is the weather in Lunapolis?"),
        AIMessage(content="Skies are clear, with a high of 120C and a low of -100C."),
        HumanMessage(content="How many cheese miners live in Lunapolis?"),
        AIMessage(content="There are 100,000 cheese miners living in Lunapolis."),
        HumanMessage(content="Do you think the cheese miners' union will strike?"),
        AIMessage(content="Yes, because they are unhappy with the new president."),
        HumanMessage(content="If you were Lunapolis' new president how would you respond to the cheese miners' union?"),
    ]},
    {"configurable": {"thread_id": "1"}},
)

pprint(response)

# %%
print(response["messages"][0].content)

# %% [markdown]
# ## Trim/delete messages

# %%
from typing import Any
from langchain.agents import AgentState
from langchain.messages import RemoveMessage
from langgraph.runtime import Runtime
from langchain.agents.middleware import before_agent
from langchain.messages import ToolMessage


@before_agent
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove all the tool messages from the state."""
    messages = state["messages"]

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    return {"messages": [RemoveMessage(id=m.id) for m in tool_messages]}


# %%
agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    middleware=[trim_messages],
)

agent2 = create_agent(
    model=model,
    checkpointer=InMemorySaver()
)
# %%
response = agent.invoke(
    {"messages": [
        HumanMessage(content="My device won't turn on. What should I do?"),
        ToolMessage(content="blorp-x7 initiating diagnostic ping…", tool_call_id="1"),
        AIMessage(content="Is the device plugged in and turned on?"),
        HumanMessage(content="Yes, it's plugged in and turned on."),
        ToolMessage(content="temp=42C voltage=2.9v … greeble complete.", tool_call_id="2"),
        AIMessage(content="Is the device showing any lights or indicators?"),
        HumanMessage(content="What's the temperature of the device?"),
    ]},
    {"configurable": {"thread_id": "2"}},
)

response2 = agent2.invoke(
    {"messages": [
        HumanMessage(content="My device won't turn on. What should I do?"),
        ToolMessage(content="blorp-x7 initiating diagnostic ping…", tool_call_id="1"),
        AIMessage(content="Is the device plugged in and turned on?"),
        HumanMessage(content="Yes, it's plugged in and turned on."),
        ToolMessage(content="temp=42C voltage=2.9v … greeble complete.", tool_call_id="2"),
        AIMessage(content="Is the device showing any lights or indicators?"),
        HumanMessage(content="What's the temperature of the device?"),
    ]},
    {"configurable": {"thread_id": "2"}},
)

pprint(response)

# %%
print(response["messages"][-1].content)

# %%
