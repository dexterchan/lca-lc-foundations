# %%
from dotenv import load_dotenv

load_dotenv(override=True)

# %%
from pathlib import Path
from typing import Dict, Any
from langchain.tools import tool
from tavily import TavilyClient
from langchain_community.utilities import SQLDatabase

tavily_client = TavilyClient()

db_path = Path(__file__).resolve().parent / "resources" / "Chinook.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information."""
    return tavily_client.search(query)


@tool
def sql_query(query: str) -> str:
    """Obtain information from the database using SQL queries."""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


# %%
from dataclasses import dataclass


@dataclass
class UserRole:
    user_role: str = "external"


# %%
from typing import Callable
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


@wrap_model_call
def dynamic_tool_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Choose available tools based on the user's role."""
    user_role = request.runtime.context.user_role

    if user_role != "internal":
        # External users only get access to web search.
        request = request.override(tools=[web_search])

    # Internal users keep access to all tools.
    return handler(request)


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
    tools=[web_search, sql_query],
    middleware=[dynamic_tool_call],
    context_schema=UserRole,
)

# %%
from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="How many artists are in the database?")]},
    context=UserRole(user_role="external"),
)

print(response["messages"][-1].content)

# %%
response = agent.invoke(
    {"messages": [HumanMessage(content="How many artists are in the database?")]},
    context=UserRole(user_role="internal"),
)

print(response["messages"][-1].content)
# %%
