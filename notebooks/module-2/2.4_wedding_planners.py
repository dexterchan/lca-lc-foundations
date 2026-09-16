# %%

from dotenv import load_dotenv

load_dotenv()

# %% [markdown]

# # 2.4 Wedding Planner
# In this lab you will build a multi-agent wedding planner.
# 
# > Note: This lab has been updated since filming to make the agent performance more robust to errors and to limit run time. In particular: 1) added error handling for MCP failures, 2) limited the number of searches. It's worth noting that, where possible, tool errors should be returned to the agent rather than failing so that the agent can adjust its invocation and try again.

# %% [markdown]

# ## Setup Tools
# 

# %%

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

RETRYABLE_MCP_CODES = {-32603}

class RetryMCPInterceptor:
    """Intercept MCP tool calls: retry transient failures, surface all errors gracefully.

    - Retryable McpError codes (e.g. -32603): retry with exponential backoff.
    - Non-retryable McpError codes (e.g. -32602): return error message immediately.
    - Any other exception (fetch failed, network errors, etc.): retry then return error message.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(code {exc.error.code}, attempt {attempt+1}/{self.max_retries}): {exc}")
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(attempt {attempt+1}/{self.max_retries}): {exc}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        print(f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )

client = MultiServerMCPClient(
    {
        "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
    },
    tool_interceptors=[RetryMCPInterceptor()],
)

tools = await client.get_tools()

# %%

from typing import Dict, Any
from tavily import TavilyClient
from langchain.tools import tool

tavily_client = TavilyClient()

@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters     
      (e.g., use 'capacite' instead of 'capacité').
    """
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}

# %%

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

@tool
def query_playlist_db(query: str) -> str:

    """Query the database for playlist information"""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"

# %% [markdown]

# ## Create State

# %%

from langchain.agents import AgentState

class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str

# %% [markdown]

# ## Create Subagents
# 

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

# Travel agent
travel_agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
    You are a travel agent. Search for flights to the desired destination wedding location.
    You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
    - Price (lowest, economy class)
    - Duration (shortest)
    - Date (time of year which you believe is best for a wedding at this location)
    To make things easy, only look for one ticket, one way.
    You may need to make multiple searches to iteratively find the best options.
    You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
    If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
    Once you have found the best options, let the user know your shortlist of options.
    """
)

# %%

# Venue agent
venue_agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt="""
    You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
    You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
    - Price (lowest)
    - Capacity (exact match)
    - Reviews (highest)
    You may need to make multiple searches to iteratively find the best options. 
    You have a suggested limit of 12 web searches. Count every web_search call you make.
    After 12 searches, you should stop searching and summarize the best options you have
    found so far.
    """
)

# %%

# Playlist agent
playlist_agent = create_agent(
    model=model,
    tools=[query_playlist_db],
    system_prompt="""
    You are a playlist specialist. Query the sql database and curate the perfect playlist for a wedding given a genre.
    Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
    If you run into errors when querying the database, try to fix them by making changes to the query.
    Do not come back empty handed, keep trying to query the db until you find a list of songs.

    This is a SQLite database. Before writing any data queries, first discover the schema.
    """
)

# %% [markdown]

# ## Main Coordinator
# 

# %%

from langchain.tools import ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command

@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """Travel agent searches for flights to the desired destination wedding location."""
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    response = await travel_agent.ainvoke({"messages": [HumanMessage(content=f"Find flights from {origin} to {destination}")]})
    return response['messages'][-1].content

@tool
def search_venues(runtime: ToolRuntime) -> str:
    """Venue agent chooses the best venue for the given location and capacity."""
    destination = runtime.state["destination"]
    capacity = runtime.state["guest_count"]
    query = f"Find wedding venues in {destination} for {capacity} guests"
    response = venue_agent.invoke({"messages": [HumanMessage(content=query)]})
    return response['messages'][-1].content

@tool
def suggest_playlist(runtime: ToolRuntime) -> str:
    """Playlist agent curates the perfect playlist for the given genre."""
    genre = runtime.state["genre"]
    query = f"Find {genre} tracks for wedding playlist"
    response = playlist_agent.invoke({"messages": [HumanMessage(content=query)]})
    return response['messages'][-1].content

@tool
def update_state(origin: str, destination: str, guest_count: str, genre: str, runtime: ToolRuntime) -> str:
    """Update the state when you know all of the values: origin, destination, guest_count, genre. 
    This tool must be called alone, without any other tool calls. It must complete and return to make,
    the information available to other tools."""
    return Command(update={
        "origin": origin, 
        "destination": destination, 
        "guest_count": guest_count, 
        "genre": genre, 
        "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)]}
        )

# %%

from langchain.agents import create_agent

coordinator = create_agent(
    model=model,
    tools=[search_flights, search_venues, suggest_playlist, update_state],
    state_schema=WeddingState,
    system_prompt="""
    You are a wedding coordinator. 
    First find all the information you need to update the state. When you have the information, update the state.
    Once that has completed and returned, you can delegate the tasks 
    to your specialists for flights, venues, and playlists.
    Once you have received their answers, coordinate the perfect wedding for me.
    """
)

# %% [markdown]

# ## Test
# 

# %%
from langchain.messages import HumanMessage

async def main():
    response = await coordinator.ainvoke(
        {
            "messages": [HumanMessage(content="I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre")],
        },
        config={"tags": ["WP"], "recursion_limit": 40},  #tag traces to make them easy to find in Langsmith. Increase number of steps the agent can take to 40.
    )

    from pprint import pprint

    pprint(response)

    print(response["messages"][-1].content)


# %%

# %% [markdown]

# link to trace: https://smith.langchain.com/public/7b5fe668-d3e3-4af4-b513-a8cacc0c9e84/r

if __name__ == "__main__":
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        result = asyncio.run(main())
    else:
        print("Python Interactive: run `result = await main()` in a new cell.")
else:
    await main()