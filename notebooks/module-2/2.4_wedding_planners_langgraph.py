# %% [markdown]


# # Wedding planner with a fixed LangGraph flow
#
# START -> collect_request (update_state) -> venue -> flights -> playlist -> output
#             |                              ^        |
#             +-> END if details are missing +--------+ flight failure
#
# Five venue attempts at most. Each specialist has its own time and call limits.
# Only failed flights return to venue search. A failed venue search stops.
# Failed playlists produce a partial plan; they do not restart the search.

# %% [markdown]

# ## Session note: what failed and why — 14 September 2026
#
# The last blocker was the playlist agent. It used its four database calls to
# inspect tables and count Jazz tracks. It reached the limit before fetching
# the actual songs, artists, lengths, and prices. More model calls also made
# the full run too slow. Venue and flights had already succeeded.
#
# Fix: build_playlist() now uses one read-only SQL query against the known
# Chinook tables. Python lists up to 12 tracks and computes the duration and
# price from those rows. This step no longer needs an AI agent. If the database
# schema (table and column layout) changes, update that query.
#
# Earlier, the venue agent's final answer was counted as a fifth tool call
# after four searches. ToolStrategy sends its final answer through a special
# tool. We now limit real tools by name, so that answer is not counted as a
# search. The model still has a six-call limit. The inner graph allows 100
# steps because its checks also use steps; steps are not the same as AI calls.
#
# A failed venue search used to restart itself and repeat the same error.
# Now it stops. Only failed flights return to venue search, with at most five
# venue attempts. The failed venue is added to rejected_venues, passed to the
# next agent prompt, and added to the web-search query as an exclusion term.
# Code also checks the returned name, since search results can still repeat it.
#
# At that time, main() had a 110-second total timeout and returned a partial plan.
# The 14 September live check: venue, flights, and playlist all succeeded in 104.2
# seconds. All 23 local tests passed before this comment-only update.
#
# Follow-up — 15 September 2026: a live venue answer sent success="true"
# instead of a JSON boolean. ToolStrategy now allows answer-format repairs
# within the existing six-call and time limits; success still uses StrictBool.
# All 26 local tests pass, including corrected true/false answers and the
# repair limit. The next live venue search passed, but Kiwi returned HTTP 503
# (service unavailable). The run returned a partial plan at the 110-second
# limit. The local database also returned 12 Jazz tracks with correct totals.

# %% [markdown]

# ## Session input restored
#
# An input agent now calls update_state to save fields from the user's message.
# It can save a partial request and ask for the missing fields on the next turn.
# The playlist still uses SQL; its genre comes from the saved request.
#
# WeddingPlanner owns the async session API. LangGraph's thread_id keeps each
# conversation's request, messages, and progress separate. One lock per session
# queues overlapping turns, while other sessions continue. Final plans and
# stopped-run replies stay in the chat history for follow-up messages.
#
# After the live benchmark, time limits were raised to 180 seconds per worker
# and 300 seconds for the full run. Remote searches need room to finish.
# A timeout still returns a partial plan.
#
# The matched benchmark also found a venue agent rejecting a suitable candidate
# solely because the requested date was unconfirmed. VenueResearch now separates
# location, advertised capacity, sources, and date availability. Python decides
# whether those facts support continuing; an unknown date does not block flights.
# Known unavailability, missing evidence, or insufficient capacity still stops.

# %%

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback
from typing import Annotated, Literal, TypedDict
from uuid import uuid4
from weakref import WeakValueDictionary

from langchain.agents import AgentState
from langchain.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import (
    BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator,
)
from pprint import pprint

class SearchResult(BaseModel):
    """An explicit success flag lets code choose the next node."""

    model_config = ConfigDict(str_strip_whitespace=True)
    success: StrictBool
    details: str = Field(
        min_length=1,
        description="Verified findings with sources, or the reason the search failed.",
    )


class VenueResult(SearchResult):
    venue_name: str = ""
    date_availability: Literal["unknown", "available", "unavailable"] = "unknown"
    arrival_airport: str = Field(
        default="", description="Three-letter airport code near the selected venue."
    )

    @model_validator(mode="after")
    def require_usable_venue(self):
        if self.success:
            if not self.venue_name:
                raise ValueError("A successful venue search needs a venue name.")
            if len(self.arrival_airport) != 3 or not self.arrival_airport.isalpha():
                raise ValueError("A successful venue search needs an airport code.")
            self.arrival_airport = self.arrival_airport.upper()
        return self


class VenueResearch(BaseModel):
    """The agent reports evidence; code decides whether planning can continue."""

    model_config = ConfigDict(str_strip_whitespace=True)
    details: str = Field(min_length=1, description="Sourced findings and uncertainties.")
    venue_name: str
    arrival_airport: str = Field(description="Nearby airport's three-letter code.")
    matches_destination: StrictBool = Field(
        description="Whether the venue is in the requested destination, ignoring date availability.",
    )
    advertised_capacity: int | None = Field(
        strict=True, gt=0,
        description="Advertised guest capacity supported by a source; null if unknown.",
    )
    source_urls: list[str] = Field(
        description="Web result URLs supporting this venue's location and capacity.",
    )
    date_availability: Literal["unknown", "available", "unavailable"] = Field(
        default="unknown",
        description=(
            "Use unknown unless a source explicitly confirms available or unavailable "
            "for the requested date. Unknown does not make a candidate unsuitable."
        ),
    )

    def to_result(self, guest_count: int) -> VenueResult:
        sources = [url.strip() for url in self.source_urls if url.strip()]
        suitable = (
            bool(self.venue_name) and self.matches_destination
            and self.advertised_capacity is not None
            and self.advertised_capacity >= guest_count
            and len(self.arrival_airport) == 3 and self.arrival_airport.isalpha()
            and bool(sources)
        )
        availability = {
            "unknown": "Not confirmed. Check with the venue before booking.",
            "available": "Reported available by the cited source; no booking made.",
            "unavailable": "Unavailable on the requested date.",
        }[self.date_availability]
        lines = [self.details]
        if sources:
            lines.append("Sources: " + ", ".join(sources))
        lines.append("Date availability: " + availability)
        if not suitable:
            lines.append(
                f"No sourced venue with room for {guest_count} guests in the requested "
                "destination and a usable arrival airport was established."
            )
        return VenueResult(
            success=suitable and self.date_availability != "unavailable",
            details="\n".join(lines), venue_name=self.venue_name,
            arrival_airport=self.arrival_airport,
            date_availability=self.date_availability,
        )


class WeddingRequest(TypedDict, total=False):
    origin: str
    destination: str
    guest_count: int
    genre: str
    travel_date: str


class RequestFields(BaseModel):
    """A message can supply some fields now and the rest on a later turn."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    origin: str | None = Field(default=None, min_length=1)
    destination: str | None = Field(default=None, min_length=1)
    guest_count: int | None = Field(default=None, gt=0, strict=True)
    genre: str | None = Field(default=None, min_length=1)
    travel_date: str | None = Field(default=None, description="Travel date, YYYY-MM-DD.")

    @field_validator("travel_date")
    @classmethod
    def valid_date(cls, value):
        return date.fromisoformat(value).isoformat() if value is not None else None


class UpdateStateInput(RequestFields):
    # ToolRuntime is injected by LangChain, hidden from the model's tool schema.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    runtime: ToolRuntime


class RequestAgentState(AgentState):
    request: WeddingRequest


@tool(args_schema=UpdateStateInput, return_direct=True)
async def update_state(
    runtime: ToolRuntime,
    origin: str | None = None,
    destination: str | None = None,
    guest_count: int | None = None,
    genre: str | None = None,
    travel_date: str | None = None,
) -> Command | ToolMessage:
    """Save known wedding fields for this session. Call alone; omit unknown fields.

    Pass only values given by the user. Omitted fields keep their saved values.
    """
    if len(runtime.state["messages"][-1].tool_calls) != 1:
        return ToolMessage(
            "Call update_state once, alone, with all fields from this message.",
            name="update_state", tool_call_id=runtime.tool_call_id, status="error",
        )
    fields = RequestFields(
        origin=origin, destination=destination, guest_count=guest_count,
        genre=genre, travel_date=travel_date,
    ).model_dump(exclude_none=True)
    request = {**runtime.state.get("request", {}), **fields}
    # Never mutate runtime.state or use a global request dictionary.
    return Command(update={
        "request": request,
        "messages": [ToolMessage(
            "Saved request: " + json.dumps(request),
            name="update_state", tool_call_id=runtime.tool_call_id,
        )],
    })


def create_request_agent(*, model=None):
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware

    return create_agent(
        model=model if model is not None else create_model(),
        tools=[update_state],
        state_schema=RequestAgentState,
        checkpointer=False,  # The outer graph owns session history.
        middleware=[ModelCallLimitMiddleware(run_limit=4, exit_behavior="error")],
        system_prompt=(
            "Read the user's wedding request and call update_state once, alone. "
            "Save any known origin, destination, guest_count, genre and travel_date. "
            "Use only facts the user gave you. Never invent missing fields or dates. "
            "Use an integer guest_count and a YYYY-MM-DD travel_date. "
            "For follow-up messages, pass only new or corrected fields; saved fields "
            "are kept. For an explicit retry, call with no changes. "
            "Call the tool even when only some fields are known. The app will ask "
            "for missing fields. If the message is unclear, ask for clarification. "
            "Do not search or plan the wedding yourself."
        ),
    )


@dataclass(frozen=True)
class VenueSearchContext:
    """Pass exclusions to the search tool without asking the model to copy them."""

    rejected_venues: tuple[str, ...] = ()


class WeddingState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    request: WeddingRequest
    attempts: int
    rejected_venues: list[str]
    failures: list[str]
    last_error: str
    venue: VenueResult | None
    flights: SearchResult | None
    playlist: SearchResult | None
    status: Literal["running", "awaiting_input", "complete", "incomplete", "cancelled"]
    output: str


Worker = Callable[[WeddingState], Awaitable[SearchResult | dict]]


def fresh_plan() -> dict:
    """Reset results for a new message, keeping its session's request and history."""
    return {
        "attempts": 0, "rejected_venues": [], "failures": [], "last_error": "",
        "venue": None, "flights": None, "playlist": None,
        "status": "running", "output": "",
    }

# %% [markdown]

# ## Graph nodes and conditional edges
# The graph owns the order and retry count. An LLM cannot change those limits.

# %%


def build_wedding_graph(
    find_venue: Worker,
    find_flights: Worker,
    make_playlist: Worker,
    *,
    max_venue_attempts: int = 5,
    step_timeout: float = 180,
    request_agent=None,
    checkpointer=None,
):
    if max_venue_attempts < 1 or step_timeout <= 0:
        raise ValueError("Attempt and timeout limits must be positive.")

    async def collect_request(state: WeddingState):
        messages = state.get("messages", [])
        request = state.get("request", {})
        try:
            result = await asyncio.wait_for(
                request_agent.ainvoke(
                    {"messages": deepcopy(messages), "request": dict(request)},
                    config={"recursion_limit": 100},
                ),
                timeout=step_timeout,
            )
        except Exception as exc:
            print("[collect_request] failed:")
            traceback.print_exc()
            return {
                **fresh_plan(), "status": "incomplete",
                "output": f"Could not read the request: {type(exc).__name__}: {exc}",
            }
        new_messages = result["messages"][len(messages):]
        saved = any(
            isinstance(msg, ToolMessage) and msg.name == "update_state"
            and msg.status == "success"
            for msg in new_messages
        )
        if saved:
            request = RequestFields.model_validate(
                result["request"]
            ).model_dump(exclude_none=True)
        missing = [name for name in RequestFields.model_fields if name not in request]
        ready = saved and not missing
        if ready:
            output_text = ""
        elif saved:
            output_text = "Please provide: " + ", ".join(missing) + "."
        else:
            reply = next(
                (msg.content for msg in reversed(new_messages)
                 if isinstance(msg, AIMessage) and msg.content),
                "Please clarify the wedding details so I can save them.",
            )
            output_text = str(reply)
        return {
            **fresh_plan(), "request": request,
            "messages": [*new_messages, *(
                [AIMessage(content=output_text)] if output_text else []
            )],
            "status": "running" if ready else "awaiting_input",
            "output": output_text,
        }

    async def run_step(worker, state, schema=SearchResult):
        try:
            result = await asyncio.wait_for(worker(state), timeout=step_timeout)
            return schema.model_validate(result)
        except TimeoutError:
            return schema(
                success=False, details=f"Search timed out after {step_timeout:g} seconds."
            )
        except Exception as exc:
            print(f"[run_step:{worker.__name__ if hasattr(worker, '__name__') else worker}] failed:")
            traceback.print_exc()
            return schema(success=False, details=f"{type(exc).__name__}: {exc}")

    async def search_venue(state: WeddingState):
        # Clear old downstream results before trying a new candidate.
        current = {
            **state,
            "attempts": state.get("attempts", 0) + 1,
            "rejected_venues": state.get("rejected_venues", []),
            "failures": state.get("failures", []),
            "last_error": state.get("last_error", ""),
            "flights": None,
            "playlist": None,
            "status": "running",
        }
        result = await run_step(find_venue, current, VenueResult)
        rejected = {name.strip().casefold() for name in current["rejected_venues"]}
        if result.success and result.venue_name.casefold() in rejected:
            result = VenueResult(
                success=False,
                details=f"Already tried {result.venue_name}; choose a different venue.",
            )
        current["venue"] = result
        current["last_error"] = "" if result.success else result.details
        if not result.success:
            current["failures"] = [
                *current["failures"], f"Venue attempt {current['attempts']}: {result.details}"
            ]
        return current

    async def search_flights(state: WeddingState):
        result = await run_step(find_flights, state)
        updates = {"flights": result, "last_error": ""}
        if not result.success:
            updates.update(
                last_error=result.details,
                rejected_venues=[
                    *state["rejected_venues"], state["venue"].venue_name
                ],
                failures=[
                    *state["failures"],
                    f"Flights for {state['venue'].venue_name}: {result.details}",
                ],
            )
        return updates

    async def playlist(state: WeddingState):
        result = await run_step(make_playlist, state)
        updates = {"playlist": result, "last_error": ""}
        if not result.success:
            updates.update(
                last_error=result.details,
                failures=[*state["failures"], f"Playlist: {result.details}"],
            )
        return updates

    def after_venue(state: WeddingState):
        if state["venue"].success:
            return "flights"
        return "output"

    def after_flights(state: WeddingState):
        if state["flights"].success:
            return "playlist"
        return "venue" if state["attempts"] < max_venue_attempts else "output"

    def output(state: WeddingState):
        complete = all(
            state.get(key) is not None and state[key].success
            for key in ("venue", "flights", "playlist")
        )
        lines = [
            "Wedding plan complete" if complete else "Wedding plan incomplete",
            f"Venue attempts: {state['attempts']} / {max_venue_attempts}",
            f"Travel date: {state['request']['travel_date']}",
        ]
        for key, label in [
            ("venue", "Venue candidate"), ("flights", "Flights"), ("playlist", "Playlist")
        ]:
            result = state.get(key)
            lines.extend(["", f"{label}: {result.details if result else 'Not run.'}"])
        if state["failures"]:
            lines.extend(["", "Search issues:", *state["failures"]])
        text = "\n".join(lines)
        return {
            "status": "complete" if complete else "incomplete",
            "output": text,
            "messages": [AIMessage(content=text)],
        }

    graph = StateGraph(WeddingState)
    graph.add_node("venue", search_venue)
    graph.add_node("flights", search_flights)
    graph.add_node("playlist", playlist)
    graph.add_node("output", output)
    if request_agent is None:
        graph.add_edge(START, "venue")
    else:
        graph.add_node("collect_request", collect_request)
        graph.add_edge(START, "collect_request")
        graph.add_conditional_edges(
            "collect_request",
            lambda state: "venue" if state["status"] == "running" else END,
            {"venue": "venue", END: END},
        )
    graph.add_conditional_edges(
        "venue", after_venue, {"flights": "flights", "output": "output"}
    )
    graph.add_conditional_edges(
        "flights", after_flights,
        {"playlist": "playlist", "venue": "venue", "output": "output"},
    )
    graph.add_edge("playlist", "output")
    graph.add_edge("output", END)
    return graph.compile(checkpointer=checkpointer)


class WeddingPlanner:
    """Share one instance among async callers in one process/event loop.

    thread_id selects saved state. A per-session lock serializes its turns;
    different sessions have different locks and can run at the same time.
    Memory storage is for this demo. For multiple server processes, supply a
    durable checkpointer and serialize each session's turns across those processes.
    """

    def __init__(self, find_venue, find_flights, make_playlist, *,
                 request_agent, checkpointer=None, **graph_options):
        if checkpointer is None:
            checkpointer = InMemorySaver(serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    (SearchResult.__module__, SearchResult.__name__),
                    (VenueResult.__module__, VenueResult.__name__),
                ],
            ))
        self.graph = build_wedding_graph(
            find_venue, find_flights, make_playlist, request_agent=request_agent,
            checkpointer=checkpointer,
            **graph_options,
        )
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._recursion_limit = 2 * graph_options.get("max_venue_attempts", 5) + 6

    def _config(self, session_id: str) -> dict:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return {
            "configurable": {"thread_id": session_id},
            "recursion_limit": self._recursion_limit,
        }

    async def aget_state(self, session_id: str) -> WeddingState:
        """Read this session's last saved step, including while it is running."""
        return (await self.graph.aget_state(self._config(session_id))).values

    async def _save_stopped(self, config: dict, *, status: str, reason: str):
        state = (await self.graph.aget_state(config)).values
        lines = ["Wedding plan incomplete", reason]
        for key in ("venue", "flights", "playlist"):
            result = state.get(key)
            lines.append(
                f"{key.title()}: {result.details if result else 'Not completed.'}"
            )
        text = "\n".join(lines)
        # Finish pending work and keep the partial reply in this session's chat.
        await self.graph.aupdate_state(config, {
            "status": status, "output": text, "messages": [AIMessage(content=text)],
        }, as_node="output")
        return (await self.graph.aget_state(config)).values

    async def ainvoke(self, message: str, *, session_id: str,
                      timeout_seconds: float = 300) -> WeddingState:
        config = self._config(session_id)
        trace_id = str(uuid4())
        config["run_id"] = trace_id
        config["run_name"] = "wedding_planner_turn"
        print(f"LangSmith trace_id: {trace_id}")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        # Keep a strong reference until all callers waiting on this lock finish.
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            try:
                async with asyncio.timeout(timeout_seconds):
                    return await self.graph.ainvoke(
                        {**fresh_plan(), "messages": [HumanMessage(content=message)]},
                        config=config,
                    )
            except TimeoutError:
                return await self._save_stopped(
                    config, status="incomplete",
                    reason=f"Total time limit reached ({timeout_seconds:g} seconds).",
                )
            except asyncio.CancelledError:
                await self._save_stopped(
                    config, status="cancelled", reason="Planning cancelled.",
                )
                raise

# %% [markdown]

# ## Live searches: DeepSeek through OpenRouter
# Search results use typed fields. Plain text without a success flag is a failure.
# The playlist uses one local, read-only query and computes its totals in Python.

# %%


def query_playlist(query: str, db_path: Path, parameters: tuple = ()) -> str:
    deadline = time.monotonic() + 5
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only = ON")
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        cursor = db.execute(query, parameters)
        return json.dumps({
            "columns": [column[0] for column in cursor.description or []],
            "rows": cursor.fetchmany(50),
        }, default=str)


def build_playlist(genre: str, db_path: Path) -> SearchResult:
    """Read tracks and compute totals directly; avoid the old schema-search loop.

    The previous agent spent its tool budget learning the database layout and
    never fetched a playlist. This fixed query uses the known Chinook schema.
    The genre is a query parameter, and the database remains read-only.
    """
    data = json.loads(query_playlist(
        """
        SELECT t.Name, ar.Name, t.Milliseconds, t.UnitPrice
        FROM Track AS t
        JOIN Album AS al ON al.AlbumId = t.AlbumId
        JOIN Artist AS ar ON ar.ArtistId = al.ArtistId
        JOIN Genre AS g ON g.GenreId = t.GenreId
        WHERE g.Name = ? COLLATE NOCASE
        ORDER BY ar.Name, t.TrackId
        LIMIT 12
        """,
        db_path, (genre,),
    ))
    rows = data["rows"]
    if not rows:
        return SearchResult(success=False, details=f"No {genre} tracks in the playlist database.")
    duration = timedelta(milliseconds=sum(row[2] for row in rows))
    price = sum((Decimal(str(row[3])) for row in rows), Decimal("0.00"))
    lines = [f"{genre.title()} playlist ({len(rows)} tracks):"]
    lines.extend(f"{index}. {row[0]} — {row[1]}" for index, row in enumerate(rows, 1))
    lines.extend([
        f"Total duration: {duration}",
        f"Total price: USD {price:.2f}",
        "Source: local Chinook.db (Track, Album, Artist, Genre).",
    ])
    return SearchResult(success=True, details="\n".join(lines))


async def ask_specialist(
    agent, prompt: str, schema, tool_names: set[str], *, context=None
):
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        # Middleware also uses graph steps. The six model-call limit below is
        # the work budget; allow enough graph steps for those calls to finish.
        config={"recursion_limit": 100},
        **({"context": context} if context is not None else {}),
    )
    parsed = schema.model_validate(result.get("structured_response"))
    if not isinstance(parsed, SearchResult) or parsed.success:
        # Do not accept an answer that skipped the actual search/database tools.
        from langchain.messages import ToolMessage

        evidence = [
            msg for msg in result["messages"]
            if isinstance(msg, ToolMessage)
            and msg.name in tool_names
            and msg.status != "error"
            and msg.content
        ]
        if not evidence:
            raise ValueError("The agent reported success without a successful tool result.")
    return parsed


def create_model():
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model="deepseek/deepseek-v4-flash-0731",
        model_provider="deepseek",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_KEY"],
        timeout=30,
        max_retries=0,
    )


def create_live_workers():
    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware, ToolCallLimitMiddleware,
    )
    from langchain.agents.structured_output import (
        StructuredOutputValidationError, ToolStrategy,
    )
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from tavily import AsyncTavilyClient

    model = create_model()
    tavily = AsyncTavilyClient(timeout=30)
    mcp_client = MultiServerMCPClient({
        "travel_server": {
            "transport": "streamable_http", "url": "https://mcp.kiwi.com"
        }
    })
    # VS Code's Interactive window may not define __file__.
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    db_path = script_dir / "resources" / "Chinook.db"
    if not db_path.is_file():
        db_path = Path.cwd() / "notebooks/module-2/resources/Chinook.db"
    if not db_path.is_file():
        raise FileNotFoundError("Cannot find notebooks/module-2/resources/Chinook.db")

    @tool
    async def web_search(query: str, runtime: ToolRuntime[VenueSearchContext]) -> dict:
        """Search the web for wedding venues, capacity, location, and nearby airports."""
        # Ask the search engine to omit rejected venue names even when the model
        # repeats its original query. Keep the final name check in search_venue.
        exclusions = [
            name.strip() for name in runtime.context.rejected_venues if name.strip()
        ]
        if exclusions:
            query += " " + " ".join(
                "-" + json.dumps(name, ensure_ascii=False) for name in exclusions
            )
        return await tavily.search(query, max_results=5)

    def specialist(tools, schema, instructions, *, context_schema=None):
        return create_agent(
            model=model,
            tools=tools,
            context_schema=context_schema,
            checkpointer=False,  # Search history belongs to this attempt only.
            # Let the model repair a malformed final answer using the evidence
            # it already has. Repairs share the same model-call and time limits.
            response_format=ToolStrategy(
                schema, handle_errors=StructuredOutputValidationError
            ),
            middleware=[
                ModelCallLimitMiddleware(run_limit=6, exit_behavior="error"),
                # Limit real tools by name. ToolStrategy's final-answer tool
                # must not count as another web search or database query.
                # On exhaustion, block the tool and let the model finish.
                *[
                    ToolCallLimitMiddleware(
                        tool_name=item.name, run_limit=4, exit_behavior="continue"
                    )
                    for item in tools
                ],
            ],
            system_prompt=(
                instructions
                + "\nUse the provided tools. Report errors and missing evidence honestly. "
                "Never invent findings. Include source URLs or database facts "
                "in details. Do not ask follow-up questions. You have at most four "
                "calls per tool. Finish as soon as you have enough evidence. If a "
                "tool hits its limit, use the results already collected to give "
                "your final structured answer and identify any missing evidence. "
                "Do not call that tool again. "
                "Boolean fields must be JSON booleans (true or false), never "
                "quoted strings. If your final answer fails validation, correct "
                "its format using the evidence already collected; do not repeat "
                "the searches. "
                "This is research only; do not book or purchase anything."
            ),
        )

    venue_agent = specialist(
        [web_search], VenueResearch,
        "Choose ONE real wedding venue in the requested destination with room for "
        "the guest count. Report its name, a nearby arrival airport's three-letter "
        "code, whether its location matches the destination, advertised guest "
        "capacity, and source URLs supporting those facts. Report only capacity "
        "that belongs to this venue, not a different venue in the same search result. "
        "This is venue research, not a booking or an availability check. A sourced "
        "venue with enough advertised capacity is a usable candidate even when "
        "the requested date is not confirmed. Keep date_availability separate: "
        "unknown unless a source explicitly confirms that date as available or "
        "unavailable. Missing availability information is not unavailability. "
        "Do not spend further searches confirming dates once a suitable candidate "
        "is supported, and do not search for flights; a separate worker does that. "
        "The rejected_venues list in the user message is mandatory: never choose "
        "any listed venue, even if it appears in search results. Find a different "
        "venue, not a spelling variant of an excluded one. Use the last flight failure to guide "
        "your next choice; consider a different nearby airport where appropriate. "
        "Keep the requested destination and travel date.",
        context_schema=VenueSearchContext,
    )
    async def find_venue(state):
        prompt = json.dumps({
            "request": state["request"],
            "attempt": state["attempts"],
            "rejected_venues": state["rejected_venues"],
            "last_flight_or_venue_error": state["last_error"],
        })
        research = await ask_specialist(
            venue_agent, prompt, VenueResearch, {"web_search"},
            context=VenueSearchContext(tuple(state["rejected_venues"])),
        )
        return research.to_result(state["request"]["guest_count"])

    async def find_flights(state):
        # Discovery failures happen inside the flight node and take the retry edge.
        tools = await mcp_client.get_tools()
        if not tools:
            raise RuntimeError("The travel MCP server returned no tools.")
        flight_agent = specialist(
            tools, SearchResult,
            "Find an actual one-way economy flight for ONE adult from the given "
            "origin to the chosen venue's arrival airport on the requested date. "
            "Prefer a low price and short duration. Give the route, date, price "
            "and source link returned by the tool. A tool error, empty results, "
            "or no matching flight means success=false. Do not claim a flight "
            "was found based only on general advice.",
        )
        prompt = json.dumps({
            "request": state["request"],
            "selected_venue": state["venue"].model_dump(),
        })
        return await ask_specialist(
            flight_agent, prompt, SearchResult, {item.name for item in tools}
        )

    async def make_playlist(state):
        return await asyncio.to_thread(
            build_playlist, state["request"]["genre"], db_path
        )

    return find_venue, find_flights, make_playlist


def create_live_planner(*, checkpointer=None) -> WeddingPlanner:
    """Create once, then reuse for each session and each follow-up message."""
    from dotenv import load_dotenv

    load_dotenv(override=True)
    return WeddingPlanner(
        *create_live_workers(), request_agent=create_request_agent(),
        checkpointer=checkpointer,
    )

# %% [markdown]

# ## Run the example
# Terminal: uv run python notebooks/module-2/2.4_wedding_planners_langgraph.py
# Python Interactive: run the definition cells, then run `result = await main()`.
# Change the message below to choose the destination, date, guests, and music.
# The input agent must call update_state before the searches can start.

# %%


async def main(timeout_seconds: float = 300, *, message: str | None = None,
               session_id: str | None = None, app: WeddingPlanner | None = None):
    if message is None:
        travel_date = (date.today() + timedelta(days=180)).isoformat()
        message = (
            "I'm from London and want a wedding in Paris for 100 guests, "
            f"with jazz music. Travel date: {travel_date}."
        )
    if app is None:
        app = create_live_planner()
    final_state = await app.ainvoke(
        message, session_id=session_id if session_id is not None else str(uuid4()),
        timeout_seconds=timeout_seconds,
    )
    pprint("\n" + final_state["output"])
    return final_state


# %% [markdown]

# ## Concurrent sessions and follow-up messages
# Use ONE planner instance within ONE running event loop. The caller supplies
# session IDs; the model cannot choose them. Reuse an ID for the same conversation.
# In a web app, bind each ID to the signed-in user's conversation on the server.
#
# ```python
# app = create_live_planner()
# alice, bob = await asyncio.gather(
#     app.ainvoke("I'm from London and like jazz.", session_id="alice"),
#     app.ainvoke("I'm from Hong Kong and like rock.", session_id="bob"),
# )
# # Both wait for missing details; each keeps its own origin and genre.
# alice = await app.ainvoke(
#     "Paris, 100 guests, travel on 2027-06-12.", session_id="alice"
# )
# bob_state = await app.aget_state("bob")  # Still waiting; Alice did not change it.
# ```
#
# Saved state uses LangGraph's thread_id. Different IDs run concurrently.
# Calls with the same ID queue behind one another. Each new message keeps the
# request and chat history, then clears prior search results and retry counts.
# aget_state reads the last saved step while a session is running.
#
# The default InMemorySaver lasts only while this app instance is alive.
# For several server processes or restart recovery, pass a durable async
# checkpointer to create_live_planner(checkpointer=...) and use a shared queue
# or lock for turns with the same session ID. The local lock covers one instance.


# %% [markdown]

# ## Start the planner
# In VS Code Python Interactive, run this in a new cell:
# ```python
# result = await main()
# ```
# The terminal entry point below uses asyncio.run only when no loop is running.

# %%

if __name__ == "__main__":
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        result = asyncio.run(main())
    else:
        print("Python Interactive: run `result = await main()` in a new cell.")
