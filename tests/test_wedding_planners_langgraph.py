"""Check the real graph with stand-ins for paid search and model calls."""

import asyncio
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from langchain.messages import AIMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "notebooks/module-2/2.4_wedding_planners_langgraph.py"
)
spec = importlib.util.spec_from_file_location("wedding_langgraph", SCRIPT)
planner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = planner
spec.loader.exec_module(planner)


REQUEST = {
    "origin": "London",
    "destination": "Paris",
    "guest_count": 100,
    "genre": "jazz",
    "travel_date": "2027-06-12",
}


def venue(name="Garden Hall", airport="CDG"):
    return {
        "success": True,
        "details": f"{name} in Paris has room for 100 guests.",
        "venue_name": name,
        "arrival_airport": airport,
    }


def venue_research(name="Garden Hall", airport="CDG", **changes):
    return {
        "details": f"{name} in Paris advertises space for 100 guests.",
        "venue_name": name,
        "arrival_airport": airport,
        "matches_destination": True,
        "advertised_capacity": 100,
        "source_urls": ["https://example.test/garden-hall"],
        "date_availability": "unknown",
        **changes,
    }


async def good_venue(state):
    return venue()


async def good_flights(state):
    return {"success": True, "details": "Flight LHR to CDG, 12 June, GBP 90."}


async def good_playlist(state):
    return {"success": True, "details": "Jazz track A; 5 minutes; USD 0.99."}


async def run_graph(graph):
    updates = [
        update
        async for update in graph.astream(
            {"request": dict(REQUEST)}, stream_mode="updates"
        )
    ]
    # Node order and final output are observable parts of the requested flow.
    order = [name for update in updates for name in update]
    state = {}
    for update in updates:
        for values in update.values():
            state.update(values)
    return order, state


class WeddingGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_runs_venue_then_flights_then_playlist_then_output(self):
        graph = planner.build_wedding_graph(
            good_venue, good_flights, good_playlist
        )
        order, state = await run_graph(graph)
        self.assertEqual(order, ["venue", "flights", "playlist", "output"])
        self.assertEqual(state["status"], "complete")
        self.assertIn("Garden Hall", state["output"])
        self.assertIn("GBP 90", state["output"])
        self.assertIn("USD 0.99", state["output"])

    async def test_failed_flight_goes_back_with_reason_and_uses_new_venue(self):
        async def choose_venue(state):
            if state["attempts"] == 1:
                return venue()
            self.assertIn("No seats", state["last_error"])
            self.assertIn("Garden Hall", state["rejected_venues"])
            self.assertIsNone(state["flights"])
            return venue("River Hall", "ORY")

        async def find_flights(state):
            if state["venue"].venue_name == "Garden Hall":
                return {"success": False, "details": "No seats to CDG."}
            self.assertEqual(state["venue"].arrival_airport, "ORY")
            return {"success": True, "details": "Flight LHR to ORY, GBP 110."}

        async def playlist(state):
            self.assertEqual(state["venue"].venue_name, "River Hall")
            self.assertIn("ORY", state["flights"].details)
            return await good_playlist(state)

        graph = planner.build_wedding_graph(choose_venue, find_flights, playlist)
        order, state = await run_graph(graph)
        self.assertEqual(
            order, ["venue", "flights", "venue", "flights", "playlist", "output"]
        )
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["attempts"], 2)
        self.assertIn("River Hall", state["output"])

    async def test_repeated_failure_stops_at_limit_without_running_playlist(self):
        async def choose_venue(state):
            return venue(f"Hall {state['attempts']}")

        async def no_flights(state):
            return {"success": False, "details": "No flights found."}

        graph = planner.build_wedding_graph(
            choose_venue, no_flights, good_playlist
        )
        order, state = await run_graph(graph)
        self.assertEqual(order, ["venue", "flights"] * 5 + ["output"])
        self.assertEqual(state["attempts"], 5)
        self.assertEqual(state["status"], "incomplete")
        self.assertIsNone(state["playlist"])
        self.assertIn("No flights found.", state["output"])

        # A second request gets a fresh retry budget.
        second_order, second_state = await run_graph(graph)
        self.assertEqual(second_order, order)
        self.assertEqual(second_state["attempts"], 5)

    async def test_flight_exception_returns_to_venue(self):
        async def choose_venue(state):
            return venue(f"Hall {state['attempts']}")

        async def find_flights(state):
            if state["attempts"] == 1:
                raise ConnectionError("Kiwi is unavailable")
            return await good_flights(state)

        graph = planner.build_wedding_graph(
            choose_venue, find_flights, good_playlist
        )
        order, state = await run_graph(graph)
        self.assertEqual(
            order, ["venue", "flights", "venue", "flights", "playlist", "output"]
        )
        self.assertEqual(state["status"], "complete")
        self.assertTrue(any("Kiwi is unavailable" in x for x in state["failures"]))

    async def test_flight_timeout_is_bounded_and_does_not_start_playlist(self):
        async def hangs(state):
            await asyncio.Event().wait()

        graph = planner.build_wedding_graph(
            good_venue, hangs, good_playlist,
            max_venue_attempts=1, step_timeout=0.02,
        )
        order, state = await asyncio.wait_for(run_graph(graph), timeout=2)
        self.assertEqual(order, ["venue", "flights", "output"])
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("timed out", state["output"].lower())

    async def test_malformed_flight_result_is_not_treated_as_success(self):
        for bad_result in [
            {"details": "Could not search"},
            {"success": "true", "details": "A guessed flight"},
            {"success": True, "details": " "},
        ]:
            with self.subTest(result=bad_result):
                async def bad_flights(state):
                    return bad_result

                graph = planner.build_wedding_graph(
                    good_venue, bad_flights, good_playlist, max_venue_attempts=1
                )
                order, state = await run_graph(graph)
                self.assertEqual(order, ["venue", "flights", "output"])
                self.assertEqual(state["status"], "incomplete")

    async def test_failed_or_incomplete_venue_stops_without_repeating_the_search(self):
        for bad_venue in [
            {"success": False, "details": "No venues found."},
            {"success": True, "details": "A venue without a name or airport"},
        ]:
            with self.subTest(result=bad_venue):
                async def choose_venue(state):
                    return bad_venue if state["attempts"] == 1 else venue()

                graph = planner.build_wedding_graph(
                    choose_venue, good_flights, good_playlist
                )
                order, state = await run_graph(graph)
                self.assertEqual(order, ["venue", "output"])
                self.assertEqual(state["status"], "incomplete")
                self.assertEqual(state["attempts"], 1)

    async def test_rejected_venue_is_not_sent_to_flights_again(self):
        async def no_flights(state):
            return {"success": False, "details": "No flights found."}

        graph = planner.build_wedding_graph(
            good_venue, no_flights, good_playlist, max_venue_attempts=2
        )
        order, state = await run_graph(graph)
        self.assertEqual(order, ["venue", "flights", "venue", "output"])
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("already tried", state["output"].lower())

    async def test_playlist_failure_returns_partial_result_without_restarting(self):
        async def failed_playlist(state):
            raise RuntimeError("Playlist database unavailable")

        graph = planner.build_wedding_graph(
            good_venue, good_flights, failed_playlist
        )
        order, state = await run_graph(graph)
        self.assertEqual(order, ["venue", "flights", "playlist", "output"])
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("Garden Hall", state["output"])
        self.assertIn("GBP 90", state["output"])
        self.assertIn("Playlist database unavailable", state["output"])

    def test_invalid_limits_are_rejected_before_running(self):
        for options in [{"max_venue_attempts": 0}, {"step_timeout": 0}]:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    planner.build_wedding_graph(
                        good_venue, good_flights, good_playlist, **options
                    )


class SpecialistBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def test_playlist_is_built_from_genre_rows_with_exact_totals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "playlist.db"
            with sqlite3.connect(path) as db:
                db.executescript("""
                    CREATE TABLE Genre (GenreId INTEGER, Name TEXT);
                    CREATE TABLE Artist (ArtistId INTEGER, Name TEXT);
                    CREATE TABLE Album (AlbumId INTEGER, ArtistId INTEGER);
                    CREATE TABLE Track (
                        TrackId INTEGER, Name TEXT, AlbumId INTEGER,
                        GenreId INTEGER, Milliseconds INTEGER, UnitPrice REAL
                    );
                    INSERT INTO Genre VALUES (1, 'Jazz'), (2, 'Rock');
                    INSERT INTO Artist VALUES (1, 'Jazz Artist'), (2, 'Rock Artist');
                    INSERT INTO Album VALUES (1, 1), (2, 2);
                    INSERT INTO Track VALUES
                        (1, 'Jazz One', 1, 1, 120000, 0.99),
                        (2, 'Jazz Two', 1, 1, 180000, 1.29),
                        (3, 'Rock One', 2, 2, 90000, 0.99);
                """)
            result = planner.build_playlist("jazz", path)
            self.assertTrue(result.success)
            self.assertIn("Jazz One", result.details)
            self.assertIn("Jazz Two", result.details)
            self.assertIn("Jazz Artist", result.details)
            self.assertIn("0:05:00", result.details)
            self.assertIn("USD 2.28", result.details)
            self.assertNotIn("Rock One", result.details)
            self.assertFalse(planner.build_playlist("' OR 1=1 --", path).success)

    async def test_main_can_be_awaited_inside_the_interactive_event_loop(self):
        request_agent = planner.create_request_agent(model=ScriptedToolModel(
            messages=iter([AIMessage(content="", tool_calls=[{
                "name": "update_state", "args": REQUEST, "id": "save-request",
            }])]),
        ))
        output = io.StringIO()
        with (
            patch.object(
                planner, "create_live_workers",
                return_value=(good_venue, good_flights, good_playlist),
            ),
            patch.object(planner, "create_request_agent", return_value=request_agent),
            patch("dotenv.load_dotenv", return_value=False),
            redirect_stdout(output),
        ):
            result = await planner.main(
                message="London to Paris, 100 guests, jazz, 12 June 2027",
                session_id="interactive-example",
            )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["request"], REQUEST)
        self.assertIn("Wedding plan complete", output.getvalue())

    async def test_main_stops_at_its_total_time_limit(self):
        request_agent = planner.create_request_agent(model=ScriptedToolModel(
            messages=iter([AIMessage(content="", tool_calls=[{
                "name": "update_state", "args": REQUEST, "id": "save-request",
            }])]),
        ))

        async def hangs(state):
            await asyncio.Event().wait()

        with (
            patch.object(planner, "create_live_workers",
                         return_value=(hangs, good_flights, good_playlist)),
            patch.object(planner, "create_request_agent", return_value=request_agent),
            patch("dotenv.load_dotenv", return_value=False),
            redirect_stdout(io.StringIO()),
        ):
            result = await asyncio.wait_for(planner.main(timeout_seconds=0.02), 2)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("time limit", result["output"].lower())

    async def test_claimed_success_needs_real_successful_tool_evidence(self):
        class AgentReply:
            def __init__(self, messages):
                self.messages = messages

            async def ainvoke(self, inputs, config):
                return {
                    "structured_response": {
                        "success": True, "details": "Flight found."
                    },
                    "messages": self.messages,
                }

        for messages in [
            [],
            [ToolMessage("search failed", name="flight_search",
                         tool_call_id="1", status="error")],
            [ToolMessage("format accepted", name="SearchResult", tool_call_id="1")],
        ]:
            with self.subTest(messages=messages):
                with self.assertRaises(ValueError):
                    await planner.ask_specialist(
                        AgentReply(messages), "Find a flight.",
                        planner.SearchResult, {"flight_search"},
                    )
        result = await planner.ask_specialist(
            AgentReply([
                ToolMessage("LHR to CDG: GBP 90", name="flight_search", tool_call_id="1")
            ]),
            "Find a flight.", planner.SearchResult, {"flight_search"},
        )
        self.assertTrue(result.success)

    def test_playlist_queries_read_rows_but_cannot_change_the_database(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "playlist.db"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE Track (Name TEXT, UnitPrice REAL)")
                db.execute("INSERT INTO Track VALUES ('Jazz tune', 0.99)")
            result = json.loads(
                planner.query_playlist("SELECT Name, UnitPrice FROM Track", path)
            )
            self.assertEqual(result["rows"], [["Jazz tune", 0.99]])
            with self.assertRaises(sqlite3.OperationalError):
                planner.query_playlist("DELETE FROM Track", path)
            remaining = json.loads(
                planner.query_playlist("SELECT COUNT(*) FROM Track", path)
            )
            self.assertEqual(remaining["rows"], [[1]])

    def test_missing_playlist_database_is_not_silently_created(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.db"
            with self.assertRaises(sqlite3.OperationalError):
                planner.query_playlist("SELECT 1", path)
            self.assertFalse(path.exists())


class ScriptedToolModel(GenericFakeChatModel):
    """Replace model I/O while keeping LangChain's real tool loop and middleware."""

    generated_count: int = 0
    seen_prompts: list[str] = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.generated_count += 1
        human_messages = [msg.content for msg in messages if msg.type == "human"]
        if human_messages:
            self.seen_prompts.append(human_messages[-1])
        return super()._generate(messages, *args, **kwargs)


class SpecialistBudgetTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def search_call(number):
        return {
            "name": "web_search",
            "args": {"query": f"Paris venue details {number}"},
            "id": f"search-{number}",
        }

    async def run_live_venue_with_scripted_model(self, responses):
        model = ScriptedToolModel(messages=iter(responses))
        search = AsyncMock(return_value={
            "results": [{
                "title": "Garden Hall",
                "url": "https://example.test/garden-hall",
                "content": "Paris venue with capacity 100, near CDG airport.",
            }]
        })
        with (
            patch.dict(os.environ, {
                "OPENROUTER_KEY": "local-check-only",
                "TAVILY_API_KEY": "local-check-only",
            }),
            patch("langchain.chat_models.init_chat_model", return_value=model),
            patch("tavily.AsyncTavilyClient.search", new=search),
        ):
            find_venue, _, _ = planner.create_live_workers()
            graph = planner.build_wedding_graph(
                find_venue, good_flights, good_playlist
            )
            order, state = await run_graph(graph)
        return order, state, search.await_count, model.generated_count

    def final_venue(self):
        return AIMessage(content="", tool_calls=[{
            "name": "VenueResearch", "args": venue_research(), "id": "final-answer"
        }])

    async def test_unknown_date_continues_to_flights_and_stays_clear_in_output(self):
        responses = [
            AIMessage(content="", tool_calls=[self.search_call(0)]),
            AIMessage(content="", tool_calls=[{
                "name": "VenueResearch",
                "args": venue_research(
                    details="Capacity 100 is advertised. The date could not be confirmed.",
                ),
                "id": "researched-venue",
            }]),
        ]
        order, state, searches, _ = await self.run_live_venue_with_scripted_model(responses)
        self.assertEqual(order, ["venue", "flights", "playlist", "output"])
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["venue"].date_availability, "unknown")
        self.assertIn("Date availability: Not confirmed", state["output"])
        self.assertIn("https://example.test/garden-hall", state["output"])
        self.assertEqual(searches, 1)

    async def test_unknown_date_does_not_excuse_an_unsuitable_or_unsourced_venue(self):
        for changes in [
            {"advertised_capacity": 99},
            {"advertised_capacity": None},
            {"matches_destination": False},
            {"source_urls": []},
            {"venue_name": ""},
            {"arrival_airport": ""},
        ]:
            with self.subTest(changes=changes):
                responses = [
                    AIMessage(content="", tool_calls=[self.search_call(0)]),
                    AIMessage(content="", tool_calls=[{
                        "name": "VenueResearch", "args": venue_research(**changes),
                        "id": "unsuitable-venue",
                    }]),
                ]
                order, state, _, _ = await self.run_live_venue_with_scripted_model(responses)
                self.assertEqual(order, ["venue", "output"])
                self.assertEqual(state["status"], "incomplete")

    async def test_known_unavailable_date_stops_before_flights(self):
        responses = [
            AIMessage(content="", tool_calls=[self.search_call(0)]),
            AIMessage(content="", tool_calls=[{
                "name": "VenueResearch",
                "args": venue_research(date_availability="unavailable"),
                "id": "unavailable-venue",
            }]),
        ]
        order, state, _, _ = await self.run_live_venue_with_scripted_model(responses)
        self.assertEqual(order, ["venue", "output"])
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("Date availability: Unavailable", state["output"])

    async def test_text_boolean_can_be_corrected_without_repeating_venue_search(self):
        responses = [
            AIMessage(content="", tool_calls=[self.search_call(0)]),
            AIMessage(content="", tool_calls=[{
                "name": "VenueResearch",
                "args": venue_research(matches_destination="true"),
                "id": "bad-answer",
            }]),
            self.final_venue(),
        ]
        order, state, searches, model_calls = await self.run_live_venue_with_scripted_model(responses)
        self.assertEqual(order, ["venue", "flights", "playlist", "output"])
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["attempts"], 1)
        self.assertEqual(searches, 1)
        self.assertEqual(model_calls, 3)

    async def test_corrected_false_answer_still_stops_before_flights(self):
        responses = [
            AIMessage(content="", tool_calls=[self.search_call(0)]),
            AIMessage(content="", tool_calls=[{
                "name": "VenueResearch",
                "args": venue_research(
                    matches_destination="false", details="No suitable venue found.",
                ),
                "id": "bad-answer",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "VenueResearch",
                "args": venue_research(
                    matches_destination=False, details="No suitable venue found.",
                ),
                "id": "corrected-answer",
            }]),
        ]
        order, state, searches, model_calls = await self.run_live_venue_with_scripted_model(responses)
        self.assertEqual(order, ["venue", "output"])
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("No suitable venue found.", state["output"])
        self.assertEqual(searches, 1)
        self.assertEqual(model_calls, 3)

    async def test_repeated_bad_answer_format_stops_at_model_limit(self):
        responses = [
            AIMessage(content="", tool_calls=[self.search_call(0)]),
            *[
                AIMessage(content="", tool_calls=[{
                    "name": "VenueResearch",
                    "args": venue_research(matches_destination="true"),
                    "id": f"bad-answer-{number}",
                }])
                for number in range(10)
            ],
        ]
        order, state, searches, model_calls = await self.run_live_venue_with_scripted_model(responses)
        self.assertEqual(order, ["venue", "output"])
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["attempts"], 1)
        self.assertEqual(searches, 1)
        self.assertEqual(model_calls, 6)

    async def test_four_parallel_searches_leave_room_for_the_structured_answer(self):
        responses = [
            AIMessage(content="", tool_calls=[self.search_call(i) for i in range(4)]),
            self.final_venue(),
        ]
        order, state, searches, _ = await self.run_live_venue_with_scripted_model(responses)
        self.assertEqual(order, ["venue", "flights", "playlist", "output"])
        self.assertEqual(state["status"], "complete")
        self.assertEqual(searches, 4)

    async def test_four_sequential_searches_fit_within_the_graph_step_budget(self):
        responses = [
            AIMessage(content="", tool_calls=[self.search_call(i)]) for i in range(4)
        ] + [self.final_venue()]
        order, state, searches, _ = await self.run_live_venue_with_scripted_model(responses)
        self.assertEqual(order, ["venue", "flights", "playlist", "output"])
        self.assertEqual(state["status"], "complete")
        self.assertEqual(searches, 4)

    async def test_fifth_search_is_blocked_but_agent_can_finish_with_existing_results(self):
        responses = [
            AIMessage(content="", tool_calls=[self.search_call(i)]) for i in range(5)
        ] + [self.final_venue()]
        order, state, searches, model_calls = await self.run_live_venue_with_scripted_model(responses)
        self.assertEqual(order, ["venue", "flights", "playlist", "output"])
        self.assertEqual(state["status"], "complete")
        self.assertEqual(searches, 4)
        self.assertEqual(model_calls, 6)

    async def test_agent_ignoring_stop_instruction_hits_model_limit_then_exits(self):
        responses = [
            AIMessage(content="", tool_calls=[self.search_call(i)]) for i in range(10)
        ]
        order, state, searches, model_calls = await self.run_live_venue_with_scripted_model(responses)
        self.assertEqual(order, ["venue", "output"])
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(searches, 4)
        self.assertEqual(model_calls, 6)


class VenueExclusionTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_flight_exclusions_reach_model_prompt_and_actual_web_query(self):
        # The model deliberately repeats the same query without exclusions.
        # Production code must add the failed venue to the outgoing search.
        query = "Paris wedding venue for 100 guests"
        responses = []
        for number, candidate in enumerate([
            venue_research(), venue_research("River Hall", "ORY"),
        ]):
            responses.extend([
                AIMessage(content="", tool_calls=[{
                    "name": "web_search", "args": {"query": query},
                    "id": f"search-{number}",
                }]),
                AIMessage(content="", tool_calls=[{
                    "name": "VenueResearch", "args": candidate,
                    "id": f"answer-{number}",
                }]),
            ])
        model = ScriptedToolModel(messages=iter(responses))
        search = AsyncMock(return_value={
            "results": [
                {"title": "Garden Hall", "content": "Paris venue, 100 guests, near CDG.",
                 "url": "https://example.test/garden"},
                {"title": "River Hall", "content": "Paris venue, 100 guests, near ORY.",
                 "url": "https://example.test/river"},
            ]
        })

        async def flights(state):
            if state["venue"].venue_name == "Garden Hall":
                return {"success": False, "details": "No seats to CDG."}
            return {"success": True, "details": "Flight to ORY, GBP 110."}

        with (
            patch.dict(os.environ, {
                "OPENROUTER_KEY": "local-check-only",
                "TAVILY_API_KEY": "local-check-only",
            }),
            patch("langchain.chat_models.init_chat_model", return_value=model),
            patch("tavily.AsyncTavilyClient.search", new=search),
        ):
            find_venue, _, _ = planner.create_live_workers()
            graph = planner.build_wedding_graph(find_venue, flights, good_playlist)
            order, state = await run_graph(graph)

        self.assertEqual(
            order, ["venue", "flights", "venue", "flights", "playlist", "output"]
        )
        self.assertEqual(state["status"], "complete")
        retry_prompt = json.loads(model.seen_prompts[2])
        self.assertEqual(retry_prompt["rejected_venues"], ["Garden Hall"])
        self.assertIn("No seats", retry_prompt["last_flight_or_venue_error"])
        self.assertEqual(search.await_args_list[0].args[0], query)
        retry_query = search.await_args_list[1].args[0]
        self.assertIn('-"Garden Hall"', retry_query)
        self.assertNotIn('-"River Hall"', retry_query)


class RunBlockTests(unittest.TestCase):
    def run_block(self, main, *, notebook=False):
        # Execute the real launch block, replacing only the paid planner call.
        block = ast.Module(body=[ast.parse(SCRIPT.read_text()).body[-1]], type_ignores=[])
        namespace = {
            "__name__": "__main__",
            "asyncio": asyncio,
            "sys": SimpleNamespace(modules={"ipykernel": object()} if notebook else {}),
            "main": main,
        }
        exec(compile(block, str(SCRIPT), "exec"), namespace)
        return namespace

    def test_terminal_runs_main_and_keeps_its_result(self):
        async def main():
            return {"status": "complete"}

        namespace = self.run_block(main)
        self.assertEqual(namespace.get("result"), {"status": "complete"})

    def test_running_loop_shows_the_await_command_without_starting_another_loop(self):
        def unexpected_main():
            self.fail("The run block should show how to await main in this event loop.")

        async def check():
            output = io.StringIO()
            with redirect_stdout(output):
                self.run_block(unexpected_main, notebook=True)
            self.assertIn("result = await main()", output.getvalue())

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
