"""Exercise real state-update tools and concurrent graphs without paid calls."""

import asyncio
import copy
import unittest

from langchain.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from test_wedding_planners_langgraph import (
    REQUEST, ScriptedToolModel, good_flights, good_venue, planner, venue,
)


class RequestModel(ScriptedToolModel):
    """Supply tool calls at the model boundary, keeping state writes real."""

    updates: dict[str, dict | None]
    conversations: list[list[str]] = []
    received_texts: list[list[str]] = []

    def _generate(self, messages, *args, **kwargs):
        self.received_texts.append([msg.content for msg in messages])
        conversation = [msg.content for msg in messages if msg.type == "human"]
        self.conversations.append(conversation)
        fields = self.updates[conversation[-1]]
        reply = AIMessage(content="Please give the wedding details.")
        if fields is not None:
            reply = AIMessage(content="", tool_calls=[{
                "name": "update_state",
                "args": copy.deepcopy(fields),
                "id": f"update-{len(self.conversations)}",
            }])
        return ChatResult(generations=[ChatGeneration(message=reply)])


async def playlist_from_request(state):
    return {"success": True, "details": f"{state['request']['genre']} playlist"}


def session_planner(updates, *, find_venue=good_venue, find_flights=good_flights,
                    make_playlist=playlist_from_request, **options):
    model = RequestModel(messages=iter([]), updates=updates)
    app = planner.WeddingPlanner(
        find_venue, find_flights, make_playlist,
        request_agent=planner.create_request_agent(model=model), **options,
    )
    return app, model


class WeddingSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_follow_up_agent_receives_previous_plan_in_chat_history(self):
        app, model = session_planner({
            "First plan": REQUEST, "Keep that music, but 80 guests": {"guest_count": 80},
        })
        first = await app.ainvoke("First plan", session_id="alice")
        second = await app.ainvoke("Keep that music, but 80 guests", session_id="alice")
        self.assertIn(first["output"], model.received_texts[-1])
        self.assertEqual(second["request"], {**REQUEST, "guest_count": 80})

    async def test_cancellation_saves_stopped_state_and_releases_session(self):
        started = asyncio.Event()

        async def flights(state):
            if state["request"]["genre"] == "jazz":
                started.set()
                await asyncio.Event().wait()
            return await good_flights(state)

        app, _ = session_planner(
            {"First plan": REQUEST, "Try rock": {"genre": "rock"}},
            find_flights=flights,
        )
        task = asyncio.create_task(app.ainvoke("First plan", session_id="alice"))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        saved = await app.aget_state("alice")
        self.assertEqual(saved["status"], "cancelled")
        self.assertIn("cancelled", saved["output"].lower())
        self.assertEqual(saved["venue"].venue_name, "Garden Hall")
        self.assertEqual(saved["messages"][-1].content, saved["output"])
        resumed = await app.ainvoke("Try rock", session_id="alice")
        self.assertEqual(resumed["status"], "complete")

    async def test_update_state_tool_saves_request_before_searches(self):
        seen_requests = []

        async def find_venue(state):
            seen_requests.append(copy.deepcopy(state["request"]))
            return venue()

        app, _ = session_planner(
            {"London to Paris, 100 guests, jazz, 12 June 2027": REQUEST},
            find_venue=find_venue,
        )
        result = await app.ainvoke(
            "London to Paris, 100 guests, jazz, 12 June 2027", session_id="alice"
        )
        self.assertEqual(seen_requests, [REQUEST])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["request"], REQUEST)
        self.assertIn("jazz playlist", result["output"])
        updates = [
            msg for msg in result["messages"]
            if isinstance(msg, ToolMessage) and msg.name == "update_state"
        ]
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].status, "success")
        self.assertEqual((await app.aget_state("alice"))["request"], REQUEST)

    async def test_missing_fields_wait_then_follow_up_keeps_saved_fields(self):
        seen_requests = []

        async def find_venue(state):
            seen_requests.append(dict(state["request"]))
            return venue()

        app, _ = session_planner({
            "I'm from London": {"origin": "London"},
            "Paris, 100 guests, jazz, 12 June 2027": {
                key: value for key, value in REQUEST.items() if key != "origin"
            },
        }, find_venue=find_venue)
        partial = await app.ainvoke("I'm from London", session_id="alice")
        self.assertEqual(partial["status"], "awaiting_input")
        self.assertEqual(partial["request"], {"origin": "London"})
        self.assertIn("genre", partial["output"])
        self.assertEqual(seen_requests, [])

        complete = await app.ainvoke(
            "Paris, 100 guests, jazz, 12 June 2027", session_id="alice"
        )
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["request"], REQUEST)
        self.assertEqual(seen_requests, [REQUEST])

    async def test_sessions_can_wait_at_different_nodes_without_mixing_state(self):
        alice_in_flights = asyncio.Event()
        bob_in_venue = asyncio.Event()
        finish = asyncio.Event()
        tokyo_request = {
            "origin": "Hong Kong", "destination": "Tokyo", "guest_count": 20,
            "genre": "rock", "travel_date": "2027-07-01",
        }

        async def find_venue(state):
            if state["request"]["destination"] == "Tokyo":
                bob_in_venue.set()
                await finish.wait()
                return venue("Tokyo Hall", "HND")
            return venue("Paris Hall", "CDG")

        async def find_flights(state):
            if state["request"]["destination"] == "Paris":
                alice_in_flights.set()
                await finish.wait()
            return {"success": True, "details": state["venue"].arrival_airport}

        app, model = session_planner(
            {"Alice's wedding": REQUEST, "Bob's wedding": tokyo_request},
            find_venue=find_venue, find_flights=find_flights,
        )
        async with asyncio.TaskGroup() as tasks:
            alice = tasks.create_task(app.ainvoke("Alice's wedding", session_id="alice"))
            bob = tasks.create_task(app.ainvoke("Bob's wedding", session_id="bob"))
            await asyncio.wait_for(
                asyncio.gather(alice_in_flights.wait(), bob_in_venue.wait()), 2
            )
            alice_state = await app.aget_state("alice")
            bob_state = await app.aget_state("bob")
            self.assertEqual(alice_state["request"], REQUEST)
            self.assertEqual(bob_state["request"], tokyo_request)
            self.assertEqual(alice_state["venue"].venue_name, "Paris Hall")
            self.assertIsNone(bob_state["venue"])
            finish.set()

        self.assertEqual(alice.result()["status"], "complete")
        self.assertEqual(bob.result()["status"], "complete")
        self.assertIn("jazz playlist", alice.result()["output"])
        self.assertIn("rock playlist", bob.result()["output"])
        self.assertCountEqual(
            model.conversations, [["Alice's wedding"], ["Bob's wedding"]]
        )

    async def test_same_session_messages_run_in_order_and_merge_the_request(self):
        first_in_flights = asyncio.Event()
        finish_first = asyncio.Event()
        requests_seen = []

        async def flights(state):
            requests_seen.append(copy.deepcopy(state["request"]))
            if state["request"]["genre"] == "jazz":
                first_in_flights.set()
                await finish_first.wait()
            return await good_flights(state)

        app, model = session_planner(
            {"First plan": REQUEST, "Change music to rock": {"genre": "rock"}},
            find_flights=flights,
        )
        async with asyncio.TaskGroup() as tasks:
            first = tasks.create_task(app.ainvoke("First plan", session_id="alice"))
            await asyncio.wait_for(first_in_flights.wait(), 2)
            second = tasks.create_task(
                app.ainvoke("Change music to rock", session_id="alice")
            )
            await asyncio.sleep(0)  # Let the second caller reach the session lock.
            self.assertEqual(model.conversations, [["First plan"]])
            finish_first.set()

        self.assertEqual(first.result()["request"]["genre"], "jazz")
        self.assertEqual(second.result()["request"], {**REQUEST, "genre": "rock"})
        self.assertEqual(second.result()["attempts"], 1)
        self.assertEqual(requests_seen, [REQUEST, {**REQUEST, "genre": "rock"}])
        self.assertEqual(
            model.conversations[-1], ["First plan", "Change music to rock"]
        )

    async def test_follow_up_resets_old_failures_and_search_results(self):
        async def flights(state):
            if state["request"]["genre"] == "jazz":
                return {"success": False, "details": "No seats."}
            self.assertEqual(state["attempts"], 1)
            self.assertEqual(state["rejected_venues"], [])
            self.assertEqual(state["failures"], [])
            self.assertIsNone(state["playlist"])
            return await good_flights(state)

        app, _ = session_planner(
            {"First plan": REQUEST, "Try rock": {"genre": "rock"}},
            find_flights=flights, max_venue_attempts=1,
        )
        failed = await app.ainvoke("First plan", session_id="alice")
        self.assertEqual(failed["status"], "incomplete")
        self.assertEqual(failed["rejected_venues"], ["Garden Hall"])
        result = await app.ainvoke("Try rock", session_id="alice")
        self.assertEqual(result["status"], "complete")
        self.assertNotIn("No seats.", result["output"])

    async def test_timeout_saves_partial_state_and_other_session_can_finish(self):
        async def flights(state):
            if state["request"]["genre"] == "jazz":
                await asyncio.Event().wait()
            return await good_flights(state)

        app, _ = session_planner(
            {"Slow plan": REQUEST, "Fast plan": {**REQUEST, "genre": "rock"},
             "Try rock": {"genre": "rock"}},
            find_flights=flights,
        )
        slow, fast = await asyncio.wait_for(asyncio.gather(
            app.ainvoke("Slow plan", session_id="alice", timeout_seconds=0.15),
            app.ainvoke("Fast plan", session_id="bob"),
        ), 2)
        self.assertEqual(slow["status"], "incomplete")
        self.assertIn("time limit", slow["output"].lower())
        self.assertEqual((await app.aget_state("alice"))["status"], "incomplete")
        self.assertEqual(fast["status"], "complete")
        recovered = await app.ainvoke("Try rock", session_id="alice")
        self.assertEqual(recovered["status"], "complete")

    async def test_answer_without_update_state_does_not_rerun_old_request(self):
        seen = []

        async def find_venue(state):
            seen.append(dict(state["request"]))
            return venue()

        app, _ = session_planner(
            {"First plan": REQUEST, "Unclear change": None}, find_venue=find_venue,
        )
        await app.ainvoke("First plan", session_id="alice")
        result = await app.ainvoke("Unclear change", session_id="alice")
        self.assertEqual(result["status"], "awaiting_input")
        self.assertIsNone(result["venue"])
        self.assertIsNone(result["playlist"])
        self.assertEqual(seen, [REQUEST])

    async def test_invalid_fields_do_not_write_partial_or_cross_session_state(self):
        for bad_fields in [
            {"guest_count": 0}, {"guest_count": True}, {"genre": " "},
            {"travel_date": "2027-02-30"}, {"session_id": "bob", "genre": "rock"},
        ]:
            with self.subTest(fields=bad_fields):
                app, _ = session_planner({"Bad request": bad_fields})
                result = await app.ainvoke("Bad request", session_id="alice")
                self.assertNotEqual(result["status"], "complete")
                self.assertEqual(result.get("request", {}), {})
                self.assertIsNone(result["venue"])
                self.assertEqual(await app.aget_state("bob"), {})

    async def test_empty_session_id_is_rejected_before_any_model_call(self):
        app, model = session_planner({"First plan": REQUEST})
        with self.assertRaises(ValueError):
            await app.ainvoke("First plan", session_id="")
        self.assertEqual(model.conversations, [])


if __name__ == "__main__":
    unittest.main()
