"""Compare both planners with one request, matched time budgets, and saved evidence.

Run from the project root. Production source files are never edited.
"""

import argparse
import ast
import asyncio
from collections import Counter
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import types
from unittest.mock import patch

from timing_helpers import Recorder, digest, load_module
from summary_helpers import summarize


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCES = {
    "original": ROOT / "notebooks/module-2/2.4_wedding_planners.py",
    "langgraph": ROOT / "notebooks/module-2/2.4_wedding_planners_langgraph.py",
}
DATABASE = SOURCES["original"].parent / "resources/Chinook.db"
REQUEST = (
    "I'm from London and want a wedding in Paris for 100 guests, "
    "with jazz music. Travel date: 2027-03-14."
)
LIMIT = 300
OUTER_LIMIT = 330
OLD_STAGES = {"update_state", "search_flights", "search_venues", "suggest_playlist"}


def read_events(folder):
    path = folder / "events.jsonl"
    if not path.exists():
        return []
    # The child may still be appending a large tool reply when we take a snapshot.
    lines = path.read_bytes().split(b"\n")[:-1]
    return [json.loads(line) for line in lines if line]


class EvidenceRecorder(Recorder):
    def stage(self, parent):
        seen = set()
        while parent is not None and parent not in seen:
            seen.add(parent)
            if parent in self.nodes:
                return self.nodes[parent]
            if self.tools.get(parent) in OLD_STAGES:
                return self.tools[parent]
            parent = self.parents.get(parent)
        return "coordinator"

    def on_chat_model_start(self, serialized, messages, **kwargs):
        super().on_chat_model_start(serialized, messages, **kwargs)
        content = "\n".join(str(msg.content) for batch in messages for msg in batch)
        self.emit(
            "model_input", id=str(kwargs["run_id"]),
            stage=self.stage(kwargs.get("parent_run_id")),
            characters=len(content), requested_date_present="2027-03-14" in content,
        )

    def on_llm_end(self, response, *, run_id, **kwargs):
        super().on_llm_end(response, run_id=run_id, **kwargs)
        for batch in response.generations:
            for generation in batch:
                message = generation.message
                self.emit(
                    "model_reply", id=str(run_id),
                    stage=self.stage(self.parents.get(run_id)),
                    content=message.content,
                    tool_calls=getattr(message, "tool_calls", []),
                    response_metadata=getattr(message, "response_metadata", {}),
                )


def original_setup_nodes(source):
    tree = ast.parse(source.read_text(), filename=str(source))
    split = next(
        i for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "response"
                for target in node.targets)
    )
    return ast.Module(body=tree.body[:split], type_ignores=[])


async def original_setup(source, recorder):
    module = types.ModuleType("matched_original")
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    code = compile(original_setup_nodes(source), str(source), "exec",
                   flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(code, vars(module))
    if inspect.isawaitable(result):
        await result
    return module.coordinator.with_config(callbacks=[recorder])


async def graph_setup(source, recorder):
    planner = load_module("matched_graph", source)
    real_query = planner.query_playlist

    def timed_query(*args, **kwargs):
        started = time.perf_counter()
        recorder.emit("sql_start")
        try:
            result = real_query(*args, **kwargs)
            recorder.emit("sql_end", duration_s=time.perf_counter() - started,
                          data=json.loads(result))
            return result
        except Exception as error:
            recorder.emit("sql_error", error=f"{type(error).__name__}: {error}")
            raise

    planner.query_playlist = timed_query
    # A step cannot expire before the common full-plan deadline.
    app = planner.WeddingPlanner(
        *planner.create_live_workers(), request_agent=planner.create_request_agent(),
        step_timeout=LIMIT,
    )
    app.graph = app.graph.with_config(callbacks=[recorder])
    return app


async def worker(folder, variant):
    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model
    from langchain.messages import HumanMessage
    from langchain_core.messages import messages_to_dict
    from tavily import TavilyClient

    load_dotenv(ROOT / ".env")
    recorder = EvidenceRecorder(folder)
    source = folder / "workdir" / SOURCES[variant].name
    database = source.parent / "resources/Chinook.db"
    recorder.emit("run_start", variant=variant, utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=digest(source), database_sha256=digest(database))

    def matched_model(*args, **kwargs):
        # Both use the same model and transport wait/retry settings.
        kwargs.update(timeout=LIMIT, max_retries=0)
        return init_chat_model(*args, **kwargs)

    def matched_tavily(*args, **kwargs):
        kwargs["timeout"] = 30  # Matches the graph's AsyncTavilyClient.
        return TavilyClient(*args, **kwargs)

    phase = "setup"
    try:
        with patch("langchain.chat_models.init_chat_model", matched_model), \
                patch("tavily.TavilyClient", matched_tavily):
            async with asyncio.timeout(60):
                app = await (
                    original_setup(source, recorder) if variant == "original"
                    else graph_setup(source, recorder)
                )
            recorder.emit("setup_end")
            phase = "planner"
            recorder.emit("planner_start")
            if variant == "original":
                async with asyncio.timeout(LIMIT):
                    state = await app.ainvoke(
                        {"messages": [HumanMessage(content=REQUEST)]},
                        config={"tags": ["matched-benchmark"], "recursion_limit": 40},
                    )
                answer = state["messages"][-1].content
                status = "returned"
            else:
                state = await app.ainvoke(
                    REQUEST, session_id=folder.name, timeout_seconds=LIMIT,
                )
                answer = state["output"]
                status = state["status"]
            serial = {
                key: value.model_dump() if hasattr(value, "model_dump") else value
                for key, value in state.items() if key != "messages"
            }
            (folder / "state.json").write_text(json.dumps(serial, indent=2) + "\n")
            (folder / "messages.json").write_text(json.dumps(
                messages_to_dict(state.get("messages", [])), indent=2, default=str,
            ) + "\n")
            (folder / "answer.txt").write_text(
                answer if isinstance(answer, str) else json.dumps(answer, indent=2)
            )
            recorder.emit("run_end", status=status, state=serial)
    except TimeoutError:
        recorder.emit("run_end", status="timeout", phase=phase)
    except Exception as error:
        traceback.print_exc()
        recorder.emit("run_end", status="error", phase=phase,
                      error=f"{type(error).__name__}: {error}")
    finally:
        recorder.emit("database_end", database_sha256=digest(database))


def manifest():
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request": REQUEST, "planner_timeout_s": LIMIT,
        "step_timeout_s": LIMIT,
        "model_transport_timeout_s": LIMIT, "model_max_retries": 0,
        "tavily_timeout_s": 30, "outer_process_cap_s": OUTER_LIMIT,
        "sources": {key: {"path": str(path), "sha256": digest(path)}
                    for key, path in SOURCES.items()},
        "database_sha256": digest(DATABASE),
        "order": ["original", "langgraph", "langgraph", "original", "original", "langgraph"],
        "controlled": [
            "Identical user message including a fixed future travel date.",
            "300 seconds for planning; no earlier individual step deadline.",
            "Same model, 300-second transport timeout, zero model retries.",
            "Both Tavily clients have a 30-second timeout.",
            "Fresh source and database copies, process, and session each run.",
            "Same evidence review; returned text and success flags alone do not pass.",
        ],
        "remaining_differences": [
            "Original specialists may overlap; graph uses venue then flights then SQL.",
            "Original discards the requested date when delegating flights; left unchanged.",
            "Original prompts and suggested 12 web searches vs graph hard 4 calls per tool.",
            "Graph has six model calls per specialist; original keeps its own graph limits.",
            "Original MCP error retries retained; graph retains its own flight retry edge.",
            "Original outputs text; graph validates a structured result.",
            "Original discovers flight tools during setup; graph discovers them during planning.",
            "Original playlist agent may write to its private database copy.",
        ],
        "interpretation": (
            "Comparison under matched time budgets, not a controlled test of the "
            "LangGraph framework. Agent prompts, call budgets and workflow still differ."
        ),
        "common_pass_criteria": {
            "venue": "Named Paris venue, evidence of room for 100 guests, and a source; no invented confirmed availability.",
            "flight": "An actual tool result supporting a priced one-way London-to-Paris flight on 2027-03-14, with a source.",
            "playlist": "Named jazz tracks supported by Chinook, with duration and cost totals matching the chosen tracks.",
            "overall": "All three evidenced parts present in the final answer. Review manually against saved tool outputs.",
        },
    }


def stop_child(process):
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_trials():
    info = manifest()
    # Never silently append to or replace an earlier experiment.
    with (HERE / "manifest.json").open("x") as stream:
        stream.write(json.dumps(info, indent=2) + "\n")
    (HERE / "request.txt").write_text(REQUEST + "\n")
    results = []
    counts = Counter()
    for sequence, variant in enumerate(info["order"], 1):
        counts[variant] += 1
        folder = HERE / f"{sequence:02d}-{variant}-{counts[variant]}"
        workdir = folder / "workdir"
        (workdir / "resources").mkdir(parents=True)
        source = SOURCES[variant]
        shutil.copy2(source, workdir / source.name)
        shutil.copy2(DATABASE, workdir / "resources/Chinook.db")
        assert digest(workdir / source.name) == info["sources"][variant]["sha256"]
        assert digest(workdir / "resources/Chinook.db") == info["database_sha256"]
        print(f"{sequence}/6: {folder.name} started; planning cap {LIMIT}s.", flush=True)
        started = time.perf_counter()
        timed_out = False
        with (folder / "console.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()),
                 "--worker", str(folder), "--variant", variant],
                cwd=workdir, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            (folder / "process.json").write_text(json.dumps({
                "parent_pid": os.getpid(), "worker_pid": process.pid,
            }) + "\n")
            while process.poll() is None:
                if time.perf_counter() - started > OUTER_LIMIT:
                    timed_out = True
                    stop_child(process)
                    break
                # A timed-out original can leave sync tool threads running.
                # Once its timeout is saved, stop this trial's own process group.
                terminal = next((e for e in reversed(read_events(folder))
                                 if e["kind"] == "run_end"), {})
                if terminal.get("status") == "timeout":
                    stop_child(process)
                    break
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        events = read_events(folder)
        result = summarize(folder, time.perf_counter() - started, timed_out, process.returncode)
        result["variant"] = variant
        result["node_spans"] = [
            e for e in events if e["kind"] in {"node_end", "node_error"}
        ]
        result["sql_calls"] = sum(e["kind"] == "sql_start" for e in events)
        result["copied_database_changed"] = (
            digest(workdir / "resources/Chinook.db") != info["database_sha256"]
        )
        result["evidence_review"] = "pending"
        results.append(result)
        (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        final = {
            **info, "results": results,
            "production_sources_unchanged": all(
                digest(path) == info["sources"][key]["sha256"]
                for key, path in SOURCES.items()
            ),
            "production_database_unchanged": digest(DATABASE) == info["database_sha256"],
        }
        (HERE / "results.json").write_text(json.dumps(final, indent=2) + "\n")
        print(f"{folder.name}: {result['status']}, {result['wall_s']:.1f}s, "
              f"{result['model_calls']} AI calls.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--variant", choices=SOURCES)
    args = parser.parse_args()
    if args.worker:
        asyncio.run(worker(args.worker, args.variant))
    else:
        run_trials()
