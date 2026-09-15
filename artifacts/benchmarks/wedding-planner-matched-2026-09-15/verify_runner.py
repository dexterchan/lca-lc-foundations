"""Offline probes for the comparison harness; no model or search calls."""

import ast
import asyncio
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
from uuid import uuid4

from langchain.messages import AIMessage

import runner


async def verify():
    for variant, source in runner.SOURCES.items():
        with tempfile.TemporaryDirectory(prefix="matched-wedding-probe-") as directory:
            folder = Path(directory)
            work = folder / "workdir"
            (work / "resources").mkdir(parents=True)
            shutil.copy2(source, work / source.name)
            shutil.copy2(runner.DATABASE, work / "resources/Chinook.db")

            class FakeApp:
                async def ainvoke(self, request, **kwargs):
                    if variant == "original":
                        assert request["messages"][0].content == runner.REQUEST
                    else:
                        assert request == runner.REQUEST
                        assert kwargs["timeout_seconds"] == runner.LIMIT
                    return {
                        "messages": [AIMessage(content="probe answer")],
                        "status": "complete", "output": "probe answer",
                    }

            async def setup(*args):
                return FakeApp()

            with patch.object(runner, "original_setup", setup), \
                    patch.object(runner, "graph_setup", setup):
                await runner.worker(folder, variant)
            events = runner.read_events(folder)
            terminal = next(e for e in events if e["kind"] == "run_end")
            assert terminal["status"] == ("returned" if variant == "original" else "complete")
            assert (folder / "answer.txt").read_text() == "probe answer"
            assert (folder / "messages.json").is_file()
            # Incomplete last records must not crash the live progress reader.
            with (folder / "events.jsonl").open("ab") as stream:
                stream.write(b'{"kind":')
            assert runner.read_events(folder) == events

    compile(runner.original_setup_nodes(runner.SOURCES["original"]),
            str(runner.SOURCES["original"]), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    with tempfile.TemporaryDirectory(prefix="matched-wedding-timing-") as directory:
        recorder = runner.EvidenceRecorder(Path(directory))
        parent, model = uuid4(), uuid4()
        recorder.tools[parent] = "search_flights"
        recorder.parents[model] = parent
        assert recorder.stage(model) == "search_flights"
        node = uuid4()
        recorder.nodes[node] = "venue"
        assert recorder.stage(node) == "venue"
    print("Offline probes passed: request, limits, answers, timing stages, partial log reads.")


if __name__ == "__main__":
    asyncio.run(verify())
