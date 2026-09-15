"""Benchmark the current session-aware planner with its normal limits."""

import argparse
import asyncio
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback

from langchain_core.callbacks import BaseCallbackHandler


ROOT = Path("/Users/dexterchan/sandbox/agentic/langchain_academy_classwork/lca-lc-foundations")
SOURCE = ROOT / "notebooks/module-2/2.4_wedding_planners_langgraph.py"
DATABASE = SOURCE.parent / "resources/Chinook.db"
ORIGINAL = ROOT / "artifacts/benchmarks/wedding-planner-original-2026-09-15"
NODES = {"collect_request", "venue", "flights", "playlist", "output"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Recorder(BaseCallbackHandler):
    run_inline = True

    def __init__(self, folder):
        self.folder = folder
        self.started = time.perf_counter()
        self.lock = threading.RLock()
        self.parents = {}
        self.nodes = {}
        self.tools = {}
        self.starts = {}

    def emit(self, kind, **fields):
        with self.lock, (self.folder / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "kind": kind,
                "elapsed_s": round(time.perf_counter() - self.started, 6),
                **fields,
            }, default=str) + "\n")

    def stage(self, parent):
        seen = set()
        while parent is not None and parent not in seen:
            seen.add(parent)
            if parent in self.nodes:
                return self.nodes[parent]
            parent = self.parents.get(parent)
        return "other"

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        with self.lock:
            self.parents[run_id] = parent_run_id
            name = kwargs.get("name", (serialized or {}).get("name"))
            if name in NODES:
                self.nodes[run_id] = name
                self.starts[run_id] = time.perf_counter()
                self.emit("node_start", id=str(run_id), name=name)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        with self.lock:
            if run_id not in self.nodes:
                return
            name = self.nodes[run_id]
            result = outputs.get(name) if isinstance(outputs, dict) else None
            self.emit("node_end", id=str(run_id), name=name,
                      duration_s=time.perf_counter() - self.starts[run_id],
                      success=getattr(result, "success", None),
                      details=getattr(result, "details", None),
                      state_status=outputs.get("status") if isinstance(outputs, dict) else None)

    def on_chain_error(self, error, *, run_id, **kwargs):
        with self.lock:
            if run_id in self.nodes:
                self.emit("node_error", id=str(run_id), name=self.nodes[run_id],
                          duration_s=time.perf_counter() - self.starts[run_id],
                          error=f"{type(error).__name__}: {error}")

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        with self.lock:
            self.parents[run_id] = parent_run_id
            self.starts[run_id] = time.perf_counter()
            self.emit("model_start", id=str(run_id), stage=self.stage(parent_run_id))

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = Counter()
        for batch in response.generations:
            for generation in batch:
                data = getattr(generation.message, "usage_metadata", None) or {}
                usage.update({key: data.get(key, 0) for key in (
                    "input_tokens", "output_tokens", "total_tokens",
                )})
        with self.lock:
            self.emit("model_end", id=str(run_id), stage=self.stage(self.parents.get(run_id)),
                      duration_s=time.perf_counter() - self.starts[run_id], usage=dict(usage))

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self.lock:
            self.emit("model_error", id=str(run_id), stage=self.stage(self.parents.get(run_id)),
                      error=f"{type(error).__name__}: {error}")

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        with self.lock:
            name = (serialized or {}).get("name", kwargs.get("name", "unknown"))
            self.parents[run_id] = parent_run_id
            self.tools[run_id] = name
            self.starts[run_id] = time.perf_counter()
            self.emit("tool_start", id=str(run_id), name=name,
                      stage=self.stage(parent_run_id), input=input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        with self.lock:
            self.emit("tool_end", id=str(run_id), name=self.tools.get(run_id),
                      stage=self.stage(run_id),
                      duration_s=time.perf_counter() - self.starts[run_id],
                      tool_status=getattr(output, "status", None),
                      output=str(getattr(output, "content", output)))

    def on_tool_error(self, error, *, run_id, **kwargs):
        with self.lock:
            self.emit("tool_error", id=str(run_id), name=self.tools.get(run_id),
                      stage=self.stage(run_id),
                      duration_s=time.perf_counter() - self.starts[run_id],
                      error=f"{type(error).__name__}: {error}")


async def worker(folder):
    from dotenv import load_dotenv

    recorder = Recorder(folder)
    copied_source = folder / "workdir" / SOURCE.name
    copied_db = copied_source.parent / "resources/Chinook.db"
    recorder.emit("run_start", utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=digest(copied_source), database_sha256=digest(copied_db))
    phase = "setup"
    try:
        load_dotenv(ROOT / ".env")
        planner = load_module("wedding_langgraph_benchmark", copied_source)
        real_query = planner.query_playlist

        def timed_query(*args, **kwargs):
            started = time.perf_counter()
            recorder.emit("sql_start")
            try:
                result = real_query(*args, **kwargs)
                recorder.emit("sql_end", duration_s=time.perf_counter() - started,
                              row_count=len(json.loads(result)["rows"]))
                return result
            except Exception as error:
                recorder.emit("sql_error", duration_s=time.perf_counter() - started,
                              error=f"{type(error).__name__}: {error}")
                raise

        planner.query_playlist = timed_query
        app = planner.create_live_planner()
        app.graph = app.graph.with_config(callbacks=[recorder])
        recorder.emit("setup_end")
        phase = "planner"
        recorder.emit("planner_start")
        message = (folder.parent / "request.txt").read_text()
        state = await planner.main(message=message, app=app, session_id=folder.name)
        serial_state = {
            key: value.model_dump() if hasattr(value, "model_dump") else value
            for key, value in state.items() if key != "messages"
        }
        (folder / "state.json").write_text(json.dumps(serial_state, indent=2) + "\n")
        (folder / "answer.txt").write_text(state["output"] + "\n")
        recorder.emit("run_end", status=state["status"], state=serial_state,
                      copied_database_unchanged=digest(copied_db) == json.loads(
                          (folder.parent / "manifest.json").read_text()
                      )["database_sha256"])
    except Exception as error:
        traceback.print_exc()
        recorder.emit("run_end", status="error", phase=phase,
                      error=f"{type(error).__name__}: {error}")


def run_trials(args):
    base = load_module("original_benchmark_helpers", ORIGINAL / "runner.py")
    args.output.mkdir(parents=True, exist_ok=False)
    travel_date = (date.today() + timedelta(days=180)).isoformat()
    request = (
        "I'm from London and want a wedding in Paris for 100 guests, "
        f"with jazz music. Travel date: {travel_date}."
    )
    (args.output / "request.txt").write_text(request)
    manifest = {
        "source": str(SOURCE), "source_sha256": digest(SOURCE),
        "database_sha256": digest(DATABASE), "prompt": request,
        "planner_timeout_s": 110, "step_timeout_s": 90,
        "outer_process_cap_s": args.timeout, "runs": args.runs, "sequential": True,
        "model": "deepseek/deepseek-v4-flash-0731",
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: version(name) for name in (
            "langchain", "langgraph", "langchain-deepseek", "mcp", "tavily-python",
        )},
        "method": (
            "Unchanged source copied beside a fresh database for every trial. "
            "Run main() with its normal limits and built-in example, freezing the "
            "example date across trials. Timing callbacks record graph nodes, "
            "models and tools; query_playlist is wrapped only to measure SQL time. "
            "Every trial gets a fresh process and session. A partial return is "
            "incomplete, even if the process exits normally."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for number in range(1, args.runs + 1):
        folder = args.output / f"run-{number}"
        workdir = folder / "workdir"
        (workdir / "resources").mkdir(parents=True)
        shutil.copy2(SOURCE, workdir / SOURCE.name)
        shutil.copy2(DATABASE, workdir / "resources/Chinook.db")
        assert digest(workdir / SOURCE.name) == manifest["source_sha256"]
        assert digest(workdir / "resources/Chinook.db") == manifest["database_sha256"]
        print(f"Run {number}/{args.runs} started; planner limit 110s.", flush=True)
        started = time.perf_counter()
        with (folder / "console.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(folder)],
                cwd=workdir, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            (folder / "process.json").write_text(json.dumps({
                "parent_pid": os.getpid(), "worker_pid": process.pid,
            }) + "\n")
            timed_out = False
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        result = base.summarize(folder, time.perf_counter() - started, timed_out, process.returncode)
        events = base.read_events(folder)
        result["node_spans"] = [
            {key: value for key, value in event.items() if key != "details"}
            for event in events if event["kind"] in {"node_end", "node_error"}
        ]
        result["sql_calls"] = sum(event["kind"] == "sql_start" for event in events)
        result["sql_s"] = sum(
            event["duration_s"] for event in events if event["kind"] == "sql_end"
        )
        results.append(result)
        (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{folder.name}: {result['status']}, {result['wall_s']:.1f}s, "
              f"{result['model_calls']} AI calls.", flush=True)
    finished = [item["wall_s"] for item in results if item["status"] == "complete"]
    report = {
        **manifest, "source_unchanged": digest(SOURCE) == manifest["source_sha256"],
        "database_unchanged": digest(DATABASE) == manifest["database_sha256"],
        "complete_runs": len(finished),
        "median_complete_wall_s": statistics.median(finished) if finished else None,
        "median_response_wall_s": statistics.median(item["wall_s"] for item in results),
        "results": results,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(Path(__file__).resolve(), args.output / "runner.py")
    print(f"Saved {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=140)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "artifacts/benchmarks/wedding-planner-langgraph-2026-09-15"
    ))
    args = parser.parse_args()
    if args.worker:
        asyncio.run(worker(args.worker))
    else:
        run_trials(args)
