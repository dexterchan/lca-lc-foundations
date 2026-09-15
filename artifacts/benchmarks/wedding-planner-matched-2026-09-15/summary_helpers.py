"""Time the original notebook script without editing its prompts or limits."""

import argparse
import ast
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import types


ROOT = Path("/Users/dexterchan/sandbox/agentic/langchain_academy_classwork/lca-lc-foundations")
SOURCE = ROOT / "notebooks/module-2/2.4_wedding_planners.py"
STAGES = {"update_state", "search_flights", "search_venues", "suggest_playlist"}


def source_hash():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


async def worker(folder):
    from langchain_core.callbacks import BaseCallbackHandler

    started = time.perf_counter()
    lock = threading.Lock()
    event_file = folder / "events.jsonl"

    def emit(kind, **fields):
        record = {"kind": kind, "elapsed_s": round(time.perf_counter() - started, 6), **fields}
        with lock, event_file.open("a") as stream:
            stream.write(json.dumps(record, default=str) + "\n")

    class Timings(BaseCallbackHandler):
        run_inline = True

        def __init__(self):
            self.parents = {}
            self.tools = {}
            self.starts = {}
            self.guard = threading.RLock()

        def stage(self, parent):
            seen = set()
            while parent is not None and parent not in seen:
                seen.add(parent)
                if self.tools.get(parent) in STAGES:
                    return self.tools[parent]
                parent = self.parents.get(parent)
            return "coordinator"

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
            with self.guard:
                self.parents[run_id] = parent_run_id

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
            with self.guard:
                self.parents[run_id] = parent_run_id
                self.starts[run_id] = time.perf_counter()
                emit("model_start", id=str(run_id), stage=self.stage(parent_run_id))

        def on_llm_end(self, response, *, run_id, **kwargs):
            usage = Counter()
            for batch in response.generations:
                for generation in batch:
                    data = getattr(generation.message, "usage_metadata", None) or {}
                    for key in ("input_tokens", "output_tokens", "total_tokens"):
                        usage[key] += data.get(key, 0)
            with self.guard:
                emit("model_end", id=str(run_id),
                     stage=self.stage(self.parents.get(run_id)),
                     duration_s=time.perf_counter() - self.starts[run_id],
                     usage=dict(usage))

        def on_llm_error(self, error, *, run_id, **kwargs):
            with self.guard:
                emit("model_error", id=str(run_id),
                     stage=self.stage(self.parents.get(run_id)), error=str(error))

        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
            with self.guard:
                name = (serialized or {}).get("name", kwargs.get("name", "unknown"))
                self.parents[run_id] = parent_run_id
                self.tools[run_id] = name
                self.starts[run_id] = time.perf_counter()
                emit("tool_start", id=str(run_id), name=name,
                     stage=name if name in STAGES else self.stage(parent_run_id),
                     input=input_str)

        def on_tool_end(self, output, *, run_id, **kwargs):
            with self.guard:
                emit("tool_end", id=str(run_id), name=self.tools.get(run_id),
                     stage=self.stage(run_id),
                     duration_s=time.perf_counter() - self.starts[run_id],
                     tool_status=getattr(output, "status", None),
                     output=str(getattr(output, "content", output)))

        def on_tool_error(self, error, *, run_id, **kwargs):
            with self.guard:
                emit("tool_error", id=str(run_id), name=self.tools.get(run_id),
                     stage=self.stage(run_id),
                     duration_s=time.perf_counter() - self.starts[run_id],
                     error=str(error))

    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    split = next(
        i for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "response"
                for target in node.targets)
    )
    module = types.ModuleType("original_wedding_benchmark")
    module.__file__ = str(SOURCE)
    sys.modules[module.__name__] = module
    namespace = vars(module)

    async def execute(nodes):
        code = compile(
            ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        result = eval(code, namespace)
        if inspect.isawaitable(result):
            await result

    phase = "setup"
    database = Path.cwd() / "resources/Chinook.db"
    emit("run_start", source_sha256=source_hash(), utc=datetime.now(timezone.utc).isoformat(),
         database_path=str(database),
         database_sha256=hashlib.sha256(database.read_bytes()).hexdigest())
    try:
        await execute(tree.body[:split])
        emit("setup_end")
        namespace["coordinator"] = namespace["coordinator"].with_config(callbacks=[Timings()])
        phase = "planner"
        emit("planner_start")
        await execute(tree.body[split:])
        response = namespace["response"]
        answer = response["messages"][-1].content
        (folder / "answer.txt").write_text(answer if isinstance(answer, str) else json.dumps(answer))
        fields = {key: response.get(key) for key in ("origin", "destination", "guest_count", "genre")}
        emit("run_end", status="returned", state=fields)
    except Exception as error:
        traceback.print_exc()
        emit("run_end", status="error", phase=phase,
             error=f"{type(error).__name__}: {error}")


def read_events(folder):
    path = folder / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(folder, wall_s, timed_out, returncode):
    events = read_events(folder)
    ended = next((item for item in reversed(events) if item["kind"] == "run_end"), {})
    status = "timeout" if timed_out else ended.get("status", "process_error")
    usage = Counter()
    model_counts = Counter()
    tool_counts = Counter()
    tool_durations = {}
    for item in events:
        if item["kind"] == "model_start":
            model_counts[item["stage"]] += 1
        elif item["kind"] == "model_end":
            usage.update(item["usage"])
        elif item["kind"] == "tool_start":
            tool_counts[item["name"]] += 1
        elif item["kind"] in {"tool_end", "tool_error"}:
            tool_durations.setdefault(item["name"], []).append(round(item["duration_s"], 3))
    setup = next((item["elapsed_s"] for item in events if item["kind"] == "setup_end"), None)
    planning = next((item["elapsed_s"] for item in events if item["kind"] == "planner_start"), None)
    return {
        "run": folder.name, "status": status, "process_returncode": returncode,
        "wall_s": round(wall_s, 3), "setup_s": setup,
        "planner_s": round(ended["elapsed_s"] - planning, 3)
        if ended and planning is not None else None,
        "model_calls": sum(model_counts.values()), "model_calls_by_stage": dict(model_counts),
        "reported_usage": dict(usage), "tool_calls": dict(tool_counts),
        "completed_tool_durations_s": tool_durations,
        "terminal_event": ended,
    }


def run_trials(args):
    args.output.mkdir(parents=True, exist_ok=True)
    digest = source_hash()
    original_db_hash = hashlib.sha256((SOURCE.parent / "resources/Chinook.db").read_bytes()).hexdigest()
    baseline = args.output / "baseline-Chinook.db"
    if not baseline.exists():
        shutil.copy2(SOURCE.parent / "resources/Chinook.db", baseline)
    summaries = [
        json.loads((args.output / f"run-{number}/summary.json").read_text())
        for number in range(1, args.start_run)
    ]
    for number in range(args.start_run, args.runs + 1):
        folder = args.output / f"run-{number}"
        folder.mkdir()
        workdir = folder / "workdir"
        (workdir / "resources").mkdir(parents=True)
        shutil.copy2(baseline, workdir / "resources/Chinook.db")
        started = time.perf_counter()
        print(f"Run {number}/{args.runs} started; cap {args.timeout:g}s.", flush=True)
        with (folder / "console.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(folder)],
                cwd=workdir, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (folder / "process.json").write_text(json.dumps({
                "parent_pid": os.getpid(), "worker_pid": process.pid,
            }) + "\n")
            timed_out = False
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Stop this benchmark's process group, including sync tool threads.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        result = summarize(folder, time.perf_counter() - started, timed_out, process.returncode)
        summaries.append(result)
        (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"{folder.name}: {result['status']}, {result['wall_s']:.1f}s, "
            f"{result['model_calls']} model calls.", flush=True,
        )
    finished = [item["wall_s"] for item in summaries if item["status"] == "returned"]
    data_policy = (
        "Run 1 used the initially clean source database; later runs use private database "
        "copies. Two shared-database trials were excluded and retained in "
        "shared-database-runs/. The original database was restored exactly. "
        if (args.output / "shared-database-runs").exists()
        else "All runs use private database copies. "
    )
    report = {
        "source": str(SOURCE), "source_sha256": digest,
        "source_unchanged": digest == source_hash(), "runs": args.runs,
        "original_database_unchanged_during_clean_reruns": original_db_hash == hashlib.sha256(
            (SOURCE.parent / "resources/Chinook.db").read_bytes()
        ).hexdigest(),
        "database_baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "wall_cap_s_per_run": args.timeout, "sequential": True,
        "prompt": "I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre",
        "method": (
            "Original source compiled with top-level-await support. "
            "Only a timing callback is attached to the coordinator. Original prompts, tools, "
            "model options and recursion limit are preserved. Each run gets a fresh process "
            "and the same starting database. " + data_policy
            + "Tool spans can overlap; token counts cover completed responses only."
        ),
        "returned_runs": len(finished),
        "median_returned_wall_s": statistics.median(finished) if finished else None,
        "results": summaries,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output / 'results.json'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "artifacts/benchmarks/wedding-planner-original-2026-09-15"
    ))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        asyncio.run(worker(args.worker))
    else:
        run_trials(args)
