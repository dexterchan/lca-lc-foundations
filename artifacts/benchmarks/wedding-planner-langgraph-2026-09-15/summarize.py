from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics


ROOT = Path("/Users/dexterchan/sandbox/agentic/langchain_academy_classwork/lca-lc-foundations")
FOLDER = ROOT / "artifacts/benchmarks/wedding-planner-langgraph-2026-09-15"
original = json.loads((
    ROOT / "artifacts/benchmarks/wedding-planner-original-2026-09-15/results.json"
).read_text())
report = json.loads((FOLDER / "results.json").read_text())


def seconds(value):
    if value is None:
        return "—"
    return f"{value * 1000:.1f}ms" if value < 0.1 else f"{value:.1f}s"


def reason(result):
    state = result["terminal_event"].get("state", {})
    text = state.get("output", "")
    if result["status"] == "complete":
        return "Venue, flight and playlist succeeded"
    if "Total time limit reached" in text:
        return "110-second total limit"
    if "Search timed out" in text:
        failed = next((name for name in ("venue", "flights", "playlist")
                       if state.get(name) and not state[name]["success"]), "search")
        return f"90-second {failed} limit"
    if result["status"] == "awaiting_input":
        return text
    return str(state.get("last_error") or text or result["terminal_event"].get("error", "No result"))[:350]


lines = [
    "# LangGraph wedding planner benchmark",
    "",
    "Three sequential live runs on 15 September 2026. Each uses a fresh process, "
    "session, and private database copy. The source and its limits were left unchanged.",
    "",
    f"**Full plans completed: {report['complete_runs']} / {report['runs']}.**",
    "",
    "| Run | Result | Time until return | AI calls started | Web calls | Flight calls | SQL calls | Reason |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
]
for result in report["results"]:
    calls = result["tool_calls"]
    flight_calls = sum(value for name, value in calls.items()
                       if name not in {"update_state", "web_search", "VenueResult", "SearchResult"})
    lines.append(
        f"| {result['run']} | {result['status']} | {seconds(result['wall_s'])} | "
        f"{result['model_calls']} | {calls.get('web_search', 0)} | "
        f"{flight_calls} | {result['sql_calls']} | {reason(result)} |"
    )
lines += [
    "",
    f"Median time until any return: **{seconds(report['median_response_wall_s'])}**. "
    f"Median time for completed plans: **{seconds(report['median_complete_wall_s'])}**.",
    "",
    "## Time by graph step",
    "",
    "| Step | Run 1 | Run 2 | Run 3 |",
    "| --- | ---: | ---: | ---: |",
]
for name in ("collect_request", "venue", "flights", "playlist"):
    values = []
    for result in report["results"]:
        spans = [item for item in result["node_spans"] if item["name"] == name]
        value = seconds(sum(item["duration_s"] for item in spans)) if spans else "Not reached"
        if any(item["kind"] == "node_error" for item in spans):
            value += " (stopped)"
        elif any(item.get("success") is False for item in spans):
            value += " (failed)"
        values.append(value)
    lines.append(f"| {name} | " + " | ".join(values) + " |")
lines += [
    "",
    "Step time includes model and tool waits. Repeated visits to a step are summed. "
    "A stopped step reached the run's time limit; it did not finish its work.",
    "",
    "## Comparison with the original",
    "",
    "| Measure | Original script | Current LangGraph script |",
    "| --- | ---: | ---: |",
    f"| Final plans | {original['returned_runs']} / 3 text plans returned | {report['complete_runs']} / 3 marked complete |",
    f"| Median time until return | {seconds(original['median_returned_wall_s'])} | {seconds(report['median_response_wall_s'])} |",
    f"| AI calls per run | {min(r['model_calls'] for r in original['results'])}–{max(r['model_calls'] for r in original['results'])} | "
    f"{min(r['model_calls'] for r in report['results'])}–{max(r['model_calls'] for r in report['results'])} |",
    "| Time limits | 300s external benchmark cap | 110s run, 90s per worker |",
    "| Order | Specialists can run together | Venue, then flights, then playlist |",
    "| Playlist | Agent queries the database | Direct read-only SQL |",
    "",
    "**A shorter time that ends with an incomplete plan is not a speed win.** "
    "This compares the two scripts as configured; it does not isolate the effect "
    "of LangGraph itself.",
    "",
    "The original chose its own flight dates. The LangGraph run uses the explicit "
    "date in the prompt below. The prompts, tool budgets, search order, and success "
    "checks differ. Three trials are a small sample and include live service delays.",
    "",
    "## Model usage",
    "",
    "| Run | Input tokens reported | Output tokens reported |",
    "| --- | ---: | ---: |",
]
for result in report["results"]:
    usage = result["reported_usage"]
    lines.append(
        f"| {result['run']} | {usage.get('input_tokens', 0):,} | "
        f"{usage.get('output_tokens', 0):,} |"
    )
lines += [
    "",
    "Counts include only completed model responses. Started calls that were "
    "cancelled may have consumed additional tokens. These are not billing totals.",
    "",
    "## Method",
    "",
    f"- Source: `{report['source']}`",
    f"- Source SHA-256: `{report['source_sha256']}`",
    f"- Database SHA-256: `{report['database_sha256']}`",
    f"- Prompt: `{report['prompt']}`",
    "- Model: `deepseek/deepseek-v4-flash-0731` through OpenRouter.",
    "- The same built-in request is used in all trials. Its date is frozen at "
    "the start of the batch.",
    "- `main()` runs with its existing 110-second limit. Setup occurs before "
    "that limit. An external 140-second cap also guards the test process.",
    "- Timing callbacks record each graph node, AI call and tool call. A wrapper "
    "around `query_playlist` records SQL time without changing its query.",
    "- A local probe confirmed the timer records all five nodes, AI stage labels, "
    "and the normal timeout state before any live runs.",
    "- `complete` requires venue, flights and playlist success. `incomplete` and "
    "`awaiting_input` are kept as separate outcomes, even with a normal process exit.",
    "- Each copied database started with the recorded hash. The original source "
    "and database hashes were checked again after the benchmark.",
    "",
    "## Files and repeat command",
    "",
    "- `results.json`: counts, times, node spans, outcomes, and environment versions.",
    "- `run-*/answer.txt`: exact final output.",
    "- `run-*/state.json`: saved request and search results.",
    "- `run-*/events.jsonl`: raw timing and tool records.",
    "- `run-*/console.log`: script output and errors.",
    "",
    "From the project root, use a new output folder:",
    "",
    "```bash",
    "uv run python artifacts/benchmarks/wedding-planner-langgraph-2026-09-15/runner.py "
    "--output /private/tmp/wedding-langgraph-benchmark-repeat --runs 3",
    "```",
    "",
]
assert report["source_unchanged"] and report["database_unchanged"]
for result in report["results"]:
    events = [json.loads(line) for line in (
        FOLDER / result["run"] / "events.jsonl"
    ).read_text().splitlines()]
    model_ids = [item["id"] for item in events if item["kind"] == "model_start"]
    assert len(model_ids) == len(set(model_ids)) == result["model_calls"]
    assert events[0]["database_sha256"] == report["database_sha256"]
    assert events[0]["source_sha256"] == report["source_sha256"]
    assert result["terminal_event"].get("copied_database_unchanged", False)
(FOLDER / "REPORT.md").write_text("\n".join(lines))
shutil.copy2("/private/tmp/summarize-langgraph-benchmark.py", FOLDER / "summarize.py")
print("\n".join(lines[:16]))
