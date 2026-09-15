import ast
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import statistics


ROOT = Path("/Users/dexterchan/sandbox/agentic/langchain_academy_classwork/lca-lc-foundations")
FOLDER = ROOT / "artifacts/benchmarks/wedding-planner-original-2026-09-15"
report = json.loads((FOLDER / "results.json").read_text())
stages = {
    "search_flights": "Flights", "search_venues": "Venues", "suggest_playlist": "Playlist",
}
audit = []
for result in report["results"]:
    folder = FOLDER / result["run"]
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    starts = {event["id"]: event for event in events if event["kind"] == "tool_start"}
    writes = []
    past_dates = 0
    positive_flight_searches = 0
    for event in events:
        if event["kind"] != "tool_end":
            continue
        if event["name"] == "query_playlist_db":
            query = ast.literal_eval(starts[event["id"]]["input"])["query"]
            command = query.lstrip().split(None, 1)[0].upper()
            if command not in {"SELECT", "PRAGMA", "EXPLAIN", "WITH"}:
                writes.append({"command": command, "query": query,
                               "reported_error": event["output"].startswith("Error")})
        if event["name"] == "search-flight":
            try:
                blocks = ast.literal_eval(event["output"])
                for block in blocks:
                    if block.get("type") == "text":
                        payload = json.loads(block["text"])
                        past_dates += "dates are in the past" in str(payload.get("error", ""))
                        positive_flight_searches += (payload.get("resultsCount") or 0) > 0
            except (ValueError, TypeError, KeyError, AttributeError):
                pass
    stage_spans = {}
    for event in events:
        if event["kind"] == "tool_end" and event["name"] in stages:
            stage_spans[event["name"]] = {
                "start_s": starts[event["id"]]["elapsed_s"],
                "end_s": event["elapsed_s"], "duration_s": event["duration_s"],
            }
    audit.append({
        "run": result["run"], "database_writes": writes,
        "past_date_flight_errors": past_dates,
        "flight_searches_with_results": positive_flight_searches,
        "stage_spans": stage_spans,
    })
(FOLDER / "observations.json").write_text(json.dumps(audit, indent=2) + "\n")
shutil.copy2("/private/tmp/benchmark-wedding-planner.py", FOLDER / "runner.py")
shutil.copy2("/private/tmp/summarize-wedding-benchmark.py", FOLDER / "summarize.py")

def duration(value):
    return f"{value:.1f}s" if value is not None else "—"

lines = [
    "# Original wedding planner benchmark",
    "",
    "Measured on 15 September 2026. Three sequential live runs, each with a "
    "300-second wall-clock cap. The original source file was not edited.",
    "",
    f"**Returned a plan: {report['returned_runs']} / {report['runs']}.** "
    f"Median total time among returned runs: **{duration(report['median_returned_wall_s'])}**.",
    "",
    "| Run | Result | Total time | AI calls | Input tokens | Output tokens |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for result in report["results"]:
    usage = result["reported_usage"]
    lines.append(
        f"| {result['run']} | {result['status']} | {duration(result['wall_s'])} | "
        f"{result['model_calls']} | {usage.get('input_tokens', 0):,} | "
        f"{usage.get('output_tokens', 0):,} |"
    )
lines += [
    "",
    "“Returned” means the script reached its final answer. It does not certify "
    "venue availability, prices, dates, or the correctness of every sentence.",
    "",
    "## Time by part",
    "",
    "| Part | Run 1 | Run 2 | Run 3 | Median |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for name, label in stages.items():
    values = [
        sum(result["completed_tool_durations_s"].get(name, []))
        if name in result["completed_tool_durations_s"] else None
        for result in report["results"]
    ]
    completed = [value for value in values if value is not None]
    lines.append(
        f"| {label} | " + " | ".join(duration(value) for value in values)
        + f" | {duration(statistics.median(completed) if completed else None)} |"
    )
lines += [
    "",
    "These spans include each specialist's AI calls and tool waits. The specialists "
    "can overlap, so do not add their times to estimate the total.",
    "",
    "## Tool calls",
    "",
    "| Run | Web searches | Flight searches | SQL calls | SQL writes | Past-date flight errors |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for result, notes in zip(report["results"], audit):
    calls = result["tool_calls"]
    lines.append(
        f"| {result['run']} | {calls.get('web_search', 0)} | "
        f"{calls.get('search-flight', 0)} | {calls.get('query_playlist_db', 0)} | "
        f"{len(notes['database_writes'])} | {notes['past_date_flight_errors']} |"
    )
lines += [
    "",
    "These are agent tool calls. They do not count extra HTTP attempts made inside "
    "the MCP retry helper. Token counts come from completed model responses; no "
    "dollar cost is inferred.",
    "",
    "## What happened",
    "",
    "- The original playlist agent can issue writes, not just reads. In run 1 it "
    "inserted a `Wedding Jazz` playlist and its track links.",
    "- That changed the shared database before the initial runs 2 and 3. Those "
    "trials are excluded here and preserved in `shared-database-runs/`.",
    "- The source database was clean before the benchmark. It was backed up after "
    "the run, then restored byte-for-byte from its original tracked version. "
    "The final runs 2 and 3 use fresh private copies of that same starting database.",
    "- In run 1, three flight searches used 2025 dates. The tool reported the "
    "current date and the agent corrected the year. Later searches returned flights.",
    "- Run 1's final reply calls 16 May 2027 a Saturday; it is a Sunday. This is "
    "one observed answer error despite the script returning normally.",
    "",
    "## Method and limits",
    "",
    f"- Source: `{report['source']}`",
    f"- Source SHA-256: `{report['source_sha256']}`",
    f"- Starting database SHA-256: `{report['database_baseline_sha256']}`",
    "- Run 1's starting database was confirmed by the clean Git status before "
    "execution. Its original event log did not record a database hash. Runs 2 "
    "and 3 explicitly record matching starting database hashes.",
    "- Model: `deepseek/deepseek-v4-flash-0731` through OpenRouter.",
    "- Request: `I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre`",
    "- The file has top-level `await`. The runner compiles it with that feature "
    "enabled, executes setup, attaches a timing callback, then executes the "
    "original request and output cells.",
    "- Original prompts, model options, tools, retry behavior, and the outer "
    "40-step limit are preserved. Each trial uses a new Python process.",
    "- Total time includes Python startup, tool discovery, agent setup, searches, "
    "and final output. `results.json` also separates setup and planner time.",
    "- Three runs are a small sample. Network delays, provider load, and different "
    "AI choices affect the result.",
    "- This is not a matched comparison with the newer LangGraph version. That "
    "version uses a fixed date, a different execution order, and tighter limits.",
    "",
    "## Files",
    "",
    "- `results.json`: timings, calls, token totals, and source checks.",
    "- `observations.json`: database writes, date errors, and specialist spans.",
    "- `environment.json`: Python, OS, and package versions.",
    "- `run-*/answer.txt`: the full answers.",
    "- `run-*/events.jsonl`: timestamped model and tool events.",
    "- `run-*/console.log`: the original output and errors.",
    "",
    "To repeat in a new output folder, run from the project root:",
    "",
    "```bash",
    "uv run python artifacts/benchmarks/wedding-planner-original-2026-09-15/runner.py "
    "--output /private/tmp/wedding-planner-benchmark-repeat --runs 3 --timeout 300",
    "```",
    "",
]
assert date(2027, 5, 16).strftime("%A") == "Sunday"
assert report["source_unchanged"]
assert report["original_database_unchanged_during_clean_reruns"]
assert hashlib.sha256(
    (ROOT / "notebooks/module-2/resources/Chinook.db").read_bytes()
).hexdigest() == report["database_baseline_sha256"]
(FOLDER / "REPORT.md").write_text("\n".join(lines))
print("\n".join(lines[:16]))
