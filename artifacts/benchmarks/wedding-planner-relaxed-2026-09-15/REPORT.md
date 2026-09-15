# LangGraph planner: relaxed time limits

The worker limit is now 180 seconds (was 90). The full-plan limit is now
300 seconds (was 110). All 37 local tests passed.

One live check returned an incomplete plan after 185.8 seconds, including
setup. The venue worker reached its 180-second limit. Flights and the
playlist did not run.

All four web searches finished, each taking 2.3–3.9 seconds. The venue
agent started another model call at 35.1 seconds. No completion was
recorded for that call before the venue step stopped at 185.0 seconds.
The extra time alone did not make this trial finish.

The source and database stayed unchanged during the check. Each trial
used private copies and a fresh session. This is one live check, not a
new three-run comparison.

See [results.json](results.json) for timings and settings,
[the saved answer](run-1/answer.txt), and [the event log](run-1/events.jsonl).

Validation command:

```sh
uv run python -m unittest discover -s tests -v
```

Live-check command used:

```sh
uv run python -u /private/tmp/benchmark-wedding-langgraph.py --runs 1 --output /Users/dexterchan/sandbox/agentic/langchain_academy_classwork/lca-lc-foundations/artifacts/benchmarks/wedding-planner-relaxed-2026-09-15
```
