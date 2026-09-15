# Wedding planners with matched time budgets

Measured on 15 September 2026. Each planner received the same London-to-Paris
request for 100 guests, jazz music, and travel on **14 March 2027**.

**The original met the core checks in 0/3 trials. The session-aware LangGraph
planner met them in 2/3, taking 98.8 and 113.2 seconds. No trial timed out.**

These are research-plan checks against saved evidence, not a claim that all
answer text is correct or that venue availability is confirmed.

| Trial | Script result | Wall time | AI calls | Shared checks | Main finding |
| --- | --- | ---: | ---: | --- | --- |
| Original 1 | Returned text | 151.3s | 21 | Fail | June flights instead of March 14 |
| LangGraph 1 | Incomplete | 119.8s | 5 | Fail | Rejected an eligible venue because date availability was unconfirmed |
| LangGraph 2 | Complete | 98.8s | 7 | Pass with notes | Correct chosen flight and song totals; extra answer details have errors |
| Original 2 | Setup error | 1.1s | 0 | Fail | Flight-service connection failed |
| Original 3 | Returned text | 148.5s | 18 | Fail | June flights instead of March 14 |
| LangGraph 3 | Complete | 113.2s | 7 | Pass with notes | Correct flight and songs; conflicting venue capacity sources disclosed |

## What this explains

The earlier “3/3 returned” result for the original script did not mean it
satisfied the same request as the newer planner. It used a request without
a fixed date and counted a final text answer as completion.

With a date specified, both returned original answers acknowledged March 14
but offered June flights. The trace shows why: `search_flights()` passes only
origin and destination to its specialist. Its prompt asks the specialist to
choose the best season. The requested date never reaches that specialist.

The newer planner has a separate issue. Its first venue agent found Hôtel de
Crillon with supporting capacity evidence, then sent `success=false` because
the exact date was not confirmed. The graph followed that flag and stopped.
The shared research criteria did not require confirmed availability. This
failure happened before the five-minute deadline; more time would not fix
that decision.

The original also failed once during `client.get_tools()`, before planning:
`httpx.ConnectError: [SSL: UNEXPECTED_MESSAGE]`. Its retry helper covers tool
calls, not this discovery step. The failed trial is retained in the results.

The useful next change is to make the venue success rule explicit: a sourced
candidate with suitable advertised capacity can pass while its date
availability remains unknown and is clearly labelled.

## Shared checks and answer limits

The criteria were saved in `manifest.json` before running:

- A named Paris venue with evidence of room for 100 guests; no invented
  confirmed date availability.
- A tool-backed, priced one-way London-to-Paris flight on 14 March 2027.
- Named jazz tracks supported by Chinook, with matching duration and price
  totals.

Both passing LangGraph trials produced 12 verified jazz tracks, totalling
**43:07.376 and USD 11.88**. Their selected flights match the saved tool data:

- LangGraph 2: easyJet U25711, SEN → CDG, 14 March, 15:55–18:05, EUR 60.
- LangGraph 3: Vueling VY8058, LHR → ORY, 14 March, 15:15–17:45, EUR 65.

Passing these core checks does not certify every sentence:

- LangGraph 2 misstates a secondary Gatwick departure as 09:45 rather than
  09:50. Its venue description also borrows a room description from a
  multi-venue search snippet without clear attribution.
- LangGraph 3 cites an advertised 200-guest seated capacity, but discloses
  another source's 60-guest figure. The actual layout needs confirmation.
- Original 1's aggregate playlist totals match its 18-track SQL selection,
  but the specialist table omits one selected track and misstates another
  track's duration. Original 3's 20-track totals match the database.

Full findings and evidence references are in [reviews.json](reviews.json).
Source evidence was reviewed from the saved tool replies; inventory and
venue claims were not independently rechecked through new network calls.

## What was matched

For this experiment only, both planners received:

- The exact same user message.
- A 300-second planning deadline, with no earlier worker deadline.
- The same model through OpenRouter, with a 300-second model transport
  timeout and zero model-client retries.
- A 30-second Tavily timeout.
- Fresh source and database copies, a fresh process, and the same review
  criteria.

Setup had a 60-second guard; an external 330-second process cap covered the
whole trial. Wall times include setup. Trial order was original, graph,
graph, original, original, graph. No trials were replaced or discarded.

The production source files and database stayed unchanged. Production
LangGraph defaults remain 180 seconds per worker, 300 seconds total, and
30 seconds for the model transport; this harness overrides them only in
the experiment.

## What still differs

This is a comparison under matched **time budgets**, not a pure framework
speed test. Both scripts use LangChain agents backed by LangGraph.

The original can run specialists together and suggests 12 venue searches.
The newer graph runs venue, then flights, then SQL, with four real calls per
tool and six model calls per specialist. Prompts, output formats, date
forwarding, retry rules, and tool-discovery timing also differ. Three trials
per planner cannot establish a stable failure rate or isolate provider load.

## Files and validation

- [comparison.json](comparison.json): grouped results and environment.
- [results.json](results.json): measurements with the completed reviews.
- [manifest.json](manifest.json): request, settings, hashes, and criteria
  recorded before execution.
- Each trial folder contains its event log, summary, audit, private source
  and database copies, and an answer when one was returned.
- `runner.py`, `timing_helpers.py`, and `summary_helpers.py` preserve the
  measurement code. The runner rejects an existing manifest to avoid
  replacing an earlier experiment.

Commands used from the project root:

```sh
uv run python artifacts/benchmarks/wedding-planner-matched-2026-09-15/verify_runner.py
uv run python -u artifacts/benchmarks/wedding-planner-matched-2026-09-15/runner.py
uv run python artifacts/benchmarks/wedding-planner-matched-2026-09-15/summarize.py
```

Offline probes passed for request forwarding, the configured total limit,
saved answers, timing-stage attribution, and concurrent log reads. The
final summary check confirmed all six outcomes have reviews and both
production source hashes and the database hash still match.
