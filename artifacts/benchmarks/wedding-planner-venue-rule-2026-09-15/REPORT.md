# Venue rule fix: validation

The venue agent now reports location, advertised capacity, source URLs, and date
availability as separate fields. Python decides whether to continue. A sourced
candidate with room for the requested guests can pass with an unknown date.
Known unavailability, insufficient capacity, a wrong destination, or missing
venue, airport, capacity, or source information still stops the plan.

The date status is saved in the session's `VenueResult` and included in the
answer. An unknown date is labelled **“Not confirmed. Check with the venue
before booking.”** The agent no longer supplies the venue's final success flag.

## Local checks

The new continuation test failed before implementation: the graph stopped after
the venue step. After the fix, **all 40 wedding tests passed** in 1.831 seconds.
They include unsuitable candidates, known unavailable dates, source requirements,
model and tool limits, structured-answer repair, venue exclusions, and concurrent
session isolation.

```sh
uv run python -m unittest discover -s tests -p 'test_wedding*.py' -v
```

A separate code review found no blockers. `git diff --check HEAD` also passed.

## Live check

One run with the normal production settings completed in **140.173 seconds**,
including setup. Limits remained 180 seconds per worker and 300 seconds total;
the model transport timeout remained 30 seconds.

| Step | Time | Result |
| --- | ---: | --- |
| Read request / update_state | 15.0s | Request saved |
| Venue | 91.0s | Shangri-La Paris, date unknown, success=true |
| Flights | 32.0s | March 14, 2027: SEN → CDG, EUR 60 |
| Playlist | 2ms | 12 jazz tracks, 43:07.376, USD 11.88 |

The saved venue state has `date_availability="unknown"` and `success=true`.
The answer retains the unconfirmed-date label. The selected flight's route,
date, price, and times match the recorded Kiwi result.

Venue capacity was supported by a third-party search result stating room for
over 100 seated guests. This is a research candidate, not confirmed availability
or an audit of every extra sentence in the generated description.

The code and database hashes stayed unchanged during the live check. No booking
was made. A single successful live check does not establish a long-term success
rate.

## Evidence

- [Final answer](run-1/answer.txt)
- [Saved state](run-1/state.json)
- [Timings and settings](results.json)
- [Tool and model event log](run-1/events.jsonl)

Command used:

```sh
uv run python -u /private/tmp/benchmark-wedding-langgraph.py --runs 1 --output /Users/dexterchan/sandbox/agentic/langchain_academy_classwork/lca-lc-foundations/artifacts/benchmarks/wedding-planner-venue-rule-2026-09-15
```
