# LangGraph wedding planner benchmark

Three sequential live runs on 15 September 2026. Each uses a fresh process, session, and private database copy. The source and its limits were left unchanged.

**Full plans completed: 1 / 3.**

| Run | Result | Time until return | AI calls started | Web calls | Flight calls | SQL calls | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| run-1 | incomplete | 97.7s | 5 | 4 | 0 | 0 | 90-second venue limit |
| run-2 | complete | 95.4s | 7 | 4 | 1 | 1 | Venue, flight and playlist succeeded |
| run-3 | incomplete | 112.0s | 6 | 4 | 1 | 0 | 110-second total limit |

Median time until any return: **97.7s**. Median time for completed plans: **95.4s**.

## Time by graph step

| Step | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| collect_request | 5.4s | 3.8s | 6.5s |
| venue | 90.0s (failed) | 58.5s | 87.9s |
| flights | Not reached | 31.1s | 15.6s (stopped) |
| playlist | Not reached | 4.1ms | Not reached |

Step time includes model and tool waits. Repeated visits to a step are summed. A stopped step reached the run's time limit; it did not finish its work.

## Comparison with the original

| Measure | Original script | Current LangGraph script |
| --- | ---: | ---: |
| Final plans | 3 / 3 text plans returned | 1 / 3 marked complete |
| Median time until return | 227.9s | 97.7s |
| AI calls per run | 18–22 | 5–7 |
| Time limits | 300s external benchmark cap | 110s run, 90s per worker |
| Order | Specialists can run together | Venue, then flights, then playlist |
| Playlist | Agent queries the database | Direct read-only SQL |

**A shorter time that ends with an incomplete plan is not a speed win.** This compares the two scripts as configured; it does not isolate the effect of LangGraph itself.

The original chose its own flight dates. The LangGraph run uses the explicit date in the prompt below. The prompts, tool budgets, search order, and success checks differ. Three trials are a small sample and include live service delays.

## Model usage

| Run | Input tokens reported | Output tokens reported |
| --- | ---: | ---: |
| run-1 | 15,151 | 1,634 |
| run-2 | 40,865 | 3,366 |
| run-3 | 18,356 | 3,732 |

Counts include only completed model responses. Started calls that were cancelled may have consumed additional tokens. These are not billing totals.

## Method

- Source: `/Users/dexterchan/sandbox/agentic/langchain_academy_classwork/lca-lc-foundations/notebooks/module-2/2.4_wedding_planners_langgraph.py`
- Source SHA-256: `190b6e401fbb398508abb8b452347270d9e5df53c1988f4857908f106f5b614b`
- Database SHA-256: `84f5d9143ac4deebdb81650ab650e226d909e660106846b119a5c47c33f94c13`
- Prompt: `I'm from London and want a wedding in Paris for 100 guests, with jazz music. Travel date: 2027-03-14.`
- Model: `deepseek/deepseek-v4-flash-0731` through OpenRouter.
- The same built-in request is used in all trials. Its date is frozen at the start of the batch.
- `main()` runs with its existing 110-second limit. Setup occurs before that limit. An external 140-second cap also guards the test process.
- Timing callbacks record each graph node, AI call and tool call. A wrapper around `query_playlist` records SQL time without changing its query.
- A local probe confirmed the timer records all five nodes, AI stage labels, and the normal timeout state before any live runs.
- `complete` requires venue, flights and playlist success. `incomplete` and `awaiting_input` are kept as separate outcomes, even with a normal process exit.
- Each copied database started with the recorded hash. The original source and database hashes were checked again after the benchmark.

## Files and repeat command

- `results.json`: counts, times, node spans, outcomes, and environment versions.
- `run-*/answer.txt`: exact final output.
- `run-*/state.json`: saved request and search results.
- `run-*/events.jsonl`: raw timing and tool records.
- `run-*/console.log`: script output and errors.

From the project root, use a new output folder:

```bash
uv run python artifacts/benchmarks/wedding-planner-langgraph-2026-09-15/runner.py --output /private/tmp/wedding-langgraph-benchmark-repeat --runs 3
```
