# Original wedding planner benchmark

Measured on 15 September 2026. Three sequential live runs, each with a 300-second wall-clock cap. The original source file was not edited.

**Returned a plan: 3 / 3.** Median total time among returned runs: **227.9s**.

| Run | Result | Total time | AI calls | Input tokens | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| run-1 | returned | 243.8s | 22 | 174,898 | 13,400 |
| run-2 | returned | 198.9s | 20 | 177,809 | 11,270 |
| run-3 | returned | 227.9s | 18 | 138,710 | 10,782 |

“Returned” means the script reached its final answer. It does not certify venue availability, prices, dates, or the correctness of every sentence.

## Time by part

| Part | Run 1 | Run 2 | Run 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Flights | 224.5s | 133.4s | 189.3s | 189.3s |
| Venues | 157.2s | 159.8s | 162.4s | 159.8s |
| Playlist | 50.3s | 138.7s | 128.5s | 128.5s |

These spans include each specialist's AI calls and tool waits. The specialists can overlap, so do not add their times to estimate the total.

## Tool calls

| Run | Web searches | Flight searches | SQL calls | SQL writes | Past-date flight errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| run-1 | 12 | 7 | 10 | 2 | 3 |
| run-2 | 12 | 6 | 8 | 0 | 3 |
| run-3 | 12 | 4 | 5 | 0 | 2 |

These are agent tool calls. They do not count extra HTTP attempts made inside the MCP retry helper. Token counts come from completed model responses; no dollar cost is inferred.

## What happened

- The original playlist agent can issue writes, not just reads. In run 1 it inserted a `Wedding Jazz` playlist and its track links.
- That changed the shared database before the initial runs 2 and 3. Those trials are excluded here and preserved in `shared-database-runs/`.
- The source database was clean before the benchmark. It was backed up after the run, then restored byte-for-byte from its original tracked version. The final runs 2 and 3 use fresh private copies of that same starting database.
- In run 1, three flight searches used 2025 dates. The tool reported the current date and the agent corrected the year. Later searches returned flights.
- Run 1's final reply calls 16 May 2027 a Saturday; it is a Sunday. This is one observed answer error despite the script returning normally.

## Method and limits

- Source: `/Users/dexterchan/sandbox/agentic/langchain_academy_classwork/lca-lc-foundations/notebooks/module-2/2.4_wedding_planners.py`
- Source SHA-256: `9810d295353f1db5c95b7cda719e1e264129d7c9a46e0361880a2eedd7a0f872`
- Starting database SHA-256: `84f5d9143ac4deebdb81650ab650e226d909e660106846b119a5c47c33f94c13`
- Run 1's starting database was confirmed by the clean Git status before execution. Its original event log did not record a database hash. Runs 2 and 3 explicitly record matching starting database hashes.
- Model: `deepseek/deepseek-v4-flash-0731` through OpenRouter.
- Request: `I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre`
- The file has top-level `await`. The runner compiles it with that feature enabled, executes setup, attaches a timing callback, then executes the original request and output cells.
- Original prompts, model options, tools, retry behavior, and the outer 40-step limit are preserved. Each trial uses a new Python process.
- Total time includes Python startup, tool discovery, agent setup, searches, and final output. `results.json` also separates setup and planner time.
- Three runs are a small sample. Network delays, provider load, and different AI choices affect the result.
- This is not a matched comparison with the newer LangGraph version. That version uses a fixed date, a different execution order, and tighter limits.

## Files

- `results.json`: timings, calls, token totals, and source checks.
- `observations.json`: database writes, date errors, and specialist spans.
- `environment.json`: Python, OS, and package versions.
- `run-*/answer.txt`: the full answers.
- `run-*/events.jsonl`: timestamped model and tool events.
- `run-*/console.log`: the original output and errors.

To repeat in a new output folder, run from the project root:

```bash
uv run python artifacts/benchmarks/wedding-planner-original-2026-09-15/runner.py --output /private/tmp/wedding-planner-benchmark-repeat --runs 3 --timeout 300
```
