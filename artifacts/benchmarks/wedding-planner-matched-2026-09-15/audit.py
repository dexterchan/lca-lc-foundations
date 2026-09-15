"""Extract timing and date evidence. Final answer quality still needs review."""

import ast
import json
from pathlib import Path

from runner import HERE, OLD_STAGES, read_events


def audit(folder):
    events = read_events(folder)
    starts = {e["id"]: e for e in events if e["kind"] in ("model_start", "tool_start")}
    ended = {e["id"] for e in events if e["kind"] in (
        "model_end", "model_error", "tool_end", "tool_error",
    )}
    flight_inputs = []
    for event in events:
        if event["kind"] != "tool_start" or event.get("name") == "search_flights":
            continue
        if "flight" not in event.get("name", "").lower():
            continue
        try:
            args = ast.literal_eval(event["input"])
        except (SyntaxError, ValueError):
            args = event["input"]
        flight_inputs.append({"elapsed_s": event["elapsed_s"], "args": args})
    data = {
        "terminal": next((e for e in reversed(events) if e["kind"] == "run_end"), None),
        "stages": [e for e in events if e["kind"] in ("node_end", "node_error")
                   or e["kind"] == "tool_end" and e.get("name") in OLD_STAGES],
        "model_durations": [
            {"stage": e["stage"], "duration_s": e["duration_s"]}
            for e in events if e["kind"] == "model_end"
        ],
        "unfinished_calls": [
            e for key, e in starts.items() if key not in ended
        ],
        "flight_queries": flight_inputs,
        "flight_model_received_requested_date": [
            e["requested_date_present"] for e in events
            if e["kind"] == "model_input"
            and e["stage"] in ("flights", "search_flights")
        ],
        "sql_queries": [
            e["input"] for e in events
            if e["kind"] == "tool_start" and e.get("name") == "query_playlist_db"
        ],
        "direct_sql_results": [e for e in events if e["kind"] == "sql_end"],
    }
    (folder / "audit.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return data


if __name__ == "__main__":
    for folder in sorted(HERE.glob("[0-9][0-9]-*")):
        result = audit(folder)
        print(folder.name, (result["terminal"] or {}).get("status", "running"),
              "flight queries:", len(result["flight_queries"]))
