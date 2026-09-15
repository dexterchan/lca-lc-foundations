"""Join saved measurements with the shared evidence review."""

from importlib.metadata import version
import json
import platform
import statistics
import sys

from audit import audit
from runner import DATABASE, HERE, SOURCES, digest


data = json.loads((HERE / "results.json").read_text())
reviews = json.loads((HERE / "reviews.json").read_text())["results"]
assert len(data["results"]) == 6
assert set(reviews) == {row["run"] for row in data["results"]}
assert all(digest(path) == data["sources"][name]["sha256"]
           for name, path in SOURCES.items())
assert digest(DATABASE) == data["database_sha256"]

for row in data["results"]:
    review = reviews[row["run"]]
    assert review["core_pass"] == all(
        review[key].startswith("pass") for key in ("venue", "flight", "playlist")
    )
    row["evidence_review"] = review
    folder = HERE / row["run"]
    evidence = audit(folder)
    assert evidence["terminal"]["status"] == row["status"]
    assert (folder / "answer.txt").is_file() == (row["status"] in ("returned", "complete", "incomplete"))
    (folder / "summary.json").write_text(json.dumps(row, indent=2) + "\n")

groups = {}
for variant in SOURCES:
    rows = [row for row in data["results"] if row["variant"] == variant]
    passed = [row for row in rows if row["evidence_review"]["core_pass"]]
    groups[variant] = {
        "trials": len(rows),
        "text_answers_returned": sum(row["status"] in ("returned", "complete", "incomplete") for row in rows),
        "core_checks_passed": len(passed),
        "passing_wall_s": [row["wall_s"] for row in passed],
        "median_passing_wall_s": statistics.median(row["wall_s"] for row in passed) if passed else None,
        "timeouts": sum(row["status"] == "timeout" for row in rows),
        "model_calls": [row["model_calls"] for row in rows],
    }

comparison = {
    "groups": groups,
    "environment_recorded_after_runs": {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: version(name) for name in (
            "langchain", "langgraph", "langchain-deepseek", "mcp", "tavily-python",
        )},
    },
    "production_sources_unchanged": True,
    "production_database_unchanged": True,
    "remaining_differences": data["remaining_differences"],
    "reviews": reviews,
}
(HERE / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
(HERE / "results.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(groups, indent=2))
print("All six outcomes reviewed; production sources and database hashes match.")
