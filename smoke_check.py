"""Run with `python smoke_check.py`; needs no model, GPU, or API key."""

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError


with tempfile.TemporaryDirectory() as directory:
    os.environ["MATH_EXPLORER_DATA_DIR"] = directory
    import graph_manager

    legacy = Path(directory) / "legacy_graph.json"
    legacy.write_text(json.dumps({"nodes": [{"id": "kept"}], "links": []}))
    graph_manager.LEGACY_GRAPH_FILE = str(legacy)
    graph = graph_manager.GraphManager()
    assert "kept" in graph.nodes
    assert any(node.get("type") == "theorem" for node in graph.nodes.values())
    assert len(graph.links) == len(graph_manager.GraphManager().links)
    assert legacy.exists()

    from server import ExploreRequest, RRSILoopRequest, app

    for model, data in (
        (ExploreRequest, {"query": "x" * 100000}),
        (ExploreRequest, {"query": "x", "max_tokens": 1000000000}),
        (RRSILoopRequest, {"steps": 1000000000}),
    ):
        try:
            model(**data)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"Accepted invalid input: {data}")
    assert app.title

print("Smoke check passed")
