#!/usr/bin/env python3
"""
FastAPI Server for 3D Math Explorer & RRSI Evolution Engine.
Serves interactive Three.js WebGL interface and endpoints for:
- Live mathematical queries (JEV -> PageIndex -> Violetto / Codex Astra -> Verification)
- RRSI recursive self-improvement steps (Proposer -> Critic -> Pruner -> Invariants)
- Dynamic 3D graph visualization state
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from math_explorer import UnifiedMathHarness

app = FastAPI(title="Math Explorer 3D // RRSI Harness API")

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Mount static directory for CSS, JS, Assets
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# Initialize Unified Math Harness (Lazy or on startup)
print("Initializing Unified Math Harness...", flush=True)
harness = UnifiedMathHarness()
print("Harness ready for server requests.", flush=True)

class ExploreRequest(BaseModel):
    query: str
    engine: Optional[str] = None
    force_arxiv: bool = False
    doc_name: Optional[str] = None
    top_k_nodes: Optional[int] = None
    max_tokens: int = 1500

@app.get("/")
async def get_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index HTML not found")
    return FileResponse(index_path)

@app.get("/api/graph")
async def get_graph():
    """Returns current 3D graph representation (nodes, links, telemetry)."""
    try:
        data = harness.gm.get_graph_data()
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/harness")
async def get_harness():
    """Returns current RRSI harness configuration and evolution history."""
    try:
        cfg = harness.rrsi.get_current_harness()
        return JSONResponse(content=cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/explore")
async def post_explore(req: ExploreRequest):
    """Executes end-to-end math exploration and appends nodes to 3D graph."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = harness.explore(
            query=req.query,
            engine=req.engine,
            force_arxiv=req.force_arxiv,
            doc_name=req.doc_name,
            top_k_nodes=req.top_k_nodes,
            max_tokens=req.max_tokens
        )
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rrsi/step")
async def post_rrsi_step():
    """Executes one regularized recursive self-improvement cycle per arXiv:2609.24972."""
    try:
        step_result = harness.rrsi.evolve_step()
        return JSONResponse(content=step_result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/graph/reset")
async def post_graph_reset():
    """Resets graph to baseline state."""
    try:
        harness.gm.reset_graph()
        return JSONResponse(content={"status": "reset_successful"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"Starting Math Explorer 3D Web Server on http://localhost:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
