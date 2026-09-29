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
import json
import time
from typing import Optional, Literal
from threading import Lock
from config import DATA_DIR, SERVER_HOST, SERVER_PORT
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
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

# ponytail: one mutating operation at a time for the shared model and JSON state;
# replace with a job queue if multiple simultaneous solves are needed.
operation_lock = Lock()
trace_lock = Lock()
trace_state = {"id": 0, "status": "idle", "title": "Waiting for a task", "events": []}


def trace_event(stage: str, message: str, detail=None):
    with trace_lock:
        events = trace_state["events"]
        events.append({
            "seq": events[-1]["seq"] + 1 if events else 1,
            "ts": time.time(),
            "stage": stage,
            "message": message,
            "detail": str(detail)[:300] if detail is not None else None,
        })
        del events[:-100]


def trace_start(title: str):
    with trace_lock:
        trace_state.update(id=str(time.time_ns()), status="running", title=title[:180], events=[])
    trace_event("task", "Task started")


def trace_finish(status: str, message: str):
    trace_event("task" if status == "done" else "error", message)
    with trace_lock:
        trace_state["status"] = status


def exclusive_operation():
    if not operation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A solver operation is already running. Retry after it finishes.")
    try:
        yield
    finally:
        operation_lock.release()


EngineName = Literal["sympy", "sympy_cas", "python_solver", "z3", "z3_smt",
                     "local_violetto", "violetto", "codex_astra", "astra"]


class ExploreRequest(BaseModel):
    query: str = Field(min_length=1, max_length=16000)
    engine: Optional[EngineName] = None
    force_arxiv: bool = False
    doc_name: Optional[str] = Field(default=None, max_length=256)
    top_k_nodes: Optional[int] = Field(default=None, ge=1, le=25)
    max_tokens: int = Field(default=1500, ge=1, le=8192)

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


@app.get("/api/trace")
async def get_trace():
    with trace_lock:
        return JSONResponse(content={**trace_state, "events": list(trace_state["events"])},
                            headers={"Cache-Control": "no-store"})

@app.post("/api/explore", dependencies=[Depends(exclusive_operation)])
def post_explore(req: ExploreRequest):
    """Executes end-to-end math exploration and appends nodes to 3D graph."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    trace_start(req.query.strip())
    trace_event("interaction", "Question submitted", req.engine or "automatic routing")
    try:
        result = harness.explore(
            query=req.query,
            engine=req.engine,
            force_arxiv=req.force_arxiv,
            doc_name=req.doc_name,
            top_k_nodes=req.top_k_nodes,
            max_tokens=req.max_tokens,
            on_event=trace_event,
        )
        trace_finish("done", "Exploration complete")
        return JSONResponse(content=result)
    except Exception as e:
        trace_finish("error", f"Exploration failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rrsi/step", dependencies=[Depends(exclusive_operation)])
def post_rrsi_step():
    """Executes one regularized recursive self-improvement cycle per arXiv:2609.24972."""
    trace_start("RRSI evolution step")
    try:
        trace_event("task", "Testing the next harness candidate")
        step_result = harness.rrsi.evolve_step(harness_runner=harness.run_with_config)
        trace_finish("done", "RRSI step complete")
        return JSONResponse(content=step_result)
    except Exception as e:
        trace_finish("error", f"RRSI step failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/graph/reset", dependencies=[Depends(exclusive_operation)])
def post_graph_reset():
    """Resets graph to baseline state."""
    try:
        harness.gm.reset_graph()
        return JSONResponse(content={"status": "reset_successful"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmark")
async def get_benchmark():
    """Returns latest benchmark evaluation results."""
    bm_path = DATA_DIR / "benchmark_results.json"
    if not os.path.exists(bm_path):
        return JSONResponse(content={"status": "not_run_yet", "results": []})
    try:
        with open(bm_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BenchmarkRunRequest(BaseModel):
    engine: Optional[EngineName] = None

@app.post("/api/benchmark/run", dependencies=[Depends(exclusive_operation)])
def post_benchmark_run(req: BenchmarkRunRequest):
    """Executes automated benchmark suite across AIME / AMC / Putnam problems."""
    from benchmark_suite import BenchmarkSuite
    trace_start("Benchmark suite")
    try:
        trace_event("task", "Running benchmark problems")
        suite = BenchmarkSuite(harness=harness)
        res = suite.run_benchmark(engine_override=req.engine)
        trace_finish("done", f"Benchmark complete: {res['solved_correctly']}/{res['total_problems']}")
        return JSONResponse(content=res)
    except Exception as e:
        trace_finish("error", f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class RRSILoopRequest(BaseModel):
    steps: int = Field(default=10, ge=1, le=100)

@app.post("/api/rrsi/loop", dependencies=[Depends(exclusive_operation)])
def post_rrsi_loop(req: RRSILoopRequest):
    """Executes multi-generation autonomous RRSI loop."""
    from rrsi_autonomous_runner import AutonomousRRSIRunner
    trace_start(f"RRSI loop · {req.steps} steps")
    try:
        trace_event("task", "Running autonomous evolution")
        runner = AutonomousRRSIRunner(target_generations=req.steps, harness=harness)
        summary = runner.run_evolution_loop()
        trace_finish("done", "RRSI loop complete")
        return JSONResponse(content=summary)
    except Exception as e:
        trace_finish("error", f"RRSI loop failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ConjectureRequest(BaseModel):
    idx: int = Field(default=0, ge=0)

@app.post("/api/conjecture/explore", dependencies=[Depends(exclusive_operation)])
def post_conjecture_explore(req: ConjectureRequest):
    """Explores an open conjecture with arXiv crawl, PageIndex tree, and Codex Astra."""
    from conjecture_crawler import ConjectureCrawler, SAMPLE_CONJECTURES
    trace_start("Conjecture exploration")
    try:
        crawler = ConjectureCrawler(harness=harness)
        target = SAMPLE_CONJECTURES[req.idx % len(SAMPLE_CONJECTURES)]
        trace_event("task", "Exploring conjecture", target.get("title") if isinstance(target, dict) else str(target))
        res = crawler.explore_conjecture(target)
        trace_finish("done", "Conjecture exploration complete")
        return JSONResponse(content=res)
    except Exception as e:
        trace_finish("error", f"Conjecture exploration failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = SERVER_PORT
    print(f"Starting Math Explorer 3D Web Server on http://localhost:{port}", flush=True)
    uvicorn.run(app, host=SERVER_HOST, port=port, log_level="info")
