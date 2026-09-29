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
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from math_explorer import UnifiedMathHarness
from benchmark_data import load_aime
from best_of_n import solve_best_of_n
from formal_claims import cegis
from lean_worker import LeanWorker
from lean_auto import autoformalize

app = FastAPI(title="Math Explorer 3D // RRSI Harness API")

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Mount static directory for CSS, JS, Assets
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# Initialize Unified Math Harness (Lazy or on startup)
print("Initializing Unified Math Harness...", flush=True)
harness = UnifiedMathHarness()
lean_worker = LeanWorker()
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
        step_result = harness.rrsi.evolve_step(harness_runner=harness.run_with_config)
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

@app.get("/api/benchmark")
async def get_benchmark():
    """Returns latest benchmark evaluation results."""
    bm_path = os.path.join(BASE_DIR, "data", "benchmark_results.json")
    if not os.path.exists(bm_path):
        return JSONResponse(content={"status": "not_run_yet", "results": []})
    try:
        with open(bm_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BenchmarkRunRequest(BaseModel):
    engine: Optional[str] = None
    dataset: str = "local"
    split: str = "evaluation"
    limit: int = 20

@app.post("/api/benchmark/run")
async def post_benchmark_run(req: BenchmarkRunRequest):
    """Executes automated benchmark suite across AIME / AMC / Putnam problems."""
    from benchmark_suite import BenchmarkSuite
    try:
        if req.dataset not in {"local", "aime"} or req.split not in {"all", "development", "evaluation"} or not 0 <= req.limit <= 1000:
            raise HTTPException(status_code=400, detail="Invalid benchmark selection")
        suite = BenchmarkSuite()
        problems = load_aime(req.split, None if req.limit == 0 else req.limit) if req.dataset == "aime" else None
        res = suite.run_benchmark(engine_override=req.engine, problems=problems)
        return JSONResponse(content=res)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class RRSILoopRequest(BaseModel):
    steps: int = 10

@app.post("/api/rrsi/loop")
async def post_rrsi_loop(req: RRSILoopRequest):
    """Executes multi-generation autonomous RRSI loop."""
    from rrsi_autonomous_runner import AutonomousRRSIRunner
    try:
        runner = AutonomousRRSIRunner(target_generations=req.steps)
        summary = runner.run_evolution_loop()
        return JSONResponse(content=summary)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ConjectureRequest(BaseModel):
    idx: int = 0

@app.post("/api/conjecture/explore")
async def post_conjecture_explore(req: ConjectureRequest):
    """Explores an open conjecture with arXiv crawl, PageIndex tree, and Codex Astra."""
    from conjecture_crawler import ConjectureCrawler, SAMPLE_CONJECTURES
    try:
        crawler = ConjectureCrawler()
        target = SAMPLE_CONJECTURES[req.idx % len(SAMPLE_CONJECTURES)]
        res = crawler.explore_conjecture(target)
        return JSONResponse(content=res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class CegisRequest(BaseModel):
    problem: str
    claim: dict
    engine: str = "local_violetto"
    max_rounds: int = 3


@app.post("/api/cegis")
def post_cegis(req: CegisRequest):
    if req.engine not in {"local_violetto", "codex_astra"} or not 0 <= req.max_rounds <= 5:
        raise HTTPException(status_code=400, detail="Invalid CEGIS options")
    backend = harness.registry.get_backend(req.engine)
    result = cegis(req.problem, req.claim, lambda prompt: backend.solve(prompt, max_tokens=500)["solution"], req.max_rounds)
    return JSONResponse(content=result)


class BestOfNRequest(BaseModel):
    query: str
    n: int = 4
    max_tokens: int = 384
    temperature: float = 0.6
    seed: Optional[int] = None


@app.post("/api/solve/best-of-n")
def post_best_of_n(req: BestOfNRequest):
    if not 1 <= req.n <= 8 or not 1 <= req.max_tokens <= 1500 or not 0.05 < req.temperature <= 2:
        raise HTTPException(status_code=400, detail="Invalid sampling options")
    result = solve_best_of_n(harness.registry.violetto.engine, req.query, n=req.n,
                             max_tokens=req.max_tokens, temperature=req.temperature, seed=req.seed)
    return JSONResponse(content=result)


class LeanProofRequest(BaseModel):
    formal_statement: str
    proof: str
    header: str = "import Init"


@app.post("/api/lean/prove")
def post_lean_prove(req: LeanProofRequest):
    return JSONResponse(content=lean_worker.prove(req.formal_statement, req.proof, req.header))


class LeanAutoRequest(BaseModel):
    problem: str
    engine: str = "codex_astra"
    header: str = "import Init"


@app.post("/api/lean/auto")
def post_lean_auto(req: LeanAutoRequest):
    if req.engine not in {"local_violetto", "codex_astra"}:
        raise HTTPException(status_code=400, detail="Invalid solver engine")
    backend = harness.registry.get_backend(req.engine)
    return JSONResponse(content=autoformalize(req.problem, lambda prompt: backend.solve(prompt, max_tokens=500)["solution"],
                                               lean_worker, req.header))


@app.post("/api/conjecture/scan")
def post_conjecture_scan():
    from conjecture_scan import scan
    return JSONResponse(content=scan())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"Starting Math Explorer 3D Web Server on http://localhost:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
