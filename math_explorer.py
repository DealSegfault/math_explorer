#!/usr/bin/env python3
"""
Unified High-Speed Mathematical Exploration Portfolio Solver.
Architecture:
- Speculative CAS / SMT Early Exit (< 5ms)
- Complete Fallback Cascade: SymPy CAS -> Z3 SMT -> Local Violetto MPS -> Codex Astra
- Fail-Closed 3-State Verification Contract (VERIFIED, REFUTED, UNVERIFIED)
- Real Empirical RRSI Evaluation via run_with_config(query, config, ground_truth, persist=False)
- Active verification_strictness knob
- Quickwit Tantivy BM25 + Parallel JEV Reranking
- Multi-Level Thread-Safe Persistent SQLite Cache
"""

import os
import sys
import time
import argparse
from typing import Optional, Dict, Any, List

from config import VIOLETTO_MODEL_PATH, TYPESAFE_KEY_PATH, DATA_DIR, PAPERS_DIR
from jev_router import JevRouter
from doc_indexer import MathDocIndexer
from arxiv_client import ArxivClient
from graph_manager import GraphManager
from rrsi_engine import RRSIEngine
from backends.registry import SolverRegistry
from verification.ensemble import VerificationEnsemble, VerificationStatus
from cache_manager import cache

class UnifiedMathHarness:
    def __init__(
        self,
        model_path: str = VIOLETTO_MODEL_PATH,
        key_path: str = TYPESAFE_KEY_PATH,
        cache_dir: str = str(DATA_DIR)
    ):
        print("=== Initializing Mathematical Exploration Harness ===", flush=True)
        self.gm = GraphManager()
        self.rrsi = RRSIEngine(graph_manager=self.gm)
        self.registry = SolverRegistry(violetto_model_path=model_path)
        self.ensemble = VerificationEnsemble()
        self.router = JevRouter(key_path=key_path)
        self.indexer = MathDocIndexer(cache_dir=cache_dir)
        self.arxiv = ArxivClient(download_dir=str(PAPERS_DIR))
        print(f"Harness ready. Active RRSI Gen: {self.rrsi.config.get('generation', 0)}. Solvers: SymPy + Z3 + Violetto + Codex Astra.\n", flush=True)

    def search_and_index_arxiv(self, query: str, max_papers: int = 1) -> Optional[str]:
        """Searches arXiv for papers, downloads PDF, and indexes into PageIndex tree."""
        print(f"\n[arXiv Pipeline] Searching for literature on: '{query}'...", flush=True)
        papers = self.arxiv.search(query, max_results=max_papers)
        if not papers:
            print("No matching arXiv papers found.", flush=True)
            return None

        paper = papers[0]
        print(f"Found arXiv paper: {paper['title']} (ID: {paper['id']})", flush=True)
        try:
            pdf_path = self.arxiv.download_pdf(paper['id'])
            doc_name = f"arxiv_{paper['id']}"
            self.indexer.index_document(pdf_path, doc_name=doc_name)
            return doc_name
        except Exception as e:
            print(f"Could not index arXiv PDF: {e}. Saving abstract note instead...", flush=True)
            md_path = os.path.join(self.arxiv.download_dir, f"{paper['id']}_abstract.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {paper['title']}\n\n## Abstract\n{paper['abstract']}\n")
            return self.indexer.index_document(md_path, doc_name=f"arxiv_{paper['id']}")

    def run_with_config(
        self,
        query: str,
        config: Dict[str, Any],
        ground_truth: Optional[str] = None,
        persist: bool = False,
        engine: Optional[str] = None,
        force_arxiv: bool = False,
        doc_name: Optional[str] = None,
        top_k_nodes: Optional[int] = None,
        max_tokens: int = 1500,
        seed: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Executes exploration under a specific candidate or baseline harness configuration.
        Implements full fallback cascade and fail-closed verification.
        When persist=False (for RRSI A/B testing), does not pollute disk graph state.
        """
        t_global_0 = time.perf_counter_ns()
        gen = config.get("generation", 0)
        top_k = top_k_nodes if top_k_nodes is not None else config.get("top_k_retrieval", 2)
        diff_threshold = config.get("jev_difficulty_threshold", 2.5)
        arxiv_threshold = config.get("jev_arxiv_threshold", 0.65)
        prompt_style = config.get("prompt_system_style", "rigorous_math_proof")
        strictness = config.get("verification_strictness", 0.75)
        temperature = config.get("violetto_temperature", 0.6)
        top_k_sampling = config.get("violetto_top_k", 50)

        # ----------------------------------------------------
        # 0. TIER 0: SPECULATIVE CAS PROBE & EARLY EXIT (< 5ms)
        # ----------------------------------------------------
        sympy_backend = self.registry.get_backend("sympy_cas")
        if sympy_backend and sympy_backend.can_handle(query) and not force_arxiv and not engine:
            t_probe_0 = time.perf_counter_ns()
            probe_result = sympy_backend.solve(query)
            t_probe_ms = round((time.perf_counter_ns() - t_probe_0) / 1e6, 2)

            if probe_result.get("is_exact") and probe_result.get("extracted_answer") is not None:
                t_ver_0 = time.perf_counter_ns()
                v_res = self.ensemble.verify(probe_result.get("solution", ""), ground_truth=ground_truth,
                                             strictness=strictness, reference_answer=probe_result.get("extracted_answer"))
                t_ver_ms = round((time.perf_counter_ns() - t_ver_0) / 1e6, 2)
                t_total_ms = round((time.perf_counter_ns() - t_global_0) / 1e6, 2)

                dummy_routing = {
                    "domain": "number_theory",
                    "difficulty_score": 0.5,
                    "needs_arxiv": False,
                    "recommended_engine": "sympy_cas"
                }
                telemetry = {
                    "early_exit": True,
                    "tier_level": 0,
                    "route_ms": 0.0,
                    "prefilter_ms": 0.0,
                    "rerank_ms": 0.0,
                    "solve_ms": t_probe_ms,
                    "verify_ms": t_ver_ms,
                    "total_ms": t_total_ms,
                    "cost_usd": 0.0
                }

                query_node_id = None
                if persist:
                    query_node_id = self.gm.record_exploration(
                        query=query,
                        routing=dummy_routing,
                        retrieved_nodes=[],
                        solver_engine="sympy_cas",
                        solution_text=probe_result.get("solution", ""),
                        tokens_or_metrics={**probe_result.get("metrics", {}), "telemetry": telemetry},
                        gate_result={"ensemble": v_res.to_dict()},
                        generation=gen,
                        symbolic_result=v_res.to_dict()
                    )

                v_dict = v_res.to_dict()
                return {
                    "query_node_id": query_node_id,
                    "query": query,
                    "routing": dummy_routing,
                    "retrieved_nodes": [],
                    "solver_engine": "sympy_cas",
                    "solution": probe_result.get("solution", ""),
                    "metrics": {**probe_result.get("metrics", {}), "telemetry": telemetry},
                    "verification": v_dict,
                    "symbolic_verification": v_dict, # backward compat
                    "verification_ensemble": v_dict, # backward compat
                    "is_verified": v_res.is_verified,
                    "generation": gen,
                    "early_exit": True,
                    "graph_stats": self.gm.get_graph_data()["stats"] if persist else {}
                }

        # ----------------------------------------------------
        # 1. TIER 1: JEV SYSTEM ONE FAST ROUTING (< 100ms)
        # ----------------------------------------------------
        t_route_0 = time.perf_counter_ns()
        routing = self.router.route_intent(query)
        t_route_ms = round((time.perf_counter_ns() - t_route_0) / 1e6, 2)
        
        backend = self.registry.select_backend(
            query=query,
            routing=routing,
            difficulty_threshold=diff_threshold,
            engine_override=engine
        )
        target_engine = backend.name

        # ----------------------------------------------------
        # 2. TIER 2: LITERATURE ACQUISITION (arXiv on demand)
        # ----------------------------------------------------
        should_fetch_arxiv = (routing.get("needs_arxiv_prob", 0.0) >= arxiv_threshold or force_arxiv)
        if should_fetch_arxiv and not doc_name:
            auto_doc = self.search_and_index_arxiv(query)
            if auto_doc:
                doc_name = auto_doc

        # ----------------------------------------------------
        # 3. TIER 3: TWO-STAGE HYBRID LITERATURE RETRIEVAL
        # ----------------------------------------------------
        context_text = ""
        top_nodes = []
        t_prefilter_ms = 0.0
        t_rerank_ms = 0.0

        if should_fetch_arxiv or doc_name or self.indexer.list_documents():
            retrieval_res = self.indexer.two_stage_retrieve(
                query=query,
                router=self.router,
                doc_name=doc_name,
                lexical_top_k=25,
                final_top_k=top_k,
                max_workers=8
            )
            top_nodes = retrieval_res.get("nodes", [])
            ret_metrics = retrieval_res.get("metrics", {})
            t_prefilter_ms = ret_metrics.get("bm25_ms", 0.0)
            t_rerank_ms = ret_metrics.get("rerank_ms", 0.0)

        if top_nodes:
            for node in top_nodes:
                title = node.get("title") or node.get("full_path")
                context_text += f"\n### {title}\n{node.get('text', '')}\n"

        # ----------------------------------------------------
        # 4. TIER 4: PORTFOLIO SOLVER WITH FALLBACK CASCADE
        # ----------------------------------------------------
        t_solve_0 = time.perf_counter_ns()
        solve_result = backend.solve(
            query=query,
            context=context_text if context_text else None,
            prompt_style=prompt_style,
            temperature=temperature,
            top_k=top_k_sampling,
            max_tokens=max_tokens,
            seed=seed,
            use_cache=use_cache
        )
        solution_text = solve_result.get("solution", "")
        metrics = solve_result.get("metrics", {})

        # FALLBACK CASCADE: If SymPy or Z3 was chosen but couldn't parse/extract answer
        if target_engine in ["sympy_cas", "z3_smt"] and (solve_result.get("extracted_answer") is None or not solve_result.get("is_exact")):
            # If SymPy failed, try Z3 if applicable
            if target_engine == "sympy_cas" and self.registry.z3.can_handle(query):
                target_engine = "z3_smt"
                backend = self.registry.z3
                solve_result = backend.solve(query=query, context=context_text, prompt_style=prompt_style)
                solution_text = solve_result.get("solution", "")
                metrics = solve_result.get("metrics", {})

            # If still unsolved by symbolic/SMT, cascade down to local Violetto
            if solve_result.get("extracted_answer") is None or not solve_result.get("is_exact"):
                target_engine = "local_violetto"
                backend = self.registry.violetto
                solve_result = backend.solve(
                    query=query,
                    context=context_text if context_text else None,
                    prompt_style=prompt_style,
                    temperature=temperature,
                    top_k=top_k_sampling,
                    max_tokens=max_tokens,
                    seed=seed,
                    use_cache=use_cache
                )
                solution_text = solve_result.get("solution", "")
                metrics = solve_result.get("metrics", {})

        t_solve_ms = round((time.perf_counter_ns() - t_solve_0) / 1e6, 2)

        # ----------------------------------------------------
        # 5. TIER 5: FAIL-CLOSED VERIFICATION & ADAPTIVE ESCALATION
        # ----------------------------------------------------
        t_ver_0 = time.perf_counter_ns()
        reference = solve_result.get("extracted_answer") if target_engine in {"sympy_cas", "z3_smt"} and solve_result.get("is_exact") else None
        v_res = self.ensemble.verify(solution_text, ground_truth=ground_truth, strictness=strictness,
                                     reference_answer=reference)
        t_ver_ms = round((time.perf_counter_ns() - t_ver_0) / 1e6, 2)

        # Adaptive Escalation: If local Violetto is UNVERIFIED or REFUTED and engine not locked
        tier_level = 1 if target_engine == "local_violetto" else (3 if target_engine == "codex_astra" else 0)
        if not v_res.is_verified and target_engine == "local_violetto" and engine is None:
            astra_backend = self.registry.get_backend("codex_astra")
            if astra_backend:
                t_esc_0 = time.perf_counter_ns()
                esc_result = astra_backend.solve(
                    query=query,
                    context=context_text,
                    prompt_style=prompt_style,
                    max_tokens=max_tokens
                )
                t_solve_ms += round((time.perf_counter_ns() - t_esc_0) / 1e6, 2)
                solution_text = esc_result.get("solution", solution_text)
                target_engine = "codex_astra"
                tier_level = 3
                # Re-verify
                v_res = self.ensemble.verify(solution_text, ground_truth=ground_truth, strictness=strictness)

        t_total_ms = round((time.perf_counter_ns() - t_global_0) / 1e6, 2)
        cost_usd = 0.05 if target_engine == "codex_astra" else 0.0

        telemetry = {
            "early_exit": False,
            "tier_level": tier_level,
            "route_ms": t_route_ms,
            "prefilter_ms": t_prefilter_ms,
            "rerank_ms": t_rerank_ms,
            "solve_ms": t_solve_ms,
            "verify_ms": t_ver_ms,
            "total_ms": t_total_ms,
            "cost_usd": cost_usd
        }

        # ----------------------------------------------------
        # 6. GRAPH PERSISTENCE (Only if persist=True)
        # ----------------------------------------------------
        query_node_id = None
        v_dict = v_res.to_dict()
        if persist:
            query_node_id = self.gm.record_exploration(
                query=query,
                routing=routing,
                retrieved_nodes=top_nodes,
                solver_engine=target_engine,
                solution_text=solution_text,
                tokens_or_metrics={**metrics, "telemetry": telemetry},
                gate_result={"ensemble": v_dict},
                generation=gen,
                symbolic_result=v_dict
            )

        return {
            "query_node_id": query_node_id,
            "query": query,
            "routing": routing,
            "retrieved_nodes": top_nodes,
            "solver_engine": target_engine,
            "solution": solution_text,
            "metrics": {**metrics, "telemetry": telemetry},
            "verification": v_dict,
            "symbolic_verification": v_dict, # backward compat
            "verification_ensemble": v_dict, # backward compat
            "is_verified": v_res.is_verified,
            "generation": gen,
            "early_exit": False,
            "graph_stats": self.gm.get_graph_data()["stats"] if persist else {}
        }

    def explore(
        self,
        query: str,
        engine: Optional[str] = None,
        force_arxiv: bool = False,
        doc_name: Optional[str] = None,
        top_k_nodes: Optional[int] = None,
        max_tokens: int = 1500,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Default public exploration interface (with persistence)."""
        cfg = self.rrsi.get_current_harness()
        return self.run_with_config(
            query=query,
            config=cfg,
            ground_truth=ground_truth,
            persist=True,
            engine=engine,
            force_arxiv=force_arxiv,
            doc_name=doc_name,
            top_k_nodes=top_k_nodes,
            max_tokens=max_tokens
        )

def main():
    parser = argparse.ArgumentParser(description="Unified Math Explorer: JEV + Tantivy + MPS Violetto + Codex Astra")
    subparsers = parser.add_subparsers(dest="command")

    # index command
    idx_parser = subparsers.add_parser("index", help="Index local document into PageIndex tree")
    idx_parser.add_argument("file", help="Path to PDF or Markdown file")
    idx_parser.add_argument("--name", help="Optional document identifier")

    # arxiv command
    arxiv_parser = subparsers.add_parser("arxiv", help="Search arXiv and index paper into PageIndex")
    arxiv_parser.add_argument("query", help="Search keywords or paper title")

    # list command
    subparsers.add_parser("list", help="List all indexed documents")

    # explore command
    exp_parser = subparsers.add_parser("explore", help="Explore a mathematical question")
    exp_parser.add_argument("query", help="Mathematical question, theorem, or conjecture")
    exp_parser.add_argument("--engine", choices=["local_violetto", "codex_astra", "sympy_cas", "z3_smt"], help="Override execution engine")
    exp_parser.add_argument("--arxiv", action="store_true", help="Force fetching literature from arXiv")
    exp_parser.add_argument("--doc", help="Filter by specific document")
    exp_parser.add_argument("--top_k", type=int, default=2, help="Number of nodes to retrieve")
    exp_parser.add_argument("--max_tokens", type=int, default=1500, help="Max tokens for local model")

    args = parser.parse_args()

    harness = UnifiedMathHarness()

    if args.command == "index":
        harness.indexer.index_document(args.file, args.name)
    elif args.command == "arxiv":
        harness.search_and_index_arxiv(args.query)
    elif args.command == "list":
        docs = harness.indexer.list_documents()
        print(f"\nIndexed Documents ({len(docs)}):")
        for d in docs:
            nodes = harness.indexer.get_document_nodes(d)
            print(f" - {d} ({len(nodes)} sections)")
    elif args.command == "explore":
        res = harness.explore(
            query=args.query,
            engine=args.engine,
            force_arxiv=args.arxiv,
            doc_name=args.doc,
            top_k_nodes=args.top_k,
            max_tokens=args.max_tokens
        )
        print(res.get("solution"))
        v = res.get("verification", {})
        print(f"\nVerification Status: {v.get('status')} (is_verified={v.get('is_verified')}, pass_rate={v.get('pass_rate')})")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
