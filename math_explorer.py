#!/usr/bin/env python3
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
from verification.ensemble import VerificationEnsemble

class UnifiedMathHarness:
    """
    Unified High-Speed Mathematical Exploration Portfolio Solver.
    Architecture:
    - Tier 1: TypeSafe JEV System One (Routing, Intent, Difficulty, <100ms)
    - Tier 2: PageIndex Hierarchical Top-Down Literature Tree Descent + arXiv
    - Tier 3: Solver Portfolio (SymPy CAS, Z3 SMT, Local Violetto 1B MPS, Codex Astra xhigh)
    - Tier 4: Verification Ensemble (SymPy CAS equality, Z3 SMT counterexamples, numerical spot-checks)
    - Engine Evolution: Empirical RRSI with paired benchmark outcomes (arXiv:2609.24972)
    """
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
        print(f"Harness ready. Active RRSI Gen: {self.rrsi.config.get('generation', 0)}. Portfolio Solvers: SymPy + Z3 + Violetto + Codex Astra.\n", flush=True)

    def search_and_index_arxiv(self, query: str, max_papers: int = 1) -> Optional[str]:
        """
        Searches arXiv for papers, downloads PDF, and indexes into PageIndex tree.
        """
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
            # Fallback to indexing abstract text
            md_path = os.path.join(self.arxiv.download_dir, f"{paper['id']}_abstract.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {paper['title']}\n\n## Abstract\n{paper['abstract']}\n")
            return self.indexer.index_document(md_path, doc_name=f"arxiv_{paper['id']}")

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
        cfg = self.rrsi.get_current_harness()
        active_gen = cfg.get("generation", 0)
        top_k = top_k_nodes if top_k_nodes is not None else cfg.get("top_k_retrieval", 2)
        diff_threshold = cfg.get("jev_difficulty_threshold", 2.5)
        arxiv_threshold = cfg.get("jev_arxiv_threshold", 0.65)

        print(f"=======================================================", flush=True)
        print(f"MATH EXPLORATION (RRSI Gen {active_gen}): {query}", flush=True)
        print(f"=======================================================\n", flush=True)

        # ----------------------------------------------------
        # 1. TIER 1: JEV SYSTEM ONE FAST ROUTING (<100ms)
        # ----------------------------------------------------
        print("[Tier 1: TypeSafe JEV System One Triage (<100ms)]", flush=True)
        routing = self.router.route_intent(query)
        print(f"  • Domain: {routing['domain']}")
        print(f"  • Difficulty Score: {routing['difficulty_score']:.2f}/4.0 (Threshold: {diff_threshold})")
        print(f"  • arXiv Literature Needed: {routing['needs_arxiv']} (prob: {routing['needs_arxiv_prob']:.2%}, Threshold: {arxiv_threshold:.0%})")
        print(f"  • JEV Recommended Tool: {routing.get('recommended_engine')}")
        
        # Portfolio Backend Selection
        backend = self.registry.select_backend(
            query=query,
            routing=routing,
            difficulty_threshold=diff_threshold,
            engine_override=engine
        )
        target_engine = backend.name
        print(f"  • Selected Portfolio Engine: {target_engine}")

        # ----------------------------------------------------
        # 2. TIER 2: LITERATURE ACQUISITION (arXiv on demand)
        # ----------------------------------------------------
        should_fetch_arxiv = (routing.get("needs_arxiv_prob", 0.0) >= arxiv_threshold or force_arxiv)
        if should_fetch_arxiv and not doc_name:
            print("\n[Tier 2: External Literature Acquisition via arXiv]", flush=True)
            auto_doc = self.search_and_index_arxiv(query)
            if auto_doc:
                doc_name = auto_doc

        # ----------------------------------------------------
        # 3. TIER 3: TRUE HIERARCHICAL TREE SEARCH (PageIndex)
        # ----------------------------------------------------
        context_text = ""
        top_nodes = []
        strategy = cfg.get("search_strategy", "hierarchical_pageindex")

        if strategy == "hierarchical_pageindex":
            print(f"\n[Tier 3: PageIndex Top-Down Hierarchical Tree Descent (beam={top_k})]", flush=True)
            top_nodes = self.indexer.hierarchical_search(
                query=query,
                router=self.router,
                doc_name=doc_name,
                branch_beam=top_k
            )
        else:
            candidates = self.indexer.get_document_nodes(doc_name) if doc_name else self.indexer.get_all_nodes()
            if candidates:
                print(f"\n[Tier 3: Flat Tree Pruning with JEV (top_k={top_k})]", flush=True)
                top_nodes = self.router.rank_nodes(query, candidates, top_k=top_k)

        if top_nodes:
            print(f"Retrieved {len(top_nodes)} Highly Relevant Tree Nodes:")
            for idx, node in enumerate(top_nodes, 1):
                score = node.get("jev_score", 0.0)
                title = node.get("title") or node.get("full_path")
                doc = node.get("doc_name", doc_name or "")
                print(f"  [{idx}] {title} (doc: {doc}) — JEV Relevance: {score:.2%}")
                context_text += f"\n### {title}\n{node.get('text', '')}\n"

        # ----------------------------------------------------
        # 4. TIER 4: PORTFOLIO SOLVER EXECUTION
        # ----------------------------------------------------
        prompt_style = cfg.get("prompt_system_style", "rigorous_math_proof")
        print(f"\n[Tier 4: Portfolio Solver Dispatch -> {target_engine} (style: {prompt_style})]", flush=True)

        solve_result = backend.solve(
            query=query,
            context=context_text if context_text else None,
            prompt_style=prompt_style,
            temperature=cfg.get("violetto_temperature", 0.6),
            top_k=cfg.get("violetto_top_k", 50),
            max_tokens=max_tokens
        )
        solution_text = solve_result.get("solution", "")
        metrics = solve_result.get("metrics", {})
        print(solution_text)

        # ----------------------------------------------------
        # 5. TIER 5: DETERMINISTIC VERIFICATION ENSEMBLE (SymPy + Z3)
        # ----------------------------------------------------
        print(f"\n[Tier 5: Deterministic Verification Ensemble (CAS + SMT + Spot Checks)]", flush=True)
        ensemble_report = self.ensemble.verify(solution_text, ground_truth=ground_truth)
        is_verified = ensemble_report.get("is_verified", False)
        pass_rate = ensemble_report.get("pass_rate", 0.0)
        ext_ans = ensemble_report.get("extracted_answer")
        gt_match = ensemble_report.get("ground_truth_matched")
        
        cas_v = ensemble_report["cas_checks"]["valid"]
        cas_t = ensemble_report["cas_checks"]["total"]
        smt_v = ensemble_report["smt_checks"]["valid"]
        smt_t = ensemble_report["smt_checks"]["total"]
        cexs = ensemble_report["smt_checks"]["counterexamples"]

        print(f"  • Formally Verified: {is_verified} (Overall Step Pass Rate: {pass_rate:.1%})")
        print(f"  • SymPy CAS Checks: {cas_v}/{cas_t} Valid Equalities")
        print(f"  • Z3 SMT Congruence Checks: {smt_v}/{smt_t} Valid")
        if cexs:
            print(f"  [!] Z3 Counterexample Discovered: {cexs[0]}")
        print(f"  • Extracted Boxed Answer: {ext_ans}")
        if ground_truth is not None:
            print(f"  • Ground Truth Match: {gt_match} (Expected: {ground_truth})")

        # ----------------------------------------------------
        # 6. GRAPH PERSISTENCE (Knowledge Graph Node Integration)
        # ----------------------------------------------------
        query_node_id = self.gm.record_exploration(
            query=query,
            routing=routing,
            retrieved_nodes=top_nodes,
            solver_engine=target_engine,
            solution_text=solution_text,
            tokens_or_metrics=metrics,
            gate_result={"ensemble": ensemble_report},
            generation=active_gen,
            symbolic_result=ensemble_report
        )

        print(f"\n=== Exploration Done (Logged to Graph Node: {query_node_id}) ===\n", flush=True)
        return {
            "query_node_id": query_node_id,
            "query": query,
            "routing": routing,
            "retrieved_nodes": top_nodes,
            "solver_engine": target_engine,
            "solution": solution_text,
            "metrics": metrics,
            "verification_ensemble": ensemble_report,
            "is_verified": is_verified,
            "generation": active_gen,
            "graph_stats": self.gm.get_graph_data()["stats"]
        }

def main():
    parser = argparse.ArgumentParser(description="Unified Math Explorer: JEV + PageIndex + arXiv + Violetto + Codex Astra")
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
    exp_parser.add_argument("--engine", choices=["local_violetto", "codex_astra"], help="Override execution engine")
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
        harness.explore(
            query=args.query,
            engine=args.engine,
            force_arxiv=args.arxiv,
            doc_name=args.doc,
            top_k_nodes=args.top_k,
            max_tokens=args.max_tokens
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
