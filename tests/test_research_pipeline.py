import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from arxiv_feed import recent_papers
from benchmark_data import load_aime, load_minif2f_test, paired_improvement_pvalue
from best_of_n import select_candidate, solve_best_of_n
from conjecture_scan import check_candidate, extract_candidates, scan
from formal_claims import cegis, check_claim
from graph_manager import GraphManager
from lean_worker import LeanWorker
from rrsi_engine import RRSIEngine
from verification.ensemble import VerificationEnsemble, VerificationStatus


class PipelineTest(unittest.TestCase):
    def test_counterexample_is_not_mistaken_for_proof(self):
        claim = {"variables": {"x": [1, 51]}, "assumptions": [], "conclusion": "Eq(Mod(x,51),x)"}
        self.assertEqual(check_claim(claim)["model"], {"x": 51})
        self.assertEqual(check_claim({**claim, "variables": {"x": [1, 50]}})["status"], "BOUNDED_VALID")
        self.assertEqual(check_claim({**claim, "assumptions": ["x > 51"]})["status"], "INVALID")
        revised = cegis("modulo", claim, lambda _: '{"assumptions": [], "conclusion": "x % 51 <= x"}')
        self.assertEqual(revised["status"], "BOUNDED_VALID")
        self.assertTrue(revised["revised"])

    def test_answer_key_does_not_certify_proof(self):
        result = VerificationEnsemble().verify(r"Answer: \boxed{8}", ground_truth="8")
        self.assertEqual(result.status, VerificationStatus.UNVERIFIED)
        self.assertTrue(result.ground_truth_matched)
        self.assertFalse(result.is_verified)

    def test_benchmarks_are_distinct(self):
        development = load_aime("development")
        evaluation = load_aime("evaluation")
        self.assertGreater(len(development) + len(evaluation), 400)
        self.assertFalse({p["id"] for p in development} & {p["id"] for p in evaluation})
        self.assertEqual(len(load_minif2f_test()), 225)
        self.assertEqual(paired_improvement_pvalue(6, 0), 1 / 64)

    def test_batched_vote_keeps_uncertainty(self):
        class Engine:
            def generate(self, *args, **kwargs):
                return [r"\boxed{02}", r"\boxed{2}", r"\boxed{3}", r"\boxed{4}"]
        result = solve_best_of_n(Engine(), "question")
        self.assertEqual(result["answer"], "2")
        self.assertEqual(result["status"], "CONSENSUS")
        self.assertFalse(result["is_formally_verified"])
        self.assertEqual(select_candidate(["no answer"])["status"], "NO_ANSWER")

    def test_feed_and_scan_checkpoint(self):
        atom = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>https://arxiv.org/abs/2609.12345v1</id><title>Open problem</title>
        <summary>Conjecture: x % 2 == 0.</summary><published>2026-09-29T00:00:00Z</published>
        </entry></feed>'''
        papers = recent_papers(1, opener=lambda request, timeout: io.BytesIO(atom))
        self.assertEqual(papers[0]["id"], "2609.12345")
        self.assertEqual(check_candidate("Conjecture: x % 2 == 0")["model"], {"x": 1})
        self.assertEqual(extract_candidates("We resolved the conjecture. The problem remains open."),
                         ["The problem remains open."])
        self.assertEqual(extract_candidates("Now, we propose the following conjecture:\nConjecture 5.2. "
                                            "The inclusion is null-homotopic and the intersection is contractible.\n"
                                            "Note that this is unknown."),
                         ["Conjecture 5.2. The inclusion is null-homotopic and the intersection is contractible."])
        self.assertEqual(extract_candidates("Section 5 discusses open problems and future directions."), [])
        self.assertEqual(extract_candidates("Conjecture 1. Every graph has property P. On the other hand, proof is unknown."),
                         ["Conjecture 1. Every graph has property P."])
        self.assertEqual(extract_candidates("Conjecture 1. Every graph has property P. 30"),
                         ["Conjecture 1. Every graph has property P."])

        class Indexer:
            def __init__(self):
                self.names = set()
            def list_documents(self):
                return list(self.names)
            def index_document(self, path, doc_name):
                self.names.add(doc_name)
            def get_document_nodes(self, name):
                return []
        class Graph:
            def __init__(self):
                self.links = []
            def add_node(self, *args, **kwargs):
                pass
            def add_link(self, *args, **kwargs):
                self.links.append(kwargs.get("label"))
        with tempfile.TemporaryDirectory() as directory, patch("conjecture_scan.ArxivClient") as client:
            client.return_value.download_pdf.return_value = "dummy.pdf"
            state = Path(directory) / "scan.json"
            indexer = Indexer()
            graph = Graph()
            first = scan(feed=lambda max_results: papers, indexer=indexer, graph=graph, state_path=state)
            second = scan(feed=lambda max_results: papers, indexer=indexer, graph=Graph(), state_path=state)
            self.assertEqual((first["new_papers"], first["new_candidates"]), (1, 1))
            self.assertEqual(second["new_papers"], 0)
            self.assertIn("refuted_on_bounds", graph.links)
            retry_state = Path(directory) / "retry.json"
            client.return_value.download_pdf.side_effect = OSError("temporary download failure")
            failed = scan(feed=lambda max_results: papers, indexer=Indexer(), graph=Graph(), state_path=retry_state)
            self.assertEqual((failed["new_papers"], failed["pending_papers"]), (0, 1))
            client.return_value.download_pdf.side_effect = None
            self.assertEqual(scan(feed=lambda max_results: papers, indexer=Indexer(),
                                  graph=Graph(), state_path=retry_state)["new_papers"], 1)
            queued_state = Path(directory) / "queued.json"
            queued_state.write_text(json.dumps({"seen": [], "pending": papers, "candidates": []}))
            def rate_limited(max_results):
                raise HTTPError("https://export.arxiv.org/api/query", 429, "Too Many Requests", None, None)
            fallback = scan(feed=rate_limited, indexer=Indexer(), graph=Graph(), state_path=queued_state)
            self.assertEqual(fallback["new_papers"], 1)
            self.assertIn("429", fallback["feed_error"])

    def test_rrsi_pairs_same_questions_without_answer_leakage(self):
        problems = [{"query": f"Question {index}", "ground_truth": "2"} for index in range(6)]
        calls = []
        with tempfile.TemporaryDirectory() as directory, patch("rrsi_engine.load_aime", return_value=problems):
            engine = RRSIEngine(config_path=str(Path(directory) / "harness.json"), graph_manager=object())
            candidate = {**engine.config, "generation": 1}

            def run(query, *, config, ground_truth, persist, seed, use_cache):
                calls.append((query, ground_truth, persist, seed, use_cache))
                answer = "2" if config["generation"] == 1 else "3"
                return {"verification": {"extracted_answer": answer},
                        "metrics": {"telemetry": {"total_ms": 1000}}}

            result = engine.evaluate_paired_benchmark(candidate, run)
            self.assertEqual(result["verdict"], "ACCEPT")
            self.assertEqual(result["p_value"], 1 / 64)
            self.assertTrue(all(ground_truth is None and not persist and not use_cache
                                for _, ground_truth, persist, _, use_cache in calls))
            self.assertEqual([seed for _, _, _, seed, _ in calls], [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

    def test_lean_rejects_axiom_escape_before_launch(self):
        worker = LeanWorker(command=["/missing/repl"])
        statement = "axiom cheat : False\ntheorem false_claim : False := sorry"
        self.assertEqual(worker.prove(statement, "exact cheat")["status"], "INVALID")
        self.assertEqual(worker.prove("theorem false_claim : False := sorry", "sorry")["status"], "INVALID")

    def test_graph_reader_sees_external_scan_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "graph.json")
            server_graph, scanner_graph = GraphManager(path), GraphManager(path)
            scanner_graph.add_node("paper", "document", "paper")
            self.assertTrue(any(node["id"] == "paper" for node in server_graph.get_graph_data()["nodes"]))
            server_graph.add_node("claim", "conjecture", "claim")
            self.assertTrue(any(node["id"] == "claim" for node in scanner_graph.get_graph_data()["nodes"]))


if __name__ == "__main__":
    unittest.main()
