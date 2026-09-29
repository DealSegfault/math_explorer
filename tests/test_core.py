from uuid import uuid4

from backends.sympy_backend import SymPyBackend
from backends.z3_backend import Z3Backend
from doc_indexer import MathDocIndexer
from jev_router import JevRouter
from math_explorer import UnifiedMathHarness
from rrsi_engine import DEFAULT_HARNESS, RRSIEngine
from backends.registry import SolverRegistry
from verification.ensemble import VerificationEnsemble


def test_z3_divisibility_negations():
    backend = Z3Backend()
    for suffix in ("not 4, not 9", "not divisible by 4, not divisible by 9"):
        query = f"Count all integers n < 1000 such that n is divisible by 6, {suffix}."
        assert backend._parse_divisibility_sieve(query) == (999, [6], [4, 9])
        assert backend.solve(query)["extracted_answer"] == "55"
    assert backend._parse_divisibility_sieve("Find a number divisible by 6, not divisible by 4") is None


def test_z3_counterexample_is_real_sat_query():
    backend = Z3Backend()
    bounds = {"n": (1, 4)}
    found = backend.find_counterexample(["n"], "Eq(n % 2, 0)", bounds)
    assert found["sat"] is True and found["model"]["n"] in (1, 3)
    proven = backend.find_counterexample(["n"], "Eq(n*n, n**2)", bounds)
    assert proven["sat"] is False and "UNSAT" in proven["message"]
    unknown = backend.find_counterexample(["n"], "Eq(sin(n), 0)", bounds)
    assert unknown["sat"] is None


def test_exact_backend_evidence_verifies_answer_without_prose_equation():
    solved = SymPyBackend().solve(
        "Count all integers n < 1000 such that n is divisible by 6, not 4, not 9.")
    result = VerificationEnsemble().verify(solved["solution"], evidence=solved["verification_evidence"])
    assert solved["extracted_answer"] == "55"
    assert result.status == "VERIFIED"
    assert VerificationEnsemble().verify("Final Answer: $\\boxed{56}$", evidence=solved["verification_evidence"]).status == "REFUTED"


def test_verifier_stays_fail_closed_without_evidence():
    verifier = VerificationEnsemble()
    assert verifier.verify("Final Answer: $\\boxed{55}$").status == "UNVERIFIED"
    assert verifier.verify("Final Answer: $\\boxed{55}$", ground_truth="56").status == "REFUTED"


def test_strictness_changes_answer_verdict_for_incomplete_steps():
    solution = "2 = 2\nx = 1\nFinal Answer: $\\boxed{55}$"
    verifier = VerificationEnsemble()
    assert verifier.verify(solution, ground_truth="55", strictness=0.4).status == "VERIFIED"
    assert verifier.verify(solution, ground_truth="55", strictness=0.75).status == "UNVERIFIED"


def test_retrieval_cache_changes_with_corpus(tmp_path):
    indexer = MathDocIndexer(cache_dir=str(tmp_path))
    indexer.registry["paper"] = {"nodes": [{"node_id": "one", "title": "Algebra", "text": "algebra"}]}
    indexer._save_registry()
    old_version = indexer.registry_version
    first = indexer.two_stage_retrieve("topology", router=None, strategy="flat_topk")
    assert first["nodes"][0]["node_id"] == "one"
    indexer.registry["paper"] = {"nodes": [{"node_id": "two", "title": "Topology", "text": "topology"}]}
    indexer._save_registry()
    assert indexer.registry_version != old_version
    second = indexer.two_stage_retrieve("topology", router=None, strategy="flat_topk")
    assert second["nodes"][0]["node_id"] == "two"


def test_search_strategy_switches_retrieval_path(tmp_path):
    harness = UnifiedMathHarness(cache_dir=str(tmp_path))
    harness.indexer.registry["paper"] = {"nodes": [
        {"node_id": "one", "title": "Gauss law", "text": "prime theorem"},
        {"node_id": "two", "title": "Eisenstein law", "text": "cubic theorem"}]}
    harness.indexer._save_registry()

    class Router:
        calls = 0

        def score_node_relevance(self, query, title, text=""):
            self.calls += 1
            return float("Gauss" in title)

    router = Router()
    harness._router = router
    common = {"evaluation_mode": "retrieval", "doc_name": "paper", "expected_node_ids": ["one"]}
    query = f"Gauss prime theorem {uuid4().hex}"
    flat = harness.run_with_config(query, dict(DEFAULT_HARNESS, search_strategy="flat_topk"), **common)
    assert router.calls == 0
    reranked = harness.run_with_config(query, dict(DEFAULT_HARNESS, search_strategy="hierarchical_pageindex"), **common)
    assert router.calls > 0
    assert flat["solver_engine"] == reranked["solver_engine"] == "retrieval_only"


def test_solver_benchmark_uses_prompt_temperature_and_seed():
    class Backend:
        name = "local_violetto"
        calls = []

        def solve(self, **kwargs):
            self.calls.append(kwargs)
            return {"solution": "Final Answer: $\\boxed{29}$", "metrics": {}, "is_exact": False}

    backend = Backend()

    class Registry:
        def get_backend(self, name):
            return None

        def select_backend(self, **kwargs):
            return backend

    harness = UnifiedMathHarness.__new__(UnifiedMathHarness)
    harness.registry = Registry()
    harness.ensemble = VerificationEnsemble()
    config = dict(DEFAULT_HARNESS, prompt_system_style="lemma_stepwise_decomposition", violetto_temperature=0.45)
    result = harness.run_with_config("What is the largest prime smaller than thirty?", config,
                                     ground_truth="29", engine="local_violetto", evaluation_mode="solver", seed=7)
    assert result["verification"]["ground_truth_matched"] is True
    assert backend.calls[0]["prompt_style"] == "lemma_stepwise_decomposition"
    assert backend.calls[0]["temperature"] == 0.45
    assert backend.calls[0]["seed"] == 7 and backend.calls[0]["use_cache"] is False


def test_rrsi_uses_retrieval_cases_and_all_latencies(tmp_path):
    engine = RRSIEngine(config_path=str(tmp_path / "harness.json"), graph_manager=object())
    candidate = dict(DEFAULT_HARNESS, top_k_retrieval=3)
    calls = []
    times = [100, 200, 300, 90, 100, 1010]

    def runner(query, config, **kwargs):
        calls.append((query, kwargs["evaluation_mode"], config["top_k_retrieval"]))
        index = len(calls) - 1
        expected = kwargs["expected_node_ids"][0]
        found = config["top_k_retrieval"] == 3
        return {"retrieved_nodes": [{"node_id": expected}] if found else [],
                "verification": {"ground_truth_matched": found},
                "metrics": {"recall_at_k": float(found), "telemetry": {
                    "total_ms": times[index], "compute_cost_units": config["top_k_retrieval"]}}}

    result = engine.evaluate_paired_benchmark(candidate, runner)
    assert len(calls) == 6 and all(mode == "retrieval" for _, mode, _ in calls)
    assert result["baseline_mean_latency_sec"] == 0.167
    assert result["candidate_mean_latency_sec"] == 0.433
    assert result["baseline_p95_latency_sec"] == 0.3
    assert result["candidate_p95_latency_sec"] == 1.01
    assert result["candidate_accuracy"] == 1.0


def test_rrsi_rejects_unmeasured_evolution(tmp_path):
    engine = RRSIEngine(config_path=str(tmp_path / "harness.json"), graph_manager=object())
    first = engine.evolve_step()
    second = engine.evolve_step()
    assert first["success"] is second["success"] is False
    assert first["critic"]["verdict"] == second["critic"]["verdict"] == "REJECT"
    assert first["proposal"]["component"] != second["proposal"]["component"]


def test_rrsi_requires_verified_llm_answers(tmp_path):
    engine = RRSIEngine(config_path=str(tmp_path / "harness.json"), graph_manager=object())
    candidate = dict(DEFAULT_HARNESS, prompt_system_style="lemma_stepwise_decomposition")

    def runner(query, config, **kwargs):
        assert kwargs["evaluation_mode"] == "solver" and kwargs["engine"] == "local_violetto"
        good = config["prompt_system_style"] == "rigorous_math_proof"
        return {"solution": config["prompt_system_style"], "solver_engine": "local_violetto",
                "verification": {"ground_truth_matched": True, "status": "VERIFIED" if good else "UNVERIFIED"},
                "metrics": {"telemetry": {"total_ms": 100, "compute_cost_units": 1}}}

    result = engine.evaluate_paired_benchmark(candidate, runner)
    assert result["candidate_accuracy"] == 0.0
    assert result["regressions"] == 4
    assert result["verdict"] == "REJECT"


def test_rrsi_holds_arxiv_gate_without_fetch_cost(tmp_path):
    engine = RRSIEngine(config_path=str(tmp_path / "harness.json"), graph_manager=object())
    candidate = dict(DEFAULT_HARNESS, jev_arxiv_threshold=0.55)

    def runner(query, config, **kwargs):
        found = kwargs["routing_override"]["needs_arxiv_prob"] >= config["jev_arxiv_threshold"]
        node = kwargs["expected_node_ids"][0]
        return {"retrieved_nodes": [{"node_id": node}] if found else [],
                "metrics": {"recall_at_k": float(found), "telemetry": {
                    "total_ms": 100, "compute_cost_units": int(found)}}}

    result = engine.evaluate_paired_benchmark(candidate, runner)
    assert result["candidate_accuracy"] == 1.0
    assert result["verdict"] == "REJECT"
    assert "unmeasured" in result["reason"]


def test_jev_plan_keeps_typed_answers_usage_and_version():
    router = JevRouter.__new__(JevRouter)
    router.api_key = "test"
    calls = []

    def evaluate_full(**kwargs):
        calls.append(kwargs)
        return {
            "model": "jev-test",
            "usage": {"input_tokens": 101, "output_tokens": 17},
            "answers": {
                "task_family": {"type": "choice", "choice": "symbolic_proof", "confidence": 0.8,
                                "probabilities": {"symbolic_proof": 0.9, "other": 0.1}},
                "domain": {"type": "choice", "choice": "algebra", "confidence": 0.9,
                           "probabilities": {"algebra": 1.0}},
                "difficulty": {"type": "score", "score": 2.7, "confidence": 0.7},
                "needs_arxiv": {"type": "noul", "noul": 0.2},
                "verification_burden": {"type": "score", "score": 2.1, "confidence": 0.6},
            },
        }

    router.evaluate_full = evaluate_full
    query = f"Prove the generated statement {uuid4().hex}"
    plan = router.plan_exploration(query)
    cached = router.plan_exploration(query)
    assert plan["plan_version"] == "route.v2"
    assert plan["usage"] == {"input_tokens": 101, "output_tokens": 17}
    assert plan["raw_answers"]["task_family"]["probabilities"]["symbolic_proof"] == 0.9
    assert plan["routing_confidence"] == 0.7
    assert cached["cached"] is True and len(calls) == 1


def test_jev_falls_back_without_becoming_a_required_dependency():
    router = JevRouter.__new__(JevRouter)
    router.api_key = ""
    plan = router.plan_exploration(f"Find a counterexample {uuid4().hex}")
    assert plan["task_family"] == "counterexample_search"
    assert plan["routing_confidence"] == 0.0
    assert plan["failure_mode"] == "FileNotFoundError"


def test_jev_retrieval_exposes_composite_signals():
    router = JevRouter.__new__(JevRouter)
    router.api_key = "test"
    router.evaluate_full = lambda **kwargs: {
        "model": "jev-test",
        "usage": {"input_tokens": 20, "output_tokens": 3},
        "answers": {
            "direct_support": {"type": "noul", "noul": 1.0},
            "contains_needed_method": {"type": "noul", "noul": 0.4},
            "matches_notation_and_scope": {"type": "noul", "noul": 0.2},
        },
    }
    result = router.assess_node_relevance(
        f"query-{uuid4().hex}", "Quadratic reciprocity", "Legendre symbols")
    assert result["score"] == 0.73
    assert result["signals"]["contains_needed_method"] == 0.4
    assert result["usage"]["input_tokens"] == 20


def test_registry_requires_confidence_before_direct_astra_route():
    class NeverExact:
        def can_handle(self, query):
            return False

    class Backend:
        def __init__(self, name):
            self.name = name

    registry = SolverRegistry.__new__(SolverRegistry)
    registry.sympy = registry.z3 = NeverExact()
    registry._violetto = Backend("local_violetto")
    registry._astra = Backend("codex_astra")
    assert registry.select_backend("hard proof", {"difficulty_score": 3.0, "routing_confidence": 0.5}).name == "local_violetto"
    assert registry.select_backend("hard proof", {"difficulty_score": 3.0, "routing_confidence": 0.8}).name == "codex_astra"
