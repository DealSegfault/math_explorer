import os
import json
import re
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from config import TYPESAFE_KEY_PATH
from cache_manager import cache

class JevRouter:
    """
    TypeSafe JEV System One semantic routing and evaluation client.
    Runs in <100ms with SQLite query and relevance caching.
    """
    API_URL = "https://api.typesafe.ai/v1/systemone"
    PLAN_VERSION = "route.v2"
    RELEVANCE_VERSION = "relevance.v2"

    def __init__(self, key_path: str = TYPESAFE_KEY_PATH):
        self.api_key = self._load_key(key_path)

    def _load_key(self, key_path: str) -> str:
        env_key = os.environ.get("TYPESAFE_API_KEY")
        if env_key:
            return env_key.strip()
        
        expanded_path = os.path.expanduser(key_path)
        if os.path.exists(expanded_path):
            with open(expanded_path, "r", encoding="utf-8") as f:
                return f.read().strip()

        return ""

    def evaluate(self, state: Any, questions: Dict[str, Any], model: str = "jev-latest") -> Dict[str, Any]:
        """Compatibility helper returning only typed answers."""
        return self.evaluate_full(state, questions, model).get("answers", {})

    def evaluate_full(self, state: Any, questions: Dict[str, Any], model: str = "jev-latest") -> Dict[str, Any]:
        """Send an evaluation request without discarding response metadata."""
        if not self.api_key:
            raise FileNotFoundError("TypeSafe API key not configured")
        payload = {
            "state": state,
            "model": model,
            "questions": questions
        }
        
        req = urllib.request.Request(
            self.API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "MathExplorer/1.0"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not isinstance(data.get("answers"), dict):
                    raise RuntimeError("JEV response did not contain typed answers")
                return data
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"JEV API Error {e.code}: {err_body}")

    @staticmethod
    def _fallback_plan(query: str, failure_mode: str) -> Dict[str, Any]:
        """Cheap local policy used only when JEV is unavailable."""
        text = query.lower()
        needs_literature = any(word in text for word in ("paper", "arxiv", "literature", "recent", "state of the art"))
        if "counterexample" in text or "disprove" in text:
            family = "counterexample_search"
        elif "prove" in text or "proof" in text:
            family = "symbolic_proof"
        elif needs_literature:
            family = "literature_grounded"
        else:
            family = "other"
        return {
            "plan_version": JevRouter.PLAN_VERSION,
            "task_family": family,
            "domain": "general",
            "difficulty_score": 2.0,
            "difficulty_confidence": 0.0,
            "verification_burden": 3.0,
            "verification_confidence": 0.0,
            "routing_confidence": 0.0,
            "needs_arxiv": needs_literature,
            "needs_arxiv_prob": float(needs_literature),
            "raw_answers": {},
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "model": None,
            "failure_mode": failure_mode,
            "cached": False,
        }

    def plan_exploration(self, query: str) -> Dict[str, Any]:
        """Ask one fan-out request for reusable planning signals; policy stays in code."""
        cache_key = cache.hash_key(self.PLAN_VERSION, query)
        cached = cache.get("route_cache", cache_key)
        if cached:
            cached["cached"] = True
            return cached

        state = {
            "query": query,
            "exact_preflight_status": "not_solved",
            "available_handlers": [
                "sympy_or_z3",
                "local_violetto",
                "codex_astra",
                "literature_retrieval",
                "counterexample_search",
            ],
        }
        questions = {
            "task_family": {
                "type": "choice",
                "instructions": "What kind of mathematical exploration is `query`?",
                "criteria": {
                    "exact_computation": "A deterministic exact computation or equation solve.",
                    "finite_search": "A bounded enumeration or constraint search.",
                    "symbolic_proof": "A self-contained theorem or proof task.",
                    "counterexample_search": "A claim should be tested or disproved by a counterexample.",
                    "literature_grounded": "The answer depends on external mathematical sources.",
                    "open_conjecture": "An open-ended or research-level conjecture exploration.",
                    "other": "None of the above.",
                },
            },
            "domain": {
                "type": "choice",
                "instructions": "Identify the primary mathematical domain of this query.",
                "criteria": {
                    "number_theory": "Number Theory, Primes, Congruences, Reciprocity",
                    "algebra": "Abstract Algebra, Groups, Rings, Fields, Linear Algebra",
                    "combinatorics": "Discrete Mathematics, Permutations, Graph Theory",
                    "analysis": "Calculus, Real Analysis, Complex Analysis, Differential Equations",
                    "geometry": "Euclidean, Differential, or Algebraic Geometry and Topology",
                    "logic": "Mathematical Logic, Set Theory, Foundations"
                }
            },
            "difficulty": {
                "type": "score",
                "instructions": "Rate the computational and theoretical difficulty of this problem.",
                "criteria": [
                    "Elementary / School Math",
                    "Competition Intermediate (AMC)",
                    "Olympiad / AIME / Putnam Hard",
                    "Advanced Research / University Frontier"
                ]
            },
            "needs_arxiv": {
                "type": "noul",
                "instructions": "Does answering `query` require external mathematical literature?",
                "criteria": {
                    "true": "A theorem, definition, recent result, or source outside the query is needed.",
                    "false": "The problem is self-contained.",
                },
            },
            "verification_burden": {
                "type": "score",
                "instructions": "How difficult will it be to verify a proposed answer to `query` deterministically?",
                "criteria": [
                    "Direct exact verification.",
                    "Bounded symbolic or finite verification.",
                    "Only partial proof checking is practical.",
                    "No complete deterministic verifier is available.",
                ],
            },
        }

        try:
            response = self.evaluate_full(state=state, questions=questions)
        except Exception as exc:
            return self._fallback_plan(query, type(exc).__name__)

        answers = response["answers"]
        family_answer = answers.get("task_family", {})
        difficulty_answer = answers.get("difficulty", {})
        verification_answer = answers.get("verification_burden", {})
        domain = answers.get("domain", {}).get("choice", "number_theory")
        difficulty_score = difficulty_answer.get("score", 2.0)
        needs_arxiv_prob = answers.get("needs_arxiv", {}).get("noul", 0.0)

        res = {
            "plan_version": self.PLAN_VERSION,
            "task_family": family_answer.get("choice", "other"),
            "domain": domain,
            "difficulty_score": difficulty_score,
            "difficulty_confidence": difficulty_answer.get("confidence", 0.0),
            "verification_burden": verification_answer.get("score", 3.0),
            "verification_confidence": verification_answer.get("confidence", 0.0),
            "routing_confidence": min(
                family_answer.get("confidence", 0.0),
                difficulty_answer.get("confidence", 0.0),
            ),
            "needs_arxiv": needs_arxiv_prob >= 0.5,
            "needs_arxiv_prob": needs_arxiv_prob,
            "raw_answers": answers,
            "usage": response.get("usage", {}),
            "model": response.get("model"),
            "failure_mode": None,
            "cached": False
        }
        cache.set("route_cache", cache_key, res)
        return res

    route_intent = plan_exploration

    @staticmethod
    def _lexical_relevance(query: str, text: str) -> float:
        query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_words:
            return 0.0
        text_words = set(re.findall(r"[a-z0-9]+", text.lower()))
        return len(query_words & text_words) / len(query_words)

    def assess_node_relevance(self, query: str, node_title: str, node_text: str = "") -> Dict[str, Any]:
        """Return inspectable relevance dimensions plus their policy-weighted score."""
        cache_key = cache.hash_key(self.RELEVANCE_VERSION, query, node_title, node_text[:800])
        cached = cache.get("jev_score", cache_key)
        if cached is not None:
            return cached

        state = {
            "query": query,
            "candidate": {"title": node_title, "text": node_text[:800]},
        }
        questions = {
            "direct_support": {
                "type": "noul",
                "instructions": "Does `candidate` directly support answering `query`?",
            },
            "contains_needed_method": {
                "type": "noul",
                "instructions": "Does `candidate` contain a theorem or method needed for `query`?",
            },
            "matches_notation_and_scope": {
                "type": "noul",
                "instructions": "Does `candidate` match the notation and mathematical scope of `query`?",
            },
        }
        try:
            response = self.evaluate_full(state=state, questions=questions)
            answers = response["answers"]
            signals = {
                key: float(answers.get(key, {}).get("noul", 0.0))
                for key in questions
            }
            failure_mode = None
            usage = response.get("usage", {})
            model = response.get("model")
        except Exception as exc:
            fallback = self._lexical_relevance(query, f"{node_title} {node_text}")
            signals = {key: fallback for key in questions}
            failure_mode = type(exc).__name__
            usage = {"input_tokens": 0, "output_tokens": 0}
            model = None

        result = {
            "score": 0.60 * signals["direct_support"]
                     + 0.25 * signals["contains_needed_method"]
                     + 0.15 * signals["matches_notation_and_scope"],
            "signals": signals,
            "usage": usage,
            "model": model,
            "failure_mode": failure_mode,
        }
        if failure_mode is None:
            cache.set("jev_score", cache_key, result)
        return result

    def score_node_relevance(self, query: str, node_title: str, node_text: str = "") -> float:
        return float(self.assess_node_relevance(query, node_title, node_text)["score"])

    def rank_nodes(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        scored = []
        for cand in candidates:
            score = self.score_node_relevance(
                query=query,
                node_title=cand.get("title", ""),
                node_text=cand.get("text", "")
            )
            cand_copy = dict(cand)
            cand_copy["jev_score"] = score
            scored.append(cand_copy)
            
        scored.sort(key=lambda x: x["jev_score"], reverse=True)
        return scored[:top_k]

    def verify_confidence_gate(self, query: str, solution: str) -> Dict[str, Any]:
        """
        Confidence gating on results with persistent caching.
        """
        cache_key = cache.hash_key("jev_gate", query, solution[-1000:])
        cached = cache.get("jev_gate", cache_key)
        if cached:
            return cached

        state = f"Query: {query}\nProposed Solution:\n{solution[-1000:]}"
        questions = {
            "is_plausible": {
                "type": "noul",
                "instructions": "Is this proposed solution mathematically coherent and plausible for the query?"
            },
            "rigor_score": {
                "type": "score",
                "instructions": "Rate the mathematical rigor and completeness of the final steps.",
                "criteria": ["Flawed / Incomplete", "Mostly sound with gaps", "Fully rigorous and clear"]
            }
        }
        res = self.evaluate(state=state, questions=questions)
        cache.set("jev_gate", cache_key, res)
        return res
