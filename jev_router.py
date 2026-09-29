import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

class JevRouter:
    """
    TypeSafe JEV System One semantic routing and evaluation client.
    Runs in <100ms: turns fuzzy natural language & math queries into typed decisions.
    """
    API_URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, key_path: str = "/Users/mac/.typesafe_key"):
        self.api_key = self._load_key(key_path)

    def _load_key(self, key_path: str) -> str:
        env_key = os.environ.get("TYPESAFE_API_KEY")
        if env_key:
            return env_key.strip()
        
        expanded_path = os.path.expanduser(key_path)
        if os.path.exists(expanded_path):
            with open(expanded_path, "r", encoding="utf-8") as f:
                return f.read().strip()
                
        raise FileNotFoundError(f"TypeSafe API key not found in {expanded_path} or TYPESAFE_API_KEY env var.")

    def evaluate(self, state: Any, questions: Dict[str, Any], model: str = "jev-latest") -> Dict[str, Any]:
        """
        Sends an evaluation request to TypeSafe System One.
        """
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
                return data.get("answers", {})
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"JEV API Error {e.code}: {err_body}")

    def route_intent(self, query: str) -> Dict[str, Any]:
        """
        High-speed multi-primitive intent and capability routing (<100ms):
        - Choice: Math domain categorization (dict criteria)
        - Score: Difficulty / Complexity (1 to 4) (list criteria)
        - Noul: Needs external literature search (arXiv)
        - Choice: Optimal execution engine (dict criteria)
        """
        questions = {
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
                "instructions": (
                    "Does this problem refer to recent academic literature, named modern conjectures, "
                    "research papers, or specialized monographs that require searching arXiv?"
                )
            },
            "recommended_engine": {
                "type": "choice",
                "instructions": "Choose the best execution engine for this query.",
                "criteria": {
                    "local_violetto": "Fast specialized local math solver on Apple Silicon MPS (<5s)",
                    "codex_astra": "Frontier deep reasoning model with xhigh thinking effort for hard proofs or open conjectures",
                    "python_solver": "Pure deterministic script, exact symbolic calculation or brute force"
                }
            }
        }
        
        answers = self.evaluate(state=query, questions=questions)
        
        domain = answers.get("domain", {}).get("choice", "number_theory")
        difficulty_score = answers.get("difficulty", {}).get("score", 2.0)
        needs_arxiv_prob = answers.get("needs_arxiv", {}).get("noul", 0.0)
        engine_id = answers.get("recommended_engine", {}).get("choice", "local_violetto")

        return {
            "domain": domain,
            "difficulty_score": difficulty_score,
            "needs_arxiv": needs_arxiv_prob >= 0.5,
            "needs_arxiv_prob": needs_arxiv_prob,
            "recommended_engine": engine_id,
            "raw_answers": answers
        }

    def score_node_relevance(self, query: str, node_title: str, node_text: str = "") -> float:
        """
        Uses JEV 'noul' primitive to score calibrated probability that a document node is relevant.
        """
        state_repr = f"Section: {node_title}\n"
        if node_text:
            state_repr += f"Content: {node_text[:800]}\n"
            
        questions = {
            "relevance": {
                "type": "noul",
                "instructions": f"Does this section contain theorems or methods needed to solve: '{query}'?"
            }
        }
        answers = self.evaluate(state=state_repr, questions=questions)
        return float(answers.get("relevance", {}).get("noul", 0.0))

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
        Confidence gating on results: checks mathematical plausibility of output.
        """
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
        return self.evaluate(state=state, questions=questions)
