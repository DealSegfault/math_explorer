#!/usr/bin/env python3
"""
Violetto Local LLM Solver Backend (Limite 1B on Apple Silicon MPS).
Accelerated with native Hugging Face MPS generation and persistent SQLite solver caching.
"""

import time
from typing import Dict, Any, Optional
from backends.base import SolverBackend
from violetto_engine import ViolettoEngine
from config import VIOLETTO_MODEL_PATH
from cache_manager import cache

class ViolettoBackend(SolverBackend):
    def __init__(self, model_path: str = VIOLETTO_MODEL_PATH):
        self.engine = ViolettoEngine(model_path=model_path)

    @property
    def name(self) -> str:
        return "local_violetto"

    def can_handle(self, query: str, domain: str = "general") -> bool:
        return True # General math reasoning

    def solve(
        self,
        query: str,
        context: Optional[str] = None,
        prompt_style: str = "standard",
        temperature: float = 0.6,
        top_k: int = 50,
        max_tokens: int = 1500,
        **kwargs
    ) -> Dict[str, Any]:
        # Check solver cache
        cache_key = cache.hash_key(self.name, query, context or "", prompt_style, temperature, top_k, max_tokens)
        cached = cache.get("solver_output", cache_key)
        if cached:
            cached["metrics"]["cached"] = True
            cached["trace"].append("Retrieved from SQLite solver cache (0ms)")
            return cached

        t0 = time.time()
        
        # Apply active prompt style formatting
        styled_prompt = query
        if prompt_style == "lemma_stepwise_decomposition":
            styled_prompt = f"Decompose this problem into lemmas step by step before concluding.\n\nProblem:\n{query}"
        elif prompt_style == "rigorous_math_proof":
            styled_prompt = f"Provide a complete, rigorous mathematical proof. State all theorems used.\n\nProblem:\n{query}"
        elif prompt_style == "olympiad_heuristic_search":
            styled_prompt = f"Approach this competition problem by analyzing small cases, finding patterns, and establishing invariants.\n\nProblem:\n{query}"

        solution = self.engine.generate(
            prompt=styled_prompt,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stream=False
        )
        elapsed = round(time.time() - t0, 3)

        result = {
            "engine": self.name,
            "solution": solution,
            "extracted_answer": None, # Extracted by verifier
            "is_exact": False,
            "metrics": {
                "execution_time_sec": elapsed,
                "hardware": "apple_silicon_mps",
                "cached": False
            },
            "trace": [f"Executed Violetto 1B on MPS (style: {prompt_style}, temp: {temperature}, elapsed: {elapsed}s)"]
        }
        cache.set("solver_output", cache_key, result)
        return result
