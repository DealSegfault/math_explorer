#!/usr/bin/env python3
"""
Codex Astra Frontier Solver Backend (gpt-6-astra reasoning effort: xhigh).
"""

import time
from typing import Dict, Any, Optional
from backends.base import SolverBackend
from codex_engine import CodexAstraEngine

class CodexAstraBackend(SolverBackend):
    def __init__(self):
        self.engine = CodexAstraEngine()

    @property
    def name(self) -> str:
        return "codex_astra"

    def can_handle(self, query: str, domain: str = "general") -> bool:
        return True # Universal frontier reasoning

    def solve(
        self,
        query: str,
        context: Optional[str] = None,
        prompt_style: str = "standard",
        **kwargs
    ) -> Dict[str, Any]:
        t0 = time.time()
        styled_prompt = query
        if prompt_style == "lemma_stepwise_decomposition":
            styled_prompt = f"Decompose into formal lemmas and establish each step rigorously.\n\nProblem:\n{query}"
        elif prompt_style == "rigorous_math_proof":
            styled_prompt = f"Write an uncompromising formal proof suitable for research publication.\n\nProblem:\n{query}"

        res = self.engine.generate(prompt=styled_prompt, context=context)
        elapsed = round(time.time() - t0, 2)

        return {
            "engine": self.name,
            "solution": res["answer"],
            "extracted_answer": None,
            "is_exact": False,
            "metrics": {
                "execution_time_sec": elapsed,
                "tokens_used": res.get("tokens_used", "unknown"),
                "model": "gpt-6-astra",
                "reasoning_effort": "xhigh"
            },
            "trace": [f"Executed Codex Astra xhigh (tokens: {res.get('tokens_used')})"]
        }
