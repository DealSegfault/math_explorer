#!/usr/bin/env python3
"""
Base interface for mathematical solver backends.
Enables pluggable execution: CAS (SymPy), SMT (Z3), Local LLM (Violetto), Frontier (Codex Astra).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class SolverBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the solver backend."""
        pass

    @abstractmethod
    def can_handle(self, query: str, domain: str = "general") -> bool:
        """Predicate checking if query can be evaluated by this solver."""
        pass

    @abstractmethod
    def solve(
        self,
        query: str,
        context: Optional[str] = None,
        prompt_style: str = "standard",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes resolution.
        Returns:
            {
                "engine": str,
                "solution": str,
                "extracted_answer": Optional[str],
                "is_exact": bool,
                "metrics": dict,
                "trace": list
            }
        """
        pass
