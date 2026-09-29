#!/usr/bin/env python3
"""
Solver Registry & Portfolio Dispatcher.
Routes mathematical tasks to the optimal engine in the portfolio:
SymPy CAS, Z3 SMT, Local Violetto 1B (MPS), or Frontier Codex Astra (xhigh).
"""

from typing import Dict, Any, Optional, List
from backends.base import SolverBackend
from backends.sympy_backend import SymPyBackend
from backends.z3_backend import Z3Backend
from backends.violetto_backend import ViolettoBackend
from backends.codex_astra_backend import CodexAstraBackend

class SolverRegistry:
    def __init__(self, violetto_model_path: Optional[str] = None):
        self.sympy = SymPyBackend()
        self.z3 = Z3Backend()
        # Initialize LLM backends lazily or on demand
        self._violetto_path = violetto_model_path
        self._violetto: Optional[ViolettoBackend] = None
        self._astra: Optional[CodexAstraBackend] = None

    @property
    def violetto(self) -> ViolettoBackend:
        if self._violetto is None:
            self._violetto = ViolettoBackend(model_path=self._violetto_path) if self._violetto_path else ViolettoBackend()
        return self._violetto

    @property
    def astra(self) -> CodexAstraBackend:
        if self._astra is None:
            self._astra = CodexAstraBackend()
        return self._astra

    def select_backend(
        self,
        query: str,
        routing: Dict[str, Any],
        difficulty_threshold: float = 2.5,
        engine_override: Optional[str] = None
    ) -> SolverBackend:
        """
        Determines the optimal backend based on query structure, JEV recommendations, and thresholds.
        """
        if engine_override:
            if engine_override in ["sympy", "sympy_cas", "python_solver"]:
                return self.sympy
            elif engine_override in ["z3", "z3_smt"]:
                return self.z3
            elif engine_override in ["local_violetto", "violetto"]:
                return self.violetto
            elif engine_override in ["codex_astra", "astra"]:
                return self.astra

        recommended = routing.get("recommended_engine", "")
        # 1. Check if JEV recommended python_solver or SymPy can handle directly
        if recommended == "python_solver" and (self.sympy.can_handle(query) or self.z3.can_handle(query)):
            if self.z3.can_handle(query) and not self.sympy.can_handle(query):
                return self.z3
            return self.sympy

        # 2. Check if query is an exact symbolic computation
        if self.sympy.can_handle(query):
            # Prioritize deterministic CAS when applicable
            return self.sympy

        # 3. SMT / constraint check
        if self.z3.can_handle(query):
            return self.z3

        # 4. LLM Routing based on difficulty threshold
        diff = routing.get("difficulty_score", 0.0)
        if diff >= difficulty_threshold:
            return self.astra
        return self.violetto

    def get_backend_by_name(self, name: str) -> Optional[SolverBackend]:
        backends = {
            "sympy": self.sympy,
            "sympy_cas": self.sympy,
            "python_solver": self.sympy,
            "z3": self.z3,
            "z3_smt": self.z3,
            "local_violetto": self.violetto,
            "violetto": self.violetto,
            "codex_astra": self.astra,
            "astra": self.astra
        }
        return backends.get(name)

    get_backend = get_backend_by_name
