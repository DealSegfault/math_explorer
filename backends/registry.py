#!/usr/bin/env python3
"""
Solver Registry & Portfolio Dispatcher.
Routes mathematical tasks to the optimal engine in the portfolio:
SymPy CAS, Z3 SMT, Local Violetto 1B (MPS), or Frontier Codex Astra (xhigh).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
from backends.base import SolverBackend
from backends.sympy_backend import SymPyBackend
from backends.z3_backend import Z3Backend

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
            from backends.violetto_backend import ViolettoBackend
            self._violetto = ViolettoBackend(model_path=self._violetto_path) if self._violetto_path else ViolettoBackend()
        return self._violetto

    @property
    def astra(self) -> CodexAstraBackend:
        if self._astra is None:
            from backends.codex_astra_backend import CodexAstraBackend
            self._astra = CodexAstraBackend()
        return self._astra

    def select_backend(
        self,
        query: str,
        routing: Dict[str, Any],
        difficulty_threshold: float = 2.5,
        confidence_threshold: float = 0.6,
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
            raise ValueError(f"Unknown engine: {engine_override}")

        # 1. Deterministic capability checks always outrank model judgments.
        if self.sympy.can_handle(query):
            return self.sympy

        # 2. SMT / constraint check
        if self.z3.can_handle(query):
            return self.z3

        # 3. JEV supplies features; code owns the confidence-gated policy.
        diff = routing.get("difficulty_score", 0.0)
        confidence = routing.get("routing_confidence", 1.0)
        if diff >= difficulty_threshold and confidence >= confidence_threshold:
            return self.astra
        return self.violetto

    def get_backend_by_name(self, name: str) -> Optional[SolverBackend]:
        attributes = {
            "sympy": "sympy", "sympy_cas": "sympy", "python_solver": "sympy",
            "z3": "z3", "z3_smt": "z3", "local_violetto": "violetto",
            "violetto": "violetto", "codex_astra": "astra", "astra": "astra"
        }
        attribute = attributes.get(name)
        return getattr(self, attribute) if attribute else None

    get_backend = get_backend_by_name
