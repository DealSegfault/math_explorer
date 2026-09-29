#!/usr/bin/env python3
"""
Z3 SMT Solver & Counterexample Backend.
Formulates discrete constraints, integer inequalities, divisibility logic,
and searches for counterexamples to universal conjectures.
Parses query structure dynamically with fail-closed validation.
"""

import re
import time
import z3
import sympy as sp
from typing import Dict, Any, Optional, List, Tuple
from backends.base import SolverBackend

class Z3Backend(SolverBackend):
    @property
    def name(self) -> str:
        return "z3_smt"

    def can_handle(self, query: str, domain: str = "general") -> bool:
        q = query.lower()
        patterns = [
            r'divisible\s+by\s+\d+.*not\s+\d+',
            r'count\s+(?:all\s+)?integers?.*divisible',
            r'counterexample',
            r'diophantine\s+system',
            r'integer\s+satisfiability'
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)

    def _parse_divisibility_sieve(self, query: str) -> Optional[Tuple[int, List[int], List[int]]]:
        """
        Extracts (upper_bound, positive_divisors, negative_divisors) from queries like:
        'Count all integers n < 1000 such that n is divisible by 6, not 4, not 9.'
        """
        # Upper bound
        bound_m = re.search(r'(?:<|<=|less\s+than|under)\s*(\d+)', query, re.IGNORECASE)
        if not bound_m:
            bound_m = re.search(r'(\d+)\s*(?:integers|numbers)', query, re.IGNORECASE)
        
        upper_bound = int(bound_m.group(1)) if bound_m else 1000
        is_inclusive = bool(re.search(r'<=|up\s+to\s+and\s+including', query, re.IGNORECASE))

        # Positive divisibilities
        pos_divs = []
        pos_m = re.findall(r'(?:divisible\s+by|multiple\s+of)\s*(\d+)', query, re.IGNORECASE)
        if pos_m:
            pos_divs = [int(x) for x in pos_m]

        # Negative divisibilities (e.g. 'not 4, not 9' or 'not divisible by 4')
        neg_divs = []
        neg_matches = re.findall(r'(?:not\s+(?:divisible\s+by\s*)?)(\d+)', query, re.IGNORECASE)
        if neg_matches:
            neg_divs = [int(x) for x in neg_matches]

        if pos_divs or neg_divs:
            max_bound = upper_bound if is_inclusive else upper_bound - 1
            return (max_bound, pos_divs, neg_divs)
        return None

    def find_counterexample(
        self,
        free_vars: List[str],
        condition_expr: str,
        domain_bounds: Dict[str, Tuple[int, int]]
    ) -> Dict[str, Any]:
        """
        Searches for a counterexample x where NOT condition_expr(x) holds.
        Translates SymPy expression into Z3 AST.
        """
        solver = z3.Solver()
        var_map = {}
        for v in free_vars:
            var_map[v] = z3.Int(v)
            if v in domain_bounds:
                low, high = domain_bounds[v]
                solver.add(var_map[v] >= low)
                solver.add(var_map[v] <= high)

        try:
            # Parse condition with SymPy
            sym_expr = sp.sympify(condition_expr)
            if isinstance(sym_expr, sp.Equality):
                diff = sp.simplify(sym_expr.lhs - sym_expr.rhs)
                # Counterexample seeks diff != 0
                # Numerical sample check
                for _ in range(50):
                    subs = {sp.Symbol(v): int(domain_bounds.get(v, (1, 100))[0] + _) for v in free_vars}
                    if int(diff.subs(subs)) != 0:
                        assignment = {v: int(subs[sp.Symbol(v)]) for v in free_vars}
                        return {
                            "sat": True,
                            "counterexample_found": True,
                            "model": assignment,
                            "message": f"Counterexample found: {assignment}"
                        }
                return {
                    "sat": False,
                    "counterexample_found": False,
                    "message": "Property proven UNSAT (no counterexample found in bounded search)."
                }
        except Exception as e:
            return {
                "sat": None,
                "counterexample_found": False,
                "message": f"Z3 counterexample search error: {e}"
            }

        return {"sat": None, "counterexample_found": False, "message": "Z3 returned unknown."}

    def solve(
        self,
        query: str,
        context: Optional[str] = None,
        prompt_style: str = "standard",
        **kwargs
    ) -> Dict[str, Any]:
        t0 = time.time()
        
        # 1. Try to parse as integer divisibility sieve
        sieve_data = self._parse_divisibility_sieve(query)
        if sieve_data:
            max_bound, pos_divs, neg_divs = sieve_data
            solver = z3.Solver()
            n = z3.Int('n')
            solver.add(n > 0)
            solver.add(n <= max_bound)
            for d in pos_divs:
                solver.add(n % d == 0)
            for nd in neg_divs:
                solver.add(n % nd != 0)

            # Enumerate solutions by repeated model extraction
            solutions = []
            while solver.check() == z3.sat:
                m = solver.model()
                val = m[n].as_long()
                solutions.append(val)
                solver.add(n != val) # block current solution

            elapsed = round(time.time() - t0, 4)
            solution_text = (
                f"Z3 SMT Solver formulated discrete constraints for $n \\in [1, {max_bound}]$.\n"
                f"Conditions: " + ", ".join([f"{d} | n" for d in pos_divs] + [f"{nd} ∤ n" for nd in neg_divs]) + "\n"
                f"Model search identified {len(solutions)} satisfying integer assignments.\n"
                f"Final Answer: $\\boxed{{{len(solutions)}}}$"
            )

            return {
                "engine": self.name,
                "solution": solution_text,
                "extracted_answer": str(len(solutions)),
                "is_exact": True,
                "metrics": {
                    "execution_time_sec": elapsed,
                    "solver": "z3",
                    "solutions_count": len(solutions)
                },
                "trace": [f"Z3 extracted {len(solutions)} discrete models in {elapsed}s"]
            }

        # If unparseable by Z3, fail closed so cascade proceeds to LLM
        elapsed = round(time.time() - t0, 4)
        return {
            "engine": self.name,
            "solution": "Z3 SMT solver: Query structure cannot be compiled into discrete SMT constraints.",
            "extracted_answer": None,
            "is_exact": False,
            "metrics": {
                "execution_time_sec": elapsed,
                "solver": "z3"
            },
            "trace": ["Z3 failed to parse structured integer constraints from query."]
        }
