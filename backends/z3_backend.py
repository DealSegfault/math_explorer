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
from formal_claims import check_claim
from typing import Dict, Any, Optional, List, Tuple
from backends.base import SolverBackend

class Z3Backend(SolverBackend):
    @property
    def name(self) -> str:
        return "z3_smt"

    def can_handle(self, query: str, domain: str = "general") -> bool:
        return self._parse_divisibility_sieve(query) is not None

    def _parse_divisibility_sieve(self, query: str) -> Optional[Tuple[int, List[int], List[int]]]:
        q = ' '.join(query.lower().split()).rstrip('. ')
        q = re.sub(r"\.?\s*give the final answer in \\boxed\{\}$", '', q).rstrip('. ')
        match = re.fullmatch(
            r'count (?:all )?(?:positive )?integers n\s*(<=|<)\s*(\d{1,12}) (?:such that n is )?divisible by (\d{1,12})(.*)', q)
        if not match:
            return None
        comparison, bound, divisor, tail = match.groups()
        negative = []
        while tail:
            clause = re.match(r'(?:,\s*(?:and )?| and )not (?:divisible by )?(\d{1,12})', tail)
            if not clause:
                return None
            negative.append(int(clause.group(1)))
            tail = tail[clause.end():]
        positive = [int(divisor)]
        if 0 in positive + negative or len(negative) > 8:
            return None
        return max(0, int(bound) - (comparison == '<')), positive, list(dict.fromkeys(negative))

    def find_counterexample(
        self,
        free_vars: List[str],
        condition_expr: str,
        domain_bounds: Dict[str, Tuple[int, int]]
    ) -> Dict[str, Any]:
        """Check the entire declared integer domain with Z3, never sample 50 points."""
        result = check_claim({"variables": {v: list(domain_bounds[v]) for v in free_vars},
                              "assumptions": [], "conclusion": condition_expr.strip()})
        return {"sat": True if result["status"] == "COUNTEREXAMPLE" else False if result["status"] == "BOUNDED_VALID" else None,
                "counterexample_found": result["status"] == "COUNTEREXAMPLE",
                "model": result.get("model"), "message": result["status"], "scope": result.get("bounds")}

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
        if sieve_data and sieve_data[0] <= 10000:
            max_bound, pos_divs, neg_divs = sieve_data
            solver = z3.Solver()
            solver.set(timeout=2000)
            n = z3.Int('n')
            solver.add(n > 0)
            solver.add(n <= max_bound)
            for d in pos_divs:
                solver.add(n % d == 0)
            for nd in neg_divs:
                solver.add(n % nd != 0)

            # Enumerate solutions by repeated model extraction
            solutions = []
            status = solver.check()
            while status == z3.sat:
                m = solver.model()
                val = m[n].as_long()
                solutions.append(val)
                solver.add(n != val) # block current solution
                if time.time() - t0 > 5:
                    status = z3.unknown
                    break
                status = solver.check()
            if status != z3.unsat:
                return {"engine": self.name, "solution": "Z3 did not complete bounded enumeration.",
                        "extracted_answer": None, "is_exact": False, "metrics": {"solver": "z3"}}

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
