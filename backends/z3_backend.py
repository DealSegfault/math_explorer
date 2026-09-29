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
from verification.expressions import parse_expression

class Z3Backend(SolverBackend):
    @property
    def name(self) -> str:
        return "z3_smt"

    def can_handle(self, query: str, domain: str = "general") -> bool:
        return self._parse_divisibility_sieve(query) is not None

    def _parse_divisibility_sieve(self, query: str) -> Optional[Tuple[int, List[int], List[int]]]:
        # Only recognize the complete bounded counting request, never a substring.
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
        upper = max(0, int(bound) - (comparison == '<'))
        return upper, positive, list(dict.fromkeys(negative))

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
        unknown = {"sat": None, "counterexample_found": False, "message": "Claim could not be checked by Z3."}
        if not free_vars or any(v not in domain_bounds for v in free_vars):
            return unknown
        try:
            variables = {v: z3.Int(v) for v in free_vars}
            symbols = {v: sp.Symbol(v, integer=True) for v in free_vars}
            claim = parse_expression(condition_expr, relational=True).xreplace(
                {sp.Symbol(name): symbol for name, symbol in symbols.items()})
            if claim.free_symbols - set(symbols.values()):
                return unknown

            def translate(node):
                if node == sp.true: return z3.BoolVal(True)
                if node == sp.false: return z3.BoolVal(False)
                if node.is_Integer: return z3.IntVal(int(node))
                if node.is_Symbol: return variables[str(node)]
                args = [translate(arg) for arg in node.args]
                if isinstance(node, sp.Equality): return args[0] == args[1]
                if isinstance(node, sp.Unequality): return args[0] != args[1]
                if isinstance(node, sp.StrictLessThan): return args[0] < args[1]
                if isinstance(node, sp.LessThan): return args[0] <= args[1]
                if isinstance(node, sp.StrictGreaterThan): return args[0] > args[1]
                if isinstance(node, sp.GreaterThan): return args[0] >= args[1]
                if isinstance(node, sp.Add): return sum(args)
                if isinstance(node, sp.Mul):
                    value = z3.IntVal(1)
                    for arg in args: value *= arg
                    return value
                if isinstance(node, sp.Mod): return args[0] % args[1]
                if isinstance(node, sp.Pow) and node.exp.is_Integer and 0 <= node.exp <= 16:
                    return args[0] ** int(node.exp)
                if isinstance(node, sp.And): return z3.And(*args)
                if isinstance(node, sp.Or): return z3.Or(*args)
                if isinstance(node, sp.Not): return z3.Not(args[0])
                raise ValueError(f"Unsupported expression: {node.func}")

            counterclaim = translate(claim)
            if not z3.is_bool(counterclaim):
                return unknown
            solver = z3.Solver()
            solver.set(timeout=5000)
            for name, variable in variables.items():
                low, high = domain_bounds[name]
                solver.add(variable >= low, variable <= high)
            solver.add(z3.Not(counterclaim))
            result = solver.check()
            if result == z3.sat:
                model = solver.model()
                assignment = {v: model.eval(variables[v], model_completion=True).as_long() for v in free_vars}
                return {"sat": True, "counterexample_found": True, "model": assignment,
                        "message": "SAT(bounds ∧ ¬claim): counterexample found."}
            if result == z3.unsat:
                return {"sat": False, "counterexample_found": False,
                        "message": "SAT(bounds ∧ ¬claim) is UNSAT."}
            return {"sat": None, "counterexample_found": False, "message": solver.reason_unknown()}
        except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError, z3.Z3Exception):
            return unknown

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
                return {"engine": self.name, "solution": "Z3 could not complete model enumeration.",
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
                "verification_evidence": {
                    "exact_result": str(len(solutions)),
                    "constraints": {"lower": 1, "upper": max_bound, "divisible_by": pos_divs, "not_divisible_by": neg_divs},
                    "witness": sorted(solutions),
                    "unsat_certificate": "Z3 exhausted the bounded model set after blocking each witness"
                },
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
