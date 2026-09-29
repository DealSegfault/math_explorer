"""Exact answers only for completely recognized problem templates."""

import re
import time
import sympy as sp
from backends.base import SolverBackend
from backends.z3_backend import Z3Backend
from verification.expressions import parse_expression


class SymPyBackend(SolverBackend):
    @property
    def name(self):
        return "sympy_cas"

    @staticmethod
    def _normalize(query):
        q = ' '.join(query.lower().split()).rstrip('.? ')
        q = re.sub(r'\.?\s*(?:give the final answer|state the count|state the final value as 1 or -1) in \\boxed\{\}$', '', q)
        return q.rstrip('.? ')

    def _parse(self, query):
        q = self._normalize(query)
        leg = re.fullmatch(r'(?:compute|calculate|evaluate) the legendre symbol \((-?\d{1,12})/(\d{1,12})\)(?: using (?:the law of|gauss\'s law of) quadratic reciprocity(?: step by step)?)?', q)
        if leg:
            a, p = map(int, leg.groups())
            if p > 2 and sp.isprime(p):
                return 'legendre', (a, p)
            return None
        div = re.fullmatch(r'how many positive integer (?:divisors|factors) of (\d{1,12}) are multiples of (\d{1,12})(?:\? note that [\d\s=^*]+)?', q)
        if div:
            n, multiple = map(int, div.groups())
            return ('divisors', (n, multiple)) if n > 0 and multiple > 0 else None
        sieve = Z3Backend()._parse_divisibility_sieve(q)
        if sieve:
            return 'sieve', sieve
        cubic = re.fullmatch(r'in the finite field f_(\d{1,12}) with (\d{1,12}) elements, determine the number of nonzero elements that are cubic residues \(cubes of nonzero elements\)', q)
        if cubic:
            p, size = map(int, cubic.groups())
            return ('cubes', (p,)) if p == size and sp.isprime(p) else None
        polynomial = re.fullmatch(r'find the number of positive integers n\s*<=\s*(\d{1,12}) such that (?:the polynomial )?x\^2\s*\+\s*x\s*\+\s*1 divides x\^\(2n\)\s*\+\s*1 in r\[x\]', q)
        if polynomial:
            return 'roots', (int(polynomial.group(1)),)
        expression = re.sub(r'^(?:compute|calculate|evaluate|simplify)\s+', '', q)
        try:
            value = parse_expression(expression)
            if value.is_number:
                return 'arithmetic', (value,)
        except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError):
            pass
        return None

    def can_handle(self, query, domain="general"):
        return self._parse(query) is not None

    def solve(self, query, context=None, prompt_style="standard", **kwargs):
        start = time.perf_counter()
        parsed = self._parse(query)
        answer = None
        explanation = "SymPy could not recognize the complete problem; no exact answer is claimed."
        if parsed:
            kind, args = parsed
            if kind == 'legendre':
                answer = sp.legendre_symbol(*args)
                explanation = f"Exact Legendre symbol for a={args[0]}, p={args[1]}, with p an odd prime."
            elif kind == 'divisors':
                n, multiple = args
                answer = sp.divisor_count(n // multiple) if n % multiple == 0 else 0
                explanation = f"Divisors of {n} divisible by {multiple} correspond to divisors of {n}/{multiple}, when the quotient is an integer."
            elif kind == 'sieve':
                from itertools import combinations
                from math import lcm
                bound, positive, negative = args
                base = lcm(*positive)
                answer = bound // base
                for size in range(1, len(negative) + 1):
                    answer += (-1)**size * sum(bound // lcm(base, *subset) for subset in combinations(negative, size))
                explanation = f"Inclusion-exclusion on positive integers from 1 through {bound}, divisible by {positive}, excluding multiples of {negative}."
            elif kind == 'cubes':
                p = args[0]
                answer = (p - 1) // sp.gcd(3, p - 1)
                explanation = f"The nonzero elements of F_{p} form a cyclic group of order {p-1}; the cube map has kernel size gcd(3, {p-1})."
            elif kind == 'roots':
                answer = 0
                explanation = "A primitive cube root of unity has order 3; no power is -1. Thus x^2+x+1 never divides x^(2n)+1."
            else:
                answer = args[0]
                explanation = "Exact evaluation of the complete arithmetic expression."
        solution = explanation
        if answer is not None:
            solution += f"\nFinal Answer: $\\boxed{{{sp.latex(answer)}}}$"
        return {"engine": self.name, "solution": solution,
                "extracted_answer": str(answer) if answer is not None else None,
                "is_exact": answer is not None,
                "verification_evidence": {"exact_result": str(answer), "method": "sympy_cas"} if answer is not None else None,
                "metrics": {"execution_time_sec": round(time.perf_counter() - start, 4), "cas": "sympy"},
                "trace": [explanation]}
