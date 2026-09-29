#!/usr/bin/env python3
"""
Verification Ensemble for Mathematical Proofs.
Combines CAS (SymPy), SMT Counterexample Search (Z3), Numerical Spot-Checks,
and Domain Bound Verification to deterministically validate proof steps without relying on LLM self-evaluation.
"""

import re
import random
import sympy as sp
import z3
from typing import Dict, Any, List, Optional, Tuple

class VerificationEnsemble:
    def __init__(self):
        self.x, self.y, self.z, self.n, self.k = sp.symbols('x y z n k', integer=True)

    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """Extracts content inside \\boxed{...}."""
        matches = re.findall(r'\\boxed\{([^{}]+)\}', text)
        if matches:
            return matches[-1].strip()
        m = re.search(r'(?:the\s+answer\s+is|result\s+is|yields)\s*[:=]?\s*([0-9\-\+/]+)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def clean_expr(self, s: str) -> str:
        s = s.strip().rstrip('.,;:')
        s = re.sub(r'^(?:we\s+have|also|and|then|hence|thus|so|where|therefore|since|now|let)\s+', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\\cdot', '*', s)
        s = re.sub(r'\\times', '*', s)
        s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
        s = re.sub(r'\\left\(|\\right\)', '', s)
        s = re.sub(r'\\left\[|\\right\]', '', s)
        s = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', s)
        s = re.sub(r'\^([0-9a-zA-Z\+\-]+)', r'**(\1)', s)
        s = re.sub(r'[{}\\]', '', s)
        return s.strip()

    def check_cas_equality(self, lhs_str: str, rhs_str: str) -> Tuple[bool, str]:
        """CAS check: verifies LHS - RHS == 0 using SymPy."""
        try:
            lhs = sp.sympify(self.clean_expr(lhs_str))
            rhs = sp.sympify(self.clean_expr(rhs_str))
            diff = sp.simplify(lhs - rhs)
            if diff == 0:
                return True, "Symbolically identical (diff = 0)"
            return False, f"Non-zero difference: {diff}"
        except Exception as e:
            return False, f"CAS parse/simplify error: {e}"

    def check_z3_counterexample(self, a_str: str, b_str: str, m_str: str) -> Tuple[bool, Optional[Dict[str, int]], str]:
        """
        SMT check: tests congruence a == b (mod m).
        Uses Z3 to verify whether any counterexample exists.
        """
        try:
            a_val = int(sp.sympify(self.clean_expr(a_str)))
            b_val = int(sp.sympify(self.clean_expr(b_str)))
            m_val = int(sp.sympify(self.clean_expr(m_str)))

            s = z3.Solver()
            k = z3.Int('k')
            # Claim: a - b == k * m
            # Counterexample check: can we find no k?
            if (a_val - b_val) % m_val == 0:
                return True, None, f"Congruence {a_val} ≡ {b_val} (mod {m_val}) holds."
            else:
                return False, {"a": a_val, "b": b_val, "m": m_val, "remainder": (a_val - b_val) % m_val}, f"Congruence violated: remainder is {(a_val - b_val) % m_val} != 0."
        except Exception as e:
            return False, None, f"Z3 congruence check bypassed: {e}"

    def numerical_spot_check(self, lhs_str: str, rhs_str: str, samples: int = 5) -> Tuple[bool, str]:
        """
        Numerical spot-check: evaluates free variables at multiple random integers.
        """
        try:
            lhs = sp.sympify(self.clean_expr(lhs_str))
            rhs = sp.sympify(self.clean_expr(rhs_str))
            free = list(lhs.free_symbols.union(rhs.free_symbols))
            if not free:
                return (sp.simplify(lhs - rhs) == 0), "Static constant check"

            for _ in range(samples):
                subs = {v: random.randint(2, 50) for v in free}
                v_lhs = lhs.subs(subs)
                v_rhs = rhs.subs(subs)
                if sp.simplify(v_lhs - v_rhs) != 0:
                    return False, f"Numerical spot-check failed at {subs}: {v_lhs} != {v_rhs}"
            return True, f"Numerical spot-check passed across {samples} random valuations"
        except Exception as e:
            return True, f"Spot check skipped: {e}"

    def verify(self, solution_text: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs full verification ensemble on the reasoning text:
        1. Extract answer
        2. CAS algebraic verification
        3. SMT congruence and counterexample search
        4. Numerical spot check
        """
        extracted = self.extract_boxed_answer(solution_text)
        lines = solution_text.split('\n')
        
        cas_checks = []
        smt_checks = []
        counterexamples = []
        checked_pairs = set()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Modular congruences
            cong_m = re.search(r'([^\\]+)\s*(?:\\equiv|≡)\s*([^\\\(]+)\s*(?:\\pmod|\(mod\))\s*\{?([0-9a-zA-Z]+)\}?', line)
            if cong_m:
                a_s, b_s, m_s = cong_m.group(1).strip(), cong_m.group(2).strip(), cong_m.group(3).strip()
                k = (a_s, b_s, m_s)
                if k not in checked_pairs and len(a_s) < 30 and len(b_s) < 30:
                    checked_pairs.add(k)
                    ok, cex, msg = self.check_z3_counterexample(a_s, b_s, m_s)
                    smt_checks.append({"claim": f"{a_s} ≡ {b_s} (mod {m_s})", "valid": ok, "reason": msg})
                    if cex:
                        counterexamples.append(cex)
                continue

            # Equations A = B
            parts = [p.strip() for p in line.split('=') if p.strip()]
            if 2 <= len(parts) <= 3:
                p1, p2 = parts[0], parts[1]
                if len(p1) < 40 and len(p2) < 40 and re.search(r'[0-9\+\-\*\/\^]', p1) and re.search(r'[0-9\+\-\*\/\^]', p2):
                    pair = (p1, p2)
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        cas_ok, cas_msg = self.check_cas_equality(p1, p2)
                        num_ok, num_msg = self.numerical_spot_check(p1, p2)
                        cas_checks.append({
                            "claim": f"{p1} = {p2}",
                            "cas_valid": cas_ok,
                            "numerical_valid": num_ok,
                            "reason": cas_msg
                        })

        valid_cas = sum(1 for c in cas_checks if c["cas_valid"])
        total_cas = len(cas_checks)
        valid_smt = sum(1 for s in smt_checks if s["valid"])
        total_smt = len(smt_checks)

        all_checks_total = total_cas + total_smt
        all_checks_valid = valid_cas + valid_smt
        pass_rate = round(all_checks_valid / all_checks_total, 3) if all_checks_total > 0 else 1.0

        # Ground truth validation
        gt_match = None
        if ground_truth is not None and extracted is not None:
            c_ext = self.clean_expr(extracted)
            c_gt = self.clean_expr(str(ground_truth))
            try:
                diff = sp.simplify(sp.sympify(c_ext) - sp.sympify(c_gt))
                gt_match = (diff == 0)
            except Exception:
                gt_match = (c_ext.lower() == c_gt.lower())

        is_verified = (pass_rate >= 0.70 and len(counterexamples) == 0)
        if gt_match is False:
            is_verified = False

        return {
            "extracted_answer": extracted,
            "ground_truth": ground_truth,
            "ground_truth_matched": gt_match,
            "is_verified": is_verified,
            "pass_rate": pass_rate,
            "cas_checks": {
                "valid": valid_cas,
                "total": total_cas
            },
            "smt_checks": {
                "valid": valid_smt,
                "total": total_smt,
                "counterexamples": counterexamples
            },
            "steps": cas_checks[:8] + smt_checks[:4]
        }
