#!/usr/bin/env python3
"""
Symbolic Verifier for Mathematical Reasoning Traces.
Leverages SymPy to extract mathematical statements, verify equalities,
check modular congruences, validate arithmetic transitions, and isolate final boxed answers.
"""

import re
import math
import sympy as sp
from verification.expressions import parse_expression
from typing import Dict, Any, List, Optional, Tuple

class SymbolicVerifier:
    def __init__(self):
        # Common mathematical symbols pre-declared
        self.x, self.y, self.z, self.n, self.k, self.p, self.q = sp.symbols('x y z n k p q', integer=True)

    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """Extracts content inside \boxed{...}."""
        # Find all occurrences of \boxed{...} accounting for nested braces
        matches = re.findall(r'\\boxed\{([^{}]+)\}', text)
        if matches:
            return matches[-1].strip()
        # Fallback regex for "The answer is X" or "Result: X"
        m = re.search(r'(?:the\s+answer\s+is|result\s+is|yields)\s*[:=]?\s*([0-9\-\+/]+)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def clean_latex_expr(self, expr_str: str) -> str:
        """Sanitizes basic LaTeX syntax and strips prose into SymPy parsable string."""
        s = expr_str.strip()
        s = s.rstrip('.,;:')
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

    def verify_arithmetic_equality(self, lhs_str: str, rhs_str: str) -> Tuple[bool, str]:
        """Checks if LHS == RHS using SymPy symbolic evaluation."""
        try:
            lhs_clean = self.clean_latex_expr(lhs_str)
            rhs_clean = self.clean_latex_expr(rhs_str)
            
            lhs_val = parse_expression(lhs_clean)
            rhs_val = parse_expression(rhs_clean)
            
            diff = sp.simplify(lhs_val - rhs_val)
            if diff == 0:
                return True, "Symbolically identical (diff = 0)"
            else:
                return False, f"Difference is non-zero: {diff}"
        except Exception as e:
            return False, f"SymPy parse error: {e}"

    def verify_congruence(self, a_str: str, b_str: str, m_str: str) -> Tuple[bool, str]:
        """Checks if a = b (mod m)."""
        try:
            a = parse_expression(self.clean_latex_expr(a_str))
            b = parse_expression(self.clean_latex_expr(b_str))
            m = parse_expression(self.clean_latex_expr(m_str))
            if (a - b) % m == 0:
                return True, f"Congruence holds modulo {m}"
            return False, f"{a} is not congruent to {b} modulo {m}"
        except Exception as e:
            return False, f"Parse error: {e}"

    def extract_and_verify_steps(self, text: str) -> List[Dict[str, Any]]:
        """
        Parses text for candidate equality steps:
        e.g. `A = B = C`, `X = Y`, or simple arithmetic like `83 - 28 = 55`.
        """
        results = []
        # Match standard equality statements
        # Regex for equations: e.g. "83 - 28 = 55" or "(-1)^30 = 1"
        eq_patterns = [
            r'([0-9\+\-\*\/\^\(\)\.\s]+)\s*=\s*([0-9\+\-\*\/\^\(\)\.\s]+)',
            r'([a-zA-Z0-9\+\-\*\/\^\(\)\s]+)\s*=\s*([a-zA-Z0-9\+\-\*\/\^\(\)\s]+)'
        ]
        
        lines = text.split('\n')
        checked_pairs = set()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for congruences: a \equiv b \pmod{m}
            cong_match = re.search(r'([^\\]+)\s*(?:\\equiv|≡)\s*([^\\\(]+)\s*(?:\\pmod|\(mod\))\s*\{?([0-9a-zA-Z]+)\}?', line)
            if cong_match:
                a_str, b_str, m_str = cong_match.group(1).strip(), cong_match.group(2).strip(), cong_match.group(3).strip()
                pair_key = (a_str, b_str, m_str)
                if pair_key not in checked_pairs and len(a_str) < 40 and len(b_str) < 40:
                    checked_pairs.add(pair_key)
                    valid, reason = self.verify_congruence(a_str, b_str, m_str)
                    results.append({
                        "type": "modular_congruence",
                        "claim": f"{a_str} ≡ {b_str} (mod {m_str})",
                        "valid": valid,
                        "reason": reason
                    })
                continue

            # Check chained equalities: A = B = C
            parts = [p.strip() for p in line.split('=') if p.strip()]
            if 2 <= len(parts) <= 4:
                for i in range(len(parts) - 1):
                    p1 = parts[i]
                    p2 = parts[i+1]
                    # Filter out non-math English sentences
                    if any(word in p1.lower() for word in ['the', 'let', 'so', 'where', 'then', 'because', 'when']):
                        # Try to isolate math at the end
                        sub_m = re.search(r'([0-9a-zA-Z\+\-\*\/\^\(\)\.\s]+)$', p1)
                        if sub_m:
                            p1 = sub_m.group(1).strip()
                    if any(word in p2.lower() for word in ['the', 'let', 'so', 'where', 'then', 'because', 'when']):
                        sub_m = re.search(r'^([0-9a-zA-Z\+\-\*\/\^\(\)\.\s]+)', p2)
                        if sub_m:
                            p2 = sub_m.group(1).strip()

                    if not p1 or not p2 or len(p1) > 40 or len(p2) > 40:
                        continue
                    # Must contain digits or operators
                    if not re.search(r'[0-9\+\-\*\/\^]', p1) or not re.search(r'[0-9\+\-\*\/\^]', p2):
                        continue

                    pair_key = (p1, p2)
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)

                    valid, reason = self.verify_arithmetic_equality(p1, p2)
                    results.append({
                        "type": "algebraic_identity",
                        "claim": f"{p1} = {p2}",
                        "valid": valid,
                        "reason": reason
                    })

        return results

    def verify_solution(self, solution_text: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs comprehensive symbolic verification of a solution:
        1. Extracts final answer
        2. Compares with ground truth if available
        3. Verifies intermediate mathematical equalities and congruences
        """
        extracted_answer = self.extract_boxed_answer(solution_text)
        steps = self.extract_and_verify_steps(solution_text)
        
        valid_steps = sum(1 for s in steps if s["valid"])
        total_steps = len(steps)
        pass_rate = round(valid_steps / total_steps, 3) if total_steps > 0 else 1.0

        # Check ground truth match
        ground_truth_matched = None
        if ground_truth is not None and extracted_answer is not None:
            # Direct string comparison or SymPy numerical equality
            clean_ans = self.clean_latex_expr(extracted_answer)
            clean_gt = self.clean_latex_expr(str(ground_truth))
            try:
                diff = sp.simplify(parse_expression(clean_ans) - parse_expression(clean_gt))
                ground_truth_matched = (diff == 0)
            except Exception:
                ground_truth_matched = (clean_ans.lower() == clean_gt.lower())

        is_sound = (pass_rate >= 0.75) if total_steps > 0 else True
        if ground_truth_matched is False:
            is_sound = False

        return {
            "extracted_answer": extracted_answer,
            "ground_truth": ground_truth,
            "ground_truth_matched": ground_truth_matched,
            "total_steps_checked": total_steps,
            "valid_steps": valid_steps,
            "step_pass_rate": pass_rate,
            "is_formally_sound": is_sound,
            "verified_steps": steps[:10]  # top 10 verified steps
        }

if __name__ == "__main__":
    verifier = SymbolicVerifier()
    sample_text = (
        "We have 83 - 28 = 55.\n"
        "Also (-1)^30 = 1.\n"
        "And 13 \\equiv 2 \\pmod{11}.\n"
        "Thus the final result is \\boxed{55}."
    )
    res = verifier.verify_solution(sample_text, ground_truth="55")
    print("Verification Result:", res)
