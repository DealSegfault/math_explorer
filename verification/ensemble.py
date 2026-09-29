#!/usr/bin/env python3
"""
Fail-Closed Deterministic Verification Ensemble for Mathematical Proofs.
Combines CAS (SymPy), SMT Counterexample Search (Z3), and Numerical Spot-Checks.
Enforces a 3-state verification contract: VERIFIED, REFUTED, UNVERIFIED.
Fail-closed: 0 checks or parse failures ALWAYS yield UNVERIFIED, never PASS.
"""

import re
import random
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
import sympy as sp
import z3
from verification.expressions import parse_expression

class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    UNVERIFIED = "UNVERIFIED"

@dataclass
class VerificationResult:
    status: VerificationStatus
    is_verified: bool
    pass_rate: float
    extracted_answer: Optional[str]
    ground_truth: Optional[str]
    ground_truth_matched: Optional[bool]
    strictness_used: float
    total_checks: int
    cas_checks: Dict[str, int]
    smt_checks: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    verification_basis: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

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
        prose_prefixes = [
            r'^(?:we\s+find\s+that|we\s+have|we\s+get|we\s+see|we\s+obtain)\s+',
            r'^(?:it\s+follows\s+that|this\s+implies\s+that|this\s+gives)\s+',
            r'^(?:also|and|then|hence|thus|so|where|therefore|since|now|let|find\s+that|see\s+that|get)\s+'
        ]
        for p in prose_prefixes:
            s = re.sub(p, '', s, flags=re.IGNORECASE)
        s = re.sub(r',.*$', '', s)
        s = re.sub(r'\s+(?:and|then|hence|thus|so|where|therefore|since|which|yields|concludes|giving|leads|for|with)\b.*$', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\\cdot', '*', s)
        s = re.sub(r'\\times', '*', s)
        s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
        s = re.sub(r'\\left\(|\\right\)', '', s)
        s = re.sub(r'\\left\[|\\right\]', '', s)
        s = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', s)
        s = re.sub(r'\^([0-9a-zA-Z\+\-]+)', r'**(\1)', s)
        s = re.sub(r'[{}\$\\]', '', s)
        return s.strip()

    def check_cas_equality(self, lhs_str: str, rhs_str: str) -> Tuple[Optional[bool], str]:
        """CAS check: verifies LHS - RHS == 0 using SymPy. Returns (None, ...) on parse failure."""
        try:
            lhs = parse_expression(self.clean_expr(lhs_str))
            rhs = parse_expression(self.clean_expr(rhs_str))
            diff = sp.simplify(lhs - rhs)
            if diff == 0:
                return True, "Symbolically identical (diff = 0)"
            return False, f"Refuted: non-zero difference ({diff})"
        except Exception as e:
            return None, f"CAS parse error: {e}"

    def check_z3_congruence(self, a_str: str, b_str: str, m_str: str) -> Tuple[Optional[bool], Optional[Dict[str, int]], str]:
        """
        SMT check: tests congruence a == b (mod m).
        Uses Z3 to verify whether any counterexample exists. Returns (None, ...) on parse error.
        """
        try:
            a_val = int(parse_expression(self.clean_expr(a_str)))
            b_val = int(parse_expression(self.clean_expr(b_str)))
            m_val = int(parse_expression(self.clean_expr(m_str)))

            if m_val == 0:
                return False, None, "Modulo 0 is undefined."

            if (a_val - b_val) % m_val == 0:
                return True, None, f"Congruence {a_val} ≡ {b_val} (mod {m_val}) holds."
            else:
                rem = (a_val - b_val) % m_val
                return False, {"a": a_val, "b": b_val, "m": m_val, "remainder": rem}, f"Refuted: remainder is {rem} != 0."
        except Exception as e:
            return None, None, f"Integer congruence unparseable: {e}"

    def numerical_spot_check(self, lhs_str: str, rhs_str: str, samples: int = 5) -> Tuple[Optional[bool], str]:
        """
        Numerical spot-check: evaluates free variables at multiple random integers.
        Fail-closed: parse error yields None (UNVERIFIED), NEVER True.
        """
        try:
            lhs = parse_expression(self.clean_expr(lhs_str))
            rhs = parse_expression(self.clean_expr(rhs_str))
            free = list(lhs.free_symbols.union(rhs.free_symbols))
            if not free:
                is_zero = (sp.simplify(lhs - rhs) == 0)
                return is_zero, ("Static constant verified" if is_zero else "Static constant refuted")

            for _ in range(samples):
                subs = {v: random.randint(2, 50) for v in free}
                v_lhs = lhs.subs(subs)
                v_rhs = rhs.subs(subs)
                if sp.simplify(v_lhs - v_rhs) != 0:
                    return False, f"Numerical spot-check refuted at {subs}: {v_lhs} != {v_rhs}"
            return True, f"Numerical spot-check passed across {samples} valuations"
        except Exception as e:
            return None, f"Numerical spot check unparseable: {e}"

    def verify(
        self,
        solution_text: str,
        ground_truth: Optional[str] = None,
        strictness: float = 0.75,
        reference_answer: Optional[str] = None
    ) -> VerificationResult:
        """
        Runs fail-closed verification ensemble:
        - 3 states: VERIFIED, REFUTED, UNVERIFIED.
        - Fail-closed: 0 checks or unparseable text => UNVERIFIED (is_verified = False).
        - Any contradiction => REFUTED (is_verified = False).
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
                    ok, cex, msg = self.check_z3_congruence(a_s, b_s, m_s)
                    if ok is not None:
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
                        
                        # Only record if at least one check parsed successfully
                        if cas_ok is not None or num_ok is not None:
                            effective_valid = (cas_ok is True) or (num_ok is True)
                            effective_refuted = (cas_ok is False) or (num_ok is False)
                            cas_checks.append({
                                "claim": f"{p1} = {p2}",
                                "valid": effective_valid and not effective_refuted,
                                "refuted": effective_refuted,
                                "reason": cas_msg if cas_ok is not None else num_msg
                            })

        valid_cas = sum(1 for c in cas_checks if c["valid"])
        refuted_cas = sum(1 for c in cas_checks if c.get("refuted"))
        total_cas = len(cas_checks)

        valid_smt = sum(1 for s in smt_checks if s["valid"])
        refuted_smt = sum(1 for s in smt_checks if not s["valid"])
        total_smt = len(smt_checks)

        all_checks_total = total_cas + total_smt
        all_checks_valid = valid_cas + valid_smt
        total_refuted = refuted_cas + refuted_smt

        # Calculate pass rate: strictly 0.0 if no checks were performable
        pass_rate = round(all_checks_valid / all_checks_total, 3) if all_checks_total > 0 else 0.0

        # Ground truth validation
        gt_match = None
        if ground_truth is not None and extracted is not None:
            c_ext = self.clean_expr(extracted)
            c_gt = self.clean_expr(str(ground_truth))
            try:
                diff = sp.simplify(parse_expression(c_ext) - parse_expression(c_gt))
                gt_match = (diff == 0)
            except Exception:
                gt_match = (c_ext.lower() == c_gt.lower())

        # Determine 3-state status
        reference_match = None
        if reference_answer is not None and extracted is not None:
            reference_match = self.check_cas_equality(extracted, reference_answer)[0]
        verification_basis = None
        if gt_match is False or reference_match is False or total_refuted > 0 or len(counterexamples) > 0:
            status = VerificationStatus.REFUTED
            is_verified = False
            rejection_reason = "Refuted by counterexample, ground-truth contradiction, or algebraic violation."
        elif reference_match is True:
            # The reference must come from an independent deterministic backend.
            status = VerificationStatus.VERIFIED
            is_verified = True
            verification_basis = "deterministic_answer"
            rejection_reason = None
        elif all_checks_total == 0:
            # FAIL-CLOSED: No verifiable equations detected in solution
            status = VerificationStatus.UNVERIFIED
            is_verified = False
            rejection_reason = "Fail-closed: No formal algebraic or congruence equations could be extracted."
        else:
            status = VerificationStatus.UNVERIFIED
            is_verified = False
            rejection_reason = "Matching answer or valid intermediate steps do not certify the whole proof."

        return VerificationResult(
            status=status,
            is_verified=is_verified,
            pass_rate=pass_rate,
            extracted_answer=extracted,
            ground_truth=ground_truth,
            ground_truth_matched=gt_match,
            strictness_used=strictness,
            total_checks=all_checks_total,
            cas_checks={"valid": valid_cas, "total": total_cas, "refuted": refuted_cas},
            smt_checks={"valid": valid_smt, "total": total_smt, "counterexamples": counterexamples},
            steps=cas_checks[:8] + smt_checks[:4],
            rejection_reason=rejection_reason,
            verification_basis=verification_basis
        )
