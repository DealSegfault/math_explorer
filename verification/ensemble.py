"""Check arithmetic steps and final answers without certifying free-form proofs."""

import re
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple, TypedDict
import sympy as sp
from verification.expressions import parse_expression


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    UNVERIFIED = "UNVERIFIED"


class VerificationEvidence(TypedDict, total=False):
    exact_result: str
    method: str
    constraints: Dict[str, Any]
    witness: List[int]
    unsat_certificate: str


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
    def extract_boxed_answer(self, text: str) -> Optional[str]:
        starts = list(re.finditer(r'\\boxed\s*\{', text))
        if starts:
            start = starts[-1].end()
            depth = 1
            for i in range(start, len(text)):
                depth += (text[i] == '{') - (text[i] == '}')
                if depth == 0:
                    return text[start:i].strip() or None
            return None
        matches = re.findall(
            r'(?:final\s+answer\s*:|the\s+answer\s+is|result\s+is)\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*[.!]?\s*$',
            text, re.IGNORECASE | re.MULTILINE)
        return matches[-1] if matches else None

    def clean_expr(self, s: str) -> str:
        s = s.strip().strip('$').strip().rstrip('.,;:')
        s = re.sub(r'^(?:we have|we get|thus|hence|therefore|so)\s+', '', s, flags=re.I)
        s = s.replace(r'\left', '').replace(r'\right', '')
        s = s.replace(r'\cdot', '*').replace(r'\times', '*')
        # Nested fractions are reduced from the inside out.
        for _ in range(8):
            previous = s
            s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
            s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
            s = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', s)
            if s == previous:
                break
        return s.replace('^', '**').strip()

    def check_cas_equality(self, lhs_str: str, rhs_str: str) -> Tuple[Optional[bool], str]:
        try:
            lhs = parse_expression(self.clean_expr(lhs_str))
            rhs = parse_expression(self.clean_expr(rhs_str))
            diff = sp.simplify(lhs - rhs)
            if diff == 0:
                return True, "Symbolically identical (diff = 0)"
            if lhs.free_symbols or rhs.free_symbols:
                return None, "Equality may depend on assumptions or variable assignments."
            if diff.is_zero is False:
                return False, f"Constant equality refuted (difference = {diff})"
            return None, "Equality could not be decided."
        except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError) as e:
            return None, f"Unsupported expression: {e}"

    def check_z3_congruence(self, a_str, b_str, m_str):
        try:
            values = [parse_expression(self.clean_expr(s)) for s in (a_str, b_str, m_str)]
            if not all(v.is_Integer for v in values):
                return None, None, "Congruence requires exact integers."
            a, b, m = map(int, values)
            if m == 0:
                return False, None, "Modulo 0 is undefined."
            remainder = (a - b) % m
            if remainder == 0:
                return True, None, "Exact integer congruence holds."
            return False, {"a": a, "b": b, "m": m, "remainder": remainder}, "Nonzero remainder."
        except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError) as e:
            return None, None, f"Unsupported congruence: {e}"

    def verify(self, solution_text: str, ground_truth: Optional[str] = None,
               strictness: float = 0.75, reference_answer: Optional[str] = None,
               evidence: Optional[VerificationEvidence] = None) -> VerificationResult:
        extracted = self.extract_boxed_answer(solution_text)
        if evidence and evidence.get("exact_result") is not None:
            exact = str(evidence["exact_result"])
            if reference_answer is None:
                reference_answer = exact
            if extracted is None:
                extracted = exact
        cas_checks, smt_checks, counterexamples = [], [], []
        checked = set()
        for line in solution_text.splitlines():
            line = line.strip().strip('$').strip()
            if not line or line.startswith('#'):
                continue
            cong = re.fullmatch(r'(.+?)\s*(?:\\equiv|≡)\s*(.+?)\s*(?:\\pmod\{([^{}]+)\}|\(mod\s+([^()]+)\))\.?', line)
            if cong:
                a, b, m1, m2 = cong.groups()
                ok, cex, reason = self.check_z3_congruence(a, b, m1 or m2)
                smt_checks.append({"claim": line, "valid": ok, "reason": reason})
                if cex:
                    counterexamples.append(cex)
                continue
            # Exclude inequalities, assignments (:=), and implications (=>).
            parts = re.split(r'(?<![<>=!:])=(?![=>])', line)
            for left, right in zip(parts, parts[1:]):
                pair = (left.strip(), right.strip())
                if pair in checked:
                    continue
                checked.add(pair)
                ok, reason = self.check_cas_equality(*pair)
                cas_checks.append({"claim": f"{left} = {right}", "valid": ok,
                                   "refuted": ok is False, "reason": reason})

        valid_cas = sum(c["valid"] is True for c in cas_checks)
        valid_smt = sum(c["valid"] is True for c in smt_checks)
        refuted_cas = sum(c["valid"] is False for c in cas_checks)
        refuted_smt = sum(c["valid"] is False for c in smt_checks)
        total = len(cas_checks) + len(smt_checks)
        pass_rate = round((valid_cas + valid_smt) / total, 3) if total else 0.0
        gt_match = self.check_cas_equality(extracted, str(ground_truth))[0] if extracted is not None and ground_truth is not None else None
        ref_match = self.check_cas_equality(extracted, str(reference_answer))[0] if extracted is not None and reference_answer is not None else None
        basis = None
        if gt_match is False or ref_match is False or refuted_cas or refuted_smt:
            status = VerificationStatus.REFUTED
            reason = "Contradiction with a reference answer or an exact arithmetic statement."
        elif (gt_match is True or ref_match is True) and (not total or pass_rate >= strictness or
                                                           (evidence and evidence.get("exact_result") is not None)):
            status = VerificationStatus.VERIFIED
            basis = "ground_truth_answer" if gt_match is True else "deterministic_answer"
            reason = None
        else:
            status = VerificationStatus.UNVERIFIED
            reason = "Answer lacks an independent reference or checked steps fall below strictness."
        return VerificationResult(
            status=status, is_verified=status == VerificationStatus.VERIFIED,
            pass_rate=pass_rate, extracted_answer=extracted, ground_truth=ground_truth,
            ground_truth_matched=gt_match, strictness_used=strictness, total_checks=total,
            cas_checks={"valid": valid_cas, "total": len(cas_checks), "refuted": refuted_cas},
            smt_checks={"valid": valid_smt, "total": len(smt_checks), "counterexamples": counterexamples},
            steps=cas_checks[:8] + smt_checks[:4], rejection_reason=reason, verification_basis=basis)
