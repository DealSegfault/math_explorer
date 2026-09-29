"""Batched Violetto sampling with exact-answer voting and CAS tie-breaking."""

import re
from collections import defaultdict

from verification.expressions import parse_expression


def _boxed(text):
    values = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if not values:
        return None
    if re.fullmatch(r"\d{1,4}", values[-1].strip()):
        return str(int(values[-1].strip()))
    try:
        value = parse_expression(values[-1].strip())
        return str(value) if value.is_number and value.is_finite else None
    except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError):
        return None


def _cas_score(text):
    valid = refuted = 0
    for line in text.splitlines():
        parts = re.split(r"(?<![<>=!:])=(?![=>])", line.strip().strip("$"))
        if len(parts) != 2:
            continue
        try:
            left, right = (parse_expression(part.strip()) for part in parts)
            if (left - right).simplify() == 0:
                valid += 1
            else:
                refuted += 1
        except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError):
            continue
    return valid, refuted


def select_candidate(samples):
    votes = defaultdict(list)
    for index, solution in enumerate(samples):
        answer = _boxed(solution)
        if answer is not None:
            valid, refuted = _cas_score(solution)
            votes[answer].append({"index": index, "valid_steps": valid, "refuted_steps": refuted})
    if not votes:
        return {"status": "NO_ANSWER", "answer": None, "votes": {}, "is_formally_verified": False}
    # Each sample contributes one vote. CAS only breaks equal vote counts.
    winner = max(votes, key=lambda answer: (len(votes[answer]), sum(v["valid_steps"] - v["refuted_steps"] for v in votes[answer])))
    winning_count = len(votes[winner])
    highest_other = max((len(items) for answer, items in votes.items() if answer != winner), default=0)
    best = max(votes[winner], key=lambda item: (item["valid_steps"] - item["refuted_steps"], -item["index"]))
    status = "SINGLE" if len(samples) == 1 else "CONSENSUS" if winning_count > highest_other and winning_count > 1 else "TIE_BREAK"
    return {"status": status,
            "answer": winner, "sample_index": best["index"], "votes": dict(votes), "is_formally_verified": False}


def solve_best_of_n(engine, query, n=4, max_tokens=384, temperature=0.6, top_k=50, context=None, seed=None):
    samples = engine.generate(query, context=context, max_tokens=max_tokens, temperature=temperature,
                              top_k=top_k, num_return_sequences=n, seed=seed)
    if isinstance(samples, str):
        samples = [samples]
    decision = select_candidate(samples)
    decision["solution"] = samples[decision["sample_index"]] if decision.get("sample_index") is not None else ""
    decision["samples"] = samples
    return decision


if __name__ == "__main__":
    selected = select_candidate(["1+1=2\n\\boxed{2}", "\\boxed{02}", "\\boxed{3}"])
    assert selected["answer"] == "2" and selected["status"] == "CONSENSUS"
