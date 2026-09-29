"""LLM proposal plus Lean kernel check; informal equivalence remains reviewable."""

import json


def autoformalize(problem, solve, worker, header="import Init"):
    prompt = (
        "Translate this mathematics problem into one Lean 4 theorem and propose a proof. "
        "Return only JSON with keys 'formal_statement' and 'proof'. "
        "formal_statement must end in ':= sorry'; proof must contain only the replacement tactics. "
        "Do not use sorry, admit, unsafe, or new axioms in the proof.\n"
        f"Available header:\n{header}\nProblem:\n{problem}"
    )
    answer = solve(prompt).strip()
    if answer.startswith("```"):
        answer = answer.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        proposal = json.loads(answer)
        result = worker.prove(proposal["formal_statement"], proposal["proof"], header=header)
        return {**result, "informal_problem": problem, "formal_statement": proposal["formal_statement"],
                "informal_equivalence_reviewed": False}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "PROPOSAL_INVALID", "reason": str(exc), "informal_problem": problem}
