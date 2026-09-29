"""Bounded integer claims and a counterexample-guided revision loop."""

import ast
import json
import re

import z3


def _compile(text, variables):
    if len(text) > 512:
        raise ValueError("Expression too long")
    tree = ast.parse(text, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return z3.IntVal(node.value)
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            if isinstance(node.op, ast.Not):
                return z3.Not(value)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BoolOp):
            values = [visit(value) for value in node.values]
            if isinstance(node.op, ast.And):
                return z3.And(*values)
            if isinstance(node.op, ast.Or):
                return z3.Or(*values)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Mod) and isinstance(node.right, ast.Constant) and type(node.right.value) is int and node.right.value > 0:
                return left % right
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and type(node.right.value) is int and 0 <= node.right.value <= 4:
                return left ** node.right.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 2 and not node.keywords:
            left, right = visit(node.args[0]), visit(node.args[1])
            if node.func.id == "Eq":
                return left == right
            if node.func.id == "Mod" and isinstance(node.args[1], ast.Constant) and type(node.args[1].value) is int and node.args[1].value > 0:
                return left % right
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = visit(node.left), visit(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Eq):
                return left == right
            if isinstance(op, ast.NotEq):
                return left != right
            if isinstance(op, ast.Lt):
                return left < right
            if isinstance(op, ast.LtE):
                return left <= right
            if isinstance(op, ast.Gt):
                return left > right
            if isinstance(op, ast.GtE):
                return left >= right
        raise ValueError("Unsupported integer or Boolean syntax")

    result = visit(tree.body)
    if not z3.is_bool(result):
        raise ValueError("Claim must be Boolean")
    return result


def check_claim(claim, timeout_ms=2000):
    """Return a counterexample or a certificate restricted to declared bounds."""
    try:
        bounds = claim["variables"]
        assumptions = claim.get("assumptions", [])
        if not isinstance(bounds, dict) or not 1 <= len(bounds) <= 4 or not isinstance(assumptions, list) or len(assumptions) > 12:
            raise ValueError("Expected 1–4 variables and at most 12 assumptions")
        variables = {}
        for name, pair in bounds.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,15}", name) or not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair) or pair[0] > pair[1]:
                raise ValueError("Invalid variable or integer bounds")
            variables[name] = z3.Int(name)
        solver = z3.Solver()
        solver.set(timeout=timeout_ms)
        for name, (low, high) in bounds.items():
            solver.add(variables[name] >= low, variables[name] <= high)
        for assumption in assumptions:
            solver.add(_compile(assumption, variables))
        assumptions_result = solver.check()
        if assumptions_result == z3.unsat:
            return {"status": "INVALID", "reason": "Empty assumption domain"}
        if assumptions_result == z3.unknown:
            return {"status": "UNKNOWN", "reason": solver.reason_unknown()}
        solver.add(z3.Not(_compile(claim["conclusion"], variables)))
        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            return {"status": "COUNTEREXAMPLE", "model": {name: model.eval(var, model_completion=True).as_long() for name, var in variables.items()}}
        if result == z3.unsat:
            return {"status": "BOUNDED_VALID", "bounds": bounds}
        return {"status": "UNKNOWN", "reason": solver.reason_unknown()}
    except (KeyError, TypeError, ValueError, SyntaxError, z3.Z3Exception) as exc:
        return {"status": "INVALID", "reason": str(exc)}


def cegis(problem, initial_claim, solve, max_rounds=3):
    """Ask a solver to revise a refuted claim; never relabel it as the original."""
    if not 0 <= max_rounds <= 5:
        raise ValueError("max_rounds must be between 0 and 5")
    claim = initial_claim
    history = []
    seen = set()
    for round_no in range(max_rounds + 1):
        result = check_claim(claim)
        history.append({"claim": claim, "check": result})
        if result["status"] != "COUNTEREXAMPLE" or round_no == max_rounds:
            return {"status": result["status"], "revised": claim != initial_claim, "claim": claim, "history": history}
        prompt = (
            "Revise the following bounded integer conjecture in response to its counterexample. "
            "Return only a JSON object with 'assumptions' (list of expressions) and 'conclusion' (expression). "
            "Keep the variable bounds fixed. A revised claim is a new conjecture, not a proof of the original.\n"
            f"Problem: {problem}\nOriginal claim: {json.dumps(initial_claim)}\n"
            f"Current claim: {json.dumps(claim)}\nCounterexample: {json.dumps(result['model'])}"
        )
        answer = solve(prompt).strip()
        if answer.startswith("```"):
            answer = answer.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            revised = json.loads(answer)
            claim = {"variables": initial_claim["variables"], "assumptions": revised["assumptions"], "conclusion": revised["conclusion"]}
            if claim["conclusion"].strip() in {item.strip() for item in claim["assumptions"]}:
                raise ValueError("Conclusion cannot be assumed")
            key = json.dumps(claim, sort_keys=True)
            if key in seen:
                raise ValueError("Repeated claim")
            seen.add(key)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {"status": "REVISION_FAILED", "reason": str(exc), "history": history}


if __name__ == "__main__":
    valid = {"variables": {"x": [1, 51]}, "assumptions": [], "conclusion": "x % 51 == x"}
    assert check_claim(valid)["model"] == {"x": 51}
    assert check_claim({**valid, "variables": {"x": [1, 50]}})["status"] == "BOUNDED_VALID"
    assert check_claim({**valid, "assumptions": ["x > 51"]})["status"] == "INVALID"
