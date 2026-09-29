"""Small arithmetic grammar for untrusted model output; never evaluate Python."""

import ast
import operator
import sympy as sp


def parse_expression(text: str, *, relational: bool = False):
    if len(text) > 512:
        raise ValueError("Expression too long")
    tree = ast.parse(text.strip().replace("^", "**"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression too complex")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub,
              ast.Mult: operator.mul, ast.Div: operator.truediv,
              ast.Pow: operator.pow, ast.Mod: operator.mod}
    functions = {"sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs}
    relations = {ast.Eq: sp.Eq, ast.NotEq: sp.Ne, ast.Lt: sp.Lt,
                 ast.LtE: sp.Le, ast.Gt: sp.Gt, ast.GtE: sp.Ge}
    if relational:
        functions.update({"Eq": sp.Eq, "Ne": sp.Ne, "Lt": sp.Lt, "Le": sp.Le,
                          "Gt": sp.Gt, "Ge": sp.Ge, "Mod": sp.Mod,
                          "And": sp.And, "Or": sp.Or, "Not": sp.Not})

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = sp.Rational(str(node.value))
        elif isinstance(node, ast.Name) and len(node.id) == 1 and node.id.isascii() and node.id.isalpha():
            value = sp.Symbol(node.id)
        elif isinstance(node, ast.Name) and node.id == "pi":
            value = sp.pi
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and (not right.is_Integer or abs(right) > 1000):
                raise ValueError("Exponent must be an integer between -1000 and 1000")
            if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
                raise ValueError("Division by zero")
            value = binary[type(node.op)](left, right)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in functions and node.args and not node.keywords):
            arity = 1 if node.func.id in ("sqrt", "Abs", "abs", "Not") else 2
            if node.func.id not in ("And", "Or") and len(node.args) != arity:
                raise ValueError("Incorrect function argument count")
            value = functions[node.func.id](*(visit(arg) for arg in node.args))
        elif relational and isinstance(node, ast.Compare) and all(type(op) in relations for op in node.ops):
            values = [visit(node.left)] + [visit(arg) for arg in node.comparators]
            value = sp.And(*(relations[type(op)](left, right)
                             for op, left, right in zip(node.ops, values, values[1:])))
        else:
            raise ValueError("Unsupported mathematical syntax")
        if value.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            raise ValueError("Non-finite value")
        if any(abs(int(n)).bit_length() > 16384 for n in value.atoms(sp.Integer)):
            raise ValueError("Value too large")
        return value

    return visit(tree.body)
