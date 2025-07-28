from typing import Dict, Tuple
from deap.gp import PrimitiveTree, Primitive


def deap_primitive_to_sympy_expr(prim: Primitive, conversion_rules: Dict, args: Tuple):
    """Convert a DEAP primitive and its arguments into the corresponding sympy
        expression.

    Args:
        prim: the primitive.
        conversion_rules: a dictionary of conversion rules.
        args: args of the primitive.

    Returns:
        the sympy-compatible expression.

    """
    prim_formatter = conversion_rules.get(prim.name, prim.format)

    return prim_formatter(*args)


def stringify_for_sympy(
    f: PrimitiveTree, conversion_rules: Dict, special_term_name: str
) -> str:
    """Returns a sympy-compatible expression.

    Args:
        f: the individual tree (DEAP format)
        conversion_rules: a dictionary of conversion rules.
        special_term_name: name of the constant placeholder.

    Returns:
        the sympy-compatible expression.
    """
    expr = ""
    stack = []
    const_idx = 0
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            if prim.name == special_term_name:
                # substitute the c placeholder with the constant value
                expr = f.consts[const_idx]
                # update the constant index
                const_idx += 1
            else:
                expr = deap_primitive_to_sympy_expr(prim, conversion_rules, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(expr)
    return expr
