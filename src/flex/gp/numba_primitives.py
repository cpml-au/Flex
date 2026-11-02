"""Opcodes of the numba-compiled tree interpreter and values of its primitives, on single
points (``unary``, ``binary``) and on blocks of points (``apply_unary``,
``apply_binary``). The definitions are those of ``numpy_primitives``; division by zero
and invalid values give inf/nan, as in numpy.
"""

from numba import njit
import numpy as np

# opcodes of the leaves (feature, constant, number), of the unary and of the binary
# primitives
VAR, CONST, NUM = 0, 1, 2
SIN, ARCSIN, COS, ARCCOS, EXP, LOG, PROT_LOG, SQRT, SQUARE, TANH = range(3, 13)
ADD, SUB, MUL, DIV, POW, PROT_POW, AQ = range(13, 20)
OPCODES = {
    "sin": SIN, "arcsin": ARCSIN, "cos": COS, "arccos": ARCCOS, "exp": EXP,
    "log": LOG, "prot_log": PROT_LOG, "sqrt": SQRT, "square": SQUARE, "tanh": TANH,
    "add": ADD, "sub": SUB, "mul": MUL, "div": DIV, "pow": POW, "prot_pow": PROT_POW,
    "aq": AQ,
}
BLOCK = 256


@njit(cache=True, error_model="numpy")
def unary(op, a):
    if op == SIN:
        return np.sin(a)
    elif op == ARCSIN:
        return np.arcsin(a)
    elif op == COS:
        return np.cos(a)
    elif op == ARCCOS:
        return np.arccos(a)
    elif op == EXP:
        return np.exp(a)
    elif op == LOG:
        return np.log(a)
    elif op == PROT_LOG:
        return np.log(np.abs(a))
    elif op == SQRT:
        return np.sqrt(a)
    elif op == SQUARE:
        return a * a
    else:
        return np.tanh(a)


@njit(cache=True, error_model="numpy")
def binary(op, a, b):
    if op == ADD:
        return a + b
    elif op == SUB:
        return a - b
    elif op == MUL:
        return a * b
    elif op == DIV:
        return a / b
    elif op == POW:
        return a**b
    elif op == PROT_POW:
        return np.abs(a) ** b
    else:
        return a / np.sqrt(1.0 + b * b)


@njit(cache=True, error_model="numpy")
def apply_unary(op, a, m):
    """a[:m] = op(a[:m])."""
    for i in range(m):
        a[i] = unary(op, a[i])


@njit(cache=True, error_model="numpy")
def apply_binary(op, a, b, m):
    """b[:m] = op(a[:m], b[:m]); the arithmetic operators have their own loops, which
    are vectorized."""
    if op == ADD:
        for i in range(m):
            b[i] = a[i] + b[i]
    elif op == SUB:
        for i in range(m):
            b[i] = a[i] - b[i]
    elif op == MUL:
        for i in range(m):
            b[i] = a[i] * b[i]
    elif op == DIV:
        for i in range(m):
            b[i] = a[i] / b[i]
    else:
        for i in range(m):
            b[i] = binary(op, a[i], b[i])
