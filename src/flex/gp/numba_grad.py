"""Exact derivatives of the numba-compiled trees with respect to their constants.

The derivatives are computed in forward mode, in the same pass as the values: each stack
entry of the interpreter holds the values of a node and their derivatives with respect
to the constants, updated at each node with the differentiation rules of its primitive
(``_unary_derivative``, ``_binary_derivatives``, ``_arithmetic_derivatives``). In these
updates a product with a zero factor is zero, also if the other factor is inf or nan.
``sse_normal_equations`` accumulates from them the terms of the normal equations used by
the fit of the constants.
"""

import numpy as np

from flex.gp.numba_primitives import (
    ADD, ARCCOS, ARCSIN, BLOCK, CONST, COS, DIV, EXP, LOG, MUL, NUM, POW, PROT_LOG,
    PROT_POW, SIN, SQRT, SQUARE, SUB, VAR, apply_binary, apply_unary, binary,
    cached_njit, unary,
)


@cached_njit(error_model="numpy")
def _product(p, d):
    """p * d, where a zero factor gives zero also if the other factor is inf or nan."""
    return 0.0 if p == 0.0 or d == 0.0 else p * d


@cached_njit(error_model="numpy")
def _quotient(n, b):
    """n / b, where a zero numerator gives zero also if b is zero or nan."""
    return 0.0 if n == 0.0 else n / b


@cached_njit(error_model="numpy")
def _unary_derivative(op, a, v):
    """Derivative of the unary primitive at a, given its value v."""
    if op == SIN:
        return np.cos(a)
    elif op == ARCSIN:
        return 1.0 / np.sqrt(1.0 - a * a)
    elif op == COS:
        return -np.sin(a)
    elif op == ARCCOS:
        return -1.0 / np.sqrt(1.0 - a * a)
    elif op == EXP:
        return v
    elif op == LOG or op == PROT_LOG:
        return 1.0 / a
    elif op == SQRT:
        return 0.5 / v
    elif op == SQUARE:
        return 2.0 * a
    else:
        return 1.0 - v * v


@cached_njit(error_model="numpy")
def _binary_derivatives(op, a, b, v):
    """Partial derivatives of the binary primitive at (a, b), given its value v."""
    if op == ADD:
        return 1.0, 1.0
    elif op == SUB:
        return 1.0, -1.0
    elif op == MUL:
        return b, a
    elif op == DIV:
        return 1.0 / b, -v / b
    elif op == POW:
        if a == 0.0:
            # v / a would be 0 / 0 and v * log(a) would be 0 * (-inf)
            da = 0.0 if b == 0.0 else b * a ** (b - 1.0)
            return da, 0.0 if b > 0.0 else np.nan
        return b * v / a, v * np.log(a)
    elif op == PROT_POW:
        if a == 0.0:
            # as for POW, but |a|^b has a kink at a = 0 when 0 < b <= 1
            da = 0.0 if b == 0.0 or b > 1.0 else np.nan
            return da, 0.0 if b > 0.0 else np.nan
        return b * v / a, v * np.log(np.abs(a))
    else:
        s = np.sqrt(1.0 + b * b)
        return 1.0 / s, -v * b / (s * s)


@cached_njit(error_model="numpy")
def _arithmetic_derivatives(op, a, b, Da, Db, za, zb, k, m):
    """Derivatives (in Db) of op(a, b) for the arithmetic operators, given those of the
    operands (Da, Db; za, zb: whether they are zero), with vectorized loops."""
    for c in range(k):
        da = Da[c]
        db = Db[c]
        if op == ADD:
            if zb:
                for i in range(m):
                    db[i] = da[i]
            elif not za:
                for i in range(m):
                    db[i] = da[i] + db[i]
        elif op == SUB:
            if zb:
                for i in range(m):
                    db[i] = da[i]
            elif za:
                for i in range(m):
                    db[i] = -db[i]
            else:
                for i in range(m):
                    db[i] = da[i] - db[i]
        elif op == MUL:
            if zb:
                for i in range(m):
                    db[i] = _product(b[i], da[i])
            elif za:
                for i in range(m):
                    db[i] = _product(a[i], db[i])
            else:
                for i in range(m):
                    db[i] = _product(b[i], da[i]) + _product(a[i], db[i])
        else:
            if zb:
                for i in range(m):
                    db[i] = _quotient(da[i], b[i])
            elif za:
                for i in range(m):
                    db[i] = _quotient(-_product(a[i] / b[i], db[i]), b[i])
            else:
                for i in range(m):
                    db[i] = _quotient(da[i] - _product(a[i] / b[i], db[i]), b[i])


@cached_njit(error_model="numpy")
def sse_normal_equations(ops, idx, nums, X, y, consts):
    """Sum of squared errors, J^T J and J^T r, where r = f - y are the residuals and J
    the Jacobian of the tree values f with respect to the constants, computed in
    forward mode (the n x k Jacobian is not stored).

    For each node and block, the partial derivatives of the primitive are computed
    first, and then the derivatives with respect to each constant are updated with a
    vectorized loop. Subtrees without constants (zero derivatives) are skipped."""
    n = X.shape[0]
    k = len(consts)
    V = np.empty((len(ops), BLOCK))
    D = np.empty((len(ops), k, BLOCK))
    # whether the derivatives of a stack entry are all zero
    Z = np.empty(len(ops), dtype=np.bool_)
    PA = np.empty(BLOCK)
    PB = np.empty(BLOCK)
    W = np.empty(BLOCK)
    total = 0.0
    JTJ = np.zeros((k, k))
    JTr = np.zeros(k)
    for start in range(0, n, BLOCK):
        m = min(BLOCK, n - start)
        sp = 0
        for j in range(len(ops) - 1, -1, -1):
            op = ops[j]
            if op == VAR:
                f = idx[j]
                for i in range(m):
                    V[sp, i] = X[start + i, f]
                Z[sp] = True
                sp += 1
            elif op == CONST:
                v = consts[idx[j]]
                for i in range(m):
                    V[sp, i] = v
                for c in range(k):
                    w = 1.0 if c == idx[j] else 0.0
                    for i in range(m):
                        D[sp, c, i] = w
                Z[sp] = False
                sp += 1
            elif op == NUM:
                v = nums[j]
                for i in range(m):
                    V[sp, i] = v
                Z[sp] = True
                sp += 1
            elif op < ADD:
                a = V[sp - 1]
                if Z[sp - 1]:
                    apply_unary(op, a, m)
                else:
                    for i in range(m):
                        v = unary(op, a[i])
                        PA[i] = _unary_derivative(op, a[i], v)
                        a[i] = v
                    for c in range(k):
                        d = D[sp - 1, c]
                        for i in range(m):
                            d[i] = _product(d[i], PA[i])
            else:
                a = V[sp - 1]
                b = V[sp - 2]
                za = Z[sp - 1]
                zb = Z[sp - 2]
                if za and zb:
                    apply_binary(op, a, b, m)
                elif op <= DIV:
                    Da, Db = D[sp - 1], D[sp - 2]
                    _arithmetic_derivatives(op, a, b, Da, Db, za, zb, k, m)
                    apply_binary(op, a, b, m)
                    Z[sp - 2] = False
                else:
                    # partial derivatives at the operands, then values (in W)
                    for i in range(m):
                        v = binary(op, a[i], b[i])
                        PA[i], PB[i] = _binary_derivatives(op, a[i], b[i], v)
                        W[i] = v
                    for c in range(k):
                        da = D[sp - 1, c]
                        db = D[sp - 2, c]
                        if za:
                            for i in range(m):
                                db[i] = _product(PB[i], db[i])
                        elif zb:
                            for i in range(m):
                                db[i] = _product(PA[i], da[i])
                        else:
                            for i in range(m):
                                db[i] = _product(PA[i], da[i]) + _product(PB[i], db[i])
                    for i in range(m):
                        b[i] = W[i]
                    Z[sp - 2] = False
                sp -= 1
        # residuals (in W) and their products with the derivatives
        for i in range(m):
            W[i] = V[0, i] - y[start + i]
            total += W[i] * W[i]
        if not Z[0]:
            for c in range(k):
                dc = D[0, c]
                s = 0.0
                for i in range(m):
                    s += dc[i] * W[i]
                JTr[c] += s
                for e in range(c + 1):
                    de = D[0, e]
                    s = 0.0
                    for i in range(m):
                        s += dc[i] * de[i]
                    JTJ[c, e] += s
    for c in range(k):
        for e in range(c):
            JTJ[e, c] = JTJ[c, e]
    return total, JTJ, JTr
