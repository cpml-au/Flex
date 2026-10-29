"""Evaluation of scalar DEAP trees with a numba-compiled interpreter, and fit of their
constants with Levenberg-Marquardt.

A tree is encoded (``encode``) in prefix order as two integer arrays: the opcode of
each node and, for the leaves, the index of the feature (column of ``X``) or of the
constant. The interpreter visits the nodes from the last to the first with a stack and
evaluates each node on blocks of data points: the opcode is dispatched once per node
and block, and the loop over the points of a block can be vectorized by the compiler.

The derivatives of the tree with respect to its constants are computed exactly in the
same pass (forward mode): each stack entry holds the values of a node and their
derivatives with respect to the constants. They are used to fit the constants by
minimizing the sum of squared errors with Levenberg-Marquardt (``fit_constants``).

The supported primitives are those of ``numpy_primitives`` (same definitions);
division by zero and invalid values give inf/nan, as in numpy.
"""

import numpy as np
from deap import gp
from numba import njit

# opcodes of the leaves, of the unary and of the binary primitives
VAR, CONST = 0, 1
SIN, ARCSIN, COS, ARCCOS, EXP, LOG, PROT_LOG, SQRT, SQUARE, TANH = range(2, 12)
ADD, SUB, MUL, DIV, POW, PROT_POW, AQ = range(12, 19)
OPCODES = {
    "sin": SIN, "arcsin": ARCSIN, "cos": COS, "arccos": ARCCOS, "exp": EXP,
    "log": LOG, "prot_log": PROT_LOG, "sqrt": SQRT, "square": SQUARE, "tanh": TANH,
    "add": ADD, "sub": SUB, "mul": MUL, "div": DIV, "pow": POW, "prot_pow": PROT_POW,
    "aq": AQ,
}
BLOCK = 256


def encode(tree, arg_names, special_term_name="c"):
    """Encodes a tree for the interpreter.

    Args:
        tree: the individual (DEAP ``PrimitiveTree``).
        arg_names: names of the arguments of the primitive set (``pset.arguments``):
            the argument at position i reads the column i of ``X``.
        special_term_name: name of the constant terminal; the i-th constant terminal
            (in prefix order) reads the i-th entry of the constants.

    Returns:
        the opcodes and the leaf indices (integer arrays) and the number of constants.
    """
    arg_index = {name: i for i, name in enumerate(arg_names)}
    ops = np.empty(len(tree), dtype=np.int64)
    idx = np.zeros(len(tree), dtype=np.int64)
    num_consts = 0
    for i, node in enumerate(tree):
        if isinstance(node, gp.Primitive):
            ops[i] = OPCODES[node.name]
        elif node.name == special_term_name:
            ops[i] = CONST
            idx[i] = num_consts
            num_consts += 1
        else:
            ops[i] = VAR
            idx[i] = arg_index[node.value]
    return ops, idx, num_consts


@njit(cache=True, error_model="numpy")
def _unary(op, a):
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


@njit(cache=True, error_model="numpy")
def _binary(op, a, b):
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
        return b * v / a, v * np.log(a)
    elif op == PROT_POW:
        return b * v / a, v * np.log(np.abs(a))
    else:
        s = np.sqrt(1.0 + b * b)
        return 1.0 / s, -v * b / (s * s)


@njit(cache=True, error_model="numpy")
def _apply_unary(op, a, m):
    """a[:m] = op(a[:m])."""
    for i in range(m):
        a[i] = _unary(op, a[i])


@njit(cache=True, error_model="numpy")
def _apply_binary(op, a, b, m):
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
            b[i] = _binary(op, a[i], b[i])


@njit(cache=True, error_model="numpy")
def _eval_block(ops, idx, X, consts, start, m, V):
    """Values of the tree at the points start, ..., start + m - 1 in V[0, :m]."""
    sp = 0
    for j in range(len(ops) - 1, -1, -1):
        op = ops[j]
        if op == VAR:
            f = idx[j]
            for i in range(m):
                V[sp, i] = X[start + i, f]
            sp += 1
        elif op == CONST:
            v = consts[idx[j]]
            for i in range(m):
                V[sp, i] = v
            sp += 1
        elif op < ADD:
            _apply_unary(op, V[sp - 1], m)
        else:
            # first operand on top of the stack, result in the slot of the second
            _apply_binary(op, V[sp - 1], V[sp - 2], m)
            sp -= 1


@njit(cache=True, error_model="numpy")
def predict(ops, idx, X, consts):
    """Values of the tree on the data X (samples x features)."""
    n = X.shape[0]
    V = np.empty((len(ops), BLOCK))
    out = np.empty(n)
    for start in range(0, n, BLOCK):
        m = min(BLOCK, n - start)
        _eval_block(ops, idx, X, consts, start, m, V)
        out[start : start + m] = V[0, :m]
    return out


@njit(cache=True, error_model="numpy")
def sse(ops, idx, X, y, consts):
    """Sum of squared errors of the tree on the data X (samples x features), y."""
    n = X.shape[0]
    V = np.empty((len(ops), BLOCK))
    total = 0.0
    for start in range(0, n, BLOCK):
        m = min(BLOCK, n - start)
        _eval_block(ops, idx, X, consts, start, m, V)
        for i in range(m):
            res = V[0, i] - y[start + i]
            total += res * res
    return total


@njit(cache=True, error_model="numpy")
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
                    db[i] = b[i] * da[i]
            elif za:
                for i in range(m):
                    db[i] = a[i] * db[i]
            else:
                for i in range(m):
                    db[i] = b[i] * da[i] + a[i] * db[i]
        else:
            if zb:
                for i in range(m):
                    db[i] = da[i] / b[i]
            elif za:
                for i in range(m):
                    db[i] = -a[i] * db[i] / (b[i] * b[i])
            else:
                for i in range(m):
                    db[i] = (da[i] - a[i] / b[i] * db[i]) / b[i]


@njit(cache=True, error_model="numpy")
def sse_normal_equations(ops, idx, X, y, consts):
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
            elif op < ADD:
                a = V[sp - 1]
                if Z[sp - 1]:
                    _apply_unary(op, a, m)
                else:
                    for i in range(m):
                        v = _unary(op, a[i])
                        PA[i] = _unary_derivative(op, a[i], v)
                        a[i] = v
                    for c in range(k):
                        d = D[sp - 1, c]
                        for i in range(m):
                            d[i] *= PA[i]
            else:
                a = V[sp - 1]
                b = V[sp - 2]
                za = Z[sp - 1]
                zb = Z[sp - 2]
                if za and zb:
                    _apply_binary(op, a, b, m)
                elif op <= DIV:
                    Da, Db = D[sp - 1], D[sp - 2]
                    _arithmetic_derivatives(op, a, b, Da, Db, za, zb, k, m)
                    _apply_binary(op, a, b, m)
                    Z[sp - 2] = False
                else:
                    # partial derivatives at the operands, then values (in W)
                    for i in range(m):
                        v = _binary(op, a[i], b[i])
                        PA[i], PB[i] = _binary_derivatives(op, a[i], b[i], v)
                        W[i] = v
                    for c in range(k):
                        da = D[sp - 1, c]
                        db = D[sp - 2, c]
                        if za:
                            for i in range(m):
                                db[i] = PB[i] * db[i]
                        elif zb:
                            for i in range(m):
                                db[i] = PA[i] * da[i]
                        else:
                            for i in range(m):
                                db[i] = PA[i] * da[i] + PB[i] * db[i]
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


@njit(cache=True)
def _solve_spd(A, b):
    """Solution of A x = b for a symmetric positive definite A (Cholesky); returns
    False if A is not positive definite."""
    k = len(b)
    L = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1):
            s = A[i, j]
            for p in range(j):
                s -= L[i, p] * L[j, p]
            if i == j:
                if not s > 0.0:
                    return np.zeros(k), False
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    z = np.empty(k)
    for i in range(k):
        s = b[i]
        for p in range(i):
            s -= L[i, p] * z[p]
        z[i] = s / L[i, i]
    x = np.empty(k)
    for i in range(k - 1, -1, -1):
        s = z[i]
        for p in range(i + 1, k):
            s -= L[p, i] * x[p]
        x[i] = s / L[i, i]
    return x, True


@njit(cache=True, error_model="numpy")
def fit_constants(ops, idx, X, y, x0, lower, upper, max_iter):
    """Fits the constants of the tree by minimizing the sum of squared errors with
    Levenberg-Marquardt (Marquardt scaling of the damping), starting from x0 and
    keeping the constants within [lower, upper] (clipping).

    Each iteration computes the Jacobian at the current constants (forward pass) and
    evaluates one trial step: the step is accepted (and the damping decreased) if it
    reduces the sum of squared errors, otherwise the damping is increased.

    Returns:
        the constants, their sum of squared errors, the number of trial steps and the
        number of forward passes (Jacobian evaluations).
    """
    x = np.minimum(np.maximum(x0, lower), upper)
    f, JTJ, JTr = sse_normal_equations(ops, idx, X, y, x)
    num_jac = 1
    num_trials = 0
    lam = 1e-3
    for _ in range(max_iter):
        finite = np.all(np.isfinite(JTJ)) and np.all(np.isfinite(JTr))
        if not (np.isfinite(f) and finite):
            break
        if f == 0.0:
            break
        A = JTJ.copy()
        for c in range(len(x)):
            A[c, c] += lam * max(JTJ[c, c], 1e-12)
        step, ok = _solve_spd(A, -JTr)
        if not ok:
            lam *= 10.0
            continue
        x_new = np.minimum(np.maximum(x + step, lower), upper)
        f_new = sse(ops, idx, X, y, x_new)
        num_trials += 1
        if f_new < f:
            x = x_new
            lam = max(lam * 0.1, 1e-12)
            f, JTJ, JTr = sse_normal_equations(ops, idx, X, y, x)
            num_jac += 1
        else:
            lam *= 10.0
    return x, f, num_trials, num_jac
