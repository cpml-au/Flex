"""Evaluation of scalar DEAP trees with a numba-compiled interpreter, and fit of their
constants with Levenberg-Marquardt.

A tree is encoded (``encode``) in prefix order as three arrays: the opcode of each
node, the index of the feature (column of ``X``) or of the constant read by each leaf,
and the value of each numerical terminal (fixed numbers, such as ``1.0`` or ephemeral
constants, which are not fitted). The interpreter visits the nodes from the last to the
first with a stack and evaluates each node on blocks of data points: the opcode is
dispatched once per node and block, and the loop over the points of a block can be
vectorized by the compiler.

The derivatives of the tree with respect to its constants are computed exactly in the
same pass (``numba_grad``). They are used to fit the constants by minimizing the sum of
squared errors with Levenberg-Marquardt (``fit_constants``).

The supported primitives are those of ``numpy_primitives`` (``numba_primitives``);
division by zero and invalid values give inf/nan, as in numpy.

User-facing helpers work on DEAP trees: ``fit_mse``, ``mse`` and ``predict_tree`` for a
single tree, and the batch functions ``fitness_batch``, ``score_batch``,
``predict_batch`` and ``assign_attributes``, which can be passed to
``GPSymbolicRegressor`` as ``fitness``, ``score_func``, ``predict_func`` and
``callback_func``.

The compiled functions are cached on disk in a folder that depends on the sources of
the three numba modules (``cached_njit``), so that changing any of them compiles all of
them again.
"""

import zlib

import numpy as np
from deap import gp

from flex.gp.numba_grad import sse_normal_equations
from flex.gp.numba_primitives import (
    ADD, BLOCK, CONST, NUM, OPCODES, VAR, apply_binary, apply_unary, cached_njit,
)


def encode(tree, arg_names, special_term_name="c", context=None):
    """Encodes a tree for the interpreter.

    Args:
        tree: the individual (DEAP ``PrimitiveTree``).
        arg_names: names of the arguments of the primitive set (``pset.arguments``):
            the argument at position i reads the column i of ``X``.
        special_term_name: name of the constant terminal; the i-th constant terminal
            (in prefix order) reads the i-th entry of the constants. The other
            terminals whose value is not an argument name are numerical terminals.
        context: values of the terminals added with a name (``pset.context``), needed
            when the tree has any.

    Returns:
        the opcodes and the leaf indices (integer arrays), the values of the
        numerical terminals (float array, zero at the other nodes) and the number of
        constants.
    """
    arg_index = {name: i for i, name in enumerate(arg_names)}
    ops = np.empty(len(tree), dtype=np.int64)
    idx = np.zeros(len(tree), dtype=np.int64)
    nums = np.zeros(len(tree))
    num_consts = 0
    for i, node in enumerate(tree):
        if isinstance(node, gp.Primitive):
            ops[i] = OPCODES[node.name]
        elif node.name == special_term_name:
            ops[i] = CONST
            idx[i] = num_consts
            num_consts += 1
        elif node.value in arg_index:
            ops[i] = VAR
            idx[i] = arg_index[node.value]
        elif isinstance(node.value, str):
            ops[i] = NUM
            nums[i] = context[node.value]
        else:
            ops[i] = NUM
            nums[i] = node.value
    return ops, idx, nums, num_consts


@cached_njit(error_model="numpy")
def _eval_block(ops, idx, nums, X, consts, start, m, V):
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
        elif op == NUM:
            v = nums[j]
            for i in range(m):
                V[sp, i] = v
            sp += 1
        elif op < ADD:
            apply_unary(op, V[sp - 1], m)
        else:
            # first operand on top of the stack, result in the slot of the second
            apply_binary(op, V[sp - 1], V[sp - 2], m)
            sp -= 1


@cached_njit(error_model="numpy")
def predict(ops, idx, nums, X, consts):
    """Values of the tree on the data X (samples x features)."""
    n = X.shape[0]
    V = np.empty((len(ops), BLOCK))
    out = np.empty(n)
    for start in range(0, n, BLOCK):
        m = min(BLOCK, n - start)
        _eval_block(ops, idx, nums, X, consts, start, m, V)
        out[start : start + m] = V[0, :m]
    return out


@cached_njit(error_model="numpy")
def sse(ops, idx, nums, X, y, consts):
    """Sum of squared errors of the tree on the data X (samples x features), y."""
    n = X.shape[0]
    V = np.empty((len(ops), BLOCK))
    total = 0.0
    for start in range(0, n, BLOCK):
        m = min(BLOCK, n - start)
        _eval_block(ops, idx, nums, X, consts, start, m, V)
        for i in range(m):
            res = V[0, i] - y[start + i]
            total += res * res
    return total




@cached_njit()
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


@cached_njit(error_model="numpy")
def fit_constants(ops, idx, nums, X, y, x0, lower, upper, max_iter):
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
    f, JTJ, JTr = sse_normal_equations(ops, idx, nums, X, y, x)
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
        f_new = sse(ops, idx, nums, X, y, x_new)
        num_trials += 1
        if f_new < f:
            x = x_new
            lam = max(lam * 0.1, 1e-12)
            f, JTJ, JTr = sse_normal_equations(ops, idx, nums, X, y, x)
            num_jac += 1
        else:
            lam *= 10.0
    return x, f, num_trials, num_jac


def _start_constants(ops, idx, nums, num_consts, bounds):
    """Start values of the fit, uniformly distributed within the bounds. The generator
    is seeded with a hash of the encoded tree, so that the values depend only on the
    tree, and not on the process or on the order in which the trees are evaluated."""
    seed = zlib.crc32(ops.tobytes() + idx.tobytes() + nums.tobytes())
    return np.random.default_rng(seed).uniform(bounds[0], bounds[1], num_consts)


def fit_mse(tree, pset, X, y, bounds=(-5.0, 5.0), max_iter=8):
    """MSE of a tree on the data X (samples x features), y, with its constants fitted
    by Levenberg-Marquardt (``fit_constants``), starting from random values
    (``_start_constants``).

    Args:
        tree: the individual (DEAP ``PrimitiveTree``), whose constants are the
            terminals named ``c``.
        pset: the primitive set of the tree; the argument at position i reads the
            column i of ``X``.
        X, y: the data.
        bounds: lower and upper bound of all the constants.
        max_iter: maximum number of iterations of Levenberg-Marquardt.

    Returns:
        the MSE (inf or nan if the tree is not finite on the data) and the fitted
        constants (empty array if the tree has none).
    """
    ops, idx, nums, num_consts = encode(tree, pset.arguments, context=pset.context)
    y = np.ravel(y)
    if num_consts == 0:
        return sse(ops, idx, nums, X, y, np.empty(0)) / len(y), np.empty(0)
    ones = np.ones(num_consts)
    x0 = _start_constants(ops, idx, nums, num_consts, bounds)
    consts, f, _, _ = fit_constants(
        ops, idx, nums, X, y, x0, bounds[0] * ones, bounds[1] * ones, max_iter
    )
    return f / len(y), consts


def mse(tree, pset, X, y, consts=()):
    """MSE of a tree with the given constants on the data X (samples x features), y."""
    ops, idx, nums, num_consts = encode(tree, pset.arguments, context=pset.context)
    if len(consts) != num_consts:
        raise ValueError(f"the tree has {num_consts} constants, got {len(consts)}")
    y = np.ravel(y)
    return sse(ops, idx, nums, X, y, np.asarray(consts, dtype=float)) / len(y)


def predict_tree(tree, pset, X, consts=()):
    """Values of a tree with the given constants on the data X (samples x features)."""
    ops, idx, nums, num_consts = encode(tree, pset.arguments, context=pset.context)
    if len(consts) != num_consts:
        raise ValueError(f"the tree has {num_consts} constants, got {len(consts)}")
    return predict(ops, idx, nums, X, np.asarray(consts, dtype=float))


# Batch functions for GPSymbolicRegressor: fitness_batch (``fitness``), score_batch
# (``score_func``), predict_batch (``predict_func``) and assign_attributes
# (``callback_func``). The options of fitness_batch are set through ``common_data``,
# which the regressor passes to all three functions.


def fitness_batch(individuals_batch, toolbox, X, y, bounds=(-5.0, 5.0), max_iter=8,
                  parsimony=0.0):
    """Fitness of each individual: its MSE with the constants fitted by ``fit_mse``
    plus ``parsimony`` times its size (tree length plus number of constants); 1e8 if
    the MSE is not finite.

    Returns:
        for each individual, a dict with the fitness (tuple) and the fitted constants,
        stored on the individual by ``assign_attributes``.
    """
    pset = toolbox.compile.keywords["pset"]
    attributes = []
    for tree in individuals_batch:
        MSE, consts = fit_mse(tree, pset, X, y, bounds, max_iter)
        if np.isfinite(MSE):
            value = MSE + parsimony * (len(tree) + len(consts))
        else:
            value = 1e8
        attributes.append({"fitness": (value,), "consts": consts})
    return attributes


def assign_attributes(individuals_batch, attributes):
    """Stores the fitness and the constants returned by ``fitness_batch`` on the
    individuals (``ind.fitness.values`` and ``ind.consts``)."""
    for ind, attr in zip(individuals_batch, attributes):
        ind.fitness.values = attr["fitness"]
        ind.consts = attr["consts"]


def score_batch(individuals_batch, toolbox, X, y, **options):
    """MSE of each individual with its stored constants (``ind.consts``).

    Since lower MSE means a better model, it is suited to the ``score`` method of
    ``GPSymbolicRegressor`` but not to validation (``validate=True``), which requires a
    score where higher is better."""
    pset = toolbox.compile.keywords["pset"]
    return [mse(ind, pset, X, y, ind.consts) for ind in individuals_batch]


def predict_batch(individuals_batch, toolbox, X, **options):
    """Predictions of each individual with its stored constants (``ind.consts``)."""
    pset = toolbox.compile.keywords["pset"]
    return [predict_tree(ind, pset, X, ind.consts) for ind in individuals_batch]
