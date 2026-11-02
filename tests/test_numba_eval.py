import copy
import pathlib
import random
from functools import partial
import warnings

import numpy as np
from deap import base, gp
from flex.gp.numba_eval import (
    _start_constants,
    assign_attributes,
    encode,
    fit_constants,
    fit_mse,
    fitness_batch,
    mse,
    predict,
    predict_batch,
    predict_tree,
    score_batch,
    sse,
)
from flex.gp import numba_primitives
from flex.gp.numba_grad import sse_normal_equations
from flex.gp.numba_primitives import _cache_dir, apply_unary
from flex.gp.numpy_primitives import numpy_primitives
from flex.gp.primitives import add_primitives_to_pset_from_dict
from flex.gp.regressor import GPSymbolicRegressor
from flex.gp.util import compile_individual_with_consts


def make_pset(num_args):
    pset = gp.PrimitiveSetTyped("Main", [float] * num_args, float)
    pset.renameArguments(ARG0="x")
    primitives = {
        "imports": {"flex.gp.numpy_primitives": ["numpy_primitives"]},
        "used": [
            {"name": name, "dimension": None, "rank": None}
            for name in numpy_primitives
        ],
    }
    pset = add_primitives_to_pset_from_dict(pset, primitives)
    pset.addTerminal(object, float, "c")
    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("clone", copy.deepcopy)
    return pset, toolbox


def test_predict_and_sse():
    warnings.filterwarnings("ignore")
    pset, toolbox = make_pset(2)
    random.seed(0)
    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 1.0, (300, 2))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2

    num_finite = 0
    num_mismatches = 0
    for _ in range(2000):
        tree = gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=1, max_=4))
        f, num_consts = compile_individual_with_consts(tree, toolbox)
        ops, idx, nums, num_consts_numba = encode(tree, pset.arguments)
        assert num_consts_numba == num_consts
        consts = rng.uniform(0.1, 1.0, max(num_consts, 1))

        expected = np.broadcast_to(f(X[:, 0], X[:, 1], consts), y.shape)
        pred = predict(ops, idx, nums, X, consts)
        finite = np.isfinite(expected)
        if not (
            np.array_equal(finite, np.isfinite(pred))
            and np.allclose(pred[finite], expected[finite], rtol=1e-10)
        ):
            num_mismatches += 1
            continue
        if finite.all():
            num_finite += 1
            assert np.isclose(
                sse(ops, idx, nums, X, y, consts),
                np.sum((y - expected) ** 2),
                rtol=1e-10,
            )
    assert num_finite > 500
    # a few trees amplify the last-bit differences between the elementary functions of
    # numba and numpy (e.g. sin of values around 1e30); a wrong primitive would make
    # many trees differ
    assert num_mismatches <= 10


def test_normal_equations():
    pset, _ = make_pset(1)
    rng = np.random.default_rng(1)
    X = rng.uniform(0.5, 1.5, (100, 1))
    y = 2.0 * np.sin(X[:, 0]) + 0.5
    exprs = [
        "add(mul(c, sin(x)), c)",
        "aq(mul(c, x), prot_pow(x, c))",
        "div(exp(mul(c, x)), add(square(x), c))",
        "tanh(sub(mul(c, arcsin(div(x, c))), sqrt(c)))",
    ]
    for expr in exprs:
        tree = gp.PrimitiveTree.from_string(expr, pset)
        ops, idx, nums, k = encode(tree, pset.arguments)
        consts = rng.uniform(2.0, 3.0, k)
        f, JTJ, JTr = sse_normal_equations(ops, idx, nums, X, y, consts)

        # Jacobian of the predictions by central finite differences
        h = 1e-6
        J = np.empty((len(y), k))
        for c in range(k):
            e = np.zeros(k)
            e[c] = h
            J[:, c] = (
                predict(ops, idx, nums, X, consts + e)
                - predict(ops, idx, nums, X, consts - e)
            ) / (2 * h)
        r = predict(ops, idx, nums, X, consts) - y
        assert np.isclose(f, np.sum(r**2), rtol=1e-12)
        assert np.allclose(JTr, J.T @ r, rtol=1e-6)
        assert np.allclose(JTJ, J.T @ J, rtol=1e-6)


def fd_jacobian(ops, idx, nums, X, consts, h):
    """Jacobian of the tree values with respect to the constants, by central finite
    differences with step h."""
    J = np.empty((len(X), len(consts)))
    for c in range(len(consts)):
        e = np.zeros(len(consts))
        e[c] = h
        J[:, c] = (
            predict(ops, idx, nums, X, consts + e)
            - predict(ops, idx, nums, X, consts - e)
        ) / (2 * h)
    return J


def test_normal_equations_random_trees():
    # arithmetic operators (with and without constants in each operand) and a unary
    # primitive; exp is left out, since nested exps reach values (e.g. 1e38) at which
    # the finite differences are no longer a reference for the derivatives
    pset = gp.PrimitiveSetTyped("Main", [float], float)
    pset.renameArguments(ARG0="x")
    primitives = {
        "imports": {"flex.gp.numpy_primitives": ["numpy_primitives"]},
        "used": [
            {"name": name, "dimension": None, "rank": None}
            for name in ["add", "sub", "mul", "div", "sin"]
        ],
    }
    pset = add_primitives_to_pset_from_dict(pset, primitives)
    pset.addTerminal(object, float, "c")
    random.seed(2)
    rng = np.random.default_rng(2)
    X = rng.uniform(0.5, 1.5, (50, 1))
    y = np.sin(X[:, 0])
    num_checked = 0
    for _ in range(500):
        tree = gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=2, max_=5))
        ops, idx, nums, k = encode(tree, pset.arguments)
        if k == 0:
            continue
        consts = rng.uniform(1.0, 2.0, k)
        f, JTJ, JTr = sse_normal_equations(ops, idx, nums, X, y, consts)
        if not (np.isfinite(f) and f < 1e6):
            continue
        J, J_fine = (fd_jacobian(ops, idx, nums, X, consts, h) for h in (1e-6, 1e-7))
        # finite differences are not a reliable reference where the derivatives are
        # undefined (infinite intermediate values) or large, or where they have not
        # converged (e.g. near a pole of a division)
        if not (np.all(np.isfinite(JTJ)) and np.abs(J).max() < 1e3):
            continue
        if not np.allclose(J, J_fine, rtol=1e-4, atol=1e-6 * np.abs(J).max()):
            continue
        r = predict(ops, idx, nums, X, consts) - y
        scale = np.max(np.abs(J.T @ J)) + 1.0
        assert np.allclose(JTr, J.T @ r, rtol=1e-5, atol=1e-5 * scale)
        assert np.allclose(JTJ, J.T @ J, rtol=1e-5, atol=1e-5 * scale)
        num_checked += 1
    assert num_checked > 200


def test_pow_derivatives_at_zero_base():
    # at a = 0 the derivatives of a^b and |a|^b exist for these exponents, although the
    # general formulas b * v / a and v * log(a) are not defined there
    pset, _ = make_pset(1)
    for number in [0.0, 1.0, 2.0]:
        pset.addTerminal(number, float)
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_nonneg = x[1:]
    c = 1.5
    p = 2.0
    cases = [
        ("pow(mul(c, x), 2.0)", [c], x, lambda x: np.c_[2 * c * x**2]),
        ("prot_pow(mul(c, x), 2.0)", [c], x, lambda x: np.c_[2 * c * x**2]),
        ("pow(mul(c, x), 1.0)", [c], x, lambda x: np.c_[x]),
        ("pow(mul(c, x), 0.0)", [c], x, lambda x: np.c_[0 * x]),
        ("prot_pow(mul(c, x), 0.0)", [c], x, lambda x: np.c_[0 * x]),
        (
            "mul(c, pow(x, c))",
            [c, p],
            x_nonneg,
            lambda x: np.c_[x**p, c * x**p * np.log(np.where(x > 0, x, 1.0))],
        ),
        (
            "mul(c, prot_pow(x, c))",
            [c, p],
            x,
            lambda x: np.c_[
                np.abs(x) ** p,
                c * np.abs(x) ** p * np.log(np.where(x != 0, np.abs(x), 1.0)),
            ],
        ),
    ]
    for expr, consts, x_case, jacobian in cases:
        X = x_case.reshape(-1, 1)
        consts = np.array(consts)
        tree = gp.PrimitiveTree.from_string(expr, pset)
        ops, idx, nums, _ = encode(tree, pset.arguments)
        y = np.linspace(-1.0, 1.0, len(X))
        f, JTJ, JTr = sse_normal_equations(ops, idx, nums, X, y, consts)
        J = jacobian(x_case)
        r = predict(ops, idx, nums, X, consts) - y
        assert np.isclose(f, r @ r), expr
        assert np.allclose(JTJ, J.T @ J), expr
        assert np.allclose(JTr, J.T @ r), expr


def test_fit_with_pow_at_zero_base():
    # a data point at x = 0 used to stop these fits before their first step
    pset, _ = make_pset(1)
    pset.addTerminal(2.0, float)
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    for expr in [
        "pow(mul(c, x), 2.0)",
        "prot_pow(mul(c, x), 2.0)",
        "mul(c, pow(x, c))",
        "mul(c, prot_pow(x, c))",
    ]:
        tree = gp.PrimitiveTree.from_string(expr, pset)
        ops, idx, nums, k = encode(tree, pset.arguments)
        ones = np.ones(k)
        y = predict(ops, idx, nums, X, 2.0 * ones)
        consts, f, num_trials, _ = fit_constants(
            ops, idx, nums, X, y, ones, -5.0 * ones, 5.0 * ones, 30
        )
        assert num_trials > 0, expr
        assert np.allclose(consts, 2.0 * ones), expr
        assert f < 1e-20, expr


def test_fit_stops_at_singular_points():
    # where a derivative is not finite the fit returns the constants reached so far, here
    # the starting ones, with their (finite) sum of squared errors
    pset, _ = make_pset(1)
    for number in [0.5, 1.0]:
        pset.addTerminal(number, float)
    cases = [
        # the derivative is infinite or does not exist at x = c = 1
        ("sqrt(sub(x, c))", 1.0, [1.0, 2.0, 3.0]),
        ("arcsin(div(x, c))", 1.0, [0.0, 0.5, 1.0]),
        ("pow(sub(x, c), 0.5)", 1.0, [1.0, 2.0, 3.0]),
        ("prot_pow(sub(x, c), 1.0)", 1.0, [1.0, 2.0, 3.0]),
        # 0^c is discontinuous at c = 0
        ("pow(x, c)", 0.0, [0.0, 1.0, 2.0]),
        # the derivative is larger than the largest float
        ("exp(mul(c, x))", 1.0, [1.0, 700.0]),
    ]
    for expr, c, x in cases:
        X = np.array(x).reshape(-1, 1)
        tree = gp.PrimitiveTree.from_string(expr, pset)
        ops, idx, nums, k = encode(tree, pset.arguments)
        x0 = np.full(k, c)
        y = predict(ops, idx, nums, X, x0) + 1.0
        f0 = sse(ops, idx, nums, X, y, x0)
        _, JTJ, _ = sse_normal_equations(ops, idx, nums, X, y, x0)
        assert not np.all(np.isfinite(JTJ)), expr
        consts, f, num_trials, num_jac = fit_constants(
            ops, idx, nums, X, y, x0, -5.0 * np.ones(k), 5.0 * np.ones(k), 8
        )
        assert np.array_equal(consts, x0), expr
        assert f == f0 and np.isfinite(f), expr
        assert num_trials == 0 and num_jac == 1, expr


def test_derivatives_of_elementary_functions():
    pset, _ = make_pset(1)
    x = np.linspace(0.5, 1.5, 300)
    X = x.reshape(-1, 1)
    y = np.sin(3.0 * x)
    a, b = 0.7, 1.3
    cases = [
        ("exp(mul(c, x))", [a], lambda: np.c_[x * np.exp(a * x)]),
        ("sin(mul(c, x))", [a], lambda: np.c_[x * np.cos(a * x)]),
        ("cos(mul(c, x))", [a], lambda: np.c_[-x * np.sin(a * x)]),
        ("log(mul(c, x))", [a], lambda: np.c_[np.full_like(x, 1.0 / a)]),
        (
            "sin(add(mul(c, x), c))",
            [a, b],
            lambda: np.c_[x * np.cos(a * x + b), np.cos(a * x + b)],
        ),
        (
            "log(add(mul(c, x), c))",
            [a, b],
            lambda: np.c_[x / (a * x + b), 1.0 / (a * x + b)],
        ),
        (
            "exp(sin(mul(c, x)))",
            [a],
            lambda: np.c_[x * np.cos(a * x) * np.exp(np.sin(a * x))],
        ),
        (
            "cos(log(add(c, x)))",
            [a],
            lambda: np.c_[-np.sin(np.log(a + x)) / (a + x)],
        ),
        ("log(exp(mul(c, x)))", [a], lambda: np.c_[x]),
        (
            "mul(cos(mul(c, x)), exp(div(x, c)))",
            [a, b],
            lambda: np.c_[
                -x * np.sin(a * x) * np.exp(x / b),
                -np.cos(a * x) * np.exp(x / b) * x / b**2,
            ],
        ),
    ]
    for expr, consts, jacobian in cases:
        consts = np.array(consts)
        tree = gp.PrimitiveTree.from_string(expr, pset)
        ops, idx, nums, _ = encode(tree, pset.arguments)
        _, JTJ, JTr = sse_normal_equations(ops, idx, nums, X, y, consts)
        J = jacobian()
        r = predict(ops, idx, nums, X, consts) - y
        assert np.allclose(JTJ, J.T @ J, rtol=1e-12), expr
        assert np.allclose(JTr, J.T @ r, rtol=1e-12), expr


def test_derivatives_of_random_trees_with_elementary_functions():
    # reference: complex step on the compiled tree, independent of the derivative rules
    warnings.filterwarnings("ignore")
    h = 1e-30
    for name in ["exp", "sin", "cos", "log"]:
        pset = gp.PrimitiveSetTyped("Main", [float], float)
        pset.renameArguments(ARG0="x")
        primitives = {
            "imports": {"flex.gp.numpy_primitives": ["numpy_primitives"]},
            "used": [
                {"name": n, "dimension": None, "rank": None}
                for n in ["add", "sub", "mul", "div", name]
            ],
        }
        pset = add_primitives_to_pset_from_dict(pset, primitives)
        pset.addTerminal(object, float, "c")
        toolbox = base.Toolbox()
        toolbox.register("compile", gp.compile, pset=pset)
        random.seed(3)
        rng = np.random.default_rng(3)
        X = rng.uniform(0.5, 1.5, (50, 1))
        y = np.sin(X[:, 0])
        num_checked = num_mismatches = 0
        for _ in range(400):
            tree = gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=1, max_=5))
            ops, idx, nums, k = encode(tree, pset.arguments)
            if k == 0:
                continue
            consts = rng.uniform(1.0, 2.0, k)
            f, JTJ, JTr = sse_normal_equations(ops, idx, nums, X, y, consts)
            if not (np.isfinite(f) and np.all(np.isfinite(JTJ))):
                continue
            compiled, _ = compile_individual_with_consts(tree, toolbox)
            J = np.empty((len(y), k))
            for c in range(k):
                z = consts.astype(complex)
                z[c] += 1j * h
                J[:, c] = np.broadcast_to(compiled(X[:, 0], z), y.shape).imag / h
            if not np.all(np.isfinite(J)) or np.abs(J).max() > 1e6:
                continue
            num_checked += 1
            r = predict(ops, idx, nums, X, consts) - y
            scale = np.max(np.abs(J.T @ J)) + 1.0
            if not (
                np.allclose(JTr, J.T @ r, rtol=1e-7, atol=1e-7 * scale)
                and np.allclose(JTJ, J.T @ J, rtol=1e-7, atol=1e-7 * scale)
            ):
                num_mismatches += 1
        assert num_checked > 150, name
        # sin or cos of a very large value (e.g. of nested exps) is not accurate in
        # double precision; a wrong derivative rule would make many trees differ
        assert num_mismatches <= 3, name


def test_zero_factor_in_the_chain_rule():
    # the true derivative exists and the chain rule has a zero factor, with an infinite
    # or undefined other factor: the derivative is correct and the fit continues
    pset, _ = make_pset(1)
    pset.addTerminal(1.0, float)

    def sqrt_derivative(x, c):
        return np.divide(x, 2 * np.sqrt(c * x), out=np.zeros_like(x), where=x > 0)

    def exp_derivative(x, c):
        u = np.divide(1.0, x, out=np.zeros_like(x), where=x > 0)
        return u * np.exp(c * u)

    cases = [
        ("sqrt(mul(c, x))", 1.0, [0.0, 1.0, 4.0], sqrt_derivative),
        ("exp(log(mul(c, x)))", 1.0, [0.0, 1.0, 2.0], lambda x, c: x),
        ("prot_pow(mul(c, x), 1.0)", 1.0, [0.0, 1.0, 2.0], lambda x, c: x),
        ("exp(div(c, x))", -1.0, [0.0, 1.0, 2.0], exp_derivative),
    ]
    for expr, c, x, derivative in cases:
        x = np.array(x)
        X = x.reshape(-1, 1)
        tree = gp.PrimitiveTree.from_string(expr, pset)
        ops, idx, nums, _ = encode(tree, pset.arguments)
        consts = np.array([c])
        y = predict(ops, idx, nums, X, consts) + 1.0
        _, JTJ, JTr = sse_normal_equations(ops, idx, nums, X, y, consts)
        J = derivative(x, c).reshape(-1, 1)
        r = predict(ops, idx, nums, X, consts) - y
        assert np.allclose(JTJ, J.T @ J), expr
        assert np.allclose(JTr, J.T @ r), expr

        target = np.array([2.0 * c])
        y = predict(ops, idx, nums, X, target)
        fitted, f, num_trials, _ = fit_constants(
            ops, idx, nums, X, y, consts, -5.0 * np.ones(1), 5.0 * np.ones(1), 30
        )
        assert num_trials > 0, expr
        assert np.allclose(fitted, target) and f < 1e-20, expr

    # a constant in another subtree is not affected by the zero factor
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    tree = gp.PrimitiveTree.from_string("add(c, sqrt(mul(c, x)))", pset)
    ops, idx, nums, k = encode(tree, pset.arguments)
    y = predict(ops, idx, nums, X, 2.0 * np.ones(k))
    fitted, f, _, _ = fit_constants(
        ops, idx, nums, X, y, np.ones(k), -5.0 * np.ones(k), 5.0 * np.ones(k), 30
    )
    assert np.allclose(fitted, [2.0, 2.0]) and f < 1e-20

    # the tree underflows to 0 where its inner functions overflow: the derivative is 0
    X = np.array([[800.0]])
    tree = gp.PrimitiveTree.from_string("div(x, exp(exp(mul(c, x))))", pset)
    ops, idx, nums, k = encode(tree, pset.arguments)
    _, JTJ, JTr = sse_normal_equations(ops, idx, nums, X, np.ones(1), np.ones(k))
    assert JTJ[0, 0] == 0.0 and JTr[0] == 0.0


def test_kink_with_zero_inner_derivative():
    # sqrt((x - c)^2) = |x - c| has no derivative at x = c, but its inner derivative is
    # zero there, as for sqrt(c * x) at x = 0, which forward mode cannot tell apart: the
    # derivative is 0 and the fit continues
    pset, _ = make_pset(1)
    tree = gp.PrimitiveTree.from_string("sqrt(square(sub(x, c)))", pset)
    ops, idx, nums, k = encode(tree, pset.arguments)
    X = np.array([[1.0], [2.0], [3.0]])
    x0 = np.ones(k)
    _, JTJ, JTr = sse_normal_equations(ops, idx, nums, X[:1], np.ones(1), x0)
    assert JTJ[0, 0] == 0.0 and JTr[0] == 0.0
    _, _, num_trials, _ = fit_constants(
        ops, idx, nums, X, np.full(3, 1.0), x0, -5.0 * x0, 5.0 * x0, 8
    )
    assert num_trials > 0


def test_zero_factor_with_a_nonzero_limit():
    # x / (x / (c1 - c2)) = c1 - c2 has derivative 1 with respect to c1, but with equal
    # constants the chain rule has the product 0 * inf, whose limit here is not zero: the
    # computed derivative is 0. The fit is protected, since a step is accepted only if
    # the error of the values of the tree decreases
    pset, _ = make_pset(1)
    tree = gp.PrimitiveTree.from_string("div(x, div(x, sub(c, c)))", pset)
    ops, idx, nums, k = encode(tree, pset.arguments)
    X = np.array([[2.0]])
    consts = np.ones(k)
    v = predict(ops, idx, nums, X, consts)
    y = v - 1.0
    _, _, JTr = sse_normal_equations(ops, idx, nums, X, y, consts)
    assert v[0] == 0.0 and np.array_equal(JTr, [0.0, 0.0])


def test_trees_that_are_not_finite_on_the_data():
    # the fit does not start, also if a limit exists at the point, as for sin(c x) / x
    # and x log(c x) at x = 0
    pset, _ = make_pset(1)
    cases = [
        ("log(mul(c, x))", 1.0, [0.0, 1.0, 2.0]),
        ("log(sub(x, c))", 1.0, [1.0, 2.0, 3.0]),
        ("log(c)", -1.0, [1.0, 2.0, 3.0]),
        ("log(c)", 0.0, [1.0, 2.0, 3.0]),
        ("mul(x, log(mul(c, x)))", 1.0, [0.0, 1.0, 2.0]),
        ("exp(div(c, x))", 1.0, [0.0, 1.0, 2.0]),
        ("exp(mul(c, x))", 1.0, [1.0, 800.0]),
        ("sin(div(c, x))", 1.0, [0.0, 1.0, 2.0]),
        ("cos(div(c, x))", 1.0, [0.0, 1.0, 2.0]),
        ("div(sin(mul(c, x)), x)", 1.0, [0.0, 1.0, 2.0]),
        ("log(cos(mul(c, x)))", 1.0, [0.0, 2.0]),
    ]
    for expr, c, x in cases:
        X = np.array(x).reshape(-1, 1)
        tree = gp.PrimitiveTree.from_string(expr, pset)
        ops, idx, nums, k = encode(tree, pset.arguments)
        x0 = np.full(k, c)
        ones = np.ones(k)
        consts, f, num_trials, num_jac = fit_constants(
            ops, idx, nums, X, np.zeros(len(x)), x0, -5.0 * ones, 5.0 * ones, 8
        )
        assert not np.isfinite(f), expr
        assert np.array_equal(consts, x0), expr
        assert num_trials == 0 and num_jac == 1, expr


def test_derivatives_near_a_pole_and_at_large_arguments():
    pset, _ = make_pset(1)
    # 1e-12 from the pole of log: large but accurate derivative
    x = -1.0 + 1e-12
    tree = gp.PrimitiveTree.from_string("log(add(c, x))", pset)
    ops, idx, nums, k = encode(tree, pset.arguments)
    X = np.array([[x]])
    y = predict(ops, idx, nums, X, np.ones(k)) - 1.0
    _, _, JTr = sse_normal_equations(ops, idx, nums, X, y, np.ones(k))
    assert np.isclose(JTr[0], 1.0 / (1.0 + x), rtol=1e-12)

    # exp(40) has a rounding error larger than the period of sin: the derivative (and the
    # value) of the tree are finite but not accurate, and the fit does not stop
    tree = gp.PrimitiveTree.from_string("sin(exp(mul(c, x)))", pset)
    ops, idx, nums, k = encode(tree, pset.arguments)
    X = np.array([[1.0], [40.0]])
    _, JTJ, _ = sse_normal_equations(ops, idx, nums, X, np.zeros(2), np.ones(k))
    assert np.all(np.isfinite(JTJ))


def test_numba_cache_depends_on_all_the_sources(tmp_path):
    sources = ["numba_primitives.py", "numba_grad.py", "numba_eval.py"]
    for source in sources:
        (tmp_path / source).write_text(source)
    folders = {_cache_dir(tmp_path)}
    for source in sources:
        (tmp_path / source).write_text(source + "changed")
        folders.add(_cache_dir(tmp_path))
    assert len(folders) == 4

    # a function of each module uses the folder of the current sources
    package_dir = pathlib.Path(numba_primitives.__file__).parent
    for func in [apply_unary, sse_normal_equations, fit_constants]:
        assert func._cache.cache_path.startswith(_cache_dir(package_dir))


def test_fit_constants():
    pset, _ = make_pset(1)
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = 3.39 * X[:, 0] ** 3 + 2.12 * X[:, 0] ** 2 + 1.78 * X[:, 0]
    tree = gp.PrimitiveTree.from_string(
        "add(mul(c, mul(x, mul(x, x))), add(mul(c, mul(x, x)), mul(c, x)))", pset
    )
    ops, idx, nums, k = encode(tree, pset.arguments)
    consts, f, num_trials, num_jac = fit_constants(
        ops, idx, nums, X, y, np.ones(k), -5.0 * np.ones(k), 5.0 * np.ones(k), 8
    )
    assert np.allclose(consts, [3.39, 2.12, 1.78])
    assert f < 1e-20
    assert num_trials <= 8 and num_jac <= num_trials + 1


def test_tree_helpers():
    pset, _ = make_pset(1)
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = 2.5 * np.sin(X[:, 0]) + 0.7
    tree = gp.PrimitiveTree.from_string("add(mul(c, sin(x)), c)", pset)
    MSE, consts = fit_mse(tree, pset, X, y)
    assert np.allclose(consts, [2.5, 0.7])
    assert MSE < 1e-20
    assert mse(tree, pset, X, y, consts) == MSE
    assert np.allclose(predict_tree(tree, pset, X, consts), y)

    # tree without constants
    tree = gp.PrimitiveTree.from_string("sin(x)", pset)
    MSE, consts = fit_mse(tree, pset, X, y)
    assert len(consts) == 0
    assert np.isclose(MSE, np.mean((np.sin(X[:, 0]) - y) ** 2))


def test_random_start_of_the_fit():
    pset, _ = make_pset(1)
    pset.addTerminal(2.0, float)
    pset.addTerminal(3.0, float)
    X = np.linspace(0.5, 1.5, 20).reshape(-1, 1)
    y = X[:, 0] / 2.0

    def start(expr):
        tree = gp.PrimitiveTree.from_string(expr, pset)
        ops, idx, nums, k = encode(tree, pset.arguments)
        return _start_constants(ops, idx, nums, k, (-5.0, 5.0))

    # within the bounds, different for each constant, and the same at each call
    x0 = start("add(c, mul(c, add(c, x)))")
    assert np.all(np.abs(x0) <= 5.0) and len(set(x0)) == 3
    assert np.array_equal(x0, start("add(c, mul(c, add(c, x)))"))
    # different trees start from different values, also if only a number differs
    assert not np.array_equal(start("add(c, 2.0)"), start("add(c, 3.0)"))

    tree = gp.PrimitiveTree.from_string("add(c, mul(c, add(c, x)))", pset)
    MSE, consts = fit_mse(tree, pset, X, y)
    MSE_again, consts_again = fit_mse(tree, pset, X, y)
    assert MSE == MSE_again and np.array_equal(consts, consts_again)

    # with equal start values c - c is zero and the tree is not finite on the data
    tree = gp.PrimitiveTree.from_string("div(x, sub(c, c))", pset)
    ops, idx, nums, k = encode(tree, pset.arguments)
    assert not np.isfinite(sse(ops, idx, nums, X, y, np.ones(k)))
    assert np.isfinite(fit_mse(tree, pset, X, y)[0])


def test_regressor_with_batch_functions():
    pset, _ = make_pset(1)
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = 2.5 * np.sin(X[:, 0]) + 0.7
    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness_batch,
        score_func=score_batch,
        predict_func=predict_batch,
        callback_func=assign_attributes,
        common_data={"parsimony": 1e-6},
        seed_str=["add(mul(c, sin(x)), c)"],
        # parents compete with the offspring, so the seeded (exact) individual survives
        overlapping_generation=True,
        num_individuals=20,
        generations=2,
        multiprocessing=False,
    )
    gpsr.fit(X, y)
    assert gpsr.score(X, y) < 1e-20
    assert np.allclose(gpsr.predict(X), y)


def test_numerical_terminals():
    # fixed numbers and ephemeral constants are evaluated as given and not fitted
    pset, toolbox = make_pset(1)
    pset.addTerminal(2.0, float)
    pset.addEphemeralConstant("half", partial(float, 0.5), float)
    half = pset.mapping["half"]()
    body = gp.PrimitiveTree.from_string("add(mul(c, sin(x)), mul(2.0, x))", pset)
    tree = gp.PrimitiveTree([pset.mapping["add"]] + list(body) + [half])
    assert str(tree) == "add(add(mul(c, sin(x)), mul(2.0, x)), 0.5)"
    X = np.linspace(0.1, 1.0, 20).reshape(-1, 1)
    y = 3.0 * np.sin(X[:, 0]) + 2.0 * X[:, 0] + 0.5

    compiled, num_consts = compile_individual_with_consts(tree, toolbox)
    assert num_consts == 1
    consts = np.array([1.7])
    expected = compiled(X[:, 0], consts)
    assert np.allclose(predict_tree(tree, pset, X, consts), expected)

    ops, idx, nums, k = encode(tree, pset.arguments)
    f, JTJ, JTr = sse_normal_equations(ops, idx, nums, X, y, consts)
    J = np.sin(X[:, 0])
    r = expected - y
    assert np.isclose(f, r @ r)
    assert np.isclose(JTr[0], J @ r) and np.isclose(JTJ[0, 0], J @ J)

    MSE, consts = fit_mse(tree, pset, X, y)
    assert np.allclose(consts, [3.0]) and MSE < 1e-20
