import copy
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
from flex.gp.numba_grad import sse_normal_equations
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
        # the true derivative is 0 at x = 0, but the chain rule gives 0 * inf = nan
        ("sqrt(mul(c, x))", 1.0, [0.0, 1.0, 2.0]),
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
