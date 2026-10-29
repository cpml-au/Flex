import copy
import random
from functools import partial
import warnings

import numpy as np
from deap import base, gp
from flex.gp.numba_eval import (
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
    sse_normal_equations,
)
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
