"""Cost of one evaluation (prediction + sum of squared errors) of the same expression,
c1*x^3 + c2*x^2 + c3*x, in flex (compiled DEAP tree, one numpy call per node), in the
numba interpreter (``flex.gp.numba_eval``) and in the backend of PySR
(SymbolicRegression.jl ``eval_tree_array``, with and without turbo mode), for several
numbers of data points. Also reports the cost of compiling the tree (flex) and of
encoding it (numba interpreter).

Run from the ``bench`` directory:

    env PYTHON_JULIACALL_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0 python eval_cost_pysr.py
"""

import copy
import timeit

import numpy as np
from deap import base, gp

import pysr  # noqa: F401  (sets up the Julia environment of PySR)
from flex.gp.primitives import add_primitives_to_pset_from_dict
from flex.gp.util import compile_individual_with_consts, load_config_data
from flex.gp.numba_eval import encode, sse
from juliacall import Main as jl

NUM_POINTS = [20, 200, 2000, 20000, 200000]

jl.seval("""
using SymbolicRegression, LoopVectorization
function make_sse(turbo)
    opts = Options(binary_operators=[+, -, *, /], unary_operators=[sin, cos, exp, log],
                   turbo=turbo)
    x1 = Node{Float64}(feature=1)
    tree = Node(1, Node(3, Node{Float64}(val=3.39), Node(3, x1, Node(3, x1, x1))),
                Node(1, Node(3, Node{Float64}(val=2.12), Node(3, x1, x1)),
                     Node(3, Node{Float64}(val=1.78), x1)))
    function sse(X, y)
        out, ok = eval_tree_array(tree, X, opts)
        s = 0.0
        @inbounds for i in eachindex(y)
            s += abs2(out[i] - y[i])
        end
        return s
    end
    return sse
end
function time_eval(sse, Xpy, ypy, nrep)
    X = Matrix{Float64}(Xpy); y = Vector{Float64}(ypy)   # native Julia arrays
    sse(X, y)  # compile
    return (@elapsed for _ in 1:nrep; sse(X, y); end) / nrep
end
""")


def main():
    _, cfg = load_config_data("Nguyen_13_consts_lbfgs.yaml")
    pset = gp.PrimitiveSetTyped("Main", [float], float)
    pset.renameArguments(ARG0="x")
    pset = add_primitives_to_pset_from_dict(pset, cfg["gp"]["primitives"])
    pset.addTerminal(object, float, "c")
    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("clone", copy.deepcopy)
    tree = gp.PrimitiveTree.from_string(
        "add(mul(c, mul(x, mul(x, x))), add(mul(c, mul(x, x)), mul(c, x)))", pset
    )
    f, _ = compile_individual_with_consts(tree, toolbox)
    ops, idx, _ = encode(tree, pset.arguments)
    c = np.array([3.39, 2.12, 1.78])
    sse_default, sse_turbo = jl.make_sse(False), jl.make_sse(True)

    nrep = 20000
    t_compile = 1e6 * timeit.timeit(
        lambda: compile_individual_with_consts(tree, toolbox), number=nrep) / nrep
    t_encode = 1e6 * timeit.timeit(
        lambda: encode(tree, pset.arguments), number=nrep) / nrep
    print(f"compile (flex) {t_compile:.2f} us, encode (numba) {t_encode:.2f} us\n")

    print(f"{'points':>8}{'flex (us)':>12}{'numba (us)':>12}{'PySR (us)':>12}"
          f"{'PySR turbo (us)':>17}")
    rng = np.random.default_rng(0)
    for n in NUM_POINTS:
        x = rng.random((n, 1))
        y = 3.39 * x**3 + 2.12 * x**2 + 1.78 * x
        nrep = max(20, 200000 // n)

        def flex_sse():
            res = y - f(x, c)
            return np.vdot(res, res)

        t_flex = 1e6 * timeit.timeit(flex_sse, number=nrep) / nrep
        yr = y.ravel()
        sse(ops, idx, x, yr, c)  # compile
        t_numba = (
            1e6 * timeit.timeit(lambda: sse(ops, idx, x, yr, c), number=nrep) / nrep
        )
        Xj, yj = np.ascontiguousarray(x.T), yr
        t_def = 1e6 * jl.time_eval(sse_default, Xj, yj, nrep)
        t_tur = 1e6 * jl.time_eval(sse_turbo, Xj, yj, nrep)
        print(f"{n:>8}{t_flex:>12.2f}{t_numba:>12.2f}{t_def:>12.2f}{t_tur:>17.2f}")


if __name__ == "__main__":
    main()
