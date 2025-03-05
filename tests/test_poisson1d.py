from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_line_mesh, build_complex_from_mesh
from dctkit.math.opt import optctrl as oc
from deap import gp
from alpine.gp import gpsymbreg as gps
from alpine.data import Dataset
from dctkit import config
import dctkit
import numpy as np
import ray
import math
from typing import Tuple, Callable, List
import os
import pytest
from alpine.gp import util

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()


def compile_individuals(toolbox, individuals_str_batch):
    return [toolbox.compile(expr=ind) for ind in individuals_str_batch]


def get_features_batch(individ_feature_extractors, individuals_str_batch):
    features_batch = [
        [fe(i) for i in individuals_str_batch] for fe in individ_feature_extractors
    ]

    indlen = features_batch[0]
    return indlen


def eval_MSE_sol(
    residual: Callable, D: Dataset, S: SimplicialComplex, u_0: C.CochainP0
) -> float:

    num_nodes = D.X.shape[1]

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    # objective: squared norm of the residual of the equation + penalty on Dirichlet
    # boundary condition on the first node
    def obj(x, y):
        penalty = 100.0 * x[0] ** 2
        u = C.CochainP0(S, x)
        f = C.CochainP0(S, y)
        r = residual(u, f)
        total_energy = C.inner(r, r) + penalty
        return total_energy

    prb = oc.OptimizationProblem(dim=num_nodes, state_dim=num_nodes, objfun=obj)

    MSE = 0.0

    u = []

    for i, curr_y in enumerate(D.y):
        # set additional arguments of the objective function
        # (apart from the vector of unknowns)
        args = {"y": curr_y}
        prb.set_obj_args(args)

        # minimize the objective
        x = prb.solve(
            x0=u_0.coeffs.flatten(), ftol_abs=1e-12, ftol_rel=1e-12, maxeval=1000
        )

        if (
            prb.last_opt_result == 1
            or prb.last_opt_result == 3
            or prb.last_opt_result == 4
        ):

            current_err = np.linalg.norm(x - D.X[i, :]) ** 2
        else:
            current_err = math.nan

        if math.isnan(current_err):
            MSE = 1e5
            break

        MSE += current_err

        u.append(x)

    MSE *= 1 / D.X.shape[0]

    return MSE, u


@ray.remote
def predict(
    individuals_str: list[str],
    toolbox,
    D: Dataset,
    S: SimplicialComplex,
    u_0: C.CochainP0,
    penalty: dict,
) -> List:

    callables = compile_individuals(toolbox, individuals_str)

    u = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        _, u[i] = eval_MSE_sol(ind, D, S, u_0)

    return u


@ray.remote
def score(
    individuals_str: list[str],
    toolbox,
    D: Dataset,
    S: SimplicialComplex,
    u_0: C.CochainP0,
    penalty: dict,
) -> List:

    callables = compile_individuals(toolbox, individuals_str)

    MSE = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        MSE[i], _ = eval_MSE_sol(ind, D, S, u_0)

    return MSE


@ray.remote
def fitness(
    individuals_str: list[str],
    toolbox,
    D: Dataset,
    S: SimplicialComplex,
    u_0: C.CochainP0,
    penalty: dict,
) -> Tuple[float,]:

    callables = compile_individuals(toolbox, individuals_str)
    indlen = get_features_batch([len], individuals_str)

    fitnesses = [None] * len(individuals_str)
    for i, ind in enumerate(callables):
        MSE, _ = eval_MSE_sol(ind, D, S, u_0)

        # add penalty on length of the tree to promote simpler solutions
        fitnesses[i] = (MSE + penalty["reg_param"] * indlen[i],)

    return fitnesses


cases = ["poisson1d_1.yaml", "poisson1d_2.yaml"]


@pytest.mark.parametrize("yamlfile", cases)
def test_poisson1d(set_test_dir, yamlfile):
    filename = os.path.join(os.path.dirname(__file__), yamlfile)

    regressor_params, config_file_data = util.load_config_data(filename)

    # generate mesh and dataset
    mesh, _ = generate_line_mesh(num_nodes=11, L=1.0)
    S = build_complex_from_mesh(mesh)
    S.get_hodge_star()
    x = S.node_coords
    num_nodes = S.num_nodes

    # generate training and test datasets
    # exact solution = x²
    u = C.CochainP0(S, np.array(x[:, 0] ** 2, dtype=dctkit.float_dtype))
    # compute source term such that u solves the discrete Poisson equation
    # Delta u + f = 0, where Delta is the discrete Laplace-de Rham operator
    f = C.laplacian(u)
    f.coeffs *= -1.0
    X_train = np.array([u.coeffs.flatten()], dtype=dctkit.float_dtype)
    y_train = np.array([f.coeffs.flatten()], dtype=dctkit.float_dtype)

    # initial guess for the unknown of the Poisson problem (cochain of nodals values)
    u_0_vec = np.zeros(num_nodes, dtype=dctkit.float_dtype)
    u_0 = C.CochainP0(S, u_0_vec)

    # define primitive set for the residual of the discrete Poisson equation
    pset = gp.PrimitiveSetTyped("RESIDUAL", [C.CochainP0, C.CochainP0], C.CochainP0)

    # rename arguments of the residual
    pset.renameArguments(ARG0="u")
    pset.renameArguments(ARG1="f")

    pset = util.add_primitives_to_pset_from_dict(
        pset, config_file_data["gp"]["primitives"]
    )

    seed_str = ["AddCP0(delP1(cobP0(u)),f)"]

    penalty = config_file_data["gp"]["penalty"]
    common_params = {"S": S, "u_0": u_0, "penalty": penalty}

    gpsr = gps.GPSymbolicRegressor(
        pset=pset,
        fitness=fitness.remote,
        error_metric=score.remote,
        predict_func=predict.remote,
        print_log=True,
        common_data=common_params,
        seed=seed_str,
        plot_history=False,
        save_best_individual=True,
        save_train_fit_history=True,
        output_path="./",
        **regressor_params
    )

    train_data = Dataset("D", X_train, y_train)

    gpsr.fit(train_data, val_data=train_data)

    u_best = gpsr.predict(train_data)

    fit_score = gpsr.score(train_data)

    gpsr.save_best_test_sols(train_data, "./")

    ray.shutdown()
    assert np.allclose(u.coeffs.flatten(), np.ravel(u_best))
    assert fit_score <= 1e-12
