import os
from dctkit import config
from deap import gp
from flex.gp.regressor import GPSymbolicRegressor
from flex.gp.util import load_config_data, compile_individuals
from flex.gp.primitives import add_primitives_to_pset_from_dict
import jax.numpy as jnp
import ray

# sets CPU device for JAX at process level
os.environ["JAX_PLATFORMS"] = "cpu"

config()


x = jnp.array([x / 10.0 for x in range(-10, 10)]).reshape(-1, 1)
y = (x**4 + x**3 + x**2 + x).ravel()


def eval_MSE_sol(individual, X, y):
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    config()
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    # WARNING: since X is a column vector, you get a 2D array as an output,
    # while y is a 1D array
    y_pred = individual(X).ravel()
    MSE = None

    if y is not None:
        MSE = jnp.mean(jnp.square(y_pred - y))
        MSE = jnp.nan_to_num(MSE, nan=1e5)

    return MSE, y_pred


def predict(individuals_str, toolbox, X):

    callables = compile_individuals(toolbox, individuals_str)

    u = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        _, u[i] = eval_MSE_sol(ind, X, None)

    return u


def score(individuals_str, toolbox, X, y):

    callables = compile_individuals(toolbox, individuals_str)

    MSE = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        MSE[i], _ = eval_MSE_sol(ind, X, y)

    return MSE


def fitness(individuals_str, toolbox, X, y):
    callables = compile_individuals(toolbox, individuals_str)

    fitnesses = [None] * len(individuals_str)
    for i, ind in enumerate(callables):
        MSE, _ = eval_MSE_sol(ind, X, y)

        fitnesses[i] = (MSE,)

    return fitnesses


def test_basic_sr(set_test_dir):
    yamlfile = "test_basic_sr.yaml"
    filename = os.path.join(os.path.dirname(__file__), yamlfile)

    regressor_params, config_file_data = load_config_data(filename)

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [
            float,
        ],
        float,
    )
    pset.addPrimitive(jnp.add, [float, float], float, "AddF")
    pset.renameArguments(ARG0="x")

    pset = add_primitives_to_pset_from_dict(pset, config_file_data["gp"]["primitives"])

    common_data = {}
    seed = [
        "AddF(AddF(AddF(MulF(MulF(x, MulF(x, x)),x), MulF(x,MulF(x, x))), MulF(x, x)), x)"  # noqa: E501
    ]

    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness,
        score_func=score,
        predict_func=predict,
        common_data=common_data,
        seed_str=seed,
        batch_size=10,
        **regressor_params
    )

    gpsr.fit(x, y)

    fit_score = gpsr.score(x, y)

    ray.shutdown()

    assert fit_score <= 1e-12
