import os
from dctkit import config
from deap import gp
from alpine.gp.regressor import GPSymbolicRegressor
from alpine.data import Dataset
from alpine.gp import util
import jax.numpy as jnp
import ray

# sets CPU device for JAX at process level
os.environ["JAX_PLATFORMS"] = "cpu"

config()


def compile_individuals(toolbox, individuals_str_batch):
    return [toolbox.compile(expr=ind) for ind in individuals_str_batch]


x = jnp.array([x / 10.0 for x in range(-10, 10)])
y = x**4 + x**3 + x**2 + x


def eval_MSE_sol(individual, X, y):
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    config()
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    y_pred = individual(X)
    MSE = None

    if y is not None:
        MSE = jnp.mean(jnp.sum(jnp.square(y_pred - y)))
        MSE = jnp.nan_to_num(MSE, nan=1e5)

    return MSE, y_pred


@ray.remote
def predict(individuals_str, toolbox, X_test):

    callables = compile_individuals(toolbox, individuals_str)

    u = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        _, u[i] = eval_MSE_sol(ind, X_test, None)

    return u


@ray.remote
def score(individuals_str, toolbox, X_test, y_test):

    callables = compile_individuals(toolbox, individuals_str)

    MSE = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        MSE[i], _ = eval_MSE_sol(ind, X_test, y_test)

    return MSE


@ray.remote
def fitness(individuals_str, toolbox, X_train, y_train):
    callables = compile_individuals(toolbox, individuals_str)

    fitnesses = [None] * len(individuals_str)
    for i, ind in enumerate(callables):
        MSE, _ = eval_MSE_sol(ind, X_train, y_train)

        fitnesses[i] = (MSE,)

    return fitnesses


def test_basic_sr(set_test_dir):
    yamlfile = "test_basic_sr.yaml"
    filename = os.path.join(os.path.dirname(__file__), yamlfile)

    regressor_params, config_file_data = util.load_config_data(filename)

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [
            float,
        ],
        float,
    )
    pset.addPrimitive(jnp.add, [float, float], float, "AddF")
    pset.renameArguments(ARG0="x")

    pset = util.add_primitives_to_pset_from_dict(
        pset, config_file_data["gp"]["primitives"]
    )

    common_data = {}
    seed = [
        "AddF(AddF(AddF(MulF(MulF(x, MulF(x, x)),x), MulF(x,MulF(x, x))), MulF(x, x)), x)"  # noqa: E501
    ]

    gpsr = GPSymbolicRegressor(
        pset=pset,
        fitness=fitness.remote,
        error_metric=score.remote,
        predict_func=predict.remote,
        common_data=common_data,
        seed=seed,
        batch_size=10,
        **regressor_params
    )

    # train_data = Dataset("true_data", x, y)
    gpsr.fit(x, y)

    fit_score = gpsr.score(x, y)

    y_pred = gpsr.predict(x)

    ray.shutdown()

    assert fit_score <= 1e-12
