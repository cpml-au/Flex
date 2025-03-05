import yaml
import os
from dctkit import config
from deap import gp
from alpine.gp.gpsymbreg import GPSymbolicRegressor
from alpine.data import Dataset
import jax.numpy as jnp
import ray

# sets CPU device for JAX at process level
os.environ["JAX_PLATFORMS"] = "cpu"

config()


def compile_individuals(toolbox, individuals_str_batch):
    return [toolbox.compile(expr=ind) for ind in individuals_str_batch]


x = jnp.array([x / 10.0 for x in range(-10, 10)])
y = x**4 + x**3 + x**2 + x


def eval_MSE_sol(individual, true_data):
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    config()
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    y_pred = individual(true_data.X)
    MSE = jnp.sum(jnp.square(y_pred - true_data.y)) / len(true_data.X)
    if jnp.isnan(MSE):
        MSE = 1e5
    return MSE, y_pred


@ray.remote
def predict(individuals_str, toolbox, true_data):

    callables = compile_individuals(toolbox, individuals_str)

    u = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        _, u[i] = eval_MSE_sol(ind, true_data)

    return u


@ray.remote
def score(individuals_str, toolbox, true_data):

    callables = compile_individuals(toolbox, individuals_str)

    MSE = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        MSE[i], _ = eval_MSE_sol(ind, true_data)

    return MSE


@ray.remote
def fitness(individuals_str, toolbox, true_data):
    callables = compile_individuals(toolbox, individuals_str)

    fitnesses = [None] * len(individuals_str)
    for i, ind in enumerate(callables):
        MSE, _ = eval_MSE_sol(ind, true_data)

        fitnesses[i] = (MSE,)

    return fitnesses


def test_basic_sr(set_test_dir):
    yamlfile = "test_basic_sr.yaml"
    filename = os.path.join(os.path.dirname(__file__), yamlfile)
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [
            float,
        ],
        float,
    )
    pset.addPrimitive(jnp.add, [float, float], float, "AddF")
    pset.renameArguments(ARG0="x")

    common_data = {}
    seed = [
        "AddF(AddF(AddF(MulF(MulF(x, MulF(x, x)),x), MulF(x,MulF(x, x))), MulF(x, x)), x)"
    ]  # noqa: E501

    gpsr = GPSymbolicRegressor(
        pset=pset,
        fitness=fitness.remote,
        error_metric=score.remote,
        predict_func=predict.remote,
        common_data=common_data,
        config_file_data=config_file_data,
        seed=seed,
        batch_size=10,
    )

    train_data = Dataset("true_data", x, y)
    gpsr.fit(train_data)

    fit_score = gpsr.score(train_data)

    ray.shutdown()

    assert fit_score <= 1e-12
