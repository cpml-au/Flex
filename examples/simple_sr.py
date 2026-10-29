import os
from deap import gp
from flex.gp.regressor import GPSymbolicRegressor
import numpy as np
import ray
from flex.gp.util import (
    get_features_batch,
    load_config_data,
)
from flex.gp.numba_eval import mse, predict_tree
from flex.gp.primitives import add_primitives_to_pset_from_dict


# Ground truth
x = np.array([x / 10.0 for x in range(-10, 10)]).reshape(-1, 1)
y = (x**4 + x**3 + x**2 + x).ravel()


def predict(individuals_batch, toolbox, X, penalty):
    pset = toolbox.compile.keywords["pset"]
    return [predict_tree(ind, pset, X) for ind in individuals_batch]


def score(individuals_batch, toolbox, X, y, penalty):
    pset = toolbox.compile.keywords["pset"]
    return [mse(ind, pset, X, y) for ind in individuals_batch]


def fitness(individuals_batch, toolbox, X, y, penalty):
    pset = toolbox.compile.keywords["pset"]

    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_batch)

    fitnesses = [None] * len(individuals_batch)
    for i, ind in enumerate(individuals_batch):
        if individ_length[i] >= 50:
            fitnesses[i] = (1e8,)
        else:
            MSE = mse(ind, pset, X, y)
            if np.isnan(MSE):
                MSE = 1e5

            fitnesses[i] = (
                MSE
                + 100000 * nested_trigs[i]
                + penalty["reg_param"] * individ_length[i],
            )

    return fitnesses


def main():
    yamlfile = "simple_sr.yaml"
    filename = os.path.join(os.path.dirname(__file__), yamlfile)

    regressor_params, config_file_data = load_config_data(filename)

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [
            float,
        ],
        float,
    )
    pset.renameArguments(ARG0="x")

    pset = add_primitives_to_pset_from_dict(pset, config_file_data["gp"]["primitives"])

    penalty = config_file_data["gp"]["penalty"]
    common_data = {"penalty": penalty}

    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness,
        score_func=score,
        predict_func=predict,
        common_data=common_data,
        print_log=True,
        batch_size=100,
        **regressor_params
    )

    gpsr.fit(x, y)

    ray.shutdown()


if __name__ == "__main__":
    main()
