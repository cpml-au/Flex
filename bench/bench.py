# import matplotlib.pyplot as plt
from deap import gp

from alpine.gp import regressor as gps
from alpine.gp import util
import numpy as np
import ray

import time

import warnings
import pygmo as pg

import re
from sklearn.metrics import r2_score
from datasets import generate_dataset

import mygrad as mg
from mygrad._utils.lock_management import mem_guard_off

from functools import partial

num_cpus = 1
num_runs = 1  # 20


def check_trig_fn(ind):
    return len(re.findall("cos", str(ind))) + len(re.findall("sin", str(ind)))


def check_nested_trig_fn(ind):
    return util.detect_nested_trigonometric_functions(str(ind))


def eval_model(individual, X, consts=[]):
    num_variables = X.shape[1]
    if num_variables > 1:
        X = [X[:, i] for i in range(num_variables)]
    else:
        X = [X]
    warnings.filterwarnings("ignore")
    y_pred = individual(*X, consts)
    return y_pred


def compute_MSE(individual, X, y, consts=[]):
    y_pred = eval_model(individual, X, consts)
    MSE = np.mean((y - y_pred) ** 2)

    if np.isnan(MSE) or np.isinf(MSE):
        MSE = 1e8

    return MSE


def eval_MSE_and_tune_constants(tree, toolbox, X, y):
    individual, num_consts = util.compile_individual_with_consts(tree, toolbox)

    if num_consts > 0:

        eval_MSE = partial(compute_MSE, individual=individual, X=X, y=y)

        x0 = np.ones(num_consts)

        class fitting_problem:
            def fitness(self, x):
                total_err = eval_MSE(consts=x)
                # return [total_err + 0.*(np.linalg.norm(x, 2))**2]
                return [total_err]

            def gradient(self, x):
                with mem_guard_off:
                    xt = mg.tensor(x, copy=False)
                    f = self.fitness(xt)[0]
                    f.backward()
                return xt.grad

            def get_bounds(self):
                return (-5.0 * np.ones(num_consts), 5.0 * np.ones(num_consts))

        # PYGMO SOLVER
        prb = pg.problem(fitting_problem())
        algo = pg.algorithm(pg.nlopt(solver="lbfgs"))
        # algo = pg.algorithm(pg.pso(gen=10))
        # pop = pg.population(prb, size=70)
        algo.extract(pg.nlopt).maxeval = 10
        pop = pg.population(prb, size=1)
        pop.push_back(x0)
        pop = algo.evolve(pop)
        MSE = pop.champion_f[0]
        consts = pop.champion_x

        if np.isinf(MSE) or np.isnan(MSE):
            MSE = 1e8
    else:
        MSE = compute_MSE(individual, X, y)
        consts = []
    return MSE, consts


def get_features_batch(
    individuals_batch,
    individ_feature_extractors=[len, check_nested_trig_fn, check_trig_fn],
):
    features_batch = [
        [fe(i) for i in individuals_batch] for fe in individ_feature_extractors
    ]

    individ_length = features_batch[0]
    nested_trigs = features_batch[1]
    num_trigs = features_batch[2]
    return individ_length, nested_trigs, num_trigs


def predict(individuals_batch, toolbox, X, penalty, fitness_scale):

    predictions = [None] * len(individuals_batch)

    for i, tree in enumerate(individuals_batch):
        callable, _ = util.compile_individual_with_consts(tree, toolbox)
        predictions[i] = eval_model(callable, X, consts=tree.consts)

    return predictions


def compute_MSEs(individuals_batch, toolbox, X, y, penalty, fitness_scale):

    total_errs = [None] * len(individuals_batch)

    for i, tree in enumerate(individuals_batch):
        callable, _ = util.compile_individual_with_consts(tree, toolbox)
        total_errs[i] = compute_MSE(callable, X, y, consts=tree.consts)

    return total_errs


def compute_attributes(individuals_batch, toolbox, X, y, penalty, fitness_scale):

    attributes = [None] * len(individuals_batch)

    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_batch)

    for i, tree in enumerate(individuals_batch):

        # Tarpeian selection
        if individ_length[i] >= 50:
            consts = None
            fitness = (1e8,)
        else:
            MSE, consts = eval_MSE_and_tune_constants(tree, toolbox, X, y)
            fitness = (
                fitness_scale
                * (
                    MSE
                    + 100000 * nested_trigs[i]
                    + 0.0 * num_trigs[i]
                    + penalty["reg_param"] * individ_length[i]
                ),
            )
        attributes[i] = {"consts": consts, "fitness": fitness}
    return attributes


def assign_attributes(individuals_batch, attributes):
    for ind, attr in zip(individuals_batch, attributes):
        ind.consts = attr["consts"]
        ind.fitness.values = attr["fitness"]


def eval(problem, cfgfile, seed=42):

    regressor_params, config_file_data = util.load_config_data(cfgfile)

    scaleXy = config_file_data["gp"]["scaleXy"]

    # generate training and test datasets
    (
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test,
        _,
        scaler_y,
        num_variables,
        _,
    ) = generate_dataset(problem, scaleXy=scaleXy, random_state=seed)

    if num_variables == 1:
        pset = gp.PrimitiveSetTyped("Main", [float], float)
        pset.renameArguments(ARG0="x")
    elif num_variables == 2:
        pset = gp.PrimitiveSetTyped("Main", [float, float], float)
        pset.renameArguments(ARG0="x")
        pset.renameArguments(ARG1="y")
    else:
        pset = gp.PrimitiveSetTyped("Main", [float] * num_variables, float)

    pset = util.add_primitives_to_pset_from_dict(
        pset, config_file_data["gp"]["primitives"]
    )

    batch_size = config_file_data["gp"]["batch_size"]
    if config_file_data["gp"]["use_constants"]:
        pset.addTerminal(object, float, "a")

    callback_func = assign_attributes
    fitness_scale = 1.0

    penalty = config_file_data["gp"]["penalty"]
    common_params = {"penalty": penalty, "fitness_scale": fitness_scale}

    gpsr = gps.GPSymbolicRegressor(
        pset_config=pset,
        fitness=compute_attributes,
        predict_func=predict,
        common_data=common_params,
        callback_func=callback_func,
        print_log=True,
        num_best_inds_str=1,
        save_best_individual=False,
        output_path="./",
        seed_str=None,
        batch_size=batch_size,
        num_cpus=num_cpus,
        **regressor_params,
    )

    tic = time.time()
    gpsr.fit(X_train_scaled, y_train_scaled)
    toc = time.time()

    best = gpsr.get_best_individual()
    if hasattr(best, "consts"):
        print("Best parameters = ", best.consts)

    print("Elapsed time = ", toc - tic)
    individuals_per_sec = (
        (gpsr.get_last_gen() + 1) * gpsr.NINDIVIDUALS * gpsr.num_islands / (toc - tic)
    )
    print("Individuals per sec = ", individuals_per_sec)

    u_best = gpsr.predict(X_test_scaled)

    # de-scale outputs before computing errors
    if scaleXy:
        u_best = scaler_y.inverse_transform(u_best.reshape(-1, 1)).flatten()

    MSE = np.mean((u_best - y_test) ** 2)
    r2_test = r2_score(y_test, u_best)
    print("MSE on the test set = ", MSE)
    print("R^2 on the test set = ", r2_test)

    pred_train = gpsr.predict(X_train_scaled)

    if scaleXy:
        pred_train = scaler_y.inverse_transform(pred_train.reshape(-1, 1)).flatten()
        y_train_scaled = scaler_y.inverse_transform(
            y_train_scaled.reshape(-1, 1)
        ).flatten()

    MSE = np.mean((pred_train - y_train_scaled) ** 2)
    r2_train = r2_score(y_train_scaled, pred_train)
    print("MSE on the training set = ", MSE)
    print("R^2 on the training set = ", r2_train)
    # Nguyen
    # if MSE <= 1e-10 or (problem == "Nguyen-13" and MSE <= 1e-5):
    #     return 1.0
    # else:
    #     return 0.0

    return r2_train, r2_test


if __name__ == "__main__":
    import argparse
    import pathlib

    # problems = [
    #     "Nguyen-1",
    #     "Nguyen-2",
    #     "Nguyen-3",
    #     "Nguyen-4",
    #     "Nguyen-5",
    #     "Nguyen-6",
    #     "Nguyen-7",
    #     "Nguyen-8",
    #     "Nguyen-9",
    #     "Nguyen-10",
    #     "Nguyen-11",
    #     "Nguyen-12",
    #     "Nguyen-13",
    # ]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "cfgfile",
        type=pathlib.Path,
        help="Path of the YAML config file for the problem.",
    )
    parser.add_argument("problem", help="Name of the PMLB or Nguyen dataset.")

    args = parser.parse_args()

    problem = args.problem
    cfgfile = args.cfgfile

    # problem = "1089_USCrime"

    seeds = [29802, 22118, 860, 15795, 21575, 5390, 11964, 6265, 23654, 11284]

    r2_tests = []

    # possibly add rmse_train, rmse_test, rmse_val
    header = ["problem", "trial", "r2_train", "r2_test", "seed"]

    with open(f"./results/{problem}.csv", "w") as f:
        for h in header:
            f.write(h)
            f.write("\n" if h == header[-1] else ";")

    for i, seed in enumerate(seeds):
        print("PROBLEM: ", problem)
        r2_train, r2_test = eval(problem=problem, cfgfile=cfgfile, seed=seed)
        r2_tests.append(r2_test)

        stats = {
            "problem": problem,
            "trial": i + 1,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "seed": seed,
        }

        with open(f"./results/{problem}.csv", "a") as f:
            for h in header:
                f.write(f"{stats[h]}")
                f.write("\n" if h == header[-1] else ";")

    print("Median Test R^2 = ", np.median(r2_tests))

    ray.shutdown()
    # Nguyen
    # ave_success_rate = 0.0
    # with open("bench_stats.txt", "w") as file:
    #     for problem in problems:
    #         success = 0.0
    #         for i in range(num_runs):
    #             print("Problem {prb}, RUN #{num}".format(prb=problem, num=i))
    #             success += bench(problem=problem)
    #         success_rate = success / num_runs * 100
    #         ave_success_rate += success_rate / len(problems)
    #         str_to_print = problem + " " + str(success_rate)
    #         print(str_to_print, file=file, flush=True)
    #     print("Average success rate = ", ave_success_rate)
