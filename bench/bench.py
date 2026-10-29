# import matplotlib.pyplot as plt
from deap import gp

from flex.gp import regressor as gps
from flex.gp import numba_eval
from flex.gp.util import load_config_data
from flex.gp.primitives import add_primitives_to_pset_from_dict
import numpy as np
import ray

import math
import time

import warnings

from sklearn.metrics import r2_score
from datasets import generate_dataset

import optuna
import sympy as sp

# silence numpy warnings (overflow, invalid values) of the evaluated models once,
# instead of at each model evaluation
warnings.filterwarnings("ignore")


def trig_features(ind):
    """Returns 1 if a sin/cos node (including arcsin/arccos) has another one in its
    subtree (0 otherwise), and the number of sin/cos nodes."""
    is_trig = [
        isinstance(node, gp.Primitive) and ("sin" in node.name or "cos" in node.name)
        for node in ind
    ]
    nested = any(
        is_trig[i] and any(is_trig[i + 1 : ind.searchSubtree(i).stop])
        for i in range(len(ind))
    )
    return int(nested), sum(is_trig)


def sanitize_array_for_metrics(arr, name, clip_value=1e150):
    """Return a finite float64 array suitable for metric computation."""
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=clip_value, neginf=-clip_value)
    arr = np.clip(arr, -clip_value, clip_value)
    return arr


def safe_r2_score(y_true, y_pred):
    """Robust R^2 computation that avoids crashing on bad numeric values."""
    try:
        return r2_score(y_true, y_pred)
    except ValueError as exc:
        print(f"Warning: r2_score failed ({exc}). Returning NaN.")
        return np.nan


def compute_fitness_value(MSE, model_size, penalty):
    # Evolution fitness: only MSE + linear size regularization.
    return MSE + penalty["reg_param"] * model_size


def compute_MSE_numba(tree, pset, X, y, consts):
    """MSE of the tree with the given constants, evaluated by the numba interpreter;
    1e8 if it is not finite."""
    MSE = numba_eval.mse(tree, pset, X, y, consts)
    return MSE if math.isfinite(MSE) else 1e8


def eval_MSE_and_fit_constants_numba(tree, toolbox, X, y):
    """MSE of the tree with its constants fitted by Levenberg-Marquardt (at most 8
    iterations, constants in [-5, 5] starting from 1), with the tree evaluated by the
    numba interpreter."""
    MSE, consts = numba_eval.fit_mse(tree, toolbox.compile.keywords["pset"], X, y)
    if not math.isfinite(MSE):
        MSE = 1e8
    return MSE, consts


def get_features_batch(individuals_batch):
    individ_length = [len(ind) for ind in individuals_batch]
    nested_trigs, num_trigs = zip(*[trig_features(ind) for ind in individuals_batch])
    return individ_length, nested_trigs, num_trigs


def predict(
    individuals_batch,
    toolbox,
    X,
    penalty,
    fitness_scale,
    tarpeian_length_threshold,
):

    pset = toolbox.compile.keywords["pset"]
    return [numba_eval.predict_tree(tree, pset, X, tree.consts)
            for tree in individuals_batch]


def compute_MSEs(
    individuals_batch,
    toolbox,
    X,
    y,
    penalty,
    fitness_scale,
    tarpeian_length_threshold,
):

    pset = toolbox.compile.keywords["pset"]
    return [compute_MSE_numba(tree, pset, X, y, tree.consts)
            for tree in individuals_batch]


def compute_attributes(
    individuals_batch,
    toolbox,
    X,
    y,
    penalty,
    fitness_scale,
    tarpeian_length_threshold,
):

    attributes = [None] * len(individuals_batch)

    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_batch)

    for i, tree in enumerate(individuals_batch):

        # Tarpeian selection
        if individ_length[i] >= tarpeian_length_threshold:
            consts = None
            fitness = (1e8,)
        else:
            MSE, consts = eval_MSE_and_fit_constants_numba(tree, toolbox, X, y)
            if MSE < 1e-12:
                # Keep fitness equal to pure error so early stopping can trigger.
                fitness = (fitness_scale * MSE,)
            else:
                num_consts = len(consts)
                model_size = individ_length[i] + num_consts
                age_penalty = penalty.get("age_reg_param", 0.0) * getattr(
                    tree, "age", 0
                )
                fitness_value = compute_fitness_value(
                    MSE=MSE,
                    model_size=model_size,
                    penalty=penalty,
                )
                fitness = (
                    fitness_scale
                    * (
                        fitness_value
                        + age_penalty
                        + 100000 * nested_trigs[i]
                        + 0.0 * num_trigs[i]
                    ),
                )
        attributes[i] = {
            "consts": consts,
            "fitness": fitness,
        }
    return attributes


def assign_attributes(individuals_batch, attributes):
    for ind, attr in zip(individuals_batch, attributes):
        ind.consts = attr["consts"]
        ind.fitness.values = attr["fitness"]


def eval(problem, cfgfile, seed=42, grid_search=False, coarse_grained_islands=False,
         remove_init_duplicates=True, custom_logger=None, multiprocessing=True,
         early_stop_fitness_threshold=1e-12):

    regressor_params, config_file_data = load_config_data(cfgfile)

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

    pset = add_primitives_to_pset_from_dict(pset, config_file_data["gp"]["primitives"])

    batch_size = config_file_data["gp"]["batch_size"]
    if config_file_data["gp"]["use_constants"]:
        pset.addTerminal(object, float, "c")

    callback_func = assign_attributes
    fitness_scale = 1.0

    penalty = config_file_data["gp"]["penalty"]
    tarpeian_length_threshold = config_file_data["gp"].get(
        "tarpeian_length_threshold", 40
    )
    common_params = {
        "penalty": penalty,
        "fitness_scale": fitness_scale,
        "tarpeian_length_threshold": tarpeian_length_threshold,
    }

    gpsr = gps.GPSymbolicRegressor(
        pset_config=pset,
        fitness=compute_attributes,
        predict_func=predict,
        score_func=compute_MSEs,
        common_data=common_params,
        callback_func=callback_func,
        custom_logger=custom_logger,
        print_log=True,
        num_best_inds_str=1,
        save_best_individual=False,
        output_path="./",
        seed_str=None,
        batch_size=batch_size,
        remove_init_duplicates=remove_init_duplicates,
        save_detailed_log=False,
        early_stop_fitness_threshold=early_stop_fitness_threshold,
        coarse_grained_islands=coarse_grained_islands,
        multiprocessing=multiprocessing,
        **regressor_params,
    )

    if grid_search:
        study = optuna.create_study(study_name=problem, direction="maximize")

        param = {
            "num_individuals": optuna.distributions.CategoricalDistribution([100, 200])
        }

        # wrap regressor inside an optuna CV search object
        best_estimator = optuna.integration.OptunaSearchCV(
            gpsr,
            param,
            cv=5,
            study=study,
            refit=True,
            n_trials=2,
            verbose=2,
            timeout=3600,
        )
    else:
        best_estimator = gpsr

    tic = time.time()
    best_estimator.fit(X_train_scaled, y_train_scaled)
    toc = time.time()

    if grid_search:
        best_estimator = best_estimator.best_estimator_
    else:
        best_estimator = best_estimator

    best_model = best_estimator.get_best_individuals(n_ind=1)[0]

    try:
        best_model_sympy = sp.simplify(best_estimator.get_best_individuals_sympy()[0])
        best_model_str = str(best_model_sympy)
    except Exception as exc:
        print(
            "Warning: failed to convert best model to SymPy "
            f"({type(exc).__name__}: {exc}). "
            "Using raw GP string instead."
        )
        best_model_str = str(best_model)
    print("Best selected model = ", best_model_str)

    if hasattr(best_model, "consts"):
        print("Best parameters = ", best_model.consts)

    print("Elapsed time = ", toc - tic)
    individuals_per_sec = (
        (best_estimator.get_last_gen() + 1)
        * gpsr.num_individuals
        * gpsr.num_islands
        / (toc - tic)
    )
    print("Individuals per sec = ", individuals_per_sec)

    u_best = best_estimator.predict(X_test_scaled)

    # de-scale outputs before computing errors
    if scaleXy:
        u_best = scaler_y.inverse_transform(u_best.reshape(-1, 1)).flatten()

    y_test_safe = sanitize_array_for_metrics(y_test, "y_test")
    u_best_safe = sanitize_array_for_metrics(u_best, "u_best")
    MSE = np.mean((u_best_safe - y_test_safe) ** 2)
    r2_test = safe_r2_score(y_test_safe, u_best_safe)
    print("MSE on the test set = ", MSE)
    print("R^2 on the test set = ", r2_test)

    pred_train = best_estimator.predict(X_train_scaled)

    if scaleXy:
        pred_train = scaler_y.inverse_transform(pred_train.reshape(-1, 1)).flatten()
        y_train_scaled = scaler_y.inverse_transform(
            y_train_scaled.reshape(-1, 1)
        ).flatten()

    y_train_safe = sanitize_array_for_metrics(y_train_scaled, "y_train")
    pred_train_safe = sanitize_array_for_metrics(pred_train, "pred_train")
    MSE = np.mean((pred_train_safe - y_train_safe) ** 2)
    r2_train = safe_r2_score(y_train_safe, pred_train_safe)
    print("MSE on the training set = ", MSE)
    print("R^2 on the training set = ", r2_train)

    return r2_train, r2_test, best_model_str, toc - tic


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "cfgfile",
        type=pathlib.Path,
        help="Path of the YAML config file for the problem.",
    )
    parser.add_argument("problem", help="Name of the PMLB or Nguyen dataset.")

    parser.add_argument(
        "-gs",
        action="store_true",
        help="Perform grid search for hyperparameter tuning.",
    )

    args = parser.parse_args()

    problem = args.problem
    cfgfile = args.cfgfile

    seeds = [29802, 22118, 860, 15795, 21575, 5390, 11964, 6265, 23654, 11284]

    r2_tests = []

    header = ["problem", "trial", "r2_train", "r2_test", "seed", "best_model"]

    with open(f"./results/{problem}.csv", "w") as f:
        for h in header:
            f.write(h)
            f.write("\n" if h == header[-1] else ";")

    for i, seed in enumerate(seeds):
        print("PROBLEM: ", problem)
        print("seed: ", seed)
        r2_train, r2_test, best_model_str, _ = eval(
            problem=problem, cfgfile=cfgfile, seed=seed, grid_search=args.gs
        )
        r2_tests.append(r2_test)

        stats = {
            "problem": problem,
            "trial": i + 1,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "seed": seed,
            "best_model": best_model_str,
        }

        with open(f"./results/{problem}.csv", "a") as f:
            for h in header:
                f.write(f"{stats[h]}")
                f.write("\n" if h == header[-1] else ";")

    print("Median Test R^2 = ", np.median(r2_tests))

    ray.shutdown()
