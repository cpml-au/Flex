from flex.gp.util import (
    get_features_batch,
    load_config_data,
)
from flex.gp.numba_eval import assign_attributes, fit_mse, predict_batch
from flex.gp import regressor as gps
from flex.gp.primitives import add_primitives_to_pset_from_dict
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pmlb import fetch_data
from sklearn.metrics import r2_score
from deap import gp
from sklearn.model_selection import RepeatedKFold, cross_val_score
import optuna
from optuna.samplers import GridSampler

# Optuna tutorial. We use the same example of simple_sr.ipynb

# set up number of cpus per ray worker
num_cpus = 1


# --- Custom generate dataset function ---
def generate_dataset(
    problem: str = "1027_ESL", random_state: int = 42, scaleXy: bool = True
):
    np.random.seed(42)
    num_variables = 1
    scaler_X = None
    scaler_y = None

    # PMLB datasets
    X, y = fetch_data(problem, return_X_y=True, local_cache_dir="./datasets")

    num_variables = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    if scaleXy:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)
    else:
        X_train_scaled = X_train
        y_train_scaled = y_train
        X_test_scaled = X_test

    y_test = y_test.flatten()
    y_train_scaled = y_train_scaled.flatten()

    num_train_points = X_train.shape[0]

    # note y_test and y_train_scaled must be flattened
    return (
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test,
        scaler_X,
        scaler_y,
        num_variables,
        num_train_points,
    )


def eval_MSE_and_tune_constants(tree, toolbox, X, y):
    # fit the constants with Levenberg-Marquardt, evaluating the tree with the numba
    # interpreter
    MSE, consts = fit_mse(tree, toolbox.compile.keywords["pset"], X, y)
    if np.isinf(MSE) or np.isnan(MSE):
        MSE = 1e8
    return MSE, consts


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


def score(individuals_batch, toolbox, X, y, penalty, fitness_scale):
    predictions = predict_batch(individuals_batch, toolbox, X)
    scores = [None] * len(individuals_batch)
    for i, prediction in enumerate(predictions):
        scores[i] = r2_score(y, prediction)
    return scores


# Custom optimization routine for Optuna.
# Although OptunaSearchCV could be used for this example,
# in practice one often needs a custom objective function.
# This shows how to define one and integrate it with Flex.
def optimize(trial, X, y, grid_search_parameters, cfgfile):
    num_variables = X.shape[1]
    params = {}
    for key in grid_search_parameters.keys():
        params[key] = trial.suggest_categorical(key, grid_search_parameters[key])

    # Create model
    regressor_params, config = load_config_data(cfgfile)
    regressor_params["num_individuals"] = params["num_individuals"]
    regressor_params["num_islands"] = params["num_islands"]

    batch_size = config["gp"]["batch_size"]
    penalty = config["gp"]["penalty"]
    fitness_scale = 1.0

    common_params = {"penalty": penalty, "fitness_scale": fitness_scale}

    initial_individuals = config["gp"].get("initial_individuals", None)
    if not initial_individuals:
        initial_individuals = None

    pset = gp.PrimitiveSetTyped("Main", [float] * num_variables, float)
    pset = add_primitives_to_pset_from_dict(pset, config["gp"]["primitives"])
    if config["gp"]["use_constants"]:
        pset.addTerminal(object, float, "c")

    gpsr = gps.GPSymbolicRegressor(
        pset_config=pset,
        fitness=compute_attributes,
        predict_func=predict_batch,
        score_func=score,
        common_data=common_params,
        callback_func=assign_attributes,
        num_best_inds_str=1,
        save_best_individual=True,
        save_train_fit_history=True,
        seed_str=initial_individuals,
        batch_size=batch_size,
        num_cpus=num_cpus,
        print_log=True,
        **regressor_params,
    )

    # Define Repeated K-Fold cross-validation
    rkf = RepeatedKFold(
        n_splits=5,  # number of folds
        n_repeats=1,  # number of repetitions
        random_state=42,
    )

    # Perform cross-validation
    scores = cross_val_score(gpsr, X, y, cv=rkf, scoring=None, n_jobs=1)

    mean_score = np.mean(scores)
    return mean_score


if __name__ == "__main__":
    grid_search_parameters = {
        "num_individuals": [250],
        "num_islands": [1],
    }

    # Define a GridSampler
    sampler = GridSampler(grid_search_parameters)
    regressor_params, config_file_data = load_config_data("simple_sr.yaml")

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
    ) = generate_dataset("1096_FacultySalaries", scaleXy=scaleXy, random_state=29802)

    # Create the study
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Run the grid search
    study.optimize(
        partial(
            optimize,
            X=X_train_scaled,
            y=y_train_scaled,
            grid_search_parameters=grid_search_parameters,
            cfgfile="simple_sr.yaml",
        )
    )

    trial = study.best_trial

    print("Accuracy: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
