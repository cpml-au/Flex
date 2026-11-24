[![Documentation Status](https://readthedocs.org/projects/flexgp/badge/?version=latest)](https://flexgp.readthedocs.io/en/latest/?badge=latest)

# Flex

_Flex_ is a Python library for **symbolic regression** via _Genetic Programming_.
It provides a high-level interface to the [`DEAP`](https://github.com/cpml-au/deap)
library, including distributed computing functionalities.

Besides solving classical symbolic regression problems involving _algebraic equations_
(see, for example, the benchmark problems contained in the
[SRBench](https://github.com/cavalab/srbench) repository), _Flex_ is specifically
designed to help identifying _symbolic_ models of _physical systems_ governed by **field equations**.
To this aim, it allows to exploit the **discrete calculus** framework defined and implemented in the library
[`dctkit`](https://github.com/cpml-au/dctkit) as a natural and effective language to express physical models
(i.e., conservation laws).

_Flex_ has been introduced in the paper [_Discovering interpretable physical models
with symbolic regression and discrete exterior calculus_](https://iopscience.iop.org/article/10.1088/2632-2153/ad1af2),
along with several benchmark problems (heat transfer, linear elasticity, Euler's
elastica). Scripts to reproduce these benchmarks can be found [here](https://github.com/cpml-au/SR-DEC_Examples).

**Features**:

- distributed computing provided by the [`ray`](https://www.ray.io) library;
- scikit-learn compatible interface;
- hyperparameter configuration via YAML files and optimization via `optuna`;
- support for custom operators (with/without strong-typing);
- benchmark suite (Nguyen and SRBench).

## Installation

Dependencies should be installed within a `conda` environment. We recommend using
[`mamba`](https://github.com/mamba-org/mamba) since it is much faster than `conda` at
solving the environment and downloading the dependencies. To create a suitable
environment based on the provided `.yaml` file, use the command

```bash
mamba env create -f environment.yaml
```

Otherwise, you can update an existing environment using the same `.yaml` file.

After activating the environment, clone the git repository and launch the following command

```bash
pip install -e .
```

to install a development version of the library.

Running the tests:

```bash
tox
```

Generating the docs:

```bash
tox -e docs
```

## Benchmarks

To run the benchmarks on the Nguyen and PMLB datasets, cd into the `bench` folder and
execute:

```bash
bash bench_Nguyen.sh
bash bench_PMLB.sh
```

Each line of these scripts runs `python bench.py <config file> <dataset>` (for example,
`python bench.py PMLB.yaml 1028_SWD`), which solves the problem with 10 random seeds
and writes the training and test $R^2$ of each run to `results/<dataset>.csv`. Then,
process the results using the `results/process_results.ipynb` notebook.

## Usage

Setting up a symbolic regression problem in _Flex_ involves two steps:

1. Define the functions that return the **prediction** and the **fitness**
   associated to an individual. These functions **must** have at least the following
   arguments in the first three positions:
   - the list of trees to be evaluated by the current worker;
   - the `toolbox` object used to compile the individual trees into callable functions;
   - the dataset features needed for the evaluation of the individuals. The name of the
     argument **must** be `X`.

Additionally, the fourth argument of the **fitness** function **must** be the dataset
labels, called `y`. For unsupervised problems, `None` must be passed as a `y` argument
to the fit method of the regressor, while the `y` argument of the fitness function must
be dropped. Any additional arguments can be set using the `common_data` argument of the
`GPSymbolicRegressor` object (see below). The entries of `common_data` are passed as
keyword arguments to the fitness, predict and score functions, so each of them must
accept all of them.

```python
def predict(individuals_batch, toolbox, X, penalty):

    predictions = [None] * len(individuals_batch)

    for i, tree in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(tree, toolbox)
        # predictions[i] = ...
    # ...
    return predictions

def fitness(individuals_batch, toolbox, X, y, penalty):

    fitnesses = [None]*len(individuals_batch)

    for i, tree in enumerate(individuals_batch):

        callable, _ = compile_individual_with_consts(tree, toolbox)
        # MSE = ...

        # each fitness MUST be a tuple (required by DEAP), unless a `callback_func`
        # stores it on the individuals
        fitnesses[i] = (MSE,)

    return fitnesses
```

_Optionally_, you can define a `score` function with the same arguments as those of the
fitness function, to compute the score (e.g. $R^2$) of the best model found during the
SR run on a given dataset. When a validation dataset is used (`validate=True`), the
score is also computed on it at each generation and the maximum is reported as the best
validation score, so a higher score must mean a better model (e.g. $R^2$, or the
negative of an error).

Inside these functions, the trees can be evaluated in two ways:

- by compiling them into Python functions with `compile_individuals` or
  `compile_individual_with_consts` of `flex.gp.util`, as in the example above. This
  works with any primitives;
- with the numba-compiled interpreter of `flex.gp.numba_eval`, which evaluates a tree
  directly, without compiling it into a Python function, and is the fastest option. It
  supports the trees built from the primitives of `numpy_primitives`. Its functions take
  the primitive set `pset` of the tree, whose i-th argument reads the i-th column of `X`:
  - `fit_mse(tree, pset, X, y, bounds=(-5.0, 5.0), max_iter=8)` fits the constants of
    the tree (the terminals named `c`), keeping them within `bounds`, and returns the
    mean squared error (MSE) and the fitted constants. The fit minimizes the MSE with
    the Levenberg-Marquardt method, using the exact derivatives of the tree with respect
    to its constants, and is described in the
    [documentation](https://flexgp.readthedocs.io/en/latest/constant_fitting.html);
  - `mse(tree, pset, X, y, consts)` returns the MSE of the tree with the given
    constants;
  - `predict_tree(tree, pset, X, consts)` returns the values of the tree with the given
    constants.

  Numerical terminals (fixed numbers added with `pset.addTerminal`, with or without a
  name, or ephemeral constants) are evaluated with their values and are not fitted.

The module `flex.gp.numba_eval` also provides predefined batch functions, which can be
passed to `GPSymbolicRegressor` in place of your own functions (the corresponding
argument of `GPSymbolicRegressor` is written in parentheses):

- `fitness_batch` (`fitness`): the fitness of an individual is its MSE after fitting its
  constants with `fit_mse`, plus `parsimony` times its size (number of nodes plus
  number of constants); it is 1e8 if the MSE is not finite. The options `bounds`,
  `max_iter` and `parsimony` (default 0) are set through `common_data`;
- `assign_attributes` (`callback_func`): stores the fitness and the fitted constants on
  each individual (`ind.consts`);
- `score_batch` (`score_func`) and `predict_batch` (`predict_func`): MSE and
  predictions of an individual with its stored constants. Since lower MSE means a
  better model, `score_batch` is suited to the `score` method but not to validation
  (`validate=True`), which requires a score where higher is better.

2. Set up and solve the symbolic regression problem. The configuration of the
   `GPSymbolicRegressor` object can be specified via the arguments of its constructor
   (see the API docs), or loaded from a YAML file.

```python
yamlfile = "simple_sr.yaml"
filename = os.path.join(os.path.dirname(__file__), yamlfile)

regressor_params, config_file_data = load_config_data(filename)

# pset = gp.PrimitiveSetTyped(...)
# pset.renameArguments(ARG0="x")

pset = add_primitives_to_pset_from_dict(
    pset, config_file_data["gp"]["primitives"]
)

# define extra common arguments of fitness, predict and score functions
penalty = config_file_data["gp"]["penalty"]
common_data = {"penalty": penalty}

# load datasets...

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

# run the symbolic regression
gpsr.fit(X_train, y_train)

# compute the prediction on the test dataset given by the best model found during the SR
pred_test = gpsr.predict(X_test)

ray.shutdown()
```

Complete example scripts can be found in the `examples` directory: `simple_sr.py`
defines its own functions, which evaluate the trees with the numba functions `mse` and
`predict_tree`, while `simple_sr_numba.py` uses the predefined batch functions.

## Citing

```
@article{Manti_2024,
    doi = {10.1088/2632-2153/ad1af2},
    url = {https://dx.doi.org/10.1088/2632-2153/ad1af2},
    year = {2024},
    publisher = {IOP Publishing},
    volume = {5},
    number = {1},
    pages = {015005},
    author = {Simone Manti and Alessandro Lucantonio},
    title = {Discovering interpretable physical models using symbolic regression and discrete exterior calculus},
    journal = {Machine Learning: Science and Technology}
}
```

## Repositories using Flex

The following projects leverage Flex for their evolution pipelines. If you have a project you would like to showcase, please submit a pull request with your repository details to be included in this list.

- [SR-DEC_Examples](https://github.com/cpml-au/SR-DEC_Examples): benchmarks for symbolic regression paired with discrete exterior calculus.
- [SR-Traffic](https://github.com/cpml-au/SR-Traffic): symbolic regression of fundamental diagrams of first-order traffic flow models.
- [SR-CLF](https://github.com/cpml-au/SR-CLF): symbolic regression of Control Lyapunov Functions.

## Acknowledgements

This work is supported by the European Union (European Research Council (ERC), ALPS, 101039481). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the ERC Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
