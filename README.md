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

The fastest way to solve a symbolic regression problem with _Flex_ is to use the
ready-made functions of the module `flex.gp.numba_eval`, which evaluate the trees with a
numba-compiled interpreter. These functions support the trees built from the primitives
of `numpy_primitives`. For other primitives, or to define a different fitness, write
your own functions as described in the section _Custom fitness, prediction and score
functions_ below.

### Symbolic regression with the numba functions

The numba-compiled interpreter evaluates a tree directly, without compiling it into a
Python function. It also fits the constants of the tree (the terminals named `c`) by
minimizing the mean squared error (MSE) with the Levenberg-Marquardt method, using the
exact derivatives of the tree with respect to its constants, which are computed in
forward mode in the module `flex.gp.numba_grad`. Numerical terminals (fixed numbers
added with `pset.addTerminal`, with or without a name, or ephemeral constants) are
evaluated with their values and are not fitted.

The module provides four functions, each passed to the `GPSymbolicRegressor` argument
written in parentheses:

- `fitness_batch` (`fitness`): the fitness of an individual is its MSE after fitting its
  constants, plus `parsimony` times its size (number of nodes plus number of constants);
  it is 1e8 if the MSE is not finite. The fit keeps the constants within `bounds` and
  performs at most `max_iter` iterations. It starts from random values, uniformly
  distributed within `bounds`, whose generator is seeded with a hash of the tree, so a
  given tree always starts from the same values, in serial and in parallel runs. If a
  derivative is not finite at some data point, the fit stops and returns the constants
  reached so far;
- `assign_attributes` (`callback_func`): stores the fitness and the fitted constants on
  each individual (`ind.consts`);
- `score_batch` (`score_func`) and `predict_batch` (`predict_func`): MSE and
  predictions of an individual with its stored constants. Since lower MSE means a
  better model, `score_batch` is suited to the `score` method but not to validation
  (`validate=True`), which requires a score where higher is better.

The options `bounds` (default `(-5.0, 5.0)`), `max_iter` (default 8) and `parsimony`
(default 0) of `fitness_batch` are set through the `common_data` argument of
`GPSymbolicRegressor`:

```python
from deap import gp
from flex.gp.numba_eval import (
    assign_attributes, fitness_batch, predict_batch, score_batch
)
from flex.gp.primitives import add_primitives_to_pset_from_dict
from flex.gp.regressor import GPSymbolicRegressor
import ray

# primitive set with one input variable, x
pset = gp.PrimitiveSetTyped("MAIN", [float], float)
pset.renameArguments(ARG0="x")
primitives = {
    "imports": {"flex.gp.numpy_primitives": ["numpy_primitives"]},
    "used": [
        {"name": "add", "dimension": None, "rank": None},
        {"name": "mul", "dimension": None, "rank": None},
        {"name": "sin", "dimension": None, "rank": None},
    ],
}
pset = add_primitives_to_pset_from_dict(pset, primitives)
# terminal whose values are fitted by fitness_batch
pset.addTerminal(object, float, "c")

# load datasets...

gpsr = GPSymbolicRegressor(
    pset_config=pset,
    fitness=fitness_batch,
    score_func=score_batch,
    predict_func=predict_batch,
    callback_func=assign_attributes,
    common_data={"parsimony": 1e-4, "max_iter": 8},
    num_individuals=200,
    num_islands=4,
    generations=50,
    batch_size=50,
)

# run the symbolic regression
gpsr.fit(X_train, y_train)

# predictions and MSE of the best model found during the run on the test dataset
pred_test = gpsr.predict(X_test)
mse_test = gpsr.score(X_test, y_test)

ray.shutdown()
```

The complete script is `examples/simple_sr_numba.py`. The configuration of
`GPSymbolicRegressor` can be specified via the arguments of its constructor (see the API
docs), or loaded from a YAML file:

```python
regressor_params, config_file_data = load_config_data("simple_sr.yaml")
pset = add_primitives_to_pset_from_dict(pset, config_file_data["gp"]["primitives"])
gpsr = GPSymbolicRegressor(pset_config=pset, ..., **regressor_params)
```

The derivatives of the trees at points where the differentiation rules are not defined,
such as `sqrt(mul(c, x))` at `x = 0`, follow conventions that are described in the
[documentation](https://flexgp.readthedocs.io/en/latest/constant_fitting.html), together
with their effect on the fit.

### Custom fitness, prediction and score functions

The functions that return the **prediction** and the **fitness** of the individuals
**must** have at least the following arguments in the first three positions:

- the list of trees to be evaluated by the current worker;
- the `toolbox` object used to compile the individual trees into callable functions;
- the dataset features needed for the evaluation of the individuals. The name of the
  argument **must** be `X`.

Additionally, the fourth argument of the **fitness** function **must** be the dataset
labels, called `y`. For unsupervised problems, `None` must be passed as a `y` argument
to the fit method of the regressor, while the `y` argument of the fitness function must
be dropped. Any additional arguments can be set using the `common_data` argument of the
`GPSymbolicRegressor` object. The entries of `common_data` are passed as keyword
arguments to the fitness, predict and score functions, so each of them must accept all
of them.

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

The functions are then passed to `GPSymbolicRegressor`:

```python
gpsr = GPSymbolicRegressor(
    pset_config=pset,
    fitness=fitness,
    score_func=score,
    predict_func=predict,
    common_data={"penalty": penalty},
    # ...
)
```

Inside these functions, the trees can be evaluated in two ways:

- with the per-tree functions of `flex.gp.numba_eval` (`pset` is the primitive set; its
  i-th argument reads the i-th column of `X`):
  - `fit_mse(tree, pset, X, y, bounds=(-5.0, 5.0), max_iter=8)` fits the constants as
    `fitness_batch` does, and returns the MSE and the fitted constants;
  - `mse(tree, pset, X, y, consts)` returns the MSE of the tree with the given
    constants;
  - `predict_tree(tree, pset, X, consts)` returns the values of the tree with the given
    constants.

  The script `examples/simple_sr.py` uses `mse` and `predict_tree` in this way;
- by compiling the trees into Python functions with `compile_individuals` or
  `compile_individual_with_consts` of `flex.gp.util`, as in the example above, which
  works with any primitives.

### Parallelization strategies

The two strategies for running the islands in parallel (`coarse_grained_islands`) are
described in the [documentation](https://flexgp.readthedocs.io/en/latest/parallelization.html).

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
