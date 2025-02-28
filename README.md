[![Documentation Status](https://readthedocs.org/projects/alpine/badge/?version=latest)](https://alpine.readthedocs.io/en/latest/?badge=latest)

# AlpineGP

_AlpineGP_ is a Python library for **symbolic regression** via _Genetic Programming_.
It provides a high-level interface to the [`DEAP`](https://github.com/alucantonio/DEAP)
library, including distributed computing functionalities.

Besides solving classical symbolic regression problems involving algebraic equations
(see, for example, the benchmark problems contained in the
[SRBench](https://github.com/cavalab/srbench) repository), _AlpineGP_ is specifically
design to help identifying _symbolic_ models of _physical systems_ governed by **field equations**.
To this aim, it allows to exploit the **discrete calculus** framework defined and implemented in the library
[`dctkit`](https://github.com/alucantonio/dctkit) as a natural and effective language to express physical models
(i.e., conservation laws).

_AlpineGP_ has been introduced in the paper [_Discovering interpretable physical models
with symbolic regression and discrete exterior calculus_](https://iopscience.iop.org/article/10.1088/2632-2153/ad1af2),
along with several benchmark problems (heat transfer, linear elasticity, Euler's
elastica). Scripts to reproduce these benchmarks can be found [here](https://github.com/alucantonio/MLST2024_SRDEC).

**Features**:
- distributed computing provided by the [`ray`](https://www.ray.io) library;
- scikit-learn compatible interface;
- hyperparameter configuration via YAML files;
- support for custom operators (with/without strong-typing);
- benchmark suite (Nguyen and interface to SRBench) 

## Benchmarks

To run the benchmarks, cd into the `bench` folder and execute:
```bash
$ ./bench.sh
```
Then process the results using the `process_results` notebook.

Results on [PMLB](https://epistasislab.github.io/pmlb/) datasets (average $R^2$ over 10 test sets):

| dataset                       |       mean |     median |        std |
|:------------------------------|-----------:|-----------:|-----------:|
| 527_analcatdata_election2000  |  0.997727  |  0.999273  | 0.00357541 |
| 663_rabe_266                  |  0.994945  |  0.995115  | 0.00134602 |
| 560_bodyfat                   |  0.988467  |  0.992938  | 0.0121634  |
| 505_tecator                   |  0.986861  |  0.986026  | 0.0039009  |
| 561_cpu                       |  0.957349  |  0.967161  | 0.0330056  |
| 690_visualizing_galaxy        |  0.963404  |  0.964137  | 0.00867664 |
| 197_cpu_act                   |  0.94309   |  0.945666  | 0.00966613 |
| 227_cpu_small                 |  0.946096  |  0.945094  | 0.00812824 |
| 523_analcatdata_neavote       |  0.936577  |  0.943564  | 0.0278365  |
| 1096_FacultySalaries          |  0.662191  |  0.894004  | 0.525012   |
| 557_analcatdata_apnea1        |  0.881416  |  0.889496  | 0.0397044  |
| 230_machine_cpu               |  0.778943  |  0.879675  | 0.273846   |
| 556_analcatdata_apnea2        |  0.863157  |  0.867148  | 0.0347729  |
| 1027_ESL                      |  0.858838  |  0.860647  | 0.0127587  |
| 695_chatfield_4               |  0.827457  |  0.830825  | 0.0677194  |
| 229_pwLinear                  |  0.810944  |  0.811717  | 0.0453826  |
| 210_cloud                     |  0.761678  |  0.786611  | 0.159399   |
| 529_pollen                    |  0.787219  |  0.782358  | 0.0118861  |
| 1089_USCrime                  |  0.739218  |  0.756442  | 0.117112   |
| 503_wind                      |  0.747271  |  0.745787  | 0.0088297  |
| 712_chscase_geyser1           |  0.751443  |  0.745605  | 0.0549794  |
| 519_vinnie                    |  0.728873  |  0.719948  | 0.0377254  |
| 228_elusage                   |  0.621403  |  0.714127  | 0.216677   |
| 659_sleuth_ex1714             |  0.562146  |  0.702428  | 0.309503   |
| 666_rmftsa_ladata             |  0.679718  |  0.672306  | 0.0620477  |
| 225_puma8NH                   |  0.66854   |  0.667771  | 0.0127414  |
| 706_sleuth_case1202           |  0.418764  |  0.568134  | 0.43742    |
| 1029_LEV                      |  0.557169  |  0.560547  | 0.0330229  |
| 547_no2                       |  0.50562   |  0.502983  | 0.0920748  |
| 485_analcatdata_vehicle       |  0.244083  |  0.47083   | 0.702171   |
| 192_vineyard                  |  0.381856  |  0.38018   | 0.200867   |
| 1030_ERA                      |  0.373955  |  0.373216  | 0.0453621  |
| 1028_SWD                      |  0.335559  |  0.343532  | 0.0556771  |
| 542_pollution                 |  0.170091  |  0.279329  | 0.254557   |
| 665_sleuth_case2002           |  0.242165  |  0.25769   | 0.146767   |
| 522_pm10                      |  0.235107  |  0.233109  | 0.0445476  |
| 678_visualizing_environmental |  0.0604016 |  0.193514  | 0.358373   |
| 687_sleuth_ex1605             | -0.0707247 | -0.0740387 | 0.372597   |

**Median test $R^2$: 0.7683**.

## Installation

Dependencies should be installed within a `conda` environment. We recommend using
[`mamba`](https://github.com/mamba-org/mamba) since it is much faster than `conda` at
solving the environment and downloading the dependencies. To create a suitable
environment based on the provided `.yaml` file, use the command

```bash
$ mamba env create -f environment.yaml
```

Otherwise, you can update an existing environment using the same `.yaml` file.

After activating the environment, clone the git repository and launch the following command

```bash
$ pip install -e .
```

to install a development version of the library.

Running the tests:

```bash
$ tox
```

Generating the docs:

```bash
$ tox -e docs
```

## Usage

Setting up a symbolic regression problem in _AlpineGP_ involves several key steps:

1. Define the function that computes the prediction associated to an _individual_
(model expression tree). Its arguments may be a _function_ obtained by parsing the
individual tree and possibly other parameters, such as the dataset needed to evaluate
the model. It returns both an _error metric_ between the prediction and the data and
the prediction itself. 
```python
def eval_MSE_sol(individual, dataset):

    # ...
    return MSE, prediction
```

1. Define the functions that return the **prediction** and the **fitness** 
   associated to an individual. These functions **must** have the same
   arguments. In particular:
   - the first argument is **always** the batch of trees to be evaluated by the
     current worker;
   - the second argument **must** be the `toolbox` object used to compile the 
     individual trees into callable functions;
   - the third argument **must** be the dataset needed for the evaluation of the
     individuals.
   Both functions **must** be decorated with `ray.remote` to support
   distributed evaluation (multiprocessing).
```python
@ray.remote
def predict(trees, toolbox, data):

    callables = compile_individuals(toolbox, trees)

    preds = [None]*len(trees)

    for i, ind in enumerate(callables):
        _, preds[i] = eval_MSE_sol(ind, data)

    return preds

@ray.remote
def fitness(trees, toolbox, true_data):
    callables = compile_individuals(toolbox, trees)

    fitnesses = [None]*len(trees)

    for i, ind in enumerate(callables):
        MSE, _ = eval_MSE_sol(ind, data)
        
        # each fitness MUST be a tuple (required by DEAP)
        fitnesses[i] = (MSE,)

    return fitnesses
```

3. Set and solve the symbolic regression problem.
```python
# read parameters from YAML file
with open("ex1.yaml") as config_file:
    config_file_data = yaml.safe_load(config_file)

# ...
# ...

# load datasets...

# define the primitive set (input/output types)
pset = gp.PrimitiveSetTyped(...)

# rename arguments of the tree function
pset.renameArguments(ARG0="u")

# define extra common arguments of fitness and predict functions
common_params = {'penalty': penalty}

# create the Symbolic Regression Problem object
gpsr = gps.GPSymbolicRegressor(pset=pset, fitness=fitness.remote,
                               predict_func=predict.remote, common_data=common_params,
                               print_log=True, 
                               config_file_data=config_file_data)

# wrap tensors corresponding to train and test data into Dataset objects (to be passed to
# fit and predict methods)
train_data = Dataset("D", X_train, y_train)
test_data = Dataset("D", X_test, y_test)

# solve the symbolic regression problem
gpsr.fit(train_data)

# compute the prediction on the test dataset given by the best model found during the SR
pred_test = gpsr.predict(test_data)
```

A complete example notebook can be found in the `examples` directory. Also check the
`simple_sr.py` script for an introductory example (use it as a template/skeleton for
defining your symbolic regression problem).

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