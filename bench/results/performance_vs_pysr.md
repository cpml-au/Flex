# Performance of flex and comparison with PySR (2026-10-28)

## Setup

- Machine: AMD Ryzen Threadripper 7960X (24 cores, 2 threads per core), 125 GB RAM.
  Runs are pinned with `taskset` to physical cores that no other job was using:
  CPU 0 for 1-core runs, CPUs 0-7 for 8-core runs (hyperthread siblings left free).
- Environment: conda env `alpine` (Python 3.12), flex at commit `019b589` plus the
  uncommitted changes listed below, PySR 2.7.2 (SymbolicRegression.jl 2.7.2,
  Julia 1.11.9).
- flex uses L-BFGS (`const_optimizer: lbfgs`, at most 8 objective evaluations, central
  finite-difference gradient) or PSO (`const_optimizer: pso`, 2 particles, 10
  generations) to tune the constants of every individual that has constants.
  The initial population is not deduplicated (`remove_init_duplicates=False`).
- PySR uses the same operators and `maxsize=40`, `precision=64`, `batching=False`
  (full dataset at each evaluation, as in flex), multithreading with one Julia
  thread per core.

## Changes to the benchmark code

All in `bench/` (`bench.py` and new scripts):

- `const_optimizer` config key: `pso` or `lbfgs`; each optimizer is built once per
  process and reused.
- Removed a pygmo population of 70 random points that was created and discarded for
  every individual (it evaluated the model 70 times per individual).
- L-BFGS gradient: central finite differences in which all the perturbed constant
  vectors (2 per constant) are evaluated in a single call, by passing the constants
  as a matrix (`compute_MSE_batch`).
- Objective of the constant optimizer: the feature and label columns are prepared
  once per individual and the MSE is computed directly in the objective.
- Trig features of an individual (number of sin/cos nodes, nested sin/cos) computed
  from the tree in one pass instead of from its string.
- `compute_MSE`: predictions and labels are flattened, so that the MSE is correct for
  any shape (before, one-feature PMLB datasets gave an n x n residual).
- Best model converted to SymPy (`get_best_individuals_sympy`, the method name was
  misspelled).
- Expression evaluations counted in `bench.EVAL_COUNTS`, and early stopping that can be
  disabled (`early_stop_fitness_threshold=None`).
- New scripts: `profile_flex.py` (cProfile of a serial run), `compare_pysr.py` (time to
  solution), `throughput_pysr.py` (evaluations per second), `eval_cost_pysr.py` (cost
  of one evaluation).

## Results

### Single-core profile of flex (Nguyen-13)

Serial run (no Ray) on CPU 0, at most 20 generations, early stopping at MSE < 1e-12.
Times measured under cProfile; only the innermost functions are reported, because
pygmo calls the Python objective from C++ and this breaks the cProfile call stack.

| Phase | PSO | L-BFGS |
|---|---|---|
| objective of the constant optimizer | 19.4% | 9.3% |
| gradient (vectorized finite differences) | - | 26.6% |
| tree compilation (`compile_individual_with_consts`) | 21.7% | 18.6% |
| variation (`varAnd`) | 20.5% | 15.8% |
| pygmo population init | 4.9% | 4.2% |
| parent selection | 3.4% | 2.5% |
| generations run | 20 | 15 (exact solution found) |
| individuals per second | 2623 | 1800 |

### Time to solution (Nguyen-13, 8 cores, 60 s budget)

Each method stops when the training MSE drops below 1e-12. flex-PSO ran 190
generations, flex-L-BFGS 140 (both calibrated to about 60 s); PySR has
`timeout_in_seconds=60`. Five seeds.

| Method | Exact solution (MSE < 1e-12) | Time, median (min-max) | R² test, median | Model size |
|---|---|---|---|---|
| flex-PSO | 0/5 | 49.2 s (45.5-49.6), budget used up | 0.99995 | 15-16 |
| flex-L-BFGS | 5/5 | 3.29 s (0.48-6.64) | 1.0 | 11-14 |
| PySR | 5/5 | 0.18 s (0.12-1.16) | 1.0 | 12 |

Model size is the number of nodes of the simplified SymPy expression.

### Throughput (evaluations per second)

Early stopping disabled, PySR in turbo mode, median of 5 seeds. Evaluations are counted
as in SymbolicRegression.jl: one per candidate expression plus the objective calls of
the constant optimizer, without the calls made to estimate gradients by finite
differences. "flex raw" also counts the model evaluations of the gradients.

Both throughputs are averages over the window from 40 s to 60 s after the start of the
search. PySR is stopped after 60 s and its value is the average over the last ~20 s
that it prints. flex runs a number of generations that makes it last more than 60 s
(run time in the table); its evaluation counts, recorded at the end of each
generation, are interpolated at 40 s and 60 s.

| Dataset (rows x features) | Cores | flex generations | flex run time | flex evals/s | flex raw evals/s | PySR evals/s | PySR / flex |
|---|---|---|---|---|---|---|---|
| Nguyen-13 (20 x 1) | 1 | 70 | 65.1 s | 2.46e4 | 2.09e5 | 3.42e5 | 13.9 |
| Nguyen-13 (20 x 1) | 8 | 220 | 67.5 s | 7.14e4 | 5.32e5 | 1.71e6 | 23.9 |
| 1028_SWD (1000 x 10) | 1 | 30 | 79.5 s | 1.40e4 | 4.68e4 | 1.15e5 | 8.2 |
| 1028_SWD (1000 x 10) | 8 | 75 | 67.3 s | 5.21e4 | 1.80e5 | 6.20e5 | 11.9 |
| 225_puma8NH (8192 x 8) | 1 | 11 | 70.0 s | 4.39e3 | 1.25e4 | 1.43e4 | 3.3 |
| 225_puma8NH (8192 x 8) | 8 | 42 | 77.8 s | 1.76e4 | 5.55e4 | 8.51e4 | 4.8 |

Speed-up from 1 to 8 cores (evals/s): flex 2.9 (Nguyen-13), 3.7 (1028_SWD), 4.0
(225_puma8NH); PySR 5.0, 5.4, 6.0. Nguyen-13 uses the operators +, -, *, /, sin, cos,
exp, log; the PMLB datasets (standardized) use add, sub, mul, aq = x/sqrt(1+y^2), sin.

### Cost of one evaluation

Prediction plus sum of squared errors of c1\*x^3 + c2\*x^2 + c3\*x (13 nodes), on 1 core.

| Points | flex (numpy) | PySR backend | PySR backend, turbo |
|---|---|---|---|
| 20 | 2.77 us | 0.45 us | 1.04 us |
| 200 | 3.25 us | 0.94 us | 0.96 us |
| 2000 | 8.76 us | 5.84 us | 10.32 us |
| 20000 | 30.96 us | 104.73 us | 103.75 us |
| 200000 | 305.97 us | 638.22 us | 484.05 us |

## Observations

- A flex tree is evaluated with one numpy call per node, which has a fixed cost of
  about 0.2 us. On small datasets this fixed cost dominates: one evaluation costs
  about 6 times more than in PySR on 20 points, about the same on 2000 points, and
  less on 20000 points or more.
- The throughput gap shrinks with the dataset size: on 1 core, from 13.9x (20 rows)
  to 8.2x (1000 rows) and 3.3x (8192 rows).
- On 225_puma8NH, flex's raw throughput (1.25e4/s) is close to PySR's counted
  throughput (1.43e4/s): the remaining factor comes from the model evaluations of the
  finite-difference gradients, since flex tunes the constants of every individual
  with constants, while PySR optimizes the constants of a fraction of the population
  (`optimize_probability=0.14`) and does not count its gradient evaluations.
- flex scales worse than PySR from 1 to 8 cores.

## Commands

Run from the `bench` directory with the `alpine` env.

```bash
# single-core profiles
env OMP_NUM_THREADS=1 taskset -c 0 python profile_flex.py Nguyen_13_consts_pso.yaml Nguyen-13 --generations 20
env OMP_NUM_THREADS=1 taskset -c 0 python profile_flex.py Nguyen_13_consts_lbfgs.yaml Nguyen-13 --generations 20

# time to solution
env OMP_NUM_THREADS=1 taskset -c 0-7 python compare_pysr.py Nguyen-13 --cores 8 --timeout 60 --num-seeds 5

# throughput, flex
env OMP_NUM_THREADS=1 taskset -c 0 python throughput_pysr.py Nguyen-13 --cores 1 --timeout 60 --flex-generations 70 --num-seeds 5 --methods flex-lbfgs
env OMP_NUM_THREADS=1 taskset -c 0-7 python throughput_pysr.py Nguyen-13 --cores 8 --timeout 60 --flex-generations 220 --num-seeds 5 --methods flex-lbfgs
env OMP_NUM_THREADS=1 taskset -c 0 python throughput_pysr.py 1028_SWD --flex-cfg PMLB_lbfgs.yaml --cores 1 --timeout 60 --flex-generations 30 --num-seeds 5 --methods flex-lbfgs
env OMP_NUM_THREADS=1 taskset -c 0-7 python throughput_pysr.py 1028_SWD --flex-cfg PMLB_lbfgs.yaml --cores 8 --timeout 60 --flex-generations 75 --num-seeds 5 --methods flex-lbfgs
env OMP_NUM_THREADS=1 taskset -c 0 python throughput_pysr.py 225_puma8NH --flex-cfg PMLB_lbfgs.yaml --cores 1 --timeout 60 --flex-generations 11 --num-seeds 5 --methods flex-lbfgs
env OMP_NUM_THREADS=1 taskset -c 0-7 python throughput_pysr.py 225_puma8NH --flex-cfg PMLB_lbfgs.yaml --cores 8 --timeout 60 --flex-generations 42 --num-seeds 5 --methods flex-lbfgs

# throughput, PySR
env OMP_NUM_THREADS=1 taskset -c 0 python throughput_pysr.py Nguyen-13 --cores 1 --timeout 60 --num-seeds 5 --methods pysr --turbo
env OMP_NUM_THREADS=1 taskset -c 0-7 python throughput_pysr.py Nguyen-13 --cores 8 --timeout 60 --num-seeds 5 --methods pysr --turbo
env OMP_NUM_THREADS=1 taskset -c 0 python throughput_pysr.py 1028_SWD --flex-cfg PMLB_lbfgs.yaml --cores 1 --timeout 60 --num-seeds 5 --methods pysr --turbo
env OMP_NUM_THREADS=1 taskset -c 0-7 python throughput_pysr.py 1028_SWD --flex-cfg PMLB_lbfgs.yaml --cores 8 --timeout 60 --num-seeds 5 --methods pysr --turbo
env OMP_NUM_THREADS=1 taskset -c 0 python throughput_pysr.py 225_puma8NH --flex-cfg PMLB_lbfgs.yaml --cores 1 --timeout 60 --num-seeds 5 --methods pysr --turbo
env OMP_NUM_THREADS=1 taskset -c 0-7 python throughput_pysr.py 225_puma8NH --flex-cfg PMLB_lbfgs.yaml --cores 8 --timeout 60 --num-seeds 5 --methods pysr --turbo

# cost of one evaluation
env PYTHON_JULIACALL_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0 python eval_cost_pysr.py
```

Result files (`results/`): `profile_Nguyen_13_consts_{pso,lbfgs}_gen20.{log,prof,yaml}`,
`compare_pysr_Nguyen-13.{csv,log}`, `throughput_<dataset>_<cores>c_<method>.{csv,log}`
(with the flex config used in `throughput_flex-lbfgs_<dataset>_<cores>c.yaml`),
`eval_cost_pysr.txt`.
