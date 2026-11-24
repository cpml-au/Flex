# Parallelization strategies

_Flex_ uses Ray to evaluate the islands in parallel with one of two strategies, selected
by the `coarse_grained_islands` argument of the `GPSymbolicRegressor` constructor (the
YAML configuration does not set this argument):

```python
gpsr = GPSymbolicRegressor(
    # ...
    num_islands=16,
    coarse_grained_islands=True,  # False (the default) selects the fine-grained strategy
)
```

**Fine-grained strategy (`coarse_grained_islands=False`, the default).** The main process
performs selection and variation for all the islands at each generation. It then splits
the new individuals of all the islands into batches of `batch_size` individuals and
sends them to the Ray workers for the evaluation of the fitness, so any free core can
evaluate an individual of any island. Statistics, validation and the early stopping
check are computed at each generation.

Use this strategy when the number of islands is smaller than the number of cores, because
all the cores can still evaluate the fitness; when individuals of different islands
differ strongly in evaluation cost, because the shared queue of batches balances the
load; when statistics, validation or early stopping are needed at each generation; and
when using `preprocess_args`, which this strategy alone supports.

**Coarse-grained strategy (`coarse_grained_islands=True`).** Each island runs as one Ray
task, which performs selection, variation and fitness evaluation for `mig_freq`
generations. The main process then collects the populations, performs the migration, and
starts the next block of generations. As a result:

- the main process and Ray communicate once per block instead of at each generation;
- each block ends when the slowest island finishes;
- statistics, validation, logging and the early stopping check are computed at the end of
  each block, so a run can continue for up to `mig_freq` generations after reaching the
  early stopping threshold;
- `batch_size` applies only to the evaluation of the initial population.

Use this strategy when selection and variation take a significant share of the run, or
when the fitness of each individual is cheap to evaluate, because these are the cases
where the communication at each generation limits the fine-grained strategy. Each island
uses one core at a time (with the default `num_cpus=1`), so set `num_islands` at least
equal to the number of cores. With more islands than cores, each island waiting in the
queue starts as soon as a core becomes free, which keeps the cores busy when some islands
are slower than others. The argument `num_individuals` is the size of each island, so
reduce it when increasing `num_islands` to keep the same total population. The benchmarks
in the `bench` folder use this strategy.

With both strategies, a run is repeatable when `random` and `numpy.random` are seeded in
the main process before calling `fit`, provided that Ray is started before the seeding,
because starting Ray can draw from `random`. With `multiprocessing=False`, the islands
run in the main process and `coarse_grained_islands` has no effect; the same holds for a
single island, which always follows the fine-grained strategy.
