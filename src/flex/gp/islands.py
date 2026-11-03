from deap import algorithms, tools, gp, base, creator
import numpy as np
import ray
import random
from typing import List, Dict, Callable


def _register_creator_classes():
    """Create the DEAP ``creator`` classes used for individuals.

    These classes must exist in every process that constructs individuals. In the
    coarse-grained island strategy the Ray workers reconstruct individuals from their
    payloads (see ``unpack_individual``) using ``creator.Individual``; importing this
    module on a worker triggers this registration, so the class is available before
    any island is evolved. The classes are configuration-independent (the primitive
    set is supplied to ``compile`` via the toolbox, not baked into the class), so
    creating them eagerly at import time is safe and idempotent.
    """
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


# Register the creator classes eagerly at import time (driver and Ray workers alike).
_register_creator_classes()


def _island_one_generation(
    pop: List,
    toolbox: base.Toolbox,
    evaluate_fn: Callable,
    callback_func: Callable | None,
    params: Dict,
):
    """Evolve a single island by one generation.

    This mirrors the per-island body of ``GPSymbolicRegressor.__evolve_islands``
    but is agnostic about *how* fitness is computed: ``evaluate_fn`` maps the list
    of invalid individuals to their per-individual results.

    Args:
        pop: the island population (a list of individuals).
        toolbox: the DEAP toolbox (selection/variation/compile registered).
        evaluate_fn: callable mapping ``invalid_inds`` to a list of per-individual
            results, aligned with ``invalid_inds``.
        callback_func: optional callback used to assign fitness/attributes.
        params: dictionary of evolution parameters.

    Returns:
        a tuple ``(pop, num_evals)`` with the evolved population and the number of
        fitness evaluations performed.
    """
    num_individuals = params["num_individuals"]
    n_elitist = params["n_elitist"]
    variation = params["variation_mechanism"]
    crossover_prob = params["crossover_prob"]
    mut_prob = params["mut_prob"]
    overlapping = params["overlapping_generation"]

    # Select the parents and clone them to form the offspring
    offspring = list(map(toolbox.clone, toolbox.select(pop)))

    # Elitism + variation
    elite_inds = tools.selBest(offspring, n_elitist)
    if variation == "varand":
        varied_offspring = algorithms.varAnd(
            offspring[: num_individuals - n_elitist],
            toolbox,
            crossover_prob,
            mut_prob,
        )
    elif variation == "varor":
        varied_offspring = algorithms.varOr(
            offspring,
            toolbox,
            num_individuals - n_elitist,
            crossover_prob,
            mut_prob,
        )
    else:
        raise ValueError(
            "variation_mechanism must be either 'varAnd' or 'varOr'. "
            f"Got: {variation}"
        )
    offspring = elite_inds + varied_offspring

    invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_inds:
        ind._newborn = True

    # Evaluate the fitness of the invalid offspring via the supplied mapper.
    results = evaluate_fn(invalid_inds)
    if callback_func is not None:
        callback_func(invalid_inds, results)
    else:
        for ind, fit in zip(invalid_inds, results):
            ind.fitness.values = fit

    # Survival selection
    if not overlapping:
        pop[:] = offspring
    else:
        pop = tools.selBest(pop + offspring, num_individuals)

    # Update individual ages after survival
    for ind in pop:
        if getattr(ind, "_newborn", False):
            ind.age = 0
        else:
            ind.age = getattr(ind, "age", 0) + 1
        if hasattr(ind, "_newborn"):
            delattr(ind, "_newborn")

    return pop, len(invalid_inds)


def pack_individual(ind):
    """Serialize an individual into a transport-safe, ``creator``-free payload.

    Individuals are *not* shipped across the Ray boundary as ``creator.Individual``
    instances: cloudpickle reconstructs DEAP's dynamically-created classes in a way
    that is incompatible with its ``MetaCreator`` metaclass (raising a metaclass
    conflict), and the behaviour is fragile across repeated round-trips. Instead each
    individual is reduced to its list of nodes (plain ``deap.gp`` ``Primitive``/
    ``Terminal`` objects, which pickle cleanly by reference), its fitness values (or
    ``None`` if not yet evaluated), and any plain instance attributes (e.g. ``age``,
    ``consts``). The ``fitness`` object itself is excluded because it is a
    ``creator``-derived class; only its raw values are transferred.

    Returns:
        a tuple ``(nodes, fitness_values, attrs)``.
    """
    fitness_values = ind.fitness.values if ind.fitness.valid else None
    attrs = {key: value for key, value in vars(ind).items() if key != "fitness"}
    return (list(ind), fitness_values, attrs)


def unpack_individual(packed):
    """Reconstruct an individual from a payload produced by ``pack_individual``.

    Args:
        packed: a ``(nodes, fitness_values, attrs)`` tuple.

    Returns:
        a ``creator.Individual`` with its fitness and attributes restored.
    """
    nodes, fitness_values, attrs = packed
    ind = creator.Individual(nodes)
    if fitness_values is not None:
        ind.fitness.values = fitness_values
    for key, value in attrs.items():
        setattr(ind, key, value)
    return ind


def evolve_island(
    seed: int,
    packed_pop: List,
    toolbox: base.Toolbox,
    fitness_func: Callable,
    fitness_args: Dict,
    callback_func: Callable | None,
    n_gens: int,
    params: Dict,
):
    """Evolve a single island for ``n_gens`` generations inside one Ray task.

    Variation, fitness and survival selection all run in this task. The task seeds
    both ``random`` and NumPy's global generator, since the variation operators draw
    from both and a Ray worker may have served other tasks before this one.

    Args:
        seed: seed of the random generators of this task, drawn on the driver.
        packed_pop: the island population as ``creator``-free payloads.
        toolbox: the lean toolbox (Ray dereferences its object reference).
        fitness_func: the fitness function, called with the list of individuals to
            evaluate.
        fitness_args: keyword arguments for ``fitness_func``, as ``ray.ObjectRef``.
        callback_func: optional callback used to assign fitness/attributes.
        n_gens: number of generations to evolve before returning to the driver.
        params: dictionary of evolution parameters.

    Returns:
        a tuple ``(packed_pop, num_evals)``.
    """
    random.seed(seed)
    np.random.seed(seed)

    data = {key: ray.get(ref) for key, ref in fitness_args.items()}

    def evaluate_fn(invalid_inds):
        return fitness_func(invalid_inds, toolbox, **data)

    pop = [unpack_individual(p) for p in packed_pop]
    num_evals = 0
    for _ in range(n_gens):
        pop, evals = _island_one_generation(
            pop, toolbox, evaluate_fn, callback_func, params
        )
        num_evals += evals
    return [pack_individual(ind) for ind in pop], num_evals
