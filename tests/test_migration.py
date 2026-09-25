from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from deap import base, creator, gp

import flex.gp.regressor as regressor


def make_regressor(**kwargs):
    return regressor.GPSymbolicRegressor(
        pset_config=gp.PrimitiveSet("MAIN", 1),
        fitness=lambda: None,
        score_func=lambda: None,
        predict_func=lambda: None,
        **kwargs,
    )


@pytest.mark.parametrize("frequency", [0, -1, 1.5, 5.0, True, "5", None])
@pytest.mark.parametrize("hybrid", [False, True])
def test_fit_rejects_invalid_migration_frequency(frequency, hybrid):
    model = make_regressor(mig_freq=frequency, coarse_grained_islands=hybrid)
    with pytest.raises(ValueError, match="mig_freq must be a positive integer"):
        model.fit(np.ones((2, 1)))


@pytest.mark.parametrize("frequency", [1, np.int64(5)])
def test_fit_accepts_positive_integer_frequency(frequency, monkeypatch):
    model = make_regressor(mig_freq=frequency)
    # Stop after validation, before dataset/toolbox setup or Ray initialization.
    validate = Mock(side_effect=RuntimeError("reached dataset validation"))
    monkeypatch.setattr(regressor, "validate_data", validate)
    with pytest.raises(RuntimeError, match="reached dataset validation"):
        model.fit(np.ones((2, 1)))
    validate.assert_called_once()


@pytest.mark.parametrize(
    "generations, expected_migrations, expected_blocks",
    [(3, [], [3]), (10, [5, 10], [5, 5]), (12, [5, 10], [5, 5, 2])],
)
def test_hybrid_migrates_only_on_schedule(
    generations, expected_migrations, expected_blocks, monkeypatch
):
    model = make_regressor(
        mig_freq=5, generations=generations, num_islands=2, num_individuals=1
    )
    individual = creator.Individual.from_string("ARG0", model.pset_config)
    individual.fitness.values = (0,)
    model._GPSymbolicRegressor__pop = [[individual], [individual]]
    model._GPSymbolicRegressor__data_store = {"common": {}, "train": {}}
    model._toolbox_ref = None
    model._GPSymbolicRegressor__logbook = SimpleNamespace(
        chapters={"fitness": SimpleNamespace(select=lambda key: [0])}
    )

    blocks = []

    def evolve(packed_pop, toolbox_ref, fitness, args, callback, n_gens, *rest):
        blocks.append(n_gens)
        return packed_pop, 0

    remote = Mock()
    remote.options.return_value.remote.side_effect = evolve
    monkeypatch.setattr(regressor, "_evolve_island_hybrid_remote", remote)
    monkeypatch.setattr(regressor.ray, "get", lambda results: results)
    migrations = []
    monkeypatch.setattr(
        regressor, "migRing",
        lambda *args, **kwargs: migrations.append(model._GPSymbolicRegressor__cgen),
    )
    stats = Mock()
    monkeypatch.setattr(model, "_GPSymbolicRegressor__stats", stats)

    model._GPSymbolicRegressor__evolve_hybrid_islands(base.Toolbox())

    assert migrations == expected_migrations
    assert blocks == [size for size in expected_blocks for _ in range(2)]
    assert model._GPSymbolicRegressor__cgen == generations
    assert stats.call_args.args[1] == generations
