Hyperparameter optimization with Optuna
=======================================

This script tunes the number of individuals and of islands of the regressor used in the
``simple_sr`` tutorial with an `Optuna <https://optuna.org>`_ grid search, scoring each
configuration by its mean :math:`R^2` in 5-fold cross-validation.

.. literalinclude:: optuna_tutorial.py
   :language: python
