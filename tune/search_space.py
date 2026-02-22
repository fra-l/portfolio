# tune/search_space.py

"""
Defines the hyperparameter search space for Bayesian optimisation.

Each entry maps a parameter name to a tuple:
  ("float",  low, high)       — continuous float
  ("int",    low, high)       — integer
  ("cat",    [choices...])    — categorical

The sampler (tune/orchestrator.py) reads this dict to build Optuna trial suggestions.
"""

SEARCH_SPACE = {
    "value_weight":         ("float", 0.0,   1.0),
    "momentum_weight":      ("float", 0.0,   1.0),   # clipped so value + momentum <= 1
    "min_r2":               ("float", 0.1,   0.7),
    "lookback_days":        ("int",   60,    504),
    "rebalance_frequency":  ("cat",   ["W",  "M", "Q"]),
    "trading_cost_pct":     ("float", 0.0005, 0.005),
}
