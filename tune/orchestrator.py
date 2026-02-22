# tune/orchestrator.py

"""
Bayesian optimisation orchestrator using Optuna (TPE sampler).

Each trial:
  1. Optuna suggests hyperparameters from SEARCH_SPACE
  2. SSHRunner ships the config to the remote machine
  3. Remote worker runs run_backtest() and returns metrics JSON
  4. compute_score() converts metrics to a scalar objective
  5. Optuna records the score and updates its model

The study is persisted to SQLite so runs can be interrupted and resumed.
"""

import optuna

from tune.search_space import SEARCH_SPACE
from tune.objective import compute_score
from tune.ssh_runner import SSHRunner


def sample_params(trial: optuna.Trial) -> dict:
    """Sample one set of hyperparameters from SEARCH_SPACE."""
    params = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "float":
            _, low, high = spec
            params[name] = trial.suggest_float(name, low, high)
        elif kind == "int":
            _, low, high = spec
            params[name] = trial.suggest_int(name, low, high)
        elif kind == "cat":
            _, choices = spec
            params[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"Unknown search space kind '{kind}' for '{name}'")
    return params


def run_study(
    n_trials: int,
    ssh_runner: SSHRunner,
    study_name: str = "portfolio-tuning",
    storage: str = "sqlite:///reports/tuning.db",
) -> optuna.Study:
    """
    Run a Bayesian optimisation study.

    Returns the completed Optuna study object (all trials accessible via study.trials).
    """
    import os
    os.makedirs("reports", exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        try:
            metrics = ssh_runner.run_trial(params)
        except Exception as exc:
            # Log and return a very bad score so Optuna skips this region
            print(f"[orchestrator] Trial {trial.number} failed: {exc}")
            raise optuna.exceptions.TrialPruned()

        score = compute_score(metrics)
        # Attach raw metrics as user attributes for later inspection
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                trial.set_user_attr(key, value)
        return score

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study
