#!/usr/bin/env python
# tune_main.py
"""
CLI entry point for Bayesian hyperparameter tuning.

Usage:
    python tune_main.py --n-trials 50 --host myserver.com --user ubuntu \
                        --repo /home/ubuntu/portfolio [--key ~/.ssh/id_rsa]

Credentials can also be supplied via env vars (TUNE_SSH_HOST, TUNE_SSH_USER,
TUNE_SSH_REPO, TUNE_SSH_KEY); CLI flags take precedence.

Outputs:
  - Best params and score printed to stdout when done
  - reports/tuning.db   — SQLite Optuna storage (all trials, resumable)
  - reports/tuning_results.csv — flat CSV of all completed trials
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter tuning for the portfolio backtest"
    )
    parser.add_argument("--n-trials",  type=int,  default=50,  help="Number of Optuna trials")
    parser.add_argument("--host",      type=str,  default=None, help="SSH host (overrides TUNE_SSH_HOST)")
    parser.add_argument("--user",      type=str,  default=None, help="SSH user (overrides TUNE_SSH_USER)")
    parser.add_argument("--repo",      type=str,  default=None, help="Repo path on remote (overrides TUNE_SSH_REPO)")
    parser.add_argument("--key",       type=str,  default=None, help="SSH key path (overrides TUNE_SSH_KEY)")
    parser.add_argument("--study",     type=str,  default="portfolio-tuning", help="Optuna study name")
    parser.add_argument("--storage",   type=str,  default="sqlite:///reports/tuning.db", help="Optuna storage URL")
    return parser.parse_args()


def main():
    args = parse_args()

    # Override env vars with CLI flags if provided
    if args.host:
        os.environ["TUNE_SSH_HOST"] = args.host
    if args.user:
        os.environ["TUNE_SSH_USER"] = args.user
    if args.repo:
        os.environ["TUNE_SSH_REPO"] = args.repo
    if args.key:
        os.environ["TUNE_SSH_KEY"] = args.key

    from tune.ssh_runner import SSHRunner
    from tune.orchestrator import run_study

    try:
        ssh_runner = SSHRunner.from_env()
    except EnvironmentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Bayesian optimisation: {args.n_trials} trials")
    print(f"  Remote:  {ssh_runner.user}@{ssh_runner.host}:{ssh_runner.repo_path}")
    print(f"  Storage: {args.storage}")
    print(f"  Study:   {args.study}\n")

    study = run_study(
        n_trials=args.n_trials,
        ssh_runner=ssh_runner,
        study_name=args.study,
        storage=args.storage,
    )

    best = study.best_trial
    print(f"\n{'='*52}")
    print(f"  TUNING COMPLETE — {len(study.trials)} trials")
    print(f"{'='*52}")
    print(f"  Best score:  {best.value:.4f}")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k:<26} {v}")
    print(f"{'='*52}")

    # Save all trials to CSV
    import pandas as pd
    import os

    os.makedirs("reports", exist_ok=True)
    rows = []
    for t in study.trials:
        row = {"trial": t.number, "score": t.value}
        row.update(t.params)
        row.update({f"metric_{k}": v for k, v in t.user_attrs.items()})
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        csv_path = "reports/tuning_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nAll trials saved to {csv_path}")


if __name__ == "__main__":
    main()
