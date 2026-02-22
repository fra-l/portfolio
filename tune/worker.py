#!/usr/bin/env python
# tune/worker.py
"""
Standalone worker script that runs one backtest trial and writes metrics to stdout.

Usage:
    python tune/worker.py < trial_config.json

Input:  JSON object on stdin with keys matching BacktestConfig fields:
          value_weight, momentum_weight, min_r2, lookback_days,
          rebalance_frequency, trading_cost_pct

Output: JSON metrics dict on stdout (all other output goes to stderr).
        Exits with code 0 on success, 1 on failure.
"""

import json
import sys
import os


def _redirect_stdout_to_stderr():
    """Replace sys.stdout with a stderr-backed writer so print() goes to stderr."""
    sys.stdout = sys.stderr


def main():
    # Redirect stdout to stderr before importing heavy modules so that any
    # library startup chatter doesn't corrupt the JSON output channel.
    real_stdout = sys.__stdout__
    _redirect_stdout_to_stderr()

    try:
        raw = real_stdout.buffer.read() if hasattr(real_stdout, "buffer") else b""
        # Re-read from the original stdin (before redirection)
        raw = sys.__stdin__.read()
        trial_config = json.loads(raw)
    except Exception as exc:
        print(f"[worker] ERROR reading stdin: {exc}", file=sys.stderr)
        sys.exit(1)

    # Build BacktestConfig from the trial params
    try:
        # Must import after stdout redirect so download progress goes to stderr
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import BacktestConfig, TradingCostConfig
        from backtest.runner import run_backtest

        value_weight    = float(trial_config.get("value_weight",    0.6))
        momentum_weight = float(trial_config.get("momentum_weight", 0.4))

        # Normalise so weights sum to at most 1.0
        total = value_weight + momentum_weight
        if total > 1.0:
            value_weight    /= total
            momentum_weight /= total

        trading_cost_pct = float(trial_config.get("trading_cost_pct", 0.001))

        config = BacktestConfig(
            min_r2=float(trial_config.get("min_r2", 0.3)),
            lookback_days=int(trial_config.get("lookback_days", 252)),
            rebalance_frequency=str(trial_config.get("rebalance_frequency", "M")),
            target_weights={
                "Value":    value_weight,
                "Momentum": momentum_weight,
            },
            trading_cost_config=TradingCostConfig(
                pct_cost=trading_cost_pct,
                min_cost=1.0,
            ),
        )

        metrics = run_backtest(config)
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"[worker] ERROR during backtest: {exc}", file=sys.stderr)
        sys.exit(1)

    # Strip private keys (objects that are not JSON-serialisable)
    output = {k: v for k, v in metrics.items() if not k.startswith("_")}

    # Write JSON to the real stdout
    real_stdout.write(json.dumps(output))
    real_stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
