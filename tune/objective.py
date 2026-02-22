# tune/objective.py

"""
Composite scoring function for portfolio backtest results.

Higher score is better. Weights:
  +0.5 × Sharpe ratio
  +0.3 × annualized return (fraction, not %)
  -0.2 × max drawdown (fraction, not %)
"""

WEIGHTS = {
    "sharpe_ratio":            0.5,
    "annualized_return_pct":   0.3,
    "max_drawdown_pct":       -0.2,
}


def compute_score(metrics: dict) -> float:
    """Return composite score from a metrics dict produced by compute_all_metrics()."""
    sharpe = metrics.get("sharpe_ratio", 0.0)
    ret    = metrics.get("annualized_return_pct", 0.0) / 100
    dd     = abs(metrics.get("max_drawdown_pct", 0.0)) / 100

    if sharpe != sharpe:   # NaN guard
        sharpe = 0.0
    if ret != ret:
        ret = 0.0
    if dd != dd:
        dd = 0.0

    return 0.5 * sharpe + 0.3 * ret - 0.2 * dd
