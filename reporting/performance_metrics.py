"""
Performance analytics for portfolio backtests.

Metrics:
  - Sharpe ratio, Sortino ratio
  - Max drawdown, drawdown duration
  - Alpha, beta, information ratio, annualized tracking error vs benchmark
  - Average monthly turnover
"""
from __future__ import annotations

import os
from typing import Union

import numpy as np
import pandas as pd


def sharpe_ratio(portfolio_returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio. risk_free is annualized (default 0%)."""
    if len(portfolio_returns) < 2:
        return float("nan")
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    if ann_vol == 0:
        return float("nan")
    return (ann_return - risk_free) / ann_vol


def sortino_ratio(portfolio_returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sortino ratio using downside deviation below zero."""
    if len(portfolio_returns) < 2:
        return float("nan")
    ann_return = portfolio_returns.mean() * 252
    downside = portfolio_returns[portfolio_returns < 0]
    if len(downside) == 0:
        return float("inf")
    downside_vol = downside.std() * np.sqrt(252)
    if downside_vol == 0:
        return float("nan")
    return (ann_return - risk_free) / downside_vol


def max_drawdown(values: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (e.g. 0.25 = 25%)."""
    if len(values) < 2:
        return 0.0
    peak = values.cummax()
    drawdown = (values - peak) / peak
    return float(-drawdown.min())


def drawdown_duration(values: pd.Series) -> int:
    """Calendar days spent in the longest continuous drawdown period."""
    if len(values) < 2:
        return 0
    peak = values.cummax()
    in_drawdown = values < peak
    max_duration = 0
    current_start = None
    for date, flag in in_drawdown.items():
        if flag and current_start is None:
            current_start = date
        elif not flag and current_start is not None:
            duration = (date - current_start).days
            max_duration = max(max_duration, duration)
            current_start = None
    if current_start is not None:
        duration = (values.index[-1] - current_start).days
        max_duration = max(max_duration, duration)
    return max_duration


def alpha_beta(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> tuple[float, float]:
    """
    OLS regression of portfolio daily returns on benchmark daily returns.
    Returns (annualized_alpha, beta).
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 10:
        return float("nan"), float("nan")
    p = portfolio_returns.loc[common].values
    b = benchmark_returns.loc[common].values
    cov = np.cov(p, b, ddof=1)
    if cov[1, 1] == 0:
        return float("nan"), float("nan")
    beta = cov[0, 1] / cov[1, 1]
    daily_alpha = p.mean() - beta * b.mean()
    return daily_alpha * 252, beta


def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized information ratio: active_return / tracking_error."""
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 2:
        return float("nan")
    active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
    te = active.std() * np.sqrt(252)
    if te == 0:
        return float("nan")
    return active.mean() * 252 / te


def annualized_tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized tracking error: std(active_returns) * sqrt(252)."""
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 2:
        return float("nan")
    active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
    return float(active.std() * np.sqrt(252))


def average_monthly_turnover(trades: list[dict], history: list[dict]) -> float:
    """Average monthly turnover as a fraction of portfolio value."""
    if not trades or not history:
        return 0.0
    trades_df = pd.DataFrame(trades)
    if "amount" not in trades_df.columns or "date" not in trades_df.columns:
        return 0.0
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    trades_df["month"] = trades_df["date"].dt.to_period("M")
    monthly_traded = trades_df.groupby("month")["amount"].sum()

    history_df = pd.DataFrame(
        [{"date": h["date"], "value": h["value"]} for h in history]
    )
    history_df["month"] = pd.to_datetime(history_df["date"]).dt.to_period("M")
    monthly_values = history_df.groupby("month")["value"].last()

    common = monthly_traded.index.intersection(monthly_values.index)
    if len(common) == 0:
        return 0.0
    turnovers = monthly_traded.loc[common] / monthly_values.loc[common]
    return float(turnovers.mean())


def compute_all_metrics(
    history: list[dict],
    trades: list[dict],
    benchmark_prices: Union[dict[str, pd.Series], pd.Series, None],
    initial_value: float,
) -> dict:
    """
    Compute all performance metrics and return as a dict.

    benchmark_prices can be:
      - dict[str, pd.Series]  — one entry per benchmark ticker (preferred)
      - pd.Series             — treated as {"SPY": series} for backward compat
    """
    if not history:
        return {}
    history_df = pd.DataFrame(
        [{"date": h["date"], "value": h["value"]} for h in history]
    ).set_index("date")
    values = history_df["value"].dropna()
    if len(values) < 2:
        return {}

    # Normalise benchmark_prices to a dict
    if isinstance(benchmark_prices, pd.Series):
        benchmark_prices = {"SPY": benchmark_prices}
    if benchmark_prices is None:
        benchmark_prices = {}

    portfolio_returns = values.pct_change().dropna()

    sharpe = sharpe_ratio(portfolio_returns)
    sortino = sortino_ratio(portfolio_returns)
    mdd = max_drawdown(values)
    dd_dur = drawdown_duration(values)
    turnover = average_monthly_turnover(trades, history)

    years = (values.index[-1] - values.index[0]).days / 365.25
    ann_return = (
        ((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1) * 100
        if years > 0 else 0.0
    )

    # Per-benchmark relative metrics
    per_benchmark = {}
    for ticker, bm_series in benchmark_prices.items():
        bm = bm_series.dropna().reindex(values.index, method="ffill").dropna()
        bm_returns = bm.pct_change().dropna()
        ann_alpha, beta = alpha_beta(portfolio_returns, bm_returns)
        ir = information_ratio(portfolio_returns, bm_returns)
        te = annualized_tracking_error(portfolio_returns, bm_returns)
        per_benchmark[ticker] = {
            "alpha_annualized": ann_alpha,
            "beta": beta,
            "information_ratio": ir,
            "annualized_tracking_error_pct": te * 100,
        }

    return {
        "annualized_return_pct": ann_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": mdd * 100,
        "max_drawdown_duration_days": dd_dur,
        "avg_monthly_turnover_pct": turnover * 100,
        "per_benchmark": per_benchmark,
    }


def _fmt(v: object, sign: bool = False, decimals: int = 2, suffix: str = "") -> str:
    """Format a numeric metric, returning 'N/A' for nan/inf."""
    if not isinstance(v, (int, float)):
        return "N/A"
    if v != v or abs(v) == float("inf"):  # nan or inf
        return "N/A"
    fmt = f"{'+'if sign else ''}.{decimals}f"
    return f"{v:{fmt}}{suffix}"


def write_summary_report(metrics: dict, output_path: str = "reports/summary.txt") -> None:
    """Write a text summary of all performance metrics to output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = [
        "=" * 52,
        "  PERFORMANCE SUMMARY",
        "=" * 52,
        f"  Annualized return:     {_fmt(metrics.get('annualized_return_pct', float('nan')), sign=True, suffix='%')}",
        f"  Sharpe ratio:          {_fmt(metrics.get('sharpe_ratio', float('nan')), sign=True)}",
        f"  Sortino ratio:         {_fmt(metrics.get('sortino_ratio', float('nan')), sign=True)}",
        f"  Max drawdown:          {_fmt(metrics.get('max_drawdown_pct', float('nan')), suffix='%')}",
        f"  Drawdown duration:     {metrics.get('max_drawdown_duration_days', 0)} days",
        "",
        "  --- Turnover ---",
        f"  Avg monthly turnover:  {_fmt(metrics.get('avg_monthly_turnover_pct', float('nan')), suffix='%')}",
    ]

    per_benchmark = metrics.get("per_benchmark", {})
    for ticker, bm in per_benchmark.items():
        lines += [
            "",
            f"  --- Benchmark-relative (vs {ticker}) ---",
            f"  Alpha (annualized):    {_fmt(bm.get('alpha_annualized', float('nan')), sign=True, decimals=4)}",
            f"  Beta:                  {_fmt(bm.get('beta', float('nan')), sign=True, decimals=3)}",
            f"  Information ratio:     {_fmt(bm.get('information_ratio', float('nan')), sign=True)}",
            f"  Tracking error (ann.): {_fmt(bm.get('annualized_tracking_error_pct', float('nan')), suffix='%')}",
        ]

    lines.append("=" * 52)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nPerformance summary written to {output_path}")
