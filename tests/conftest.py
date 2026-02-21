"""
Shared pytest fixtures providing synthetic market data for unit and integration tests.
No network calls are made — all data is generated deterministically from fixed seeds.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from data.market_data import MarketData

# ---------------------------------------------------------------------------
# Tickers and factor configuration
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
FACTORS = ["MKT", "Value", "Momentum"]

# Factor betas per ticker: [MKT, Value, Momentum]
# Designed so each stock has high R² in factor regressions (≥ 0.3 threshold).
_BETAS = {
    "AAPL": [1.2,  0.3,  0.5],
    "MSFT": [0.9,  0.2,  0.4],
    "GOOG": [1.1, -0.1,  0.3],
    "AMZN": [0.8,  0.6, -0.2],
    "META": [1.0,  0.4,  0.1],
}

# AMZN and META carry a negative daily drift so they accumulate unrealized losses
# by the end of 2022, enabling tax-loss harvesting tests.
_DRIFTS = {
    "AAPL": 0.0,
    "MSFT": 0.0,
    "GOOG": 0.0,
    "AMZN": -0.001,
    "META": -0.0008,
}

# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def warmup_dates():
    """
    Full date range including ~18 months of warmup before the backtest start.
    Required so the 252-calendar-day lookback window is fully populated on day 1.
    """
    return pd.bdate_range("2020-06-01", "2023-12-29")


@pytest.fixture(scope="session")
def trading_dates():
    """Backtest date range: ~2 years of business days (2022-01-03 to 2023-12-29)."""
    return pd.bdate_range("2022-01-03", "2023-12-29")


# ---------------------------------------------------------------------------
# Market data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def small_factor_returns(warmup_dates):
    """
    Deterministic Fama-French-style 3-factor returns over the full warmup+backtest
    period. No network calls.
    """
    np.random.seed(0)
    n = len(warmup_dates)
    return pd.DataFrame(
        {
            "MKT":      np.random.normal(0.0003, 0.010, n),
            "Value":    np.random.normal(0.0001, 0.007, n),
            "Momentum": np.random.normal(0.0002, 0.008, n),
        },
        index=warmup_dates,
    )


@pytest.fixture(scope="session")
def small_prices(warmup_dates, small_factor_returns):
    """
    5-ticker price DataFrame covering the full warmup+backtest period.
    Returns are constructed as a linear combination of factor returns plus low
    idiosyncratic noise, so each stock achieves R² ≥ 0.7 in the factor regression.
    AMZN and META carry a deliberate negative drift so they develop unrealized
    losses by the end of 2022.
    """
    np.random.seed(42)
    n = len(warmup_dates)
    factor_arr = small_factor_returns.values  # shape (n, 3)
    prices = {}
    for ticker, betas in _BETAS.items():
        idio = np.random.normal(0, 0.004, n)   # small idiosyncratic noise → high R²
        ret = factor_arr @ np.array(betas) + _DRIFTS[ticker] + idio
        prices[ticker] = 100.0 * np.cumprod(1 + ret)
    return pd.DataFrame(prices, index=warmup_dates)


@pytest.fixture(scope="session")
def small_spy_prices(warmup_dates, small_factor_returns):
    """Deterministic SPY price series derived from the MKT factor."""
    spy_returns = small_factor_returns["MKT"].values
    spy = 400.0 * np.cumprod(1 + spy_returns)
    return pd.Series(spy, index=warmup_dates, name="SPY")


@pytest.fixture(scope="session")
def small_market_data(small_prices, trading_dates):
    """
    MarketData built directly from synthetic prices (no yfinance).

    - ``prices``: trimmed to the backtest start date (2022-01-03) so the engine
      iterates only over the intended window.
    - ``returns``: computed from the full price history (including warmup) so the
      252-day lookback in the strategy is always populated.
    """
    returns_full = small_prices.pct_change().dropna(how="all")
    prices = small_prices.loc[small_prices.index >= trading_dates[0]]
    return MarketData(prices=prices, returns=returns_full)
