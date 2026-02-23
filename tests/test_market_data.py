"""
Unit tests for MarketData.

Covers:
- get_price() for a known date with a valid price
- get_price() NaN fallback: returns last known non-NaN price
- get_price() returns 0.0 when no prior valid price exists
- get_returns() returns the correct slice for a given date range
- get_returns() handles partial date overlap gracefully
- Warmup period alignment: prices trimmed to backtest start, returns keep full history
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from data.market_data import MarketData


# ---------------------------------------------------------------------------
# Minimal fixture helpers (no network calls)
# ---------------------------------------------------------------------------

def _make_market_data():
    """Two-ticker MarketData with a NaN entry for deterministic testing."""
    dates = pd.bdate_range("2022-01-03", "2022-01-10")
    prices = pd.DataFrame(
        {
            "AAPL": [150.0, 151.0, float("nan"), 153.0, 154.0, 155.0],
            "MSFT": [300.0, 301.0, 302.0, 303.0, 304.0, 305.0],
        },
        index=dates,
    )
    returns = prices.pct_change().dropna(how="all")
    return MarketData(prices=prices, returns=returns)


# ---------------------------------------------------------------------------
# get_price — happy path
# ---------------------------------------------------------------------------

class TestGetPriceHappyPath:
    def test_returns_correct_price_for_known_date(self):
        md = _make_market_data()
        date = pd.Timestamp("2022-01-03")
        assert md.get_price("AAPL", date) == pytest.approx(150.0)

    def test_returns_correct_price_for_different_ticker(self):
        md = _make_market_data()
        date = pd.Timestamp("2022-01-04")
        assert md.get_price("MSFT", date) == pytest.approx(301.0)


# ---------------------------------------------------------------------------
# get_price — NaN fallback
# ---------------------------------------------------------------------------

class TestGetPriceNaNFallback:
    def test_nan_date_falls_back_to_last_known_price(self):
        """2022-01-05 is NaN for AAPL; should return the previous valid price (151.0)."""
        md = _make_market_data()
        date = pd.Timestamp("2022-01-05")
        assert md.get_price("AAPL", date) == pytest.approx(151.0)

    def test_nan_fallback_ignores_later_valid_prices(self):
        """Fallback uses only prices up to and including the requested date."""
        dates = pd.bdate_range("2022-01-03", "2022-01-07")
        prices = pd.DataFrame(
            {"AAPL": [float("nan"), 100.0, float("nan"), 102.0, 103.0]},
            index=dates,
        )
        md = MarketData(prices=prices, returns=prices.pct_change().dropna(how="all"))
        # 2022-01-05 is NaN; last known before it is 100.0 (2022-01-04)
        assert md.get_price("AAPL", pd.Timestamp("2022-01-05")) == pytest.approx(100.0)

    def test_returns_zero_when_no_prior_valid_price(self):
        """If the first price entry is NaN and there is no prior data, return 0.0."""
        dates = pd.bdate_range("2022-01-03", "2022-01-05")
        prices = pd.DataFrame(
            {"AAPL": [float("nan"), 100.0, 101.0]},
            index=dates,
        )
        md = MarketData(prices=prices, returns=prices.pct_change().dropna(how="all"))
        assert md.get_price("AAPL", pd.Timestamp("2022-01-03")) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_returns
# ---------------------------------------------------------------------------

class TestGetReturns:
    def test_returns_correct_slice(self):
        md = _make_market_data()
        start = pd.Timestamp("2022-01-04")
        end = pd.Timestamp("2022-01-06")
        result = md.get_returns(["MSFT"], start, end)
        assert list(result.index) == list(md.returns.loc[start:end].index)
        assert "MSFT" in result.columns

    def test_returns_only_requested_tickers(self):
        md = _make_market_data()
        result = md.get_returns(["AAPL"], pd.Timestamp("2022-01-04"), pd.Timestamp("2022-01-10"))
        assert list(result.columns) == ["AAPL"]
        assert "MSFT" not in result.columns

    def test_partial_date_overlap_does_not_raise(self):
        """Requesting a window that extends beyond the stored data should return whatever is available."""
        md = _make_market_data()
        # end extends past last date in data
        result = md.get_returns(
            ["AAPL"],
            pd.Timestamp("2022-01-04"),
            pd.Timestamp("2099-01-01"),
        )
        assert len(result) > 0

    def test_start_before_data_returns_available_rows(self):
        md = _make_market_data()
        result = md.get_returns(
            ["MSFT"],
            pd.Timestamp("2000-01-01"),
            pd.Timestamp("2022-01-06"),
        )
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Warmup period alignment (using conftest session fixtures)
# ---------------------------------------------------------------------------

class TestWarmupAlignment:
    def test_prices_start_at_backtest_start(self, small_market_data, trading_dates):
        """prices should be trimmed to the first trading date."""
        assert small_market_data.prices.index[0] == trading_dates[0]

    def test_returns_include_warmup_history(self, small_market_data, trading_dates):
        """returns must start before the first trading date (warmup included)."""
        assert small_market_data.returns.index[0] < trading_dates[0]

    def test_prices_subset_of_returns_dates(self, small_market_data):
        """Every date in prices must also be present in returns."""
        price_dates = set(small_market_data.prices.index)
        return_dates = set(small_market_data.returns.index)
        assert price_dates.issubset(return_dates)
