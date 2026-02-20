import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from universe.universe_selector import UniverseSelector


def _make_data(loadings, r2_values, vol_values=None):
    """
    Helper: build exposures DataFrame, r2 Series and optional volatility Series.

    loadings   : dict  {ticker: {"MKT": x, "Value": y, "Momentum": z}}
    r2_values  : dict  {ticker: float}
    vol_values : dict  {ticker: float} | None
    """
    tickers = list(loadings)
    exposures = pd.DataFrame(loadings).T          # index=tickers, cols=factors
    r2 = pd.Series(r2_values)
    volatility = pd.Series(vol_values) if vol_values else None
    return exposures, r2, volatility


# ---------------------------------------------------------------------------

def test_r2_filter_only():
    """Existing behaviour: stocks below min_r2 are excluded."""
    exposures, r2, _ = _make_data(
        {"AAPL": {"MKT": 1.0, "Value": 0.3}, "MSFT": {"MKT": 0.9, "Value": 0.1}},
        {"AAPL": 0.5, "MSFT": 0.2},
    )
    sel = UniverseSelector(min_r2=0.3)
    result = sel.select(exposures, r2)
    assert result == ["AAPL"]


def test_min_factor_loading():
    """Stocks with loading below the minimum threshold are excluded."""
    exposures, r2, _ = _make_data(
        {"AAPL": {"Value": 0.15}, "MSFT": {"Value": 0.05}},
        {"AAPL": 0.6, "MSFT": 0.6},
    )
    sel = UniverseSelector(min_r2=0.3, min_factor_loading={"Value": 0.1})
    result = sel.select(exposures, r2)
    assert "AAPL" in result
    assert "MSFT" not in result


def test_max_factor_loading():
    """Stocks with loading above the maximum threshold are excluded."""
    exposures, r2, _ = _make_data(
        {"AAPL": {"MKT": 1.2}, "MSFT": {"MKT": 2.0}},
        {"AAPL": 0.6, "MSFT": 0.6},
    )
    sel = UniverseSelector(min_r2=0.3, max_factor_loading={"MKT": 1.5})
    result = sel.select(exposures, r2)
    assert "AAPL" in result
    assert "MSFT" not in result


def test_volatility_filter():
    """Stocks with annualised volatility above the cap are excluded."""
    exposures, r2, volatility = _make_data(
        {"AAPL": {"MKT": 1.0}, "TSLA": {"MKT": 1.0}},
        {"AAPL": 0.6, "TSLA": 0.6},
        {"AAPL": 0.25, "TSLA": 0.80},
    )
    sel = UniverseSelector(min_r2=0.3, max_volatility=0.5)
    result = sel.select(exposures, r2, volatility=volatility)
    assert "AAPL" in result
    assert "TSLA" not in result


def test_volatility_none_skips_filter():
    """When volatility=None, max_volatility filter does not exclude anything."""
    exposures, r2, _ = _make_data(
        {"AAPL": {"MKT": 1.0}, "TSLA": {"MKT": 1.0}},
        {"AAPL": 0.6, "TSLA": 0.6},
    )
    sel = UniverseSelector(min_r2=0.3, max_volatility=0.1)  # very tight cap
    result = sel.select(exposures, r2, volatility=None)
    # Both pass because volatility=None skips the filter
    assert set(result) == {"AAPL", "TSLA"}


def test_combined_filters():
    """All three new filters apply together (AND logic)."""
    exposures, r2, volatility = _make_data(
        {
            "A": {"Value": 0.3, "MKT": 1.0},   # passes all
            "B": {"Value": 0.05, "MKT": 1.0},  # fails min_factor_loading
            "C": {"Value": 0.3, "MKT": 2.0},   # fails max_factor_loading
            "D": {"Value": 0.3, "MKT": 1.0},   # fails volatility
        },
        {"A": 0.6, "B": 0.6, "C": 0.6, "D": 0.6},
        {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.9},
    )
    sel = UniverseSelector(
        min_r2=0.3,
        min_factor_loading={"Value": 0.1},
        max_factor_loading={"MKT": 1.5},
        max_volatility=0.5,
    )
    result = sel.select(exposures, r2, volatility=volatility)
    assert result == ["A"]


def test_unknown_factor_in_filter_is_ignored():
    """A filter referencing a factor not in exposures does not crash."""
    exposures, r2, _ = _make_data(
        {"AAPL": {"Value": 0.3}},
        {"AAPL": 0.6},
    )
    sel = UniverseSelector(
        min_r2=0.3,
        min_factor_loading={"NonExistentFactor": 0.5},
    )
    result = sel.select(exposures, r2)
    assert result == ["AAPL"]   # filter silently skipped, stock kept
