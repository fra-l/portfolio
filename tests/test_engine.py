"""
Unit tests for BacktestEngine.

Uses a no-op strategy stub so tests are isolated from strategy logic.
Fixtures come from conftest.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from backtest.engine import BacktestEngine
from portfolio.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Minimal strategy stub
# ---------------------------------------------------------------------------

class _NoopStrategy:
    """Strategy that does nothing — lets the engine record history untouched."""

    def __init__(self, market_data):
        self.market_data = market_data

    def on_date(self, date):
        pass


# ---------------------------------------------------------------------------
# History recording
# ---------------------------------------------------------------------------

class TestHistoryRecording:
    def test_history_length_matches_date_count(self, small_market_data):
        portfolio = Portfolio(cash=10_000.0)
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=_NoopStrategy(small_market_data),
        )
        engine.run(small_market_data.prices.index)
        assert len(engine.history) == len(small_market_data.prices.index)

    def test_history_entry_has_required_keys(self, small_market_data):
        portfolio = Portfolio(cash=5_000.0)
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=_NoopStrategy(small_market_data),
        )
        engine.run(small_market_data.prices.index[:3])
        for entry in engine.history:
            assert "date" in entry
            assert "value" in entry
            assert "leverage" in entry
            assert "allocations" in entry

    def test_allocations_always_contains_cash(self, small_market_data):
        portfolio = Portfolio(cash=7_500.0)
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=_NoopStrategy(small_market_data),
        )
        engine.run(small_market_data.prices.index[:5])
        for entry in engine.history:
            assert "cash" in entry["allocations"]

    def test_dates_in_history_match_input(self, small_market_data):
        portfolio = Portfolio(cash=1_000.0)
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=_NoopStrategy(small_market_data),
        )
        dates = small_market_data.prices.index[:10]
        engine.run(dates)
        recorded_dates = [e["date"] for e in engine.history]
        for expected, got in zip(dates, recorded_dates):
            assert expected == got

    def test_empty_history_before_run(self, small_market_data):
        portfolio = Portfolio(cash=1_000.0)
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=_NoopStrategy(small_market_data),
        )
        assert engine.history == []


# ---------------------------------------------------------------------------
# Cash-only portfolio (no positions)
# ---------------------------------------------------------------------------

class TestCashOnlyPortfolio:
    def test_value_equals_cash_when_no_positions(self, small_market_data):
        cash = 15_000.0
        portfolio = Portfolio(cash=cash)
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=_NoopStrategy(small_market_data),
        )
        engine.run(small_market_data.prices.index[:10])
        for entry in engine.history:
            assert abs(entry["value"] - cash) < 1e-6, (
                f"Cash-only portfolio value changed on {entry['date']}: "
                f"expected {cash}, got {entry['value']}"
            )

    def test_leverage_is_one_without_margin(self, small_market_data):
        portfolio = Portfolio(cash=10_000.0)
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=_NoopStrategy(small_market_data),
        )
        engine.run(small_market_data.prices.index[:5])
        for entry in engine.history:
            assert abs(entry["leverage"] - 1.0) < 1e-9, (
                f"Leverage should be 1.0 without margin, got {entry['leverage']}"
            )

    def test_cash_allocation_equals_full_value(self, small_market_data):
        cash = 8_000.0
        portfolio = Portfolio(cash=cash)
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=_NoopStrategy(small_market_data),
        )
        engine.run(small_market_data.prices.index[:3])
        for entry in engine.history:
            assert abs(entry["allocations"]["cash"] - cash) < 1e-6


# ---------------------------------------------------------------------------
# Strategy interaction
# ---------------------------------------------------------------------------

class TestStrategyInteraction:
    def test_strategy_on_date_called_for_every_date(self, small_market_data):
        called_dates = []

        class _RecordingStrategy:
            def __init__(self, md):
                self.market_data = md

            def on_date(self, date):
                called_dates.append(date)

        portfolio = Portfolio(cash=1_000.0)
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=_RecordingStrategy(small_market_data),
        )
        dates = small_market_data.prices.index[:20]
        engine.run(dates)
        assert len(called_dates) == len(dates)
        for expected, got in zip(dates, called_dates):
            assert expected == got

    def test_run_with_zero_dates(self, small_market_data):
        portfolio = Portfolio(cash=1_000.0)
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=_NoopStrategy(small_market_data),
        )
        engine.run([])  # should not raise
        assert engine.history == []
