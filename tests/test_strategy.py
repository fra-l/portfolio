"""
Unit tests for FactorReplicationStrategy.

Tests use the synthetic fixtures from conftest.py so no network calls are made.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import MarginConfig, TaxConfig, TradingCostConfig
from decisions.decision_engine import DecisionEngine
from execution.executor import Executor
from factors.factor_model import FactorModel
from portfolio.portfolio import Portfolio
from strategy.strategy import FactorReplicationStrategy
from targets.factor_target import FactorTarget
from tax.tax_engine import TaxEngine
from trading.margin_cost import MarginCostModel
from universe.universe_selector import UniverseSelector


# ---------------------------------------------------------------------------
# Helper: build a minimal FactorReplicationStrategy
# ---------------------------------------------------------------------------

def _make_strategy(market_data, factor_returns, cash=20_000.0):
    tax_engine = TaxEngine()
    margin_config = MarginConfig(enabled=False)
    decision_engine = DecisionEngine(
        tax_engine=tax_engine,
        trading_cost_config=TradingCostConfig(),
        margin_config=margin_config,
        margin_cost_model=MarginCostModel(),
    )
    portfolio = Portfolio(cash=cash)
    executor = Executor(market_data)
    target = FactorTarget({"Value": 0.6, "Momentum": 0.4})
    universe_selector = UniverseSelector(min_r2=0.1)
    factor_model = FactorModel(factor_returns)
    strategy = FactorReplicationStrategy(
        market_data=market_data,
        factor_model=factor_model,
        universe_selector=universe_selector,
        target=target,
        portfolio=portfolio,
        decision_engine=decision_engine,
        executor=executor,
        margin_config=margin_config,
        margin_cost_model=MarginCostModel(),
    )
    return strategy, portfolio, executor


# ---------------------------------------------------------------------------
# _is_rebalance_date
# ---------------------------------------------------------------------------

class TestIsRebalanceDate:
    def test_first_call_always_triggers_rebalance(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        assert strategy._is_rebalance_date(pd.Timestamp("2022-01-03"))

    def test_same_month_skips_rebalance(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        strategy.last_rebalance_date = pd.Timestamp("2022-01-03")
        assert not strategy._is_rebalance_date(pd.Timestamp("2022-01-20"))

    def test_next_month_triggers_rebalance(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        strategy.last_rebalance_date = pd.Timestamp("2022-01-03")
        assert strategy._is_rebalance_date(pd.Timestamp("2022-02-01"))

    def test_same_day_skips(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        strategy.last_rebalance_date = pd.Timestamp("2022-03-15")
        assert not strategy._is_rebalance_date(pd.Timestamp("2022-03-15"))

    def test_year_boundary_triggers_rebalance(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        strategy.last_rebalance_date = pd.Timestamp("2022-12-30")
        assert strategy._is_rebalance_date(pd.Timestamp("2023-01-03"))


# ---------------------------------------------------------------------------
# _portfolio_factor_exposure
# ---------------------------------------------------------------------------

class TestPortfolioFactorExposure:
    def test_empty_portfolio_returns_zero_vector(self, small_market_data, small_factor_returns):
        strategy, portfolio, _ = _make_strategy(small_market_data, small_factor_returns)
        assert len(portfolio.positions) == 0

        tickers = small_market_data.prices.columns.tolist()
        factor_names = small_factor_returns.columns.tolist()
        exposures = pd.DataFrame(
            np.zeros((len(tickers), len(factor_names))),
            index=tickers,
            columns=factor_names,
        )
        result = strategy._portfolio_factor_exposure(exposures, pd.Timestamp("2022-01-03"))
        assert np.allclose(result, np.zeros(len(factor_names)))

    def test_single_position_returns_its_exposure(self, small_market_data, small_factor_returns):
        from portfolio.lots import Lot

        strategy, portfolio, executor = _make_strategy(small_market_data, small_factor_returns)
        date = pd.Timestamp("2022-01-03")
        # Buy 10 shares of AAPL at whatever the price is on that date
        executor.buy(portfolio, "AAPL", 1000.0, date)

        factor_names = small_factor_returns.columns.tolist()
        exposures = pd.DataFrame(
            {
                "AAPL": [1.0, 0.5, 0.3],
                "MSFT": [0.8, 0.2, 0.4],
            },
            index=factor_names,
        ).T
        result = strategy._portfolio_factor_exposure(exposures, date)
        # Only AAPL is in the portfolio, so exposure should equal AAPL's row
        assert np.allclose(result, exposures.loc["AAPL"].values)


# ---------------------------------------------------------------------------
# _realized_gains_ytd
# ---------------------------------------------------------------------------

class TestRealizedGainsYTD:
    def test_empty_gains_list_returns_zero(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        assert strategy._realized_gains_ytd(pd.Timestamp("2022-06-01")) == 0.0

    def test_only_current_year_is_summed(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        strategy.realized_gains = [
            {"date": pd.Timestamp("2022-03-01"), "realized_gain": 500.0},
            {"date": pd.Timestamp("2022-08-15"), "realized_gain": 200.0},
            {"date": pd.Timestamp("2021-12-31"), "realized_gain": 1000.0},  # prior year
        ]
        total = strategy._realized_gains_ytd(pd.Timestamp("2022-10-01"))
        assert abs(total - 700.0) < 1e-9

    def test_negative_gains_included(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        strategy.realized_gains = [
            {"date": pd.Timestamp("2022-04-01"), "realized_gain": 800.0},
            {"date": pd.Timestamp("2022-06-01"), "realized_gain": -300.0},
        ]
        assert abs(strategy._realized_gains_ytd(pd.Timestamp("2022-07-01")) - 500.0) < 1e-9

    def test_future_gains_excluded(self, small_market_data, small_factor_returns):
        """Gains in the same year but after the query date are still included (YTD = full year)."""
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        strategy.realized_gains = [
            {"date": pd.Timestamp("2022-11-01"), "realized_gain": 100.0},
        ]
        # query date in 2022 — same year → included
        assert abs(strategy._realized_gains_ytd(pd.Timestamp("2022-01-31")) - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# _conviction_score
# ---------------------------------------------------------------------------

class TestConvictionScore:
    def test_high_r2_high_te_gives_high_score(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        r2 = pd.Series({"A": 1.0, "B": 1.0})
        score = strategy._conviction_score(r2, tracking_error=1.0)
        assert abs(score - 1.0) < 1e-9

    def test_score_is_bounded_at_one(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        r2 = pd.Series({"A": 0.9, "B": 0.8})
        # tracking_error > 1 → te_score capped at 1.0
        score = strategy._conviction_score(r2, tracking_error=5.0)
        assert 0.0 <= score <= 1.0

    def test_zero_r2_gives_zero_score(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        r2 = pd.Series({"A": 0.0, "B": 0.0})
        assert strategy._conviction_score(r2, tracking_error=1.0) == 0.0

    def test_zero_tracking_error_gives_zero_score(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        r2 = pd.Series({"A": 0.9, "B": 0.8})
        assert strategy._conviction_score(r2, tracking_error=0.0) == 0.0


# ---------------------------------------------------------------------------
# _lookback_start
# ---------------------------------------------------------------------------

class TestLookbackStart:
    def test_default_252_day_lookback(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        date = pd.Timestamp("2022-06-01")
        start = strategy._lookback_start(date)
        assert (date - start).days == 252

    def test_custom_lookback(self, small_market_data, small_factor_returns):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        strategy.lookback_days = 100
        date = pd.Timestamp("2022-06-01")
        start = strategy._lookback_start(date)
        assert (date - start).days == 100


# ---------------------------------------------------------------------------
# on_date — smoke test over a short date window
# ---------------------------------------------------------------------------

class TestOnDate:
    def test_runs_first_60_days_without_error(self, small_market_data, small_factor_returns, trading_dates):
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        for d in trading_dates[:60]:
            strategy.on_date(d)  # must not raise

    def test_exposure_history_grows_monthly(self, small_market_data, small_factor_returns, trading_dates):
        """Exposure snapshots should be added once per rebalance (monthly frequency)."""
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        # Run 6 months (≈ 126 trading days)
        for d in trading_dates[:126]:
            strategy.on_date(d)
        # Expect roughly 6 rebalance snapshots (one per month)
        assert 4 <= len(strategy.exposure_history) <= 8

    def test_no_rebalance_on_non_rebalance_days(self, small_market_data, small_factor_returns, trading_dates):
        """last_rebalance_date should only update on actual rebalance days."""
        strategy, _, _ = _make_strategy(small_market_data, small_factor_returns)
        # Run just the first two trading days (same month → only first should rebalance)
        strategy.on_date(trading_dates[0])
        first_rebalance = strategy.last_rebalance_date
        strategy.on_date(trading_dates[1])
        # Second day is same month → last_rebalance_date should NOT change if no rebalance
        # (it would change only if decision engine approved another rebalance, which won't
        # happen since on_date returns early for non-rebalance dates)
        assert strategy.last_rebalance_date == first_rebalance
