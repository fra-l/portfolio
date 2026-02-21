"""
Integration test: runs the full pipeline end-to-end on 2 years of synthetic
price/factor data.  No yfinance calls are made — all data comes from the
conftest fixtures.

Assertions:
  - Portfolio value is positive throughout
  - Realized gains list is maintained
  - Trades are recorded
  - Any harvest events that fire are in November or December
  - Exposure history is captured at each rebalance
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from backtest.engine import BacktestEngine
from config import MarginConfig, TaxConfig, TradingCostConfig
from decisions.decision_engine import DecisionEngine
from execution.executor import Executor
from factors.factor_model import FactorModel
from portfolio.portfolio import Portfolio
from strategy.strategy import FactorReplicationStrategy
from targets.factor_target import FactorTarget
from tax.tax_engine import TaxEngine
from tax.tax_harvesting import TaxHarvestingEngine
from trading.margin_cost import MarginCostModel
from universe.universe_selector import UniverseSelector


# ---------------------------------------------------------------------------
# Module-scoped backtest fixture (runs once for all tests in this file)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backtest_result(small_market_data, small_factor_returns, small_spy_prices):
    """
    Run the full pipeline once and return (backtest_engine, strategy, executor).
    Uses min_r2=0.1 (relaxed) so all synthetic tickers pass the universe filter.
    """
    tax_config = TaxConfig()
    tax_engine = TaxEngine(config=tax_config)
    margin_config = MarginConfig(enabled=False)
    margin_cost_model = MarginCostModel()
    trading_cost_config = TradingCostConfig()
    decision_engine = DecisionEngine(
        tax_engine=tax_engine,
        trading_cost_config=trading_cost_config,
        margin_config=margin_config,
        margin_cost_model=margin_cost_model,
    )
    portfolio = Portfolio(cash=20_000.0)
    executor = Executor(small_market_data)
    harvester = TaxHarvestingEngine(
        config=tax_config,
        tax_engine=tax_engine,
        executor=executor,
    )
    factor_model = FactorModel(small_factor_returns)
    universe_selector = UniverseSelector(min_r2=0.1)
    target = FactorTarget({"Value": 0.6, "Momentum": 0.4})

    strategy = FactorReplicationStrategy(
        market_data=small_market_data,
        factor_model=factor_model,
        universe_selector=universe_selector,
        target=target,
        portfolio=portfolio,
        decision_engine=decision_engine,
        executor=executor,
        harvester=harvester,
        margin_config=margin_config,
        margin_cost_model=margin_cost_model,
    )
    backtest = BacktestEngine(portfolio=portfolio, strategy=strategy)
    backtest.run(small_market_data.prices.index)
    return backtest, strategy, executor


# ---------------------------------------------------------------------------
# Pipeline integrity
# ---------------------------------------------------------------------------

class TestPipelineRuns:
    def test_history_is_populated(self, backtest_result):
        backtest, _, _ = backtest_result
        assert len(backtest.history) > 0

    def test_history_length_matches_price_dates(self, backtest_result, small_market_data):
        backtest, _, _ = backtest_result
        assert len(backtest.history) == len(small_market_data.prices.index)

    def test_portfolio_value_positive(self, backtest_result):
        backtest, _, _ = backtest_result
        final_value = backtest.history[-1]["value"]
        assert final_value > 0, f"Final portfolio value should be positive, got {final_value}"

    def test_all_daily_values_positive(self, backtest_result):
        backtest, _, _ = backtest_result
        for entry in backtest.history:
            assert entry["value"] > 0, f"Negative portfolio value on {entry['date']}: {entry['value']}"

    def test_history_entry_schema(self, backtest_result):
        backtest, _, _ = backtest_result
        for entry in backtest.history:
            assert "date" in entry
            assert "value" in entry
            assert "leverage" in entry
            assert "allocations" in entry
            assert "cash" in entry["allocations"]

    def test_leverage_without_margin_is_one(self, backtest_result):
        backtest, _, _ = backtest_result
        for entry in backtest.history:
            assert abs(entry["leverage"] - 1.0) < 1e-6, (
                f"Leverage should be 1.0 without margin, got {entry['leverage']} on {entry['date']}"
            )

    def test_positions_non_negative_shares(self, backtest_result):
        backtest, _, _ = backtest_result
        for ticker, position in backtest.portfolio.positions.items():
            assert position.total_shares() >= 0, f"{ticker} has negative shares"


# ---------------------------------------------------------------------------
# Realized gains tracking
# ---------------------------------------------------------------------------

class TestRealizedGains:
    def test_realized_gains_is_list(self, backtest_result):
        _, strategy, _ = backtest_result
        assert isinstance(strategy.realized_gains, list)

    def test_realized_gain_record_schema(self, backtest_result):
        _, strategy, _ = backtest_result
        for record in strategy.realized_gains:
            assert "date" in record
            assert "ticker" in record
            assert "realized_gain" in record

    def test_trades_were_executed(self, backtest_result):
        _, _, executor = backtest_result
        assert len(executor.trades) > 0, "At least some trades should have been executed"

    def test_buy_trades_have_positive_amounts(self, backtest_result):
        _, _, executor = backtest_result
        buys = [t for t in executor.trades if t["type"] == "buy"]
        assert len(buys) > 0, "At least one buy trade should exist"
        for t in buys:
            assert t["amount"] > 0, f"Buy amount must be positive, got {t['amount']}"


# ---------------------------------------------------------------------------
# Tax-loss harvest events
# ---------------------------------------------------------------------------

class TestHarvestEvents:
    def test_harvest_events_only_in_nov_dec(self, backtest_result):
        """Any harvest events that fired must fall in November or December."""
        _, strategy, _ = backtest_result
        harvest_events = [r for r in strategy.realized_gains if r.get("is_harvest")]
        for event in harvest_events:
            assert event["date"].month in {11, 12}, (
                f"Harvest event on {event['date']} is not in Nov/Dec"
            )

    def test_harvest_events_have_non_negative_tax_saved(self, backtest_result):
        _, strategy, _ = backtest_result
        harvest_events = [r for r in strategy.realized_gains if r.get("is_harvest")]
        for event in harvest_events:
            assert event.get("tax_saved", 0) >= 0, (
                f"tax_saved must be non-negative, got {event.get('tax_saved')}"
            )


# ---------------------------------------------------------------------------
# Factor exposure history
# ---------------------------------------------------------------------------

class TestExposureHistory:
    def test_exposure_history_populated(self, backtest_result):
        _, strategy, _ = backtest_result
        assert len(strategy.exposure_history) > 0, (
            "strategy.exposure_history should be populated at each rebalance date"
        )

    def test_exposure_history_schema(self, backtest_result):
        _, strategy, _ = backtest_result
        for snap in strategy.exposure_history:
            assert "date" in snap
            assert "exposure" in snap
            assert "factors" in snap
            assert len(snap["exposure"]) == len(snap["factors"])

    def test_exposure_entries_match_factor_names(self, backtest_result):
        _, strategy, _ = backtest_result
        for snap in strategy.exposure_history:
            for name in snap["factors"]:
                assert isinstance(name, str)
            for val in snap["exposure"]:
                assert isinstance(val, float)
