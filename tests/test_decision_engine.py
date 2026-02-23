"""
Unit tests for DecisionEngine.should_rebalance and trading cost calculation.

Verifies that TradingCostConfig values (pct_cost, min_cost) are respected and
that the rebalance gate correctly compares expected_improvement against the
combined tax + trading cost.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from config import TaxConfig, TradingCostConfig
from decisions.decision_engine import DecisionEngine
from tax.tax_engine import TaxEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine(pct_cost: float = 0.001, min_cost: float = 1.0) -> DecisionEngine:
    """Return a DecisionEngine with zero-gain tax engine and given cost config."""
    tax_engine = TaxEngine(config=TaxConfig())
    trading_cost_config = TradingCostConfig(pct_cost=pct_cost, min_cost=min_cost)
    return DecisionEngine(tax_engine=tax_engine, trading_cost_config=trading_cost_config)


# ---------------------------------------------------------------------------
# _trading_cost
# ---------------------------------------------------------------------------

class TestTradingCost:
    def test_pct_cost_applied_when_above_min(self):
        de = _engine(pct_cost=0.01, min_cost=1.0)
        # 10_000 * 0.01 = 100 > min_cost 1.0
        assert abs(de._trading_cost(10_000.0) - 100.0) < 1e-9

    def test_min_cost_floor_applied_for_small_trade(self):
        de = _engine(pct_cost=0.001, min_cost=5.0)
        # 100 * 0.001 = 0.10 < min_cost 5.0  → floor kicks in
        assert abs(de._trading_cost(100.0) - 5.0) < 1e-9

    def test_zero_trade_value_returns_min_cost(self):
        de = _engine(pct_cost=0.001, min_cost=2.0)
        assert abs(de._trading_cost(0.0) - 2.0) < 1e-9

    def test_negative_trade_value_uses_abs(self):
        de = _engine(pct_cost=0.01, min_cost=1.0)
        assert abs(de._trading_cost(-10_000.0) - 100.0) < 1e-9

    def test_default_config_used_when_none_injected(self):
        tax_engine = TaxEngine()
        de = DecisionEngine(tax_engine=tax_engine)
        # Default: pct_cost=0.001, min_cost=1.0
        # 2000 * 0.001 = 2.0 > 1.0
        assert abs(de._trading_cost(2_000.0) - 2.0) < 1e-9

    def test_custom_pct_cost_reflected(self):
        de = _engine(pct_cost=0.005, min_cost=1.0)
        # 1000 * 0.005 = 5.0
        assert abs(de._trading_cost(1_000.0) - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# should_rebalance — no tax (zero unrealized gain)
# ---------------------------------------------------------------------------

class TestShouldRebalanceNoTax:
    def test_rebalances_when_improvement_exceeds_trading_cost(self):
        de = _engine(pct_cost=0.001, min_cost=1.0)
        # trade_value=1000 → cost=1.0, improvement=5.0 → approve
        assert de.should_rebalance(
            tracking_error=0.1,
            unrealized_gain=0.0,
            expected_improvement=5.0,
            trade_value=1_000.0,
        )

    def test_does_not_rebalance_when_improvement_equals_cost(self):
        de = _engine(pct_cost=0.001, min_cost=1.0)
        # trade_value=1000 → cost=1.0, improvement=1.0 → not strictly greater
        assert not de.should_rebalance(
            tracking_error=0.1,
            unrealized_gain=0.0,
            expected_improvement=1.0,
            trade_value=1_000.0,
        )

    def test_does_not_rebalance_when_improvement_below_cost(self):
        de = _engine(pct_cost=0.01, min_cost=1.0)
        # trade_value=1000 → cost=10.0, improvement=5.0 → reject
        assert not de.should_rebalance(
            tracking_error=0.5,
            unrealized_gain=0.0,
            expected_improvement=5.0,
            trade_value=1_000.0,
        )

    def test_min_cost_floor_affects_gate(self):
        de = _engine(pct_cost=0.001, min_cost=20.0)
        # pct would be 0.1 but floor is 20.0; improvement=10.0 → reject
        assert not de.should_rebalance(
            tracking_error=0.1,
            unrealized_gain=0.0,
            expected_improvement=10.0,
            trade_value=100.0,
        )

    def test_zero_trade_value_uses_min_cost(self):
        de = _engine(pct_cost=0.001, min_cost=3.0)
        # trade_value=0 → cost=min_cost=3.0; improvement=5.0 → approve
        assert de.should_rebalance(
            tracking_error=0.0,
            unrealized_gain=0.0,
            expected_improvement=5.0,
            trade_value=0.0,
        )


# ---------------------------------------------------------------------------
# should_rebalance — with tax
# ---------------------------------------------------------------------------

class TestShouldRebalanceWithTax:
    def test_tax_added_to_trading_cost(self):
        de = _engine(pct_cost=0.001, min_cost=1.0)
        # unrealized_gain=5000 → tax = 5000 * 0.27 = 1350
        # trade_value=1000 → trading_cost = 1.0
        # total cost = 1351.0; improvement must exceed that
        assert not de.should_rebalance(
            tracking_error=0.5,
            unrealized_gain=5_000.0,
            expected_improvement=1_000.0,
            trade_value=1_000.0,
        )

    def test_rebalances_when_improvement_exceeds_tax_plus_trading_cost(self):
        de = _engine(pct_cost=0.001, min_cost=1.0)
        # unrealized_gain=100 → tax = 100 * 0.27 = 27.0
        # trade_value=1000 → trading_cost = 1.0; total = 28.0
        assert de.should_rebalance(
            tracking_error=0.5,
            unrealized_gain=100.0,
            expected_improvement=50.0,
            trade_value=1_000.0,
        )

    def test_negative_gain_no_tax(self):
        de = _engine(pct_cost=0.001, min_cost=1.0)
        # unrealized_gain=-500 → tax=0; trading_cost=1.0; improvement=2.0 → approve
        assert de.should_rebalance(
            tracking_error=0.1,
            unrealized_gain=-500.0,
            expected_improvement=2.0,
            trade_value=1_000.0,
        )
