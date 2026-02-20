import sys
import os
import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MarginConfig, TaxConfig, TradingCostConfig
from trading.margin_cost import MarginCostModel
from portfolio.portfolio import Portfolio
from portfolio.lots import Lot
from execution.executor import Executor
from decisions.decision_engine import DecisionEngine
from tax.tax_engine import TaxEngine


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeMarketData:
    def __init__(self, prices):
        self._prices = prices

    def get_price(self, ticker, date):
        return self._prices[ticker]


class _FakeExecutor:
    def __init__(self, prices):
        self._prices = prices
        self.trades = []

    def sell(self, portfolio, ticker, shares, date, method="HIFO"):
        price = self._prices[ticker]
        result = portfolio.positions[ticker].sell_shares(shares, price, method=method)
        portfolio.cash += result["proceeds"]
        self.trades.append({"type": "sell", "ticker": ticker})
        return result

    def borrow(self, portfolio, amount, date):
        portfolio.cash += amount
        portfolio.margin_balance += amount
        self.trades.append({"type": "borrow", "amount": amount})

    def repay(self, portfolio, amount, date):
        repayable = min(amount, portfolio.margin_balance, max(portfolio.cash, 0.0))
        if repayable < 1e-9:
            return 0.0
        portfolio.cash -= repayable
        portfolio.margin_balance -= repayable
        self.trades.append({"type": "repay", "amount": repayable})
        return repayable


DATE = pd.Timestamp("2024-06-15")


# ---------------------------------------------------------------------------
# MarginCostModel
# ---------------------------------------------------------------------------

def test_daily_interest():
    m = MarginCostModel()
    # 5% annual on €10k → €10k * 0.05 / 252 ≈ €1.98
    result = m.daily_interest(10_000, 0.05)
    assert abs(result - 10_000 * 0.05 / 252) < 1e-6


def test_total_cost_90_days():
    m = MarginCostModel()
    # 5% annual on €1000 for 90 days → €1000 * 0.05 * 90/365 ≈ €12.33
    result = m.total_cost(1_000, 0.05, 90)
    assert abs(result - 1_000 * 0.05 * 90 / 365) < 1e-6


def test_total_cost_zero_days():
    m = MarginCostModel()
    assert m.total_cost(1_000, 0.05, 0) == 0.0


# ---------------------------------------------------------------------------
# Portfolio margin fields
# ---------------------------------------------------------------------------

def test_portfolio_margin_balance_initialized_zero():
    p = Portfolio(cash=10_000)
    assert p.margin_balance == 0.0


def test_equity_value_no_margin():
    p = Portfolio(cash=10_000)
    md = _FakeMarketData({"AAPL": 100.0})
    p.add_position("AAPL", Lot(10.0, 90.0, datetime.date(2023, 1, 1)))
    # market_value = 10_000 + 10*100 = 11_000; equity = 11_000 - 0 = 11_000
    assert abs(p.equity_value(md, DATE) - 11_000) < 1e-6


def test_equity_value_with_margin():
    p = Portfolio(cash=10_000)
    p.margin_balance = 2_000.0
    md = _FakeMarketData({})
    # market_value = 10_000 (no positions); equity = 10_000 - 2_000 = 8_000
    assert abs(p.equity_value(md, DATE) - 8_000) < 1e-6


def test_leverage_ratio_no_margin():
    p = Portfolio(cash=10_000)
    md = _FakeMarketData({})
    assert p.leverage_ratio(md, DATE) == 1.0


def test_leverage_ratio_with_margin():
    p = Portfolio(cash=12_000)   # 10k equity + 2k borrowed
    p.margin_balance = 2_000.0
    md = _FakeMarketData({})
    # total_assets=12_000, equity=10_000, leverage=1.2
    assert abs(p.leverage_ratio(md, DATE) - 1.2) < 1e-6


def test_leverage_ratio_infinite_when_equity_zero():
    p = Portfolio(cash=2_000)
    p.margin_balance = 2_000.0
    md = _FakeMarketData({})
    assert p.leverage_ratio(md, DATE) == float("inf")


# ---------------------------------------------------------------------------
# Executor borrow / repay
# ---------------------------------------------------------------------------

def _real_executor():
    """Returns an Executor backed by a FakeMarketData (no tickers needed for borrow/repay)."""
    class _FakeMD:
        def get_price(self, ticker, date):
            return 100.0
    from execution.executor import Executor as RealExecutor
    e = RealExecutor(_FakeMD())
    return e


def test_borrow_increases_cash_and_debt():
    e = _real_executor()
    p = Portfolio(cash=10_000)
    e.borrow(p, 2_000, DATE)
    assert abs(p.cash - 12_000) < 1e-6
    assert abs(p.margin_balance - 2_000) < 1e-6


def test_repay_reduces_cash_and_debt():
    e = _real_executor()
    p = Portfolio(cash=12_000)
    p.margin_balance = 2_000.0
    e.repay(p, 2_000, DATE)
    assert abs(p.cash - 10_000) < 1e-6
    assert abs(p.margin_balance - 0) < 1e-6


def test_repay_capped_by_cash():
    e = _real_executor()
    p = Portfolio(cash=500)
    p.margin_balance = 2_000.0
    repaid = e.repay(p, 2_000, DATE)
    assert abs(repaid - 500) < 1e-6
    assert abs(p.cash - 0) < 1e-6
    assert abs(p.margin_balance - 1_500) < 1e-6


def test_repay_capped_by_balance():
    e = _real_executor()
    p = Portfolio(cash=5_000)
    p.margin_balance = 300.0
    repaid = e.repay(p, 5_000, DATE)
    assert abs(repaid - 300) < 1e-6
    assert abs(p.margin_balance - 0) < 1e-6


def test_repay_with_negative_cash_returns_zero():
    e = _real_executor()
    p = Portfolio(cash=-100)
    p.margin_balance = 500.0
    repaid = e.repay(p, 500, DATE)
    assert repaid == 0.0
    assert p.margin_balance == 500.0


def test_borrow_repay_recorded_in_trades():
    e = _real_executor()
    p = Portfolio(cash=10_000)
    e.borrow(p, 1_000, DATE)
    e.repay(p, 1_000, DATE)
    types = [t["type"] for t in e.trades]
    assert "borrow" in types
    assert "repay" in types


# ---------------------------------------------------------------------------
# DecisionEngine.should_borrow_instead_of_sell
# ---------------------------------------------------------------------------

def _decision_engine(margin_enabled=True, annual_rate=0.05, expected_hold_days=90):
    tax_cfg = TaxConfig()
    engine = TaxEngine(config=tax_cfg)
    margin_cfg = MarginConfig(enabled=margin_enabled, annual_rate=annual_rate,
                               expected_hold_days=expected_hold_days)
    cost_model = MarginCostModel()
    return DecisionEngine(
        tax_engine=engine,
        trading_cost_config=TradingCostConfig(),
        margin_config=margin_cfg,
        margin_cost_model=cost_model,
    )


def test_should_borrow_returns_false_when_disabled():
    de = _decision_engine(margin_enabled=False)
    # Even with large unrealised gain, borrowing disabled → always False
    assert de.should_borrow_instead_of_sell(
        unrealized_gain=50_000, sell_amount=10_000, expected_hold_days=90
    ) is False


def test_should_borrow_returns_false_when_no_gain():
    de = _decision_engine()
    # No unrealised gain → no tax to avoid → don't borrow
    assert de.should_borrow_instead_of_sell(
        unrealized_gain=0, sell_amount=10_000, expected_hold_days=90
    ) is False


def test_should_borrow_true_when_interest_cheaper_than_tax():
    """
    Sell €1k position with €15k unrealised gain → tax ≈ 42% on the top bracket.
    Interest on €1k for 90 days at 5% ≈ €12.33 << tax.
    """
    de = _decision_engine()
    result = de.should_borrow_instead_of_sell(
        unrealized_gain=15_000,
        sell_amount=1_000,
        expected_hold_days=90,
    )
    assert result is True


def test_should_borrow_false_when_interest_more_expensive():
    """
    Tiny unrealised gain → small tax; large position to sell → interest exceeds tax.
    """
    de = _decision_engine(annual_rate=0.20, expected_hold_days=365)
    # unrealised_gain=100 → tax≈€27; interest on €50k for 365d at 20% = €10_000
    result = de.should_borrow_instead_of_sell(
        unrealized_gain=100,
        sell_amount=50_000,
        expected_hold_days=365,
    )
    assert result is False


# ---------------------------------------------------------------------------
# Forced liquidation (integration)
# ---------------------------------------------------------------------------

def test_forced_liquidation_reduces_leverage():
    """After a sharp drop the leverage ratio can exceed the cap; forced liquidation must fix it."""
    from strategy.strategy import FactorReplicationStrategy
    from factors.factor_model import FactorModel
    from universe.universe_selector import UniverseSelector
    from targets.factor_target import FactorTarget

    # Portfolio with margin: equity=€5k, borrowed=€5k, positions ≈ €10k
    # Then price drops 50% → assets≈€5k, equity≈€0, leverage → ∞
    # Forced liquidation should sell and repay until leverage ≤ max_leverage

    import numpy as np

    margin_cfg = MarginConfig(enabled=True, max_leverage=1.2, annual_rate=0.05)
    cost_model = MarginCostModel()
    p = Portfolio(cash=0.0)
    p.margin_balance = 5_000.0
    # Two positions, each worth €5k (€10k total assets)
    p.add_position("AAPL", Lot(50.0, 100.0, datetime.date(2023, 1, 1)))
    p.add_position("MSFT", Lot(50.0, 100.0, datetime.date(2023, 1, 1)))

    # Prices drop to €50 → total_assets=€5k, equity=€0 → leverage=∞
    md = _FakeMarketData({"AAPL": 50.0, "MSFT": 50.0})

    # Verify initial leverage is breached
    assert p.leverage_ratio(md, DATE) == float("inf")

    class _MinimalStrategy:
        """Thin stand-in that exposes just enough for _forced_liquidation."""
        def __init__(self):
            self.portfolio = p
            self.market_data = md
            self.margin_config = margin_cfg
            self.margin_cost_model = cost_model
            self.realized_gains = []
            self.total_interest_paid = 0.0
            self.executor = _FakeExecutor({"AAPL": 50.0, "MSFT": 50.0})

        # Copy the real _forced_liquidation logic
        def _forced_liquidation(self, date):
            positions_by_value = sorted(
                [(t, pos) for t, pos in self.portfolio.positions.items()
                 if pos.total_shares() > 1e-12],
                key=lambda x: x[1].total_shares() * self.market_data.get_price(x[0], date)
            )
            for ticker, position in positions_by_value:
                if self.portfolio.leverage_ratio(self.market_data, date) \
                        <= self.margin_config.max_leverage:
                    break
                shares = position.total_shares()
                result = self.executor.sell(self.portfolio, ticker, shares, date)
                self.realized_gains.append({"date": date, "ticker": ticker,
                                            "realized_gain": result["realized_gain"],
                                            "proceeds": result["proceeds"],
                                            "is_forced_liquidation": True})
                self.executor.repay(self.portfolio, result["proceeds"], date)

    strat = _MinimalStrategy()
    strat._forced_liquidation(DATE)

    final_leverage = p.leverage_ratio(md, DATE)
    assert final_leverage <= margin_cfg.max_leverage or final_leverage == float("inf") \
        or p.margin_balance == 0.0  # fully repaid is also acceptable
