"""
Unit tests for Executor.

Covers:
- buy() creates a Lot with the correct shares and cost basis
- buy() decrements portfolio cash by the invested amount
- buy() logs the trade correctly
- Fractional share calculation in buy()
- sell() with a single lot: correct realized_gain
- sell() HIFO with multiple lots: highest cost basis consumed first
- sell() partial sell: residual shares remain in the position
- Multiple lots across a partial sell (multi-lot HIFO scenario)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from execution.executor import Executor
from portfolio.lots import Lot
from portfolio.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Minimal stub so tests don't need a real MarketData object
# ---------------------------------------------------------------------------

class _FakeMarketData:
    """Stub that returns a configurable fixed price for any ticker/date."""

    def __init__(self, price: float):
        self._price = price

    def get_price(self, ticker, date):
        return self._price


def _make_executor(price: float):
    return Executor(_FakeMarketData(price))


# ---------------------------------------------------------------------------
# buy()
# ---------------------------------------------------------------------------

class TestBuy:
    def test_creates_lot_with_correct_shares(self):
        portfolio = Portfolio(cash=1_000.0)
        executor = _make_executor(price=100.0)
        executor.buy(portfolio, "AAPL", 500.0, pd.Timestamp("2022-01-03"))
        position = portfolio.positions["AAPL"]
        assert len(position.lots) == 1
        assert position.lots[0].shares == pytest.approx(5.0)

    def test_creates_lot_with_correct_cost_basis(self):
        portfolio = Portfolio(cash=1_000.0)
        executor = _make_executor(price=200.0)
        executor.buy(portfolio, "MSFT", 400.0, pd.Timestamp("2022-01-03"))
        lot = portfolio.positions["MSFT"].lots[0]
        assert lot.cost_basis == pytest.approx(200.0)

    def test_decrements_cash_by_invested_amount(self):
        portfolio = Portfolio(cash=1_000.0)
        executor = _make_executor(price=50.0)
        executor.buy(portfolio, "GOOG", 300.0, pd.Timestamp("2022-01-03"))
        assert portfolio.cash == pytest.approx(700.0)

    def test_fractional_shares_computed_correctly(self):
        """euro_amount / price should yield fractional shares when not evenly divisible."""
        portfolio = Portfolio(cash=1_000.0)
        executor = _make_executor(price=75.0)
        executor.buy(portfolio, "AMZN", 100.0, pd.Timestamp("2022-01-03"))
        expected_shares = 100.0 / 75.0
        assert portfolio.positions["AMZN"].lots[0].shares == pytest.approx(expected_shares)

    def test_trade_logged_correctly(self):
        portfolio = Portfolio(cash=2_000.0)
        executor = _make_executor(price=100.0)
        date = pd.Timestamp("2022-02-01")
        executor.buy(portfolio, "AAPL", 1_000.0, date)
        assert len(executor.trades) == 1
        trade = executor.trades[0]
        assert trade["type"] == "buy"
        assert trade["ticker"] == "AAPL"
        assert trade["price"] == pytest.approx(100.0)
        assert trade["amount"] == pytest.approx(1_000.0)
        assert trade["date"] == date

    def test_multiple_buys_accumulate_lots(self):
        portfolio = Portfolio(cash=5_000.0)
        executor = _make_executor(price=100.0)
        date = pd.Timestamp("2022-01-03")
        executor.buy(portfolio, "AAPL", 1_000.0, date)
        executor.buy(portfolio, "AAPL", 500.0, date)
        assert len(portfolio.positions["AAPL"].lots) == 2


# ---------------------------------------------------------------------------
# sell()
# ---------------------------------------------------------------------------

class TestSell:
    def _portfolio_with_lot(self, cost_basis: float, shares: float, price_now: float):
        """Helper: portfolio with one AAPL lot, executor whose price equals price_now."""
        portfolio = Portfolio(cash=0.0)
        portfolio.add_position("AAPL", Lot(shares=shares, cost_basis=cost_basis,
                                           purchase_date=pd.Timestamp("2022-01-03")))
        executor = _make_executor(price=price_now)
        return portfolio, executor

    def test_single_lot_realized_gain(self):
        """Selling shares at a higher price than cost basis produces correct gain."""
        portfolio, executor = self._portfolio_with_lot(cost_basis=100.0, shares=10.0, price_now=150.0)
        result = executor.sell(portfolio, "AAPL", 10.0, pd.Timestamp("2022-06-01"))
        assert result["realized_gain"] == pytest.approx(10.0 * (150.0 - 100.0))

    def test_single_lot_proceeds_added_to_cash(self):
        portfolio, executor = self._portfolio_with_lot(cost_basis=100.0, shares=5.0, price_now=120.0)
        executor.sell(portfolio, "AAPL", 5.0, pd.Timestamp("2022-06-01"))
        assert portfolio.cash == pytest.approx(5.0 * 120.0)

    def test_sell_logs_trade(self):
        portfolio, executor = self._portfolio_with_lot(cost_basis=100.0, shares=3.0, price_now=110.0)
        executor.sell(portfolio, "AAPL", 3.0, pd.Timestamp("2022-06-01"))
        assert len(executor.trades) == 1
        assert executor.trades[0]["type"] == "sell"

    def test_hifo_selects_highest_cost_basis_first(self):
        """HIFO must consume the lot with the highest cost basis before cheaper lots."""
        portfolio = Portfolio(cash=0.0)
        # Lot A: cheap (cost_basis=50), Lot B: expensive (cost_basis=200)
        portfolio.add_position("AAPL", Lot(shares=5.0, cost_basis=50.0,
                                           purchase_date=pd.Timestamp("2021-01-01")))
        portfolio.add_position("AAPL", Lot(shares=5.0, cost_basis=200.0,
                                           purchase_date=pd.Timestamp("2022-01-01")))
        executor = _make_executor(price=120.0)

        # Sell 5 shares — HIFO should consume the 200-basis lot first (loss of 80/share)
        result = executor.sell(portfolio, "AAPL", 5.0, pd.Timestamp("2022-06-01"))
        expected_gain = 5.0 * (120.0 - 200.0)
        assert result["realized_gain"] == pytest.approx(expected_gain)

    def test_hifo_partial_sell_leaves_residual_shares(self):
        """Selling fewer shares than the lot contains leaves the rest in place."""
        portfolio = Portfolio(cash=0.0)
        portfolio.add_position("AAPL", Lot(shares=10.0, cost_basis=100.0,
                                           purchase_date=pd.Timestamp("2022-01-01")))
        executor = _make_executor(price=130.0)
        executor.sell(portfolio, "AAPL", 6.0, pd.Timestamp("2022-06-01"))
        remaining = portfolio.positions["AAPL"].total_shares()
        assert remaining == pytest.approx(4.0)

    def test_hifo_multi_lot_partial_sell(self):
        """
        Multi-lot HIFO scenario: three lots with different cost bases.
        Selling enough shares to exhaust the most expensive lot and partially
        consume the next must produce the correct blended realized gain.
        """
        portfolio = Portfolio(cash=0.0)
        # Lot ordering (by cost basis, descending): C=300, B=200, A=100
        portfolio.add_position("AAPL", Lot(shares=4.0, cost_basis=100.0,
                                           purchase_date=pd.Timestamp("2020-01-01")))  # Lot A
        portfolio.add_position("AAPL", Lot(shares=4.0, cost_basis=200.0,
                                           purchase_date=pd.Timestamp("2021-01-01")))  # Lot B
        portfolio.add_position("AAPL", Lot(shares=4.0, cost_basis=300.0,
                                           purchase_date=pd.Timestamp("2022-01-01")))  # Lot C

        current_price = 250.0
        executor = _make_executor(price=current_price)

        # Sell 6 shares: exhaust Lot C (4 shares at cb=300) + 2 shares from Lot B (cb=200)
        result = executor.sell(portfolio, "AAPL", 6.0, pd.Timestamp("2022-06-01"))
        expected_gain = 4.0 * (current_price - 300.0) + 2.0 * (current_price - 200.0)
        assert result["realized_gain"] == pytest.approx(expected_gain)
        assert result["shares_sold"] == pytest.approx(6.0)
        # Remaining: 2 shares from Lot B + 4 shares from Lot A = 6 shares
        assert portfolio.positions["AAPL"].total_shares() == pytest.approx(6.0)
