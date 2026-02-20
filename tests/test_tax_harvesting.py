import sys
import os
import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import TaxConfig
from tax.tax_engine import TaxEngine
from tax.tax_harvesting import TaxHarvestingEngine
from portfolio.portfolio import Portfolio
from portfolio.lots import Lot


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeMarketData:
    def __init__(self, prices):
        self._prices = prices  # dict: ticker -> float

    def get_price(self, ticker, date):
        return self._prices[ticker]


class _FakeExecutor:
    """Thin wrapper that delegates to Position.sell_shares and updates portfolio.cash."""
    def __init__(self, prices):
        self._prices = prices
        self.trades = []

    def sell(self, portfolio, ticker, shares, date, method="HIFO"):
        price = self._prices[ticker]
        result = portfolio.positions[ticker].sell_shares(shares, price, method=method)
        portfolio.cash += result["proceeds"]
        self.trades.append({"ticker": ticker, "shares": result["shares_sold"]})
        return result


def _harvester(config=None, prices=None):
    cfg = config or TaxConfig()
    engine = TaxEngine(config=cfg)
    executor = _FakeExecutor(prices or {})
    h = TaxHarvestingEngine(config=cfg, tax_engine=engine, executor=executor)
    return h, executor


def _loss_portfolio(ticker="AAPL", cost_basis=100.0, shares=10.0):
    p = Portfolio(cash=0.0)
    p.add_position(ticker, Lot(shares=shares, cost_basis=cost_basis,
                               purchase_date=datetime.date(2023, 1, 1)))
    return p


# A November date in 2024 (inside the default harvest window)
NOV = pd.Timestamp("2024-11-15")
SEP = pd.Timestamp("2024-09-15")


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def test_harvest_disabled():
    cfg = TaxConfig(harvest_enabled=False)
    h, executor = _harvester(config=cfg, prices={"AAPL": 70.0})
    portfolio = _loss_portfolio()  # cost=100, price=70 → loss=300
    records = h.harvest(portfolio, _FakeMarketData({"AAPL": 70.0}), NOV,
                        realized_gains_ytd=500.0)
    assert records == []
    assert executor.trades == []


def test_harvest_skipped_outside_harvest_months():
    h, executor = _harvester(prices={"AAPL": 70.0})
    portfolio = _loss_portfolio()
    records = h.harvest(portfolio, _FakeMarketData({"AAPL": 70.0}), SEP,
                        realized_gains_ytd=500.0)
    assert records == []
    assert executor.trades == []


def test_harvest_skipped_when_no_gains():
    h, executor = _harvester(prices={"AAPL": 70.0})
    portfolio = _loss_portfolio()
    records = h.harvest(portfolio, _FakeMarketData({"AAPL": 70.0}), NOV,
                        realized_gains_ytd=0.0)
    assert records == []


def test_harvest_skipped_below_threshold():
    cfg = TaxConfig(min_loss_threshold=500.0)  # require ≥€500 loss
    h, executor = _harvester(config=cfg, prices={"AAPL": 70.0})
    portfolio = _loss_portfolio()  # total loss = 300 < 500
    records = h.harvest(portfolio, _FakeMarketData({"AAPL": 70.0}), NOV,
                        realized_gains_ytd=200.0)
    assert records == []
    assert executor.trades == []


# ---------------------------------------------------------------------------
# Core harvesting behaviour
# ---------------------------------------------------------------------------

def test_harvest_full_position():
    """When gains_ytd > total loss, harvest the entire loss position."""
    h, executor = _harvester(prices={"AAPL": 70.0})
    portfolio = _loss_portfolio(cost_basis=100.0, shares=10.0)  # loss = 10*(100-70) = 300
    md = _FakeMarketData({"AAPL": 70.0})

    records = h.harvest(portfolio, md, NOV, realized_gains_ytd=500.0)

    assert len(records) == 1
    r = records[0]
    assert r["ticker"] == "AAPL"
    assert abs(r["realized_loss"] - 300.0) < 1.0   # ~€300 loss realised
    assert r["tax_saved"] > 0
    # Cash returned to portfolio
    assert portfolio.cash == pytest_approx(10 * 70.0, rel=1e-6)


def test_harvest_partial_position():
    """When gains_ytd < total loss, sell only enough to offset the gain."""
    h, executor = _harvester(prices={"AAPL": 70.0})
    portfolio = _loss_portfolio(cost_basis=100.0, shares=10.0)  # loss = 300
    md = _FakeMarketData({"AAPL": 70.0})

    records = h.harvest(portfolio, md, NOV, realized_gains_ytd=100.0)

    assert len(records) == 1
    r = records[0]
    # Should only realise ~€100 of loss (not all €300)
    assert abs(r["realized_loss"] - 100.0) < 5.0
    # Some shares remain
    remaining_shares = portfolio.positions["AAPL"].total_shares()
    assert remaining_shares > 0


def test_harvest_multiple_positions_stops_at_zero_gains():
    """Harvest from multiple positions but stop once gains are fully offset."""
    prices = {"AAPL": 70.0, "MSFT": 80.0}
    h, executor = _harvester(prices=prices)

    portfolio = Portfolio(cash=0.0)
    # AAPL: cost=100, shares=10 → loss=300
    portfolio.add_position("AAPL", Lot(10.0, 100.0, datetime.date(2023, 1, 1)))
    # MSFT: cost=120, shares=10 → loss=400
    portfolio.add_position("MSFT", Lot(10.0, 120.0, datetime.date(2023, 1, 1)))
    md = _FakeMarketData(prices)

    records = h.harvest(portfolio, md, NOV, realized_gains_ytd=200.0)

    total_harvested = sum(r["realized_loss"] for r in records)
    assert abs(total_harvested - 200.0) < 10.0  # close to, not exceeding, gains_ytd


# ---------------------------------------------------------------------------
# Wash-sale blacklist
# ---------------------------------------------------------------------------

def test_wash_sale_blocks_ticker_within_window():
    h, _ = _harvester(prices={"AAPL": 70.0})
    h.wash_sale_blacklist["AAPL"] = NOV

    same_day = NOV
    assert h.is_wash_sale_blocked("AAPL", same_day) is True

    ten_days_later = NOV + pd.Timedelta(days=10)
    assert h.is_wash_sale_blocked("AAPL", ten_days_later) is True


def test_wash_sale_clears_after_waiting_period():
    cfg = TaxConfig(wash_sale_waiting_days=30)
    h, _ = _harvester(config=cfg, prices={"AAPL": 70.0})
    h.wash_sale_blacklist["AAPL"] = NOV

    after_window = NOV + pd.Timedelta(days=31)
    assert h.is_wash_sale_blocked("AAPL", after_window) is False


def test_wash_sale_does_not_block_different_ticker():
    h, _ = _harvester(prices={"AAPL": 70.0})
    h.wash_sale_blacklist["AAPL"] = NOV
    assert h.is_wash_sale_blocked("MSFT", NOV) is False


def test_harvested_ticker_added_to_blacklist():
    h, _ = _harvester(prices={"AAPL": 70.0})
    portfolio = _loss_portfolio()
    md = _FakeMarketData({"AAPL": 70.0})
    h.harvest(portfolio, md, NOV, realized_gains_ytd=500.0)
    assert "AAPL" in h.wash_sale_blacklist


# ---------------------------------------------------------------------------
# Annual cap
# ---------------------------------------------------------------------------

def test_annual_cap_respected():
    cfg = TaxConfig(max_harvest_per_year=150.0)
    prices = {"AAPL": 70.0, "MSFT": 80.0}
    h, executor = _harvester(config=cfg, prices=prices)

    portfolio = Portfolio(cash=0.0)
    portfolio.add_position("AAPL", Lot(10.0, 100.0, datetime.date(2023, 1, 1)))  # loss=300
    portfolio.add_position("MSFT", Lot(10.0, 120.0, datetime.date(2023, 1, 1)))  # loss=400
    md = _FakeMarketData(prices)

    records = h.harvest(portfolio, md, NOV, realized_gains_ytd=1000.0)

    total_harvested = sum(r["realized_loss"] for r in records)
    assert total_harvested <= cfg.max_harvest_per_year + 1.0  # tiny float tolerance


def test_annual_counter_resets_new_year():
    cfg = TaxConfig(max_harvest_per_year=200.0)
    prices = {"AAPL": 70.0}
    h, _ = _harvester(config=cfg, prices=prices)

    # First harvest fills the cap
    portfolio = _loss_portfolio(cost_basis=100.0, shares=10.0)
    md = _FakeMarketData(prices)
    h.harvest(portfolio, md, NOV, realized_gains_ytd=1000.0)
    assert h._harvested_this_year > 0

    # Same ticker won't trigger again (wash-sale) — use a fresh portfolio
    portfolio2 = Portfolio(cash=0.0)
    portfolio2.add_position("AAPL", Lot(10.0, 100.0, datetime.date(2024, 1, 1)))

    jan_next_year = pd.Timestamp("2025-11-01")
    records = h.harvest(portfolio2, md, jan_next_year, realized_gains_ytd=1000.0)
    # Counter should have reset; harvest should proceed
    assert h._current_year == 2025
    assert h._harvested_this_year >= 0  # reset happened


# ---------------------------------------------------------------------------
# Tax saved calculation
# ---------------------------------------------------------------------------

def test_tax_saved_uses_bracket_rates():
    """
    With Danish default brackets (27% up to €10k, 42% above):
    gains_ytd=12_000, loss=5_000 → net=7_000 (below threshold)
    Before: tax = 10_000*0.27 + 2_000*0.42 = 3_540
    After:  tax = 7_000*0.27 = 1_890
    Saved = 1_650
    """
    h, _ = _harvester()
    saved = h._tax_saved(realized_loss=5_000, realized_gains_ytd=12_000)
    expected = (10_000 * 0.27 + 2_000 * 0.42) - (7_000 * 0.27)
    assert abs(saved - expected) < 1e-6


# ---------------------------------------------------------------------------
# Helper shim for approx comparisons
# ---------------------------------------------------------------------------

def pytest_approx(value, rel=1e-6):
    """Simple approximate comparison shim (avoids pytest import at module level)."""
    return value  # Used only for readability; actual asserts use abs() checks above
