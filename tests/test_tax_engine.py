import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import TaxConfig
from tax.tax_engine import TaxEngine


def _engine(brackets):
    cfg = TaxConfig(brackets=brackets)
    return TaxEngine(config=cfg)


# --- zero / negative gain ---

def test_zero_gain():
    engine = TaxEngine()
    assert engine.tax_due(0) == 0.0


def test_negative_gain_returns_zero():
    engine = TaxEngine()
    assert engine.tax_due(-500) == 0.0


# --- single bracket (flat rate) ---

def test_single_bracket_flat():
    engine = _engine([(float("inf"), 0.30)])
    assert engine.tax_due(1_000) == 300.0
    assert engine.tax_due(50_000) == 15_000.0


# --- two brackets (Danish default) ---

def test_danish_default_below_threshold():
    engine = TaxEngine()  # default Danish config
    assert engine.tax_due(5_000) == 5_000 * 0.27


def test_danish_default_exactly_at_threshold():
    engine = TaxEngine()
    assert engine.tax_due(10_000) == 10_000 * 0.27


def test_danish_default_above_threshold():
    engine = TaxEngine()
    gain = 15_000
    expected = 10_000 * 0.27 + 5_000 * 0.42
    assert abs(engine.tax_due(gain) - expected) < 1e-9


def test_custom_two_bracket():
    engine = _engine([(20_000.0, 0.20), (float("inf"), 0.35)])
    # 20k at 20%, 5k at 35%
    expected = 20_000 * 0.20 + 5_000 * 0.35
    assert abs(engine.tax_due(25_000) - expected) < 1e-9


# --- three brackets ---

def test_three_brackets_all_bands():
    engine = _engine([
        (10_000.0, 0.10),
        (50_000.0, 0.25),
        (float("inf"), 0.45),
    ])
    # 10k @ 10% + 40k @ 25% + 10k @ 45%
    expected = 10_000 * 0.10 + 40_000 * 0.25 + 10_000 * 0.45
    assert abs(engine.tax_due(60_000) - expected) < 1e-9


def test_three_brackets_only_first_band():
    engine = _engine([
        (10_000.0, 0.10),
        (50_000.0, 0.25),
        (float("inf"), 0.45),
    ])
    assert abs(engine.tax_due(5_000) - 500.0) < 1e-9


def test_three_brackets_first_two_bands():
    engine = _engine([
        (10_000.0, 0.10),
        (50_000.0, 0.25),
        (float("inf"), 0.45),
    ])
    # 10k @ 10% + 10k @ 25%
    expected = 10_000 * 0.10 + 10_000 * 0.25
    assert abs(engine.tax_due(20_000) - expected) < 1e-9


# --- TaxConfig harvesting defaults ---

def test_taxconfig_default_harvest_fields():
    cfg = TaxConfig()
    assert cfg.harvest_enabled is True
    assert cfg.harvest_months == [11, 12]
    assert cfg.min_loss_threshold == 50.0
    assert cfg.wash_sale_waiting_days == 30
    assert cfg.max_harvest_per_year == float("inf")


def test_taxconfig_custom_harvest_fields():
    cfg = TaxConfig(
        harvest_enabled=False,
        harvest_months=[12],
        min_loss_threshold=200.0,
        wash_sale_waiting_days=60,
        max_harvest_per_year=5_000.0,
    )
    assert cfg.harvest_enabled is False
    assert cfg.harvest_months == [12]
    assert cfg.min_loss_threshold == 200.0
    assert cfg.wash_sale_waiting_days == 60
    assert cfg.max_harvest_per_year == 5_000.0
