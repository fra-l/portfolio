from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaxConfig:
    # Progressive rate brackets: list of (upper_limit, rate) pairs.
    # The last entry's upper_limit must be float("inf").
    # Default: Danish capital gains rates — 27% up to €10k, 42% above.
    brackets: list[tuple[float, float]] = field(default_factory=lambda: [
        (10_000.0, 0.27),
        (float("inf"), 0.42),
    ])

    # Tax harvesting settings
    harvest_enabled: bool = True
    harvest_months: list[int] = field(default_factory=lambda: [11, 12])  # Nov–Dec
    min_loss_threshold: float = 50.0        # Minimum unrealized loss (€) worth harvesting
    wash_sale_waiting_days: int = 30        # Days before repurchasing the same ticker
    max_harvest_per_year: float = float("inf")  # Cap on losses harvested annually (€)


@dataclass
class TradingCostConfig:
    pct_cost: float = 0.001
    min_cost: float = 1.0

@dataclass
class MarginConfig:
    enabled: bool = False
    max_leverage: float = 1.2
    annual_rate: float = 0.05
    conviction_threshold: float = 0.8   # min [0,1] conviction score to trigger borrowing
    expected_hold_days: int = 90        # horizon for tax-aware borrow vs. sell comparison


@dataclass
class BacktestConfig:
    # Data
    start: str = "2020-01-01"
    lookback_days: int = 252
    # Universe
    regions: list[str] = field(default_factory=lambda: ["US", "Europe", "Asia-Pacific"])
    cap_tiers: list[str] = field(default_factory=lambda: ["mega", "large"])
    min_adv: float = 1e8
    # Factor model
    min_r2: float = 0.3
    # Target
    target_weights: dict[str, float] = field(default_factory=lambda: {"Value": 0.6, "Momentum": 0.4})
    # Strategy
    rebalance_frequency: str = "M"
    # Portfolio
    initial_cash: float = 20_000.0
    # Benchmarks — any Yahoo Finance ticker(s) to compare against; first is the primary
    benchmark_tickers: list[str] = field(default_factory=lambda: ["SPY"])
    # Sub-configs (use defaults if not set)
    tax_config: TaxConfig = field(default_factory=TaxConfig)
    trading_cost_config: TradingCostConfig = field(default_factory=TradingCostConfig)
    margin_config: MarginConfig = field(default_factory=MarginConfig)
