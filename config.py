from dataclasses import dataclass

@dataclass
class TaxConfig:
    lower_rate: float = 0.27
    upper_rate: float = 0.42
    threshold: float = 10_000.0

@dataclass
class TradingCostConfig:
    pct_cost: float = 0.001
    min_cost: float = 1.0

@dataclass
class MarginConfig:
    enabled: bool = False
    max_leverage: float = 1.2
    annual_rate: float = 0.05
