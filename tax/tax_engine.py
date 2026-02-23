from __future__ import annotations

from typing import Optional

from config import TaxConfig


class TaxEngine:
    def __init__(self, config: Optional[TaxConfig] = None) -> None:
        if config is None:
            config = TaxConfig()
        self.config = config

    def tax_due(self, realized_gain: float) -> float:
        if realized_gain <= 0:
            return 0.0
        remaining = realized_gain
        tax = 0.0
        prev_limit = 0.0
        for upper_limit, rate in self.config.brackets:
            band = min(remaining, upper_limit - prev_limit)
            if band <= 0:
                break
            tax += band * rate
            remaining -= band
            prev_limit = upper_limit
            if remaining <= 0:
                break
        return tax
