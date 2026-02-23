from __future__ import annotations

from typing import Any, Optional

from config import TradingCostConfig


class DecisionEngine:
    def __init__(
        self,
        tax_engine: Any,
        trading_cost_config: Optional[TradingCostConfig] = None,
        margin_config: Any = None,
        margin_cost_model: Any = None,
    ) -> None:
        self.tax_engine = tax_engine
        if trading_cost_config is None:
            trading_cost_config = TradingCostConfig()
        self.trading_cost_config = trading_cost_config
        self.margin_config = margin_config
        self.margin_cost_model = margin_cost_model

    def _trading_cost(self, trade_value: float) -> float:
        return max(abs(trade_value) * self.trading_cost_config.pct_cost, self.trading_cost_config.min_cost)

    def should_rebalance(
        self,
        tracking_error: float,
        unrealized_gain: float,
        expected_improvement: float,
        trade_value: float = 0.0,
    ) -> bool:
        cost = self.tax_engine.tax_due(unrealized_gain) + self._trading_cost(trade_value)
        return expected_improvement > cost

    def should_borrow_instead_of_sell(
        self,
        unrealized_gain: float,
        sell_amount: float,
        expected_hold_days: int,
    ) -> bool:
        """
        Return True if the interest cost of borrowing `sell_amount` for `expected_hold_days`
        is less than the capital gains tax that would be triggered by selling.
        Only active when MarginConfig.enabled is True.
        """
        if self.margin_config is None or not self.margin_config.enabled:
            return False
        if self.margin_cost_model is None:
            return False
        tax_cost = self.tax_engine.tax_due(unrealized_gain)
        if tax_cost <= 0:
            return False
        interest_cost = self.margin_cost_model.total_cost(
            sell_amount, self.margin_config.annual_rate, expected_hold_days
        )
        return interest_cost < tax_cost
