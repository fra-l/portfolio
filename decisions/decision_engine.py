from config import TradingCostConfig

class DecisionEngine:
    def __init__(self, tax_engine, trading_cost_config=None):
        self.tax_engine = tax_engine
        if trading_cost_config is None:
            trading_cost_config = TradingCostConfig()
        self.trading_cost_config = trading_cost_config

    def _trading_cost(self, trade_value):
        return max(abs(trade_value) * self.trading_cost_config.pct_cost, self.trading_cost_config.min_cost)

    def should_rebalance(self, tracking_error, unrealized_gain, expected_improvement, trade_value=0.0):
        cost = self.tax_engine.tax_due(unrealized_gain) + self._trading_cost(trade_value)
        return expected_improvement > cost
