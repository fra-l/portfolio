class DecisionEngine:
    def __init__(self, tax_engine, trading_cost):
        self.tax_engine = tax_engine
        self.trading_cost = trading_cost

    def should_rebalance(self, tracking_error, unrealized_gain, expected_improvement):
        cost = self.tax_engine.tax_due(unrealized_gain) + self.trading_cost
        return expected_improvement > cost