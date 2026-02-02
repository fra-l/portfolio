class TradingCostModel:
    def cost(self, trade_value, pct_cost, min_cost):
        return max(abs(trade_value) * pct_cost, min_cost)
