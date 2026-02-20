class MarginCostModel:
    _TRADING_DAYS = 252

    def daily_interest(self, margin_balance, annual_rate):
        """Interest owed for one trading day on the outstanding balance."""
        return margin_balance * annual_rate / self._TRADING_DAYS

    def total_cost(self, amount, annual_rate, days):
        """Simple (non-compounding) interest on `amount` over `days` calendar days."""
        return amount * annual_rate * days / 365
