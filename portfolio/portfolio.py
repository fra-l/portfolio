from .position import Position

class Portfolio:
    def __init__(self, cash):
        self.cash = cash
        self.positions = {}
        self.margin_balance = 0.0   # outstanding borrowed amount (debt)

    def add_position(self, ticker, lot):
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker)
        self.positions[ticker].lots.append(lot)

    def market_value(self, market_data, date):
        """Total assets = cash (including borrowed funds) + stock positions."""
        value = self.cash
        for ticker, position in self.positions.items():
            price = market_data.get_price(ticker, date)
            value += position.total_shares() * price
        return value

    def equity_value(self, market_data, date):
        """Net equity = total assets minus margin debt."""
        return self.market_value(market_data, date) - self.margin_balance

    def leverage_ratio(self, market_data, date):
        """Total assets / equity. Returns 1.0 when no debt outstanding."""
        if self.margin_balance <= 0:
            return 1.0
        equity = self.equity_value(market_data, date)
        if equity <= 0:
            return float("inf")
        return self.market_value(market_data, date) / equity