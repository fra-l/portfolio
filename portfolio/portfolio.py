from .position import Position

class Portfolio:
    def __init__(self, cash):
        self.cash = cash
        self.positions = {}

    def add_position(self, ticker, lot):
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker)
        self.positions[ticker].lots.append(lot)

    def market_value(self, market_data, date):
        """
        Total portfolio value = cash + market value of all positions
        """
        value = self.cash

        for ticker, position in self.positions.items():
            price = market_data.get_price(ticker, date)
            value += position.total_shares() * price

        return value