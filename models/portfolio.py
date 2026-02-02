class Portfolio:
    def __init__(self, cash=0.0):
        self.cash = cash
        self.positions = {}

    def value(self, prices):
        return self.cash + sum(
            qty * prices[sym] for sym, qty in self.positions.items()
        )
