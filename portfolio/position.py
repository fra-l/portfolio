class Position:
    def __init__(self, ticker):
        self.ticker = ticker
        self.lots = []

    def total_shares(self):
        return sum(l.shares for l in self.lots)