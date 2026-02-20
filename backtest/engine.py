class BacktestEngine:
    def __init__(self, portfolio, strategy):
        self.portfolio = portfolio
        self.strategy = strategy
        self.history = []

    def run(self, dates):
        for d in dates:
            self.strategy.on_date(d)
            value = self.portfolio.market_value(self.strategy.market_data, d)
            self.history.append({"date": d, "value": value})