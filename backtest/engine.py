class BacktestEngine:
    def __init__(self, portfolio, strategy):
        self.portfolio = portfolio
        self.strategy = strategy

    def run(self, dates):
        for d in dates:
            self.strategy.on_date(d)