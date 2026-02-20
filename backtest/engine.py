class BacktestEngine:
    def __init__(self, portfolio, strategy):
        self.portfolio = portfolio
        self.strategy = strategy
        self.history = []

    def run(self, dates):
        market_data = self.strategy.market_data
        for d in dates:
            self.strategy.on_date(d)
            value = self.portfolio.market_value(market_data, d)
            leverage = self.portfolio.leverage_ratio(market_data, d)
            allocations = {"cash": self.portfolio.cash}
            for ticker, position in self.portfolio.positions.items():
                price = market_data.get_price(ticker, d)
                allocations[ticker] = position.total_shares() * price
            self.history.append({"date": d, "value": value,
                                  "leverage": leverage, "allocations": allocations})