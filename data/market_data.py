import pandas as pd

class MarketData:
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns

    def get_price(self, ticker, date):
        return self.prices.loc[date, ticker]

    def get_returns(self, tickers, start, end):
        return self.returns.loc[start:end, tickers]