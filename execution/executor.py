from portfolio.lots import Lot

class Executor:
    def __init__(self, market_data):
        self.market_data = market_data

    def buy(self, portfolio, ticker, euro_amount, date):
        price = self.market_data.get_price(ticker, date)
        shares = euro_amount / price
        portfolio.cash -= euro_amount
        portfolio.add_position(ticker, Lot(shares, price))