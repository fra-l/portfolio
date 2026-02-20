from portfolio.lots import Lot

class Executor:
    def __init__(self, market_data):
        self.market_data = market_data
        self.trades = []

    def buy(self, portfolio, ticker, euro_amount, date):
        price = self.market_data.get_price(ticker, date)
        shares = euro_amount / price
        portfolio.cash -= euro_amount
        portfolio.add_position(ticker, Lot(shares, price, purchase_date=date))
        self.trades.append({"type": "buy", "date": date, "ticker": ticker,
                            "shares": shares, "price": price, "amount": euro_amount})

    def sell(self, portfolio, ticker, shares, date, method="HIFO"):
        price = self.market_data.get_price(ticker, date)
        position = portfolio.positions[ticker]
        result = position.sell_shares(shares, price, method=method)
        portfolio.cash += result["proceeds"]
        self.trades.append({"type": "sell", "date": date, "ticker": ticker,
                            "shares": result["shares_sold"], "price": price,
                            "amount": result["proceeds"]})
        return result

    def borrow(self, portfolio, amount, date):
        """Draw down `amount` from the margin facility. Adds to cash and to debt."""
        portfolio.cash += amount
        portfolio.margin_balance += amount
        self.trades.append({"type": "borrow", "date": date, "ticker": None,
                            "shares": 0, "price": 0, "amount": amount})

    def repay(self, portfolio, amount, date):
        """Repay up to `amount` of margin debt from available cash. Returns amount repaid."""
        repayable = min(amount, portfolio.margin_balance, max(portfolio.cash, 0.0))
        if repayable < 1e-9:
            return 0.0
        portfolio.cash -= repayable
        portfolio.margin_balance -= repayable
        self.trades.append({"type": "repay", "date": date, "ticker": None,
                            "shares": 0, "price": 0, "amount": repayable})
        return repayable