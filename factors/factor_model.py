import pandas as pd
from sklearn.linear_model import LinearRegression

class FactorModel:
    def __init__(self, factor_returns):
        self.factor_returns = factor_returns

    def estimate_exposures(self, stock_returns):
        exposures = {}
        r2 = {}
        for stock in stock_returns.columns:
            y = stock_returns[stock].values
            X = self.factor_returns.loc[stock_returns.index].values
            model = LinearRegression().fit(X, y)
            exposures[stock] = model.coef_
            r2[stock] = model.score(X, y)
        return pd.DataFrame(exposures, index=self.factor_returns.columns).T, pd.Series(r2)