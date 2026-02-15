import pandas as pd
from sklearn.linear_model import LinearRegression

class FactorModel:
    def __init__(self, factor_returns):
        self.factor_returns = factor_returns

    def estimate_exposures(self, stock_returns):
        # Find common dates between stock returns and factor returns
        common_dates = stock_returns.index.intersection(self.factor_returns.index)

        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between stock returns and factor returns")

        exposures = {}
        r2 = {}
        for stock in stock_returns.columns:
            y = stock_returns.loc[common_dates, stock].values
            X = self.factor_returns.loc[common_dates].values
            model = LinearRegression().fit(X, y)
            exposures[stock] = model.coef_
            r2[stock] = model.score(X, y)
        return pd.DataFrame(exposures, index=self.factor_returns.columns).T, pd.Series(r2)