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
        factor_X = self.factor_returns.loc[common_dates]
        for stock in stock_returns.columns:
            y = stock_returns.loc[common_dates, stock]
            valid = y.notna()
            if valid.sum() < 30:
                continue
            y_clean = y[valid].values
            X_clean = factor_X[valid].values
            model = LinearRegression().fit(X_clean, y_clean)
            exposures[stock] = model.coef_
            r2[stock] = model.score(X_clean, y_clean)
        return pd.DataFrame(exposures, index=self.factor_returns.columns).T, pd.Series(r2)