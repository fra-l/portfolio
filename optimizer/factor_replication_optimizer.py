import cvxpy as cp
import numpy as np

class FactorReplicationOptimizer:
    def optimize(self, exposures, target_exposures, budget=None):
        tickers = list(exposures.index)
        factors = list(exposures.columns)
        n = len(tickers)

        X = exposures.values
        target = np.array([target_exposures.get(f, 0.0) for f in factors])

        w = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(X.T @ w - target))
        constraints = [cp.sum(w) == 1, w >= 0]

        cp.Problem(objective, constraints).solve()

        weights = {ticker: float(w.value[i]) for i, ticker in enumerate(tickers)}

        if budget is not None:
            return {ticker: weight * budget for ticker, weight in weights.items()}
        return weights
