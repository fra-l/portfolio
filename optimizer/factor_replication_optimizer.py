import cvxpy as cp
import numpy as np

class FactorReplicationOptimizer:
    def optimize(self, symbols, prices, factor_model, target_exposures, budget):
        factors = list(target_exposures.keys())
        n = len(symbols)

        X = np.array([
            [factor_model.exposure(sym)[f] for f in factors]
            for sym in symbols
        ])

        target = np.array([target_exposures[f] for f in factors])
        w = cp.Variable(n)

        objective = cp.Minimize(cp.sum_squares(X.T @ w - target))
        constraints = [cp.sum(w) == 1, w >= 0]

        cp.Problem(objective, constraints).solve()

        return {
            sym: float(w.value[i] * budget / prices[sym])
            for i, sym in enumerate(symbols)
        }
