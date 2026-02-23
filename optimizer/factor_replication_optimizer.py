import logging

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


class FactorReplicationOptimizer:
    def optimize(self, exposures, target_exposures, budget=None):
        tickers = list(exposures.index)
        factors = list(exposures.columns)
        n = len(tickers)

        if n == 0:
            logger.warning("Optimizer called with empty universe; returning None.")
            return None

        X = exposures.values
        target = np.array([target_exposures.get(f, 0.0) for f in factors])

        w = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(X.T @ w - target))
        constraints = [cp.sum(w) == 1, w >= 0]

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
        except Exception as exc:
            logger.warning("CVXPY solver raised an exception: %s", exc)
            return None

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            logger.warning(
                "CVXPY solver returned non-optimal status '%s'; "
                "falling back to equal-weight allocation.",
                problem.status,
            )
            return None

        if w.value is None:
            logger.warning(
                "CVXPY solver status is '%s' but w.value is None; "
                "falling back to equal-weight allocation.",
                problem.status,
            )
            return None

        logger.debug(
            "Optimizer solved (status=%s, objective=%.6f)",
            problem.status,
            float(problem.value),
        )

        weights = {ticker: float(w.value[i]) for i, ticker in enumerate(tickers)}

        if budget is not None:
            return {ticker: weight * budget for ticker, weight in weights.items()}
        return weights
