"""
Unit tests for FactorReplicationOptimizer.

Verifies that optimizer allocations reflect factor loadings and differ from
equal-weight when stocks have distinct factor exposures.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from optimizer.factor_replication_optimizer import FactorReplicationOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exposures(data: dict, factors: list[str]) -> pd.DataFrame:
    """Build a (n_stocks x n_factors) exposures DataFrame."""
    return pd.DataFrame(data, index=factors).T


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestOptimizerWeights:
    def test_weights_sum_to_one(self):
        exposures = _make_exposures(
            {"A": [1.0, 0.5], "B": [0.5, 1.0], "C": [0.8, 0.3]},
            ["Value", "Momentum"],
        )
        opt = FactorReplicationOptimizer()
        weights = opt.optimize(exposures, {"Value": 0.6, "Momentum": 0.4})
        assert weights is not None
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_weights_are_non_negative(self):
        exposures = _make_exposures(
            {"A": [1.0, 0.2], "B": [0.1, 0.9]},
            ["Value", "Momentum"],
        )
        opt = FactorReplicationOptimizer()
        weights = opt.optimize(exposures, {"Value": 0.6, "Momentum": 0.4})
        assert weights is not None
        for ticker, w in weights.items():
            assert w >= -1e-6, f"Negative weight for {ticker}: {w}"

    def test_budget_scales_weights(self):
        exposures = _make_exposures(
            {"A": [1.0, 0.5], "B": [0.3, 0.8]},
            ["Value", "Momentum"],
        )
        budget = 50_000.0
        opt = FactorReplicationOptimizer()
        allocations = opt.optimize(
            exposures, {"Value": 0.6, "Momentum": 0.4}, budget=budget
        )
        assert allocations is not None
        assert abs(sum(allocations.values()) - budget) < 1e-3


# ---------------------------------------------------------------------------
# Factor-driven allocation (not equal-weight)
# ---------------------------------------------------------------------------

class TestOptimizerVsEqualWeight:
    def test_high_value_stock_overweighted_for_value_target(self):
        """
        Stock A has high Value exposure; stock B has low Value exposure.
        With a strong Value target the optimizer should allocate more to A.
        """
        exposures = _make_exposures(
            {"A": [0.9, 0.1], "B": [0.1, 0.9]},
            ["Value", "Momentum"],
        )
        opt = FactorReplicationOptimizer()
        # Pure value target
        weights = opt.optimize(exposures, {"Value": 1.0, "Momentum": 0.0})
        assert weights is not None
        assert weights["A"] > weights["B"], (
            f"Expected A (high Value) to outweigh B (low Value), "
            f"got A={weights['A']:.4f}, B={weights['B']:.4f}"
        )

    def test_high_momentum_stock_overweighted_for_momentum_target(self):
        """
        Stock A has high Momentum exposure; stock B has low Momentum exposure.
        With a pure Momentum target the optimizer should favour A.
        """
        exposures = _make_exposures(
            {"A": [0.1, 0.9], "B": [0.9, 0.1]},
            ["Value", "Momentum"],
        )
        opt = FactorReplicationOptimizer()
        weights = opt.optimize(exposures, {"Value": 0.0, "Momentum": 1.0})
        assert weights is not None
        assert weights["A"] > weights["B"], (
            f"Expected A (high Momentum) to outweigh B, "
            f"got A={weights['A']:.4f}, B={weights['B']:.4f}"
        )

    def test_optimizer_differs_from_equal_weight_when_exposures_vary(self):
        """
        When stocks have heterogeneous factor exposures the optimizer must
        produce a distribution that is meaningfully different from equal-weight.
        """
        exposures = _make_exposures(
            {
                "A": [0.9, 0.1],
                "B": [0.5, 0.5],
                "C": [0.1, 0.9],
            },
            ["Value", "Momentum"],
        )
        target = {"Value": 0.8, "Momentum": 0.2}
        opt = FactorReplicationOptimizer()
        weights = opt.optimize(exposures, target)
        assert weights is not None

        equal = 1.0 / 3.0
        max_diff = max(abs(w - equal) for w in weights.values())
        assert max_diff > 0.05, (
            "Optimizer weights are suspiciously close to equal-weight "
            f"(max deviation {max_diff:.4f})"
        )


# ---------------------------------------------------------------------------
# Allocation reduces tracking error vs equal-weight
# ---------------------------------------------------------------------------

class TestOptimizerReducesTrackingError:
    def test_optimizer_tracking_error_le_equal_weight(self):
        """
        Optimizer weights should achieve a portfolio factor exposure at least
        as close to the target as the naive equal-weight allocation.
        """
        exposures = _make_exposures(
            {
                "A": [0.8, 0.2],
                "B": [0.3, 0.7],
                "C": [0.5, 0.5],
                "D": [0.9, 0.1],
            },
            ["Value", "Momentum"],
        )
        target_weights = {"Value": 0.7, "Momentum": 0.3}
        target_vec = np.array([target_weights["Value"], target_weights["Momentum"]])

        opt = FactorReplicationOptimizer()
        opt_weights = opt.optimize(exposures, target_weights)
        assert opt_weights is not None

        tickers = list(exposures.index)
        X = exposures.values  # (n, 2)

        opt_w = np.array([opt_weights[t] for t in tickers])
        opt_exposure = X.T @ opt_w
        opt_te = np.linalg.norm(opt_exposure - target_vec)

        eq_w = np.ones(len(tickers)) / len(tickers)
        eq_exposure = X.T @ eq_w
        eq_te = np.linalg.norm(eq_exposure - target_vec)

        assert opt_te <= eq_te + 1e-6, (
            f"Optimizer TE ({opt_te:.6f}) should be ≤ equal-weight TE ({eq_te:.6f})"
        )


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------

class TestOptimizerEdgeCases:
    def test_single_stock_gets_full_weight(self):
        exposures = _make_exposures(
            {"A": [0.5, 0.5]},
            ["Value", "Momentum"],
        )
        opt = FactorReplicationOptimizer()
        weights = opt.optimize(exposures, {"Value": 0.6, "Momentum": 0.4})
        assert weights is not None
        assert abs(weights["A"] - 1.0) < 1e-4

    def test_empty_universe_returns_none(self):
        exposures = pd.DataFrame(columns=["Value", "Momentum"])
        opt = FactorReplicationOptimizer()
        result = opt.optimize(exposures, {"Value": 0.6, "Momentum": 0.4})
        assert result is None

    def test_unknown_factor_in_target_treated_as_zero(self):
        """Factors in target_exposures not present in exposures columns default to 0."""
        exposures = _make_exposures(
            {"A": [0.8], "B": [0.4]},
            ["Value"],
        )
        opt = FactorReplicationOptimizer()
        # "Momentum" not in exposures columns → should be ignored (treated as 0 target)
        weights = opt.optimize(exposures, {"Value": 0.6, "Momentum": 0.4})
        assert weights is not None
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_budget_none_returns_fractional_weights(self):
        exposures = _make_exposures(
            {"A": [0.9, 0.1], "B": [0.1, 0.9]},
            ["Value", "Momentum"],
        )
        opt = FactorReplicationOptimizer()
        weights = opt.optimize(exposures, {"Value": 0.5, "Momentum": 0.5})
        assert weights is not None
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-4
        # No ticker should have weight > 1
        for w in weights.values():
            assert w <= 1.0 + 1e-6
