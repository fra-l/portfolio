"""
Unit tests for reporting/performance_metrics.py.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from reporting.performance_metrics import (
    alpha_beta,
    annualized_tracking_error,
    average_monthly_turnover,
    compute_all_metrics,
    information_ratio,
    max_drawdown,
    drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    write_summary_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bdate_series(values, start="2022-01-03"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_constant_return_positive_sharpe(self):
        r = _bdate_series([0.001] * 252)
        assert sharpe_ratio(r) > 0

    def test_negative_constant_return_negative_sharpe(self):
        r = _bdate_series([-0.001] * 252)
        assert sharpe_ratio(r) < 0

    def test_zero_volatility_returns_nan(self):
        r = _bdate_series([0.0] * 100)
        assert math.isnan(sharpe_ratio(r))

    def test_single_observation_returns_nan(self):
        r = pd.Series([0.01])
        assert math.isnan(sharpe_ratio(r))

    def test_zero_risk_free_equals_default(self):
        np.random.seed(1)
        r = _bdate_series(np.random.normal(0.0005, 0.015, 252))
        assert abs(sharpe_ratio(r, risk_free=0.0) - sharpe_ratio(r)) < 1e-12

    def test_higher_risk_free_lowers_sharpe(self):
        r = _bdate_series([0.001] * 252)
        assert sharpe_ratio(r, risk_free=0.05) < sharpe_ratio(r, risk_free=0.0)


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_all_positive_returns_inf(self):
        r = _bdate_series([0.001] * 100)
        assert sortino_ratio(r) == float("inf")

    def test_mixed_returns_finite(self):
        np.random.seed(2)
        r = _bdate_series(np.random.normal(0.0005, 0.015, 252))
        s = sortino_ratio(r)
        assert math.isfinite(s)

    def test_single_observation_returns_nan(self):
        r = pd.Series([0.01])
        assert math.isnan(sortino_ratio(r))

    def test_higher_downside_lower_sortino(self):
        """A series with more downside should have a lower Sortino ratio."""
        np.random.seed(3)
        base = np.random.normal(0.001, 0.010, 252)
        noisy = base - np.abs(np.random.normal(0, 0.005, 252))
        s_base = sortino_ratio(_bdate_series(base))
        s_noisy = sortino_ratio(_bdate_series(noisy))
        assert s_base > s_noisy


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_monotonically_rising_gives_zero(self):
        v = pd.Series([100, 110, 120, 130], index=pd.date_range("2022-01-01", periods=4))
        assert max_drawdown(v) == 0.0

    def test_simple_50_percent_drawdown(self):
        v = pd.Series([200, 100], index=pd.date_range("2022-01-01", periods=2))
        assert abs(max_drawdown(v) - 0.5) < 1e-9

    def test_recover_after_drawdown(self):
        v = pd.Series([100, 80, 90, 120], index=pd.date_range("2022-01-01", periods=4))
        # peak=100, trough=80, dd = 20/100 = 0.20
        assert abs(max_drawdown(v) - 0.20) < 1e-9

    def test_short_series_returns_zero(self):
        v = pd.Series([100], index=pd.date_range("2022-01-01", periods=1))
        assert max_drawdown(v) == 0.0

    def test_result_is_non_negative(self):
        np.random.seed(4)
        v = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.01, 252)),
                      index=pd.bdate_range("2022-01-03", periods=252))
        assert max_drawdown(v) >= 0.0


# ---------------------------------------------------------------------------
# drawdown_duration
# ---------------------------------------------------------------------------

class TestDrawdownDuration:
    def test_no_drawdown_zero_duration(self):
        v = pd.Series([100, 110, 120], index=pd.date_range("2022-01-01", periods=3))
        assert drawdown_duration(v) == 0

    def test_closed_drawdown_duration(self):
        idx = pd.date_range("2022-01-01", periods=5, freq="D")
        # Drawdown starts on day 1, recovers on day 4
        v = pd.Series([100, 90, 85, 95, 105], index=idx)
        dur = drawdown_duration(v)
        assert dur > 0

    def test_open_drawdown_counts_to_last_date(self):
        idx = pd.date_range("2022-01-01", periods=4, freq="D")
        v = pd.Series([100, 90, 80, 70], index=idx)
        dur = drawdown_duration(v)
        # Drawdown starts on Jan 2 (first date below peak=100).
        # duration = (last_date - current_start).days = (Jan 4 - Jan 2) = 2 calendar days.
        assert dur == 2

    def test_short_series_returns_zero(self):
        v = pd.Series([100], index=pd.date_range("2022-01-01", periods=1))
        assert drawdown_duration(v) == 0


# ---------------------------------------------------------------------------
# alpha_beta
# ---------------------------------------------------------------------------

class TestAlphaBeta:
    def test_identical_returns_zero_alpha_unit_beta(self):
        np.random.seed(5)
        idx = pd.bdate_range("2022-01-03", periods=200)
        b = pd.Series(np.random.normal(0, 0.01, 200), index=idx)
        alpha, beta = alpha_beta(b, b)
        assert abs(beta - 1.0) < 1e-6
        assert abs(alpha) < 1e-2  # essentially zero after annualization

    def test_too_few_observations_returns_nan(self):
        p = pd.Series([0.01, 0.02, 0.03])
        b = pd.Series([0.01, 0.02, 0.03])
        alpha, beta = alpha_beta(p, b)
        assert math.isnan(alpha)
        assert math.isnan(beta)

    def test_returns_tuple_of_two(self):
        np.random.seed(6)
        idx = pd.bdate_range("2022-01-03", periods=50)
        p = pd.Series(np.random.normal(0, 0.01, 50), index=idx)
        b = pd.Series(np.random.normal(0, 0.01, 50), index=idx)
        result = alpha_beta(p, b)
        assert len(result) == 2

    def test_no_common_dates_returns_nan(self):
        p = pd.Series([0.01] * 20, index=pd.bdate_range("2022-01-03", periods=20))
        b = pd.Series([0.01] * 20, index=pd.bdate_range("2023-01-03", periods=20))
        alpha, beta = alpha_beta(p, b)
        assert math.isnan(alpha)


# ---------------------------------------------------------------------------
# information_ratio
# ---------------------------------------------------------------------------

class TestInformationRatio:
    def test_identical_returns_nan(self):
        idx = pd.bdate_range("2022-01-03", periods=100)
        r = pd.Series([0.001] * 100, index=idx)
        assert math.isnan(information_ratio(r, r))

    def test_constant_outperformance_positive_ir(self):
        idx = pd.bdate_range("2022-01-03", periods=252)
        np.random.seed(7)
        b = pd.Series(np.random.normal(0.0003, 0.01, 252), index=idx)
        p = b + 0.0005  # constant daily outperformance
        ir = information_ratio(p, b)
        assert ir > 0

    def test_constant_underperformance_negative_ir(self):
        idx = pd.bdate_range("2022-01-03", periods=252)
        np.random.seed(8)
        b = pd.Series(np.random.normal(0.0003, 0.01, 252), index=idx)
        p = b - 0.0005
        assert information_ratio(p, b) < 0


# ---------------------------------------------------------------------------
# annualized_tracking_error
# ---------------------------------------------------------------------------

class TestAnnualizedTrackingError:
    def test_identical_returns_zero_te(self):
        idx = pd.bdate_range("2022-01-03", periods=100)
        r = pd.Series([0.001] * 100, index=idx)
        assert annualized_tracking_error(r, r) == 0.0

    def test_te_is_positive(self):
        np.random.seed(9)
        idx = pd.bdate_range("2022-01-03", periods=100)
        p = pd.Series(np.random.normal(0, 0.01, 100), index=idx)
        b = pd.Series(np.random.normal(0, 0.01, 100), index=idx)
        assert annualized_tracking_error(p, b) > 0

    def test_higher_divergence_higher_te(self):
        idx = pd.bdate_range("2022-01-03", periods=252)
        np.random.seed(10)
        b = pd.Series(np.random.normal(0, 0.01, 252), index=idx)
        p_close = b + np.random.normal(0, 0.001, 252)   # small tracking
        p_far = b + np.random.normal(0, 0.02, 252)       # large tracking
        te_close = annualized_tracking_error(_bdate_series(p_close.values), b)
        te_far = annualized_tracking_error(_bdate_series(p_far.values), b)
        assert te_far > te_close

    def test_short_series_returns_nan(self):
        p = pd.Series([0.01])
        b = pd.Series([0.01])
        assert math.isnan(annualized_tracking_error(p, b))


# ---------------------------------------------------------------------------
# average_monthly_turnover
# ---------------------------------------------------------------------------

class TestAverageMonthlyTurnover:
    def test_no_trades_returns_zero(self):
        assert average_monthly_turnover([], []) == 0.0

    def test_empty_trades_list_returns_zero(self):
        idx = pd.bdate_range("2022-01-03", periods=20)
        history = [{"date": d, "value": 10_000.0} for d in idx]
        assert average_monthly_turnover([], history) == 0.0

    def test_basic_turnover_is_non_negative(self):
        idx = pd.bdate_range("2022-01-03", periods=252)
        history = [{"date": d, "value": 20_000.0} for d in idx]
        trades = [
            {"type": "buy", "date": pd.Timestamp("2022-01-05"), "ticker": "A",
             "shares": 10, "price": 100, "amount": 1000.0},
        ]
        to = average_monthly_turnover(trades, history)
        assert to >= 0.0

    def test_known_turnover_calculation(self):
        # Portfolio always at 10000. One month, 1000 traded → turnover = 10%
        idx = pd.bdate_range("2022-01-03", periods=22)
        history = [{"date": d, "value": 10_000.0} for d in idx]
        trades = [
            {"type": "buy", "date": pd.Timestamp("2022-01-05"),
             "ticker": "A", "shares": 5, "price": 200, "amount": 1000.0},
        ]
        to = average_monthly_turnover(trades, history)
        assert abs(to - 0.10) < 1e-6


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_returns_dict(self, small_spy_prices, trading_dates):
        history = [{"date": d, "value": 20_000 + i * 5} for i, d in enumerate(trading_dates[:100])]
        metrics = compute_all_metrics(history, [], small_spy_prices, 20_000.0)
        assert isinstance(metrics, dict)

    def test_expected_keys_present(self, small_spy_prices, trading_dates):
        history = [{"date": d, "value": 20_000.0 + i} for i, d in enumerate(trading_dates[:200])]
        metrics = compute_all_metrics(history, [], small_spy_prices, 20_000.0)
        expected_keys = [
            "annualized_return_pct", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "max_drawdown_duration_days",
            "alpha_annualized", "beta", "information_ratio",
            "annualized_tracking_error_pct", "avg_monthly_turnover_pct",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_empty_history_returns_empty_dict(self, small_spy_prices):
        result = compute_all_metrics([], [], small_spy_prices, 20_000.0)
        assert result == {}

    def test_single_entry_returns_empty_dict(self, small_spy_prices, trading_dates):
        history = [{"date": trading_dates[0], "value": 20_000.0}]
        result = compute_all_metrics(history, [], small_spy_prices, 20_000.0)
        assert result == {}

    def test_monotonic_increase_positive_return(self, small_spy_prices, trading_dates):
        history = [{"date": d, "value": 20_000 + i * 10} for i, d in enumerate(trading_dates)]
        metrics = compute_all_metrics(history, [], small_spy_prices, 20_000.0)
        assert metrics["annualized_return_pct"] > 0
        assert metrics["max_drawdown_pct"] == 0.0


# ---------------------------------------------------------------------------
# write_summary_report
# ---------------------------------------------------------------------------

class TestWriteSummaryReport:
    def test_creates_file(self, tmp_path, small_spy_prices, trading_dates):
        history = [{"date": d, "value": 20_000.0 + i} for i, d in enumerate(trading_dates[:100])]
        metrics = compute_all_metrics(history, [], small_spy_prices, 20_000.0)
        out = str(tmp_path / "summary.txt")
        write_summary_report(metrics, output_path=out)
        assert os.path.exists(out)

    def test_file_contains_header(self, tmp_path, small_spy_prices, trading_dates):
        history = [{"date": d, "value": 20_000.0 + i} for i, d in enumerate(trading_dates[:100])]
        metrics = compute_all_metrics(history, [], small_spy_prices, 20_000.0)
        out = str(tmp_path / "summary2.txt")
        write_summary_report(metrics, output_path=out)
        content = open(out).read()
        assert "PERFORMANCE SUMMARY" in content

    def test_file_contains_sharpe_label(self, tmp_path, small_spy_prices, trading_dates):
        history = [{"date": d, "value": 20_000.0 + i * 5} for i, d in enumerate(trading_dates[:200])]
        metrics = compute_all_metrics(history, [], small_spy_prices, 20_000.0)
        out = str(tmp_path / "summary3.txt")
        write_summary_report(metrics, output_path=out)
        content = open(out).read()
        assert "Sharpe" in content

    def test_nan_metrics_handled_gracefully(self, tmp_path):
        """write_summary_report must not crash when metrics contain nan values."""
        metrics = {
            "annualized_return_pct": float("nan"),
            "sharpe_ratio": float("nan"),
            "sortino_ratio": float("inf"),
            "max_drawdown_pct": 0.0,
            "max_drawdown_duration_days": 0,
            "alpha_annualized": float("nan"),
            "beta": float("nan"),
            "information_ratio": float("nan"),
            "annualized_tracking_error_pct": float("nan"),
            "avg_monthly_turnover_pct": 0.0,
        }
        out = str(tmp_path / "summary_nan.txt")
        write_summary_report(metrics, output_path=out)  # must not raise
        assert os.path.exists(out)

    def test_creates_parent_directory(self, tmp_path):
        metrics = {
            "annualized_return_pct": 5.0,
            "sharpe_ratio": 1.0,
            "sortino_ratio": 1.5,
            "max_drawdown_pct": 10.0,
            "max_drawdown_duration_days": 30,
            "alpha_annualized": 0.01,
            "beta": 0.95,
            "information_ratio": 0.5,
            "annualized_tracking_error_pct": 3.0,
            "avg_monthly_turnover_pct": 2.0,
        }
        nested = str(tmp_path / "nested" / "dir" / "summary.txt")
        write_summary_report(metrics, output_path=nested)
        assert os.path.exists(nested)
