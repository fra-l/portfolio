# main.py

from config import BacktestConfig
from backtest.runner import run_backtest
from reporting.backtest_report import print_backtest_results
from reporting.performance_metrics import write_summary_report
from reporting.charts import plot_results


def main():
    config = BacktestConfig()
    metrics = run_backtest(config)

    print_backtest_results(metrics)

    write_summary_report(metrics, output_path="reports/summary.txt")

    plot_results(
        history=metrics["_backtest"].history,
        trades=metrics["_executor"].trades,
        benchmark_prices=metrics["_benchmark_prices"],
        initial_value=metrics["_initial_value"],
        exposure_history=metrics["_strategy"].exposure_history,
    )


if __name__ == "__main__":
    ## entry point
    main()
