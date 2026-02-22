# reporting/backtest_report.py

"""
Prints the human-readable BACKTEST RESULTS block to stdout.

Extracted from main.py so that main() stays a thin entry point and the
same output can be reused from other callers (e.g. the tuning pipeline).
"""

import pandas as pd


def print_backtest_results(metrics: dict) -> None:
    """Print the BACKTEST RESULTS summary block using the metrics dict returned by run_backtest()."""
    backtest      = metrics["_backtest"]
    strategy      = metrics["_strategy"]
    portfolio     = metrics["_portfolio"]
    market_data   = metrics["_market_data"]
    initial_value = metrics["_initial_value"]

    history = pd.DataFrame(backtest.history).set_index("date")
    final_value      = history["value"].iloc[-1]
    total_return_pct = (final_value / initial_value - 1) * 100
    years            = (history.index[-1] - history.index[0]).days / 365.25
    ann_return_pct   = ((final_value / initial_value) ** (1 / years) - 1) * 100

    harvest_events  = [r for r in strategy.realized_gains if r.get("is_harvest")]
    total_realized  = sum(g["realized_gain"] for g in strategy.realized_gains)
    total_tax_saved = sum(r.get("tax_saved", 0.0) for r in harvest_events)
    last_date       = market_data.prices.index[-1]
    peak_leverage   = max((h.get("leverage", 1.0) for h in backtest.history), default=1.0)

    print(f"\n{'='*52}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*52}")
    print(f"  Period:           {history.index[0].date()} — {history.index[-1].date()}")
    print(f"  Initial value:    €{initial_value:>12,.2f}")
    print(f"  Final value:      €{final_value:>12,.2f}")
    print(f"  Total return:     {total_return_pct:>+11.1f}%")
    print(f"  Annualized:       {ann_return_pct:>+11.1f}%")
    print(f"  Realized gains:   €{total_realized:>12,.2f}")
    print(f"  Rebalances:       {len(strategy.realized_gains) - len(harvest_events):>12}")
    print(f"  Harvest events:   {len(harvest_events):>12}")
    print(f"  Est. tax saved:   €{total_tax_saved:>12,.2f}")
    print(f"  Margin balance:   €{portfolio.margin_balance:>12,.2f}")
    print(f"  Peak leverage:    {peak_leverage:>11.2f}×")
    print(f"  Interest paid:    €{strategy.total_interest_paid:>12,.2f}")
    print(f"\n  Final positions:")
    for ticker, position in sorted(portfolio.positions.items()):
        shares = position.total_shares()
        if shares < 1e-6:
            continue
        price = market_data.get_price(ticker, last_date)
        value = shares * price
        print(f"    {ticker:<14}  {shares:>10.4f} shares   €{value:>10,.2f}")
    print(f"    {'Cash':<14}  {'':>17}   €{portfolio.cash:>10,.2f}")
    print(f"{'='*52}")
