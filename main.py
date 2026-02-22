# main.py

from config import TaxConfig, TradingCostConfig, MarginConfig
from trading.margin_cost import MarginCostModel
from data.market_data import MarketData
from factors.factor_model import FactorModel
from universe.universe_selector import UniverseSelector
from targets.factor_target import FactorTarget
from portfolio.portfolio import Portfolio
from tax.tax_engine import TaxEngine
from decisions.decision_engine import DecisionEngine
from execution.executor import Executor
from optimizer.factor_replication_optimizer import FactorReplicationOptimizer
from backtest.engine import BacktestEngine
from tax.tax_harvesting import TaxHarvestingEngine

from universe.ticker_universe import TickerUniverse
from strategy.strategy import FactorReplicationStrategy


def main():

    # --------------------------------------------------
    # 1. Load data (NO logic here)
    # --------------------------------------------------
    ticker_universe = TickerUniverse()
    tickers = ticker_universe.select(
        regions=["US", "Europe", "Asia-Pacific"],
        cap_tiers=["mega", "large"],
        min_adv=1e8,
    )
    market_data, factor_returns, spy_prices = MarketData.from_tickers(tickers, start="2020-01-01")

    # --------------------------------------------------
    # 2. Initialize models
    # --------------------------------------------------
    factor_model = FactorModel(factor_returns)
    universe_selector = UniverseSelector(min_r2=0.3)

    target = FactorTarget({
        "Value": 0.6,
        "Momentum": 0.4
    })

    # --------------------------------------------------
    # 3. Portfolio & infrastructure
    # --------------------------------------------------
    portfolio = Portfolio(cash=20_000)

    tax_config = TaxConfig()
    tax_engine = TaxEngine(config=tax_config)
    margin_config = MarginConfig()          # enabled=False by default; set enabled=True to use
    margin_cost_model = MarginCostModel()
    decision_engine = DecisionEngine(
        tax_engine=tax_engine,
        trading_cost_config=TradingCostConfig(),
        margin_config=margin_config,
        margin_cost_model=margin_cost_model,
    )

    executor = Executor(market_data)
    harvester = TaxHarvestingEngine(
        config=tax_config,
        tax_engine=tax_engine,
        executor=executor,
    )
    optimizer = FactorReplicationOptimizer()

    # --------------------------------------------------
    # 4. Strategy (THIS is the brain)
    # --------------------------------------------------
    strategy = FactorReplicationStrategy(
        market_data=market_data,
        factor_model=factor_model,
        universe_selector=universe_selector,
        target=target,
        portfolio=portfolio,
        decision_engine=decision_engine,
        executor=executor,
        optimizer=optimizer,
        harvester=harvester,
        margin_config=margin_config,
        margin_cost_model=margin_cost_model,
    )

    # --------------------------------------------------
    # 5. Backtest runner
    # --------------------------------------------------
    backtest = BacktestEngine(
        portfolio=portfolio,
        strategy=strategy
    )

    backtest.run(market_data.prices.index)

    # --------------------------------------------------
    # 6. Reporting / output
    # --------------------------------------------------
    import pandas as pd

    history = pd.DataFrame(backtest.history).set_index("date")
    initial_value = 20_000.0
    final_value = history["value"].iloc[-1]
    total_return_pct = (final_value / initial_value - 1) * 100
    years = (history.index[-1] - history.index[0]).days / 365.25
    ann_return_pct = ((final_value / initial_value) ** (1 / years) - 1) * 100
    total_realized = sum(g["realized_gain"] for g in strategy.realized_gains)
    harvest_events = [r for r in strategy.realized_gains if r.get("is_harvest")]
    total_tax_saved = sum(r.get("tax_saved", 0.0) for r in harvest_events)
    last_date = market_data.prices.index[-1]

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
    peak_leverage = max((h.get("leverage", 1.0) for h in backtest.history), default=1.0)
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

    from reporting.performance_metrics import compute_all_metrics, write_summary_report
    metrics = compute_all_metrics(
        history=backtest.history,
        trades=executor.trades,
        spy_prices=spy_prices,
        initial_value=initial_value,
    )
    write_summary_report(metrics, output_path="reports/summary.txt")

    from reporting.charts import plot_results
    plot_results(
        history=backtest.history,
        trades=executor.trades,
        spy_prices=spy_prices,
        initial_value=initial_value,
        exposure_history=strategy.exposure_history,
    )


if __name__ == "__main__":
    ## entry point
    main()
