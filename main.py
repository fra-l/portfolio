# main.py

from config import TaxConfig, TradingCostConfig
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

from strategy.strategy import FactorReplicationStrategy


def main():

    # --------------------------------------------------
    # 1. Load data (NO logic here)
    # --------------------------------------------------
    tickers = [
        # US Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # US Financials
        "JPM", "BAC", "GS", "BRK-B", "V", "MA",
        # US Healthcare
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY",
        # US Consumer
        "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE",
        # US Industrials
        "CAT", "HON", "UPS", "BA", "GE",
        # US Energy
        "XOM", "CVX", "COP",
        # US Utilities / REITs
        "NEE", "DUK", "AMT",
        # Europe
        "NESN.SW", "NOVO-B.CO", "ASML", "SAP", "MC.PA", "SHEL", "AZN",
        # Asia-Pacific
        "TM", "SONY", "9988.HK",
        # Materials
        "LIN", "BHP",
    ]
    market_data, factor_returns = MarketData.from_tickers(tickers, start="2020-01-01")

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

    tax_engine = TaxEngine(config=TaxConfig())
    decision_engine = DecisionEngine(
        tax_engine=tax_engine,
        trading_cost_config=TradingCostConfig(),
    )

    executor = Executor(market_data)
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
    print(f"  Rebalances:       {len(strategy.realized_gains):>12}")
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


if __name__ == "__main__":
    main()
