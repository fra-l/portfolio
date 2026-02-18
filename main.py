# main.py

from data.market_data import MarketData
from factors.factor_model import FactorModel
from universe.universe_selector import UniverseSelector
from targets.factor_target import FactorTarget
from portfolio.portfolio import Portfolio
from tax.tax_engine import TaxEngine
from decisions.decision_engine import DecisionEngine
from execution.executor import Executor
from backtest.engine import BacktestEngine

from strategy.strategy import FactorReplicationStrategy


def main():

    # --------------------------------------------------
    # 1. Load data (NO logic here)
    # --------------------------------------------------
    tickers = ["AAPL", "MSFT", "JNJ"]
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

    tax_engine = TaxEngine()
    decision_engine = DecisionEngine(
        tax_engine=tax_engine,
        trading_cost=15  # flat estimate, example
    )

    executor = Executor(market_data)

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
        executor=executor
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
    print("Final cash:", portfolio.cash)
    print("Final positions:", portfolio.positions.keys())


if __name__ == "__main__":
    main()
