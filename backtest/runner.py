# backtest/runner.py

from config import BacktestConfig
from data.market_data import MarketData
from factors.factor_model import FactorModel
from universe.universe_selector import UniverseSelector
from universe.ticker_universe import TickerUniverse
from targets.factor_target import FactorTarget
from portfolio.portfolio import Portfolio
from tax.tax_engine import TaxEngine
from tax.tax_harvesting import TaxHarvestingEngine
from trading.margin_cost import MarginCostModel
from decisions.decision_engine import DecisionEngine
from execution.executor import Executor
from optimizer.factor_replication_optimizer import FactorReplicationOptimizer
from strategy.strategy import FactorReplicationStrategy
from backtest.engine import BacktestEngine
from reporting.performance_metrics import compute_all_metrics


def run_backtest(config: BacktestConfig) -> dict:
    """Runs the full pipeline and returns the metrics dict from compute_all_metrics()."""

    # 1. Ticker universe + data download
    print("Selecting ticker universe...")
    ticker_universe = TickerUniverse()
    tickers = ticker_universe.select(
        regions=config.regions,
        cap_tiers=config.cap_tiers,
        min_adv=config.min_adv,
    )
    print(f"  {len(tickers)} tickers selected")
    market_data, factor_returns, benchmark_prices = MarketData.from_tickers(
        tickers, start=config.start, benchmark_tickers=config.benchmark_tickers
    )

    # 2. Model initialisation
    print("Initializing models...")
    factor_model = FactorModel(factor_returns)
    universe_selector = UniverseSelector(min_r2=config.min_r2)
    target = FactorTarget(config.target_weights)

    # 3. Portfolio & infrastructure
    portfolio = Portfolio(cash=config.initial_cash)

    tax_engine = TaxEngine(config=config.tax_config)
    margin_cost_model = MarginCostModel()
    decision_engine = DecisionEngine(
        tax_engine=tax_engine,
        trading_cost_config=config.trading_cost_config,
        margin_config=config.margin_config,
        margin_cost_model=margin_cost_model,
    )
    executor = Executor(market_data)
    harvester = TaxHarvestingEngine(
        config=config.tax_config,
        tax_engine=tax_engine,
        executor=executor,
    )
    optimizer = FactorReplicationOptimizer()

    # 4. Strategy
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
        margin_config=config.margin_config,
        margin_cost_model=margin_cost_model,
        rebalance_frequency=config.rebalance_frequency,
        lookback_days=config.lookback_days,
    )

    # 5. Backtest
    backtest = BacktestEngine(portfolio=portfolio, strategy=strategy)

    n_days = len(market_data.prices.index)
    start_date = market_data.prices.index[0].date()
    end_date = market_data.prices.index[-1].date()
    print(f"\nRunning backtest  ({start_date} → {end_date},  {n_days} trading days)")
    backtest.run(market_data.prices.index)

    # 6. Compute and return metrics
    metrics = compute_all_metrics(
        history=backtest.history,
        trades=executor.trades,
        benchmark_prices=benchmark_prices,
        initial_value=config.initial_cash,
    )

    # Attach extra context the caller may want for reporting
    metrics["_backtest"] = backtest
    metrics["_strategy"] = strategy
    metrics["_portfolio"] = portfolio
    metrics["_market_data"] = market_data
    metrics["_executor"] = executor
    metrics["_benchmark_prices"] = benchmark_prices
    metrics["_initial_value"] = config.initial_cash

    return metrics
