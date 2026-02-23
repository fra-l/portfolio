# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Guidelines
- Default branch is `main`.
- Always use `main` for code modifications, diffs, and commits.
- Do not create new branches.

## Project Overview

A factor-based portfolio management application. Given a fixed set of stock tickers, the system builds a portfolio weighted by user-chosen factor exposures (Value, Momentum, etc.). It downloads live market data and Fama-French factor returns, estimates each stock's factor loadings via regression, and constructs portfolios that target desired factor tilts. A backtest engine simulates the strategy over historical dates.

**Current scope:** Fixed ticker input, factor regression, universe filtering, tax-aware rebalancing decisions, and a backtest loop. Future work includes smarter stock selection, proper sell logic, and connecting the CVXPY optimizer.

## Running the Code

```bash
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Dependencies: `numpy`, `pandas`, `cvxpy`, `scikit-learn`, `yfinance`, `requests`

## Architecture

Data flows top-to-bottom through `main.py`, which wires all components together.

### Core Components

1. **Data Layer** (`data/market_data.py`)
   - `MarketData.from_tickers(tickers, start)` — class method that downloads live data:
     - Stock prices via **yfinance** (`yf.download`)
     - Fama-French 3-factor (MKT, HML, SMB) + Momentum (UMD) from Kenneth French's website
   - Downloads extra lookback history (252 + 60 days before `start`) so the factor model has a full window on day one
   - Returns `(MarketData, factor_returns_df)` — factor_returns is a separate DataFrame, not stored inside MarketData
   - `MarketData` stores `prices` and `returns` DataFrames, with `get_price(ticker, date)` and `get_returns(tickers, start, end)` accessors
   - Old CSV stubs (`data/prices.csv`, `data/returns.csv`, `data/factor_returns.csv`) are leftover and unused

2. **Factor Model** (`factors/factor_model.py`)
   - `FactorModel.estimate_exposures(stock_returns)` — OLS regression per stock against factor returns
   - Returns `(exposures_df, r2_series)` — factor loadings (betas) and R-squared values
   - Uses `sklearn.linear_model.LinearRegression`
   - Factor columns in practice: `["MKT", "Value", "Momentum"]`

3. **Universe Selection** (`universe/universe_selector.py`)
   - `UniverseSelector.select(exposures, r2)` — keeps stocks with R-squared >= threshold (default 0.3)
   - Returns a list of ticker strings

4. **Target Specification** (`targets/factor_target.py`)
   - `FactorTarget({"Value": 0.6, "Momentum": 0.4})` — desired factor weights
   - `vector(factor_names)` returns an ordered list aligned with the model's factor ordering (unspecified factors get 0.0)

5. **Portfolio Management** (`portfolio/`)
   - `Portfolio` (`portfolio.py`): holds `cash` and `positions` dict (ticker -> Position), has `market_value(market_data, date)`
   - `Position` (`position.py`): groups `Lot` objects for one ticker, has `total_shares()`
   - `Lot` (`lots.py`): tracks `shares` and `cost_basis` for one purchase
   - Lot-based tracking enables future tax-efficient selling (HIFO, FIFO, etc.)

6. **Tax Engine** (`tax/tax_engine.py`)
   - `TaxEngine.tax_due(realized_gain)` — progressive capital gains using `TaxConfig.brackets`
   - Default brackets: 27% up to €10k, 42% above (Danish rates)
   - Accepts `TaxConfig` at construction; defaults to `TaxConfig()` if not provided

7. **Decision Engine** (`decisions/decision_engine.py`)
   - `DecisionEngine.should_rebalance(tracking_error, unrealized_gain, expected_improvement, trade_value=0.0)`
   - Gate: rebalance only if `expected_improvement > tax_cost + trading_cost`
   - Trading cost computed as `max(abs(trade_value) * pct_cost, min_cost)` via `TradingCostConfig`
   - Accepts `TradingCostConfig` at construction; defaults to `TradingCostConfig()` if not provided
   - Note: `tracking_error` parameter is accepted but unused in the cost formula

8. **Executor** (`execution/executor.py`)
   - `Executor.buy(portfolio, ticker, euro_amount, date)` — buys fractional shares, creates a new Lot
   - **No sell logic exists**

9. **Strategy** (`strategy/strategy.py`)
   - `FactorReplicationStrategy.on_date(date)` — called each trading day:
     - Skips non-rebalance dates (monthly frequency)
     - Estimates factor exposures over trailing 252-day window
     - Filters universe by R-squared
     - Computes portfolio vs. target factor exposure, tracking error (L2 norm)
     - Monetizes improvement as `tracking_error * portfolio_value`
     - If `DecisionEngine` approves: splits remaining cash equally across universe stocks
   - **Does not sell**, only deploys available cash

10. **Backtest Engine** (`backtest/engine.py`)
    - `BacktestEngine.run(dates)` — simple loop calling `strategy.on_date(d)` for each date
    - No performance tracking or reporting beyond final print in `main.py`

### Disconnected / Unused Components

- **`optimizer/factor_replication_optimizer.py`** — CVXPY-based optimizer that minimizes `||X.T @ w - target||^2` with constraints `sum(w)=1, w>=0`. Not imported anywhere. The strategy does naive equal-weight instead.

### Known Implementation Gaps

- **No selling**: only buys, never sells positions
- **Equal-weight allocation**: ignores factor loadings when allocating; the optimizer exists but is not connected
- **Price lookup bug in strategy**: `_portfolio_factor_exposure` uses `market_data.prices[ticker].iloc[-1]` (last price in full series) instead of the price on the current backtest date
- **MKT in target vector**: target has no MKT weight (defaults to 0.0), but MKT appears in factor exposures, creating persistent tracking error on the market factor
- **No `__init__.py` files**: modules are imported directly by path

## Configuration

`config.py` defines:
- `TaxConfig`: configurable progressive brackets; default: 27% up to €10k, 42% above. Used by `TaxEngine`.
- `TradingCostConfig`: `pct_cost=0.001`, `min_cost=1.0`. Used by `DecisionEngine`.
- `MarginConfig`: `enabled=False`, `max_leverage=1.2`, `annual_rate=0.05`. Used by `DecisionEngine` and `FactorReplicationStrategy`.
- `BacktestConfig`: top-level config aggregating all sub-configs; passed from `runner.py` through the full pipeline.

Tax rates model Danish progressive capital gains: 27% up to 10k, 42% above.

## Module Dependencies

- `FactorReplicationStrategy` depends on all other components
- `Portfolio` -> `Position` -> `Lot` (hierarchical tracking)
- `DecisionEngine` -> `TaxEngine` + `TradingCostConfig`
- `FactorModel` -> `sklearn.linear_model.LinearRegression`
- `MarketData.from_tickers` -> `yfinance`, `requests` (Fama-French download)

## Debugging

- `.vscode/launch.json`: runs `main.py` via debugpy
- `.zed/debug.json`: runs active file

Set breakpoints in `strategy/strategy.py` at `on_date()` or in `main.py`.
