# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ETF Replication Framework v2 for research and educational purposes. The system replicates factor ETFs (Value, Momentum, Quality, etc.) using individual stocks, with support for fractional shares, progressive capital-gains tax modeling, and tax-aware rebalancing decisions.

## Running the Code

**Setup:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Run backtest:**
```bash
python main.py
```

## Architecture

The codebase follows a clean separation of concerns with data flow from top to bottom:

### Core Components Flow

1. **Data Layer** (`data/`)
   - `MarketData`: Wrapper for prices and returns DataFrames
   - CSV files: `prices.csv`, `returns.csv`, `factor_returns.csv`

2. **Factor Model** (`factors/factor_model.py`)
   - `FactorModel.estimate_exposures()`: Runs linear regression to estimate each stock's factor exposures
   - Returns exposure coefficients and R² values for universe filtering

3. **Universe Selection** (`universe/universe_selector.py`)
   - `UniverseSelector.select()`: Filters stocks by minimum R² threshold (default 0.3)
   - Ensures only stocks well-explained by factors are included

4. **Target Specification** (`targets/factor_target.py`)
   - `FactorTarget`: Defines desired factor exposure weights (e.g., 60% Value, 40% Momentum)

5. **Portfolio Management** (`portfolio/`)
   - `Portfolio`: Holds cash and positions
   - `Position`: Groups multiple lots for a single ticker
   - `Lot`: Tracks shares and cost basis for tax calculations
   - Lot-based tracking enables tax-efficient selling (HIFO, FIFO, etc.)

6. **Tax & Decision Logic**
   - `TaxEngine` (`tax/tax_engine.py`): Progressive tax model (27% up to €10k, 42% above)
   - `DecisionEngine` (`decisions/decision_engine.py`): Rebalancing decision based on tracking error improvement vs. tax + trading costs

7. **Execution** (`execution/executor.py`)
   - `Executor.buy()`: Executes fractional share purchases, creates lots

8. **Strategy** (`strategy/strategy.py`)
   - `FactorReplicationStrategy`: Orchestrates the entire flow
   - `on_date()`: Called for each date in backtest, decides whether to rebalance
   - Computes current portfolio factor exposure vs. target exposure
   - Delegates rebalancing decision to DecisionEngine

9. **Backtest Engine** (`backtest/engine.py`)
   - `BacktestEngine.run()`: Iterates through dates, calls strategy.on_date()

### Key Design Patterns

- **Lot-based position tracking**: Each purchase creates a separate lot with cost basis, enabling sophisticated tax optimization
- **Factor exposure calculation**: Portfolio-level factor exposure is value-weighted sum of individual stock exposures
- **Tax-aware rebalancing**: Only rebalance if expected tracking error improvement exceeds tax liability + trading costs
- **Lookback window**: Factor exposures estimated using trailing 252 days (1 year) of returns

## Configuration

`config.py` contains dataclasses for:
- `TaxConfig`: Progressive tax rates and threshold
- `TradingCostConfig`: Percentage and minimum flat cost
- `MarginConfig`: Leverage settings (currently disabled)

Tax rates default to Danish progressive capital gains: 27% up to €10k, 42% above.

## Module Dependencies

When modifying code, be aware of these key dependencies:
- `FactorReplicationStrategy` depends on all other components
- `Portfolio` uses `Position` and `Lot` for hierarchical tracking
- `DecisionEngine` requires `TaxEngine` for cost calculation
- `FactorModel` uses sklearn's LinearRegression

## Important Implementation Details

- **Fractional shares**: The system supports fractional shares (executor divides euro_amount by price)
- **Tax calculation**: Uses lot-based tracking, but current strategy doesn't implement sell logic yet
- **Rebalancing logic**: Currently buys equally across universe; does NOT use the optimizer in `optimizer/factor_replication_optimizer.py`
- **Date handling**: Uses pandas timestamps; rebalance frequency set via pandas period codes ("M" for monthly)
- **No selling yet**: Current implementation only buys on rebalance, never sells positions

## Debugging

VSCode and Zed debug configurations exist:
- `.vscode/launch.json`
- `.zed/debug.json`

To debug, set breakpoints in `strategy/strategy.py` at `on_date()` or in `main.py`.
