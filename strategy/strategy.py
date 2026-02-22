# strategy.py

import numpy as np
import pandas as pd

class FactorReplicationStrategy:

    def __init__(
        self,
        market_data,
        factor_model,
        universe_selector,
        target,
        portfolio,
        decision_engine,
        executor,
        optimizer=None,
        harvester=None,
        margin_config=None,
        margin_cost_model=None,
        rebalance_frequency="M",
        lookback_days=252
    ):
        self.market_data = market_data
        self.factor_model = factor_model
        self.universe_selector = universe_selector
        self.target = target
        self.portfolio = portfolio
        self.decision_engine = decision_engine
        self.executor = executor
        self.optimizer = optimizer
        self.harvester = harvester
        self.margin_config = margin_config
        self.margin_cost_model = margin_cost_model
        self.rebalance_frequency = rebalance_frequency
        self.lookback_days = lookback_days

        self.last_rebalance_date = None
        self.realized_gains = []
        self.total_interest_paid = 0.0
        self.exposure_history = []  # list of {date, exposure, factors} snapshots

    def on_date(self, date):
        # Margin: daily interest accrual and margin-call check (every day)
        if self.margin_config is not None and self.margin_config.enabled \
                and self.portfolio.margin_balance > 0:
            interest = self.margin_cost_model.daily_interest(
                self.portfolio.margin_balance, self.margin_config.annual_rate
            )
            self.portfolio.cash -= interest
            self.total_interest_paid += interest

            if self.portfolio.leverage_ratio(self.market_data, date) \
                    > self.margin_config.max_leverage:
                self._forced_liquidation(date)

        # Tax-loss harvesting — runs in configured months, independent of rebalance gate
        if self.harvester is not None:
            ytd_gains = self._realized_gains_ytd(date)
            harvest_records = self.harvester.harvest(
                portfolio=self.portfolio,
                market_data=self.market_data,
                date=date,
                realized_gains_ytd=ytd_gains,
            )
            for r in harvest_records:
                self.realized_gains.append({
                    "date": r["date"],
                    "ticker": r["ticker"],
                    "realized_gain": -r["realized_loss"],
                    "proceeds": r["proceeds"],
                    "is_harvest": True,
                    "tax_saved": r["tax_saved"],
                })

        if not self._is_rebalance_date(date):
            return
        end = date
        start = self._lookback_start(date)
        returns = self.market_data.get_returns(
            tickers=self.market_data.returns.columns,
            start=start,
            end=end
        )
        exposures, r2 = self.factor_model.estimate_exposures(returns)
        volatility = returns.std() * (252 ** 0.5)
        universe = self.universe_selector.select(exposures, r2, volatility=volatility)
        exposures = exposures.loc[universe]
        current_exposure = self._portfolio_factor_exposure(exposures, date)
        # Snapshot portfolio factor exposure on every rebalance date (all factors)
        self.exposure_history.append({
            "date": date,
            "exposure": current_exposure.tolist(),
            "factors": list(exposures.columns),
        })
        # Tracking error is computed only over factors explicitly managed by the
        # target (e.g. Value, Momentum).  Un-managed factors such as MKT are
        # excluded: because we never set a MKT target, including it would
        # introduce a permanent, irrecoverable tracking-error component that the
        # strategy can never close.
        factor_names = list(exposures.columns)
        managed = [f for f in factor_names if f in self.target.target_weights]
        managed_idx = np.array([factor_names.index(f) for f in managed])
        managed_current = current_exposure[managed_idx]
        managed_target = np.array([self.target.target_weights[f] for f in managed])
        tracking_error = np.linalg.norm(managed_current - managed_target)
        unrealized_gain = self._estimate_unrealized_gains(date)
        portfolio_value = self.portfolio.market_value(
            self.market_data,
            date
        )
        expected_improvement = tracking_error * portfolio_value
        rebalance = self.decision_engine.should_rebalance(
            tracking_error=tracking_error,
            unrealized_gain=unrealized_gain,
            expected_improvement=expected_improvement
        )
        if not rebalance:
            return

        # High-conviction overlay: borrow up to the leverage cap when signals are strong
        if self.margin_config is not None and self.margin_config.enabled:
            conviction = self._conviction_score(r2, tracking_error)
            if conviction >= self.margin_config.conviction_threshold:
                equity = self.portfolio.equity_value(self.market_data, date)
                headroom = equity * (self.margin_config.max_leverage - 1) \
                           - self.portfolio.margin_balance
                if headroom > 1.0:
                    self.executor.borrow(self.portfolio, headroom, date)

        self._rebalance_portfolio(universe, exposures, date)
        self.last_rebalance_date = date
        print(f"    [rebalance] {date.date()}  universe={len(universe)}  TE={tracking_error:.3f}  value=€{portfolio_value:,.0f}", flush=True)

        # Repay any margin balance that can be covered by spare cash
        if self.margin_config is not None and self.margin_config.enabled \
                and self.portfolio.margin_balance > 0:
            self.executor.repay(self.portfolio, max(self.portfolio.cash, 0.0), date)

    def _is_rebalance_date(self, date):
        if self.last_rebalance_date is None:
            return True
        return date.to_period(self.rebalance_frequency) != \
               self.last_rebalance_date.to_period(self.rebalance_frequency)

    def _lookback_start(self, date):
        return date - pd.Timedelta(days=self.lookback_days)

    def _portfolio_factor_exposure(self, exposures, date):
        total_value = 0
        weighted_exposure = np.zeros(exposures.shape[1])

        for ticker, position in self.portfolio.positions.items():
            if ticker not in exposures.index:
                continue

            price = self.market_data.get_price(ticker, date)
            value = position.total_shares() * price
            total_value += value

            weighted_exposure += value * exposures.loc[ticker].values

        if total_value == 0:
            return weighted_exposure

        return weighted_exposure / total_value

    def _realized_gains_ytd(self, date):
        return sum(
            r["realized_gain"]
            for r in self.realized_gains
            if r["date"].year == date.year
        )

    def _estimate_unrealized_gains(self, date):
        gain = 0
        for ticker, position in self.portfolio.positions.items():
            price = self.market_data.get_price(ticker, date)
            for lot in position.lots:
                gain += (price - lot.cost_basis) * lot.shares
        return max(gain, 0)

    def _conviction_score(self, r2_series, tracking_error):
        """
        Score in [0, 1]:  high mean R² (reliable model) × high tracking error
        (large room for improvement). Capped at 1.0.
        """
        mean_r2 = float(r2_series.mean()) if hasattr(r2_series, "mean") else float(r2_series)
        te_score = min(tracking_error, 1.0)
        return mean_r2 * te_score

    def _forced_liquidation(self, date):
        """
        Sell the smallest positions (cheapest to unwind) until leverage is
        back within the configured cap, repaying margin from each set of proceeds.
        """
        positions_by_value = sorted(
            [(t, p) for t, p in self.portfolio.positions.items()
             if p.total_shares() > 1e-12],
            key=lambda x: x[1].total_shares() * self.market_data.get_price(x[0], date)
        )
        for ticker, position in positions_by_value:
            if self.portfolio.leverage_ratio(self.market_data, date) \
                    <= self.margin_config.max_leverage:
                break
            shares = position.total_shares()
            result = self.executor.sell(self.portfolio, ticker, shares, date)
            self.realized_gains.append({
                "date": date,
                "ticker": ticker,
                "realized_gain": result["realized_gain"],
                "proceeds": result["proceeds"],
                "is_forced_liquidation": True,
            })
            self.executor.repay(self.portfolio, result["proceeds"], date)

    def _rebalance_portfolio(self, universe, exposures, date):
        portfolio_value = self.portfolio.market_value(self.market_data, date)

        if self.optimizer is not None:
            target_allocations = self.optimizer.optimize(
                exposures=exposures.loc[universe],
                target_exposures=self.target.target_weights,
                budget=portfolio_value,
            )
        else:
            equal_weight = portfolio_value / len(universe)
            target_allocations = {ticker: equal_weight for ticker in universe}

        # Sell over-allocated or exited positions first
        # (or borrow instead if that is cheaper than triggering capital gains tax)
        for ticker, position in list(self.portfolio.positions.items()):
            price = self.market_data.get_price(ticker, date)
            current_value = position.total_shares() * price
            target_value = target_allocations.get(ticker, 0.0)

            if current_value > target_value + 1e-6:
                sell_amount = current_value - target_value
                unrealized = sum(
                    (price - lot.cost_basis) * lot.shares for lot in position.lots
                )
                if self.decision_engine.should_borrow_instead_of_sell(
                    unrealized_gain=max(unrealized, 0.0),
                    sell_amount=sell_amount,
                    expected_hold_days=self.margin_config.expected_hold_days
                    if self.margin_config else 90,
                ):
                    # Borrow to fund the shortfall rather than crystallise a taxable gain
                    equity = self.portfolio.equity_value(self.market_data, date)
                    headroom = equity * (self.margin_config.max_leverage - 1) \
                               - self.portfolio.margin_balance
                    borrow_amount = min(sell_amount, max(headroom, 0.0))
                    if borrow_amount > 1.0:
                        self.executor.borrow(self.portfolio, borrow_amount, date)
                else:
                    shares_to_sell = sell_amount / price
                    result = self.executor.sell(
                        portfolio=self.portfolio,
                        ticker=ticker,
                        shares=shares_to_sell,
                        date=date,
                    )
                    self.realized_gains.append({
                        "date": date,
                        "ticker": ticker,
                        "realized_gain": result["realized_gain"],
                        "proceeds": result["proceeds"],
                    })

        # Buy under-allocated positions using available cash
        for ticker in universe:
            # Skip tickers recently harvested (wash-sale waiting period)
            if self.harvester is not None and self.harvester.is_wash_sale_blocked(ticker, date):
                continue
            price = self.market_data.get_price(ticker, date)
            current_value = 0.0
            if ticker in self.portfolio.positions:
                current_value = self.portfolio.positions[ticker].total_shares() * price

            shortfall = target_allocations.get(ticker, 0.0) - current_value
            buy_amount = min(shortfall, self.portfolio.cash)
            if buy_amount > 1e-6:
                self.executor.buy(
                    portfolio=self.portfolio,
                    ticker=ticker,
                    euro_amount=buy_amount,
                    date=date,
                )
