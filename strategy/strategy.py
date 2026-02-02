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
        self.rebalance_frequency = rebalance_frequency
        self.lookback_days = lookback_days

        self.last_rebalance_date = None

    def on_date(self, date):
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
        universe = self.universe_selector.select(exposures, r2)
        exposures = exposures.loc[universe]
        current_exposure = self._portfolio_factor_exposure(exposures)
        target_exposure = np.array(
            self.target.vector(exposures.columns)
        )
        tracking_error = np.linalg.norm(
            current_exposure - target_exposure
        )
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
        self._rebalance_portfolio(universe, date)
        self.last_rebalance_date = date

    def _is_rebalance_date(self, date):
        if self.last_rebalance_date is None:
            return True
        return date.to_period(self.rebalance_frequency) != \
               self.last_rebalance_date.to_period(self.rebalance_frequency)

    def _lookback_start(self, date):
        return date - pd.Timedelta(days=self.lookback_days)

    def _portfolio_factor_exposure(self, exposures):
        total_value = 0
        weighted_exposure = np.zeros(exposures.shape[1])

        for ticker, position in self.portfolio.positions.items():
            if ticker not in exposures.index:
                continue

            price = self.market_data.prices[ticker].iloc[-1]
            value = position.total_shares() * price
            total_value += value

            weighted_exposure += value * exposures.loc[ticker].values

        if total_value == 0:
            return weighted_exposure

        return weighted_exposure / total_value

    def _estimate_unrealized_gains(self, date):
        gain = 0
        for ticker, position in self.portfolio.positions.items():
            price = self.market_data.get_price(ticker, date)
            for lot in position.lots:
                gain += (price - lot.cost_basis) * lot.shares
        return max(gain, 0)

    def _rebalance_portfolio(self, universe, date):
        allocation = self.portfolio.cash / len(universe)
        for ticker in universe:
            self.executor.buy(
                portfolio=self.portfolio,
                ticker=ticker,
                euro_amount=allocation,
                date=date
            )
