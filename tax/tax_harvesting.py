class TaxHarvestingEngine:
    """
    Proactively realizes losses to offset capital gains, reducing net tax owed
    under the configured progressive bracket schedule.
    """

    def __init__(self, config, tax_engine, executor):
        self.config = config
        self.tax_engine = tax_engine
        self.executor = executor

        self._harvested_this_year = 0.0
        self._current_year = None
        self.wash_sale_blacklist = {}   # ticker -> date of last harvest sell

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def harvest(self, portfolio, market_data, date, realized_gains_ytd):
        """
        Scan all positions for harvestable losses and execute sells.

        Gates applied in order:
          1. harvest_enabled is True
          2. date.month in harvest_months
          3. realized_gains_ytd > 0  (nothing to offset otherwise)
          4. Per-position: net unrealized loss >= min_loss_threshold
          5. Per-position: not in wash-sale blacklist
          6. Annual cap not exceeded

        Returns a list of harvest records:
            {"date", "ticker", "shares_sold", "proceeds",
             "realized_loss", "tax_saved"}
        """
        self._reset_annual_if_needed(date)

        if not self.config.harvest_enabled:
            return []
        if date.month not in self.config.harvest_months:
            return []
        if realized_gains_ytd <= 0:
            return []

        records = []
        remaining_gains = realized_gains_ytd

        for ticker, position in list(portfolio.positions.items()):
            if remaining_gains <= 0:
                break

            annual_room = self.config.max_harvest_per_year - self._harvested_this_year
            if annual_room <= 0:
                break

            if position.total_shares() < 1e-12:
                continue
            if self.is_wash_sale_blocked(ticker, date):
                continue

            current_price = market_data.get_price(ticker, date)
            total_pnl = sum(
                (current_price - lot.cost_basis) * lot.shares
                for lot in position.lots
            )

            if total_pnl >= -self.config.min_loss_threshold:
                continue  # position not in sufficient loss

            available_loss = abs(total_pnl)
            loss_to_harvest = min(available_loss, remaining_gains, annual_room)

            if loss_to_harvest < self.config.min_loss_threshold:
                continue

            shares_to_sell = self._shares_for_target_loss(
                position, current_price, loss_to_harvest
            )
            if shares_to_sell < 1e-9:
                continue

            result = self.executor.sell(
                portfolio=portfolio,
                ticker=ticker,
                shares=shares_to_sell,
                date=date,
                method="HIFO",
            )

            realized_loss = abs(result["realized_gain"])
            tax_saved = self._tax_saved(realized_loss, remaining_gains)

            self.wash_sale_blacklist[ticker] = date
            self._harvested_this_year += realized_loss
            remaining_gains = max(remaining_gains - realized_loss, 0.0)

            records.append({
                "date": date,
                "ticker": ticker,
                "shares_sold": result["shares_sold"],
                "proceeds": result["proceeds"],
                "realized_loss": realized_loss,
                "tax_saved": tax_saved,
            })

        return records

    def is_wash_sale_blocked(self, ticker, date):
        """Return True if ticker was recently harvested and is in the waiting window."""
        if ticker not in self.wash_sale_blacklist:
            return False
        days_since = (date - self.wash_sale_blacklist[ticker]).days
        return days_since < self.config.wash_sale_waiting_days

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_annual_if_needed(self, date):
        if self._current_year != date.year:
            self._harvested_this_year = 0.0
            self._current_year = date.year

    def _shares_for_target_loss(self, position, current_price, target_loss):
        """
        Compute the minimum shares to sell (HIFO order — highest cost basis first,
        same as executor default) to realize `target_loss` (positive magnitude).
        Lots that are at a gain are skipped.
        """
        hifo_lots = sorted(position.lots, key=lambda l: l.cost_basis, reverse=True)
        shares_to_sell = 0.0
        remaining_needed = target_loss

        for lot in hifo_lots:
            loss_per_share = lot.cost_basis - current_price
            if loss_per_share <= 0:
                continue  # lot is at a gain; HIFO will still process it in sequence
            lot_total_loss = loss_per_share * lot.shares
            if lot_total_loss >= remaining_needed:
                shares_to_sell += remaining_needed / loss_per_share
                remaining_needed = 0.0
                break
            else:
                shares_to_sell += lot.shares
                remaining_needed -= lot_total_loss

        return shares_to_sell

    def _tax_saved(self, realized_loss, realized_gains_ytd):
        """Tax reduction from offsetting `realized_loss` against `realized_gains_ytd`."""
        before = self.tax_engine.tax_due(realized_gains_ytd)
        after = self.tax_engine.tax_due(max(0.0, realized_gains_ytd - realized_loss))
        return before - after
