class UniverseSelector:
    def __init__(self, min_r2=0.3, min_factor_loading=None,
                 max_factor_loading=None, max_volatility=None):
        """
        Parameters
        ----------
        min_r2 : float
            Minimum R² from factor regression to include a stock.
        min_factor_loading : dict[str, float] | None
            e.g. {"Value": 0.1} — keep only stocks whose loading on each
            specified factor is >= the given value.
        max_factor_loading : dict[str, float] | None
            e.g. {"MKT": 1.5} — exclude stocks whose loading exceeds the value.
        max_volatility : float | None
            Maximum annualised return volatility (e.g. 0.5 = 50%).
            Requires `volatility` to be passed to select().
        """
        self.min_r2 = min_r2
        self.min_factor_loading = min_factor_loading or {}
        self.max_factor_loading = max_factor_loading or {}
        self.max_volatility = max_volatility

    def select(self, exposures, r2, volatility=None):
        """
        Return a list of tickers that pass all filters.

        Parameters
        ----------
        exposures : pd.DataFrame
            Factor loadings, index=tickers, columns=factor names.
        r2 : pd.Series
            R² per ticker from the factor regression.
        volatility : pd.Series | None
            Annualised return volatility per ticker. Required when
            max_volatility is set; ignored otherwise.
        """
        mask = r2 >= self.min_r2

        for factor, min_val in self.min_factor_loading.items():
            if factor in exposures.columns:
                mask = mask & (exposures[factor] >= min_val)

        for factor, max_val in self.max_factor_loading.items():
            if factor in exposures.columns:
                mask = mask & (exposures[factor] <= max_val)

        if self.max_volatility is not None and volatility is not None:
            aligned = volatility.reindex(exposures.index).fillna(float("inf"))
            mask = mask & (aligned <= self.max_volatility)

        return exposures[mask].index.tolist()
