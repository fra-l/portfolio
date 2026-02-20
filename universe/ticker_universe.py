import yfinance as yf
import pandas as pd

# ---------------------------------------------------------------------------
# Cap-tier definitions (USD)
# ---------------------------------------------------------------------------
_CAP_TIERS = {
    "mega":  (200e9,   float("inf")),
    "large": (10e9,    200e9),
    "mid":   (2e9,     10e9),
    "small": (0,       2e9),
}

# ---------------------------------------------------------------------------
# Static catalog
# Fields: ticker, sector (GICS), region, currency
# ---------------------------------------------------------------------------
_CATALOG = [
    # US — Technology
    {"ticker": "AAPL",     "sector": "Technology",             "region": "US",           "currency": "USD"},
    {"ticker": "MSFT",     "sector": "Technology",             "region": "US",           "currency": "USD"},
    {"ticker": "NVDA",     "sector": "Technology",             "region": "US",           "currency": "USD"},
    # US — Communication Services
    {"ticker": "GOOGL",    "sector": "Communication Services", "region": "US",           "currency": "USD"},
    {"ticker": "META",     "sector": "Communication Services", "region": "US",           "currency": "USD"},
    # US — Consumer Discretionary
    {"ticker": "AMZN",     "sector": "Consumer Discretionary", "region": "US",           "currency": "USD"},
    {"ticker": "TSLA",     "sector": "Consumer Discretionary", "region": "US",           "currency": "USD"},
    {"ticker": "MCD",      "sector": "Consumer Discretionary", "region": "US",           "currency": "USD"},
    {"ticker": "NKE",      "sector": "Consumer Discretionary", "region": "US",           "currency": "USD"},
    # US — Consumer Staples
    {"ticker": "PG",       "sector": "Consumer Staples",       "region": "US",           "currency": "USD"},
    {"ticker": "KO",       "sector": "Consumer Staples",       "region": "US",           "currency": "USD"},
    {"ticker": "PEP",      "sector": "Consumer Staples",       "region": "US",           "currency": "USD"},
    {"ticker": "WMT",      "sector": "Consumer Staples",       "region": "US",           "currency": "USD"},
    {"ticker": "COST",     "sector": "Consumer Staples",       "region": "US",           "currency": "USD"},
    # US — Financials
    {"ticker": "JPM",      "sector": "Financials",             "region": "US",           "currency": "USD"},
    {"ticker": "BAC",      "sector": "Financials",             "region": "US",           "currency": "USD"},
    {"ticker": "GS",       "sector": "Financials",             "region": "US",           "currency": "USD"},
    {"ticker": "BRK-B",    "sector": "Financials",             "region": "US",           "currency": "USD"},
    {"ticker": "V",        "sector": "Financials",             "region": "US",           "currency": "USD"},
    {"ticker": "MA",       "sector": "Financials",             "region": "US",           "currency": "USD"},
    # US — Health Care
    {"ticker": "JNJ",      "sector": "Health Care",            "region": "US",           "currency": "USD"},
    {"ticker": "UNH",      "sector": "Health Care",            "region": "US",           "currency": "USD"},
    {"ticker": "PFE",      "sector": "Health Care",            "region": "US",           "currency": "USD"},
    {"ticker": "ABBV",     "sector": "Health Care",            "region": "US",           "currency": "USD"},
    {"ticker": "MRK",      "sector": "Health Care",            "region": "US",           "currency": "USD"},
    {"ticker": "LLY",      "sector": "Health Care",            "region": "US",           "currency": "USD"},
    # US — Industrials
    {"ticker": "CAT",      "sector": "Industrials",            "region": "US",           "currency": "USD"},
    {"ticker": "HON",      "sector": "Industrials",            "region": "US",           "currency": "USD"},
    {"ticker": "UPS",      "sector": "Industrials",            "region": "US",           "currency": "USD"},
    {"ticker": "BA",       "sector": "Industrials",            "region": "US",           "currency": "USD"},
    {"ticker": "GE",       "sector": "Industrials",            "region": "US",           "currency": "USD"},
    # US — Energy
    {"ticker": "XOM",      "sector": "Energy",                 "region": "US",           "currency": "USD"},
    {"ticker": "CVX",      "sector": "Energy",                 "region": "US",           "currency": "USD"},
    {"ticker": "COP",      "sector": "Energy",                 "region": "US",           "currency": "USD"},
    # US — Utilities
    {"ticker": "NEE",      "sector": "Utilities",              "region": "US",           "currency": "USD"},
    {"ticker": "DUK",      "sector": "Utilities",              "region": "US",           "currency": "USD"},
    # US — Real Estate
    {"ticker": "AMT",      "sector": "Real Estate",            "region": "US",           "currency": "USD"},
    # US — Materials
    {"ticker": "LIN",      "sector": "Materials",              "region": "US",           "currency": "USD"},
    # Europe
    {"ticker": "NESN.SW",  "sector": "Consumer Staples",       "region": "Europe",       "currency": "CHF"},
    {"ticker": "NOVO-B.CO","sector": "Health Care",            "region": "Europe",       "currency": "DKK"},
    {"ticker": "ASML",     "sector": "Technology",             "region": "Europe",       "currency": "USD"},
    {"ticker": "SAP",      "sector": "Technology",             "region": "Europe",       "currency": "USD"},
    {"ticker": "MC.PA",    "sector": "Consumer Discretionary", "region": "Europe",       "currency": "EUR"},
    {"ticker": "SHEL",     "sector": "Energy",                 "region": "Europe",       "currency": "USD"},
    {"ticker": "AZN",      "sector": "Health Care",            "region": "Europe",       "currency": "USD"},
    # Asia-Pacific
    {"ticker": "TM",       "sector": "Consumer Discretionary", "region": "Asia-Pacific", "currency": "USD"},
    {"ticker": "SONY",     "sector": "Technology",             "region": "Asia-Pacific", "currency": "USD"},
    {"ticker": "9988.HK",  "sector": "Consumer Discretionary", "region": "Asia-Pacific", "currency": "HKD"},
    {"ticker": "BHP",      "sector": "Materials",              "region": "Asia-Pacific", "currency": "USD"},
]


class TickerUniverse:
    """
    Builds a filtered ticker list from a static catalog (sector, region, currency)
    enriched with live market-cap and average-daily-volume data from yfinance.

    Usage
    -----
    universe = TickerUniverse()
    tickers = universe.select(
        regions=["US", "Europe"],
        sectors=["Technology", "Financials"],
        cap_tiers=["mega", "large"],
        min_adv=1e8,
    )
    """

    def __init__(self, catalog=None, adv_lookback="3mo"):
        """
        Parameters
        ----------
        catalog : list[dict] | None
            Custom catalog. Each dict must have keys: ticker, sector, region, currency.
            Defaults to the built-in catalog.
        adv_lookback : str
            yfinance period string used to compute average daily volume (e.g. "3mo", "6mo").
        """
        self._df = pd.DataFrame(catalog or _CATALOG)
        self._adv_lookback = adv_lookback
        self._live: pd.DataFrame | None = None  # lazily populated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        regions=None,
        sectors=None,
        currencies=None,
        cap_tiers=None,
        min_adv=None,
        min_market_cap=None,
    ):
        """
        Return a list of tickers matching all specified criteria.

        Within each list argument criteria are OR-combined;
        across arguments they are AND-combined.

        Parameters
        ----------
        regions : list[str] | None
            e.g. ["US", "Europe", "Asia-Pacific"]
        sectors : list[str] | None
            GICS sector names, e.g. ["Technology", "Health Care"]
        currencies : list[str] | None
            e.g. ["USD", "EUR"]
        cap_tiers : list[str] | None
            Subset of {"mega", "large", "mid", "small"}
        min_adv : float | None
            Minimum average daily dollar volume (USD).
        min_market_cap : float | None
            Minimum market capitalisation (USD).
        """
        df = self._df.copy()

        # --- Static filters ---
        if regions:
            df = df[df["region"].isin(regions)]
        if sectors:
            df = df[df["sector"].isin(sectors)]
        if currencies:
            df = df[df["currency"].isin(currencies)]

        tickers = df["ticker"].tolist()
        if not tickers:
            return []

        # --- Live filters ---
        needs_live = cap_tiers or min_adv is not None or min_market_cap is not None
        if needs_live:
            self._ensure_live()
            live = self._live[self._live.index.isin(tickers)].copy()

            if cap_tiers:
                def _in_tier(mc):
                    return any(
                        _CAP_TIERS[t][0] <= mc < _CAP_TIERS[t][1]
                        for t in cap_tiers
                    )
                live = live[live["market_cap"].apply(_in_tier)]

            if min_adv is not None:
                live = live[live["adv"] >= min_adv]

            if min_market_cap is not None:
                live = live[live["market_cap"] >= min_market_cap]

            tickers = [t for t in tickers if t in live.index]

        return tickers

    @property
    def all_tickers(self):
        """All tickers in the catalog, with no filtering applied."""
        return self._df["ticker"].tolist()

    def summary(self):
        """
        Return a DataFrame with static metadata plus cached live data
        (if it has already been fetched).
        """
        df = self._df.copy()
        if self._live is not None:
            df = df.join(self._live, on="ticker")
        return df.set_index("ticker")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_live(self):
        """
        Fetch market cap and ADV from yfinance for every ticker in the catalog.
        Results are cached so subsequent calls are instant.
        """
        if self._live is not None:
            return

        tickers = self._df["ticker"].tolist()
        print(f"[TickerUniverse] Fetching live market cap + ADV for {len(tickers)} tickers…")

        # --- ADV: bulk price+volume download ---
        raw = yf.download(tickers, period=self._adv_lookback,
                          auto_adjust=True, progress=False)
        close = raw["Close"]
        volume = raw["Volume"]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
            volume = volume.to_frame(tickers[0])
        dollar_vol = close * volume
        adv = dollar_vol.mean().rename("adv")

        # --- Market cap: fast_info per ticker ---
        market_caps = {}
        for ticker in tickers:
            try:
                mc = yf.Ticker(ticker).fast_info.market_cap
                market_caps[ticker] = float(mc) if mc else 0.0
            except Exception:
                market_caps[ticker] = 0.0

        mc_series = pd.Series(market_caps, name="market_cap")
        self._live = pd.concat([mc_series, adv], axis=1)
        self._live.index.name = "ticker"
