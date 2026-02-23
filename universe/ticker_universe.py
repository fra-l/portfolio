import concurrent.futures
import warnings

import requests
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

    def __init__(self, catalog=None, adv_lookback="3mo", max_workers=8, request_timeout=30):
        """
        Parameters
        ----------
        catalog : list[dict] | None
            Custom catalog. Each dict must have keys: ticker, sector, region, currency.
            Defaults to the built-in catalog.
        adv_lookback : str
            yfinance period string used to compute average daily volume (e.g. "3mo", "6mo").
        max_workers : int
            Maximum number of threads used to fetch per-ticker fundamentals in parallel.
            Default is 8.
        request_timeout : float
            Seconds to wait for each individual ticker fetch before giving up and emitting
            a warning.  Default is 30.
        """
        self._df = pd.DataFrame(catalog or _CATALOG)
        self._adv_lookback = adv_lookback
        self._max_workers = max_workers
        self._request_timeout = request_timeout
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
        max_pe=None,
        max_pb=None,
        min_dividend_yield=None,
        max_beta=None,
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
        max_pe : float | None
            Maximum trailing P/E ratio. Tickers with no reported P/E are excluded.
        max_pb : float | None
            Maximum price-to-book ratio. Tickers with no reported P/B are excluded.
        min_dividend_yield : float | None
            Minimum dividend yield (e.g. 0.01 = 1%). Tickers with no yield are excluded.
        max_beta : float | None
            Maximum 5-year monthly beta. Tickers with no reported beta are excluded.
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
        needs_live = (
            cap_tiers
            or min_adv is not None
            or min_market_cap is not None
            or max_pe is not None
            or max_pb is not None
            or min_dividend_yield is not None
            or max_beta is not None
        )
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

            if max_pe is not None:
                live = live[live["pe"].notna() & (live["pe"] <= max_pe)]

            if max_pb is not None:
                live = live[live["pb"].notna() & (live["pb"] <= max_pb)]

            if min_dividend_yield is not None:
                live = live[live["dividend_yield"].notna() & (live["dividend_yield"] >= min_dividend_yield)]

            if max_beta is not None:
                live = live[live["beta"].notna() & (live["beta"] <= max_beta)]

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

    def _fetch_one(self, ticker: str) -> dict:
        """Fetch fundamentals for a single ticker and return a flat dict."""
        _nan = float("nan")
        fallback = {
            "ticker": ticker,
            "market_cap": 0.0,
            "pe": _nan,
            "pb": _nan,
            "dividend_yield": _nan,
            "beta": _nan,
        }
        try:
            t = yf.Ticker(ticker)
            info = t.info
            mc = t.fast_info.market_cap
            return {
                "ticker": ticker,
                "market_cap": float(mc) if mc else 0.0,
                "pe":             float(info["trailingPE"])    if info.get("trailingPE")    is not None else _nan,
                "pb":             float(info["priceToBook"])   if info.get("priceToBook")   is not None else _nan,
                "dividend_yield": float(info["dividendYield"]) if info.get("dividendYield") is not None else _nan,
                "beta":           float(info["beta"])          if info.get("beta")          is not None else _nan,
            }
        except (KeyError, TypeError, ValueError) as exc:
            warnings.warn(
                f"[TickerUniverse] Data parse error for {ticker!r}: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
        except requests.exceptions.HTTPError as exc:
            warnings.warn(
                f"[TickerUniverse] HTTP error fetching {ticker!r}: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
        except requests.exceptions.RequestException as exc:
            warnings.warn(
                f"[TickerUniverse] Network error fetching {ticker!r}: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
        return fallback

    def _ensure_live(self):
        """
        Fetch market cap and per-ticker fundamentals from yfinance for every ticker in the
        catalog, using a thread pool for parallelism.  Results are cached so subsequent
        calls are instant.
        """
        if self._live is not None:
            return

        tickers = self._df["ticker"].tolist()
        print(
            f"[TickerUniverse] Fetching live data for {len(tickers)} tickers "
            f"(max_workers={self._max_workers}, timeout={self._request_timeout}s)…"
        )

        # --- ADV: bulk price+volume download (unchanged — already vectorised) ---
        raw = yf.download(tickers, period=self._adv_lookback,
                          auto_adjust=True, progress=False)
        close = raw["Close"]
        volume = raw["Volume"]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
            volume = volume.to_frame(tickers[0])
        dollar_vol = close * volume
        adv = dollar_vol.mean().rename("adv")

        # --- Per-ticker fundamentals: parallel fetch ---
        rows: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_ticker = {pool.submit(self._fetch_one, t): t for t in tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    rows[ticker] = future.result(timeout=self._request_timeout)
                except concurrent.futures.TimeoutError:
                    warnings.warn(
                        f"[TickerUniverse] Timeout waiting for {ticker!r} "
                        f"(>{self._request_timeout}s); using NaN fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    rows[ticker] = {
                        "ticker": ticker,
                        "market_cap": 0.0,
                        "pe": float("nan"),
                        "pb": float("nan"),
                        "dividend_yield": float("nan"),
                        "beta": float("nan"),
                    }

        # Preserve catalog order
        ordered = [rows[t] for t in tickers]
        fundamentals = pd.DataFrame(ordered).set_index("ticker")

        self._live = pd.concat([fundamentals, adv], axis=1)
        self._live.index.name = "ticker"
