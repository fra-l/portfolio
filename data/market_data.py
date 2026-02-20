import io
import zipfile

import pandas as pd
import requests
import yfinance as yf

_FF_FACTORS_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
_FF_MOMENTUM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


def _fetch_ff_csv(url, start, end):
    """Download and parse a Fama-French daily factors CSV from Kenneth French's website."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = next(n for n in z.namelist() if n.upper().endswith(".CSV"))
        content = z.open(csv_name).read().decode("latin-1")

    lines = content.splitlines()

    # Find the first line containing an 8-digit date (data start)
    data_start = next(
        i for i, line in enumerate(lines)
        if line.split(",")[0].strip().isdigit() and len(line.split(",")[0].strip()) == 8
    )

    # The last non-blank line before data_start is the column header
    header_idx = data_start - 1
    while header_idx >= 0 and not lines[header_idx].strip():
        header_idx -= 1

    # Find data end (first blank or non-date line after data starts)
    data_end = next(
        (i for i in range(data_start, len(lines)) if not lines[i].split(",")[0].strip()),
        len(lines)
    )

    header = [c.strip() for c in lines[header_idx].split(",")]
    df = pd.read_csv(
        io.StringIO("\n".join(lines[data_start:data_end])),
        header=None, names=header
    )

    date_col = df.columns[0]
    df.index = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d")
    df = df.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


class MarketData:
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns

    def get_price(self, ticker, date):
        return self.prices.loc[date, ticker]

    def get_returns(self, tickers, start, end):
        return self.returns.loc[start:end, tickers]

    @classmethod
    def from_tickers(cls, tickers, start="2020-01-01", end=None, lookback_days=252):
        # Download extra history so the lookback window is fully populated
        # from the very first backtest date.
        warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")

        raw = yf.download(tickers, start=warmup_start, end=end, auto_adjust=True, progress=False)
        prices_full = raw["Close"] if len(tickers) > 1 else raw["Close"].to_frame(tickers[0])
        prices_full = prices_full.dropna(how="all")
        prices_full.index = prices_full.index.tz_localize(None)

        # Compute returns over the full history (needed for lookback)
        returns_full = prices_full.pct_change().dropna(how="all")

        # Download Fama-French daily factors (Value = HML, Momentum = UMD)
        ff = _fetch_ff_csv(_FF_FACTORS_URL, warmup_start, end)
        mom = _fetch_ff_csv(_FF_MOMENTUM_URL, warmup_start, end)

        factor_returns = pd.DataFrame({
            "MKT": ff["Mkt-RF"] / 100,
            "Value": ff["HML"] / 100,
            "Momentum": mom[mom.columns[0]] / 100,
        })

        # Align returns and factor_returns to their common dates
        common_dates = returns_full.index.intersection(factor_returns.index)
        returns_full = returns_full.loc[common_dates]
        factor_returns = factor_returns.loc[common_dates]

        # Prices used for backtest iteration start from `start`;
        # returns keep full history so the lookback window is always populated.
        start_ts = pd.Timestamp(start)
        prices = prices_full.loc[prices_full.index >= start_ts]

        return cls(prices, returns_full), factor_returns
