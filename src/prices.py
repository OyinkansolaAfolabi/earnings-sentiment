"""Price download and event-window returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GAP_DAYS = 5  # if the first trading day on/after the call is further away, data is missing


def fetch_closes(tickers: list[str], start: str, end: str, benchmark: str = "SPY") -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker, plus the benchmark."""
    import yfinance as yf  # lazy import

    symbols = sorted(set(tickers) | {benchmark})
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close.sort_index()


def event_return(close: pd.Series, call_date: pd.Timestamp, pre: int = 1, post: int = 1) -> float:
    """Return from the close `pre` trading days before the first trading day on/after
    `call_date` (day T) to the close `post` trading days after T.

    With pre=post=1 the window is close(T-1) -> close(T+1). This brackets the market
    reaction whether the call happened before the open (reaction on T) or after the
    close (reaction on T+1), without needing the call's time of day.
    """
    s = close.dropna()
    if s.empty:
        return np.nan
    pos = int(s.index.searchsorted(call_date, side="left"))
    if pos - pre < 0 or pos + post >= len(s):
        return np.nan
    if (s.index[pos] - call_date).days > MAX_GAP_DAYS:
        return np.nan
    return float(s.iloc[pos + post] / s.iloc[pos - pre] - 1.0)


def add_returns(df: pd.DataFrame, closes: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """Add raw, benchmark and abnormal (stock minus benchmark) event-window returns."""
    df = df.copy()
    rets, bench, ab = [], [], []
    for _, row in df.iterrows():
        r = event_return(closes[row["ticker"]], row["date"]) if row["ticker"] in closes else np.nan
        b = event_return(closes[benchmark], row["date"]) if benchmark in closes else np.nan
        rets.append(r)
        bench.append(b)
        ab.append(r - b if not (np.isnan(r) or np.isnan(b)) else np.nan)
    df["ret"], df["bench_ret"], df["abnormal_ret"] = rets, bench, ab
    return df
