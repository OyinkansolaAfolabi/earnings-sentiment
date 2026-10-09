"""Load earnings-call transcripts and split them into prepared remarks vs Q&A."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Phrases that mark the start of the Q&A portion. The earliest match located
# past `min_frac` of the text wins (the operator's intro near the top often
# *mentions* a Q&A session, which must not be mistaken for the real boundary).
_QA_PATTERNS = [
    r"question[\s\-]*and[\s\-]*answer\s+session",
    r"questions?\s*(?:&|and)\s*answers?",
    r"\bq\s*&\s*a\b",
    r"\[operator instructions\]",
    r"first\s+question\s+(?:\w+\s+){0,3}from",
    r"we\s+will\s+now\s+(?:begin|take|open)\s+(?:the\s+)?(?:question|q\s*&\s*a|floor|line)",
    r"open(?:ing)?\s+(?:up\s+)?(?:the\s+|our\s+)?(?:call|line|floor)\s+(?:up\s+)?(?:to|for)\s+questions",
    # Fallback: 'Operator: Our next question is from ...' (lands at most one question late)
    r"operator:\s*(?:\w+\s+){0,6}?question\s+(?:\w+\s+){0,3}from",
]
_QA_RE = re.compile("|".join(f"(?:{p})" for p in _QA_PATTERNS), re.IGNORECASE)


def split_sections(text: str, min_frac: float = 0.10) -> tuple[str, str, bool]:
    """Return (prepared_remarks, qa, ok). ok=False when no boundary was found."""
    if not isinstance(text, str) or not text.strip():
        return "", "", False
    cutoff = int(len(text) * min_frac)
    for m in _QA_RE.finditer(text):
        if m.start() >= cutoff:
            return text[: m.start()].strip(), text[m.start():].strip(), True
    return text.strip(), "", False


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()
    df = df.dropna(subset=["ticker", "date"])
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first")

    if {"prepared", "qa"}.issubset(df.columns):
        prep = df["prepared"].fillna("").astype(str)
        qa = df["qa"].fillna("").astype(str)
        df["prepared"], df["qa"] = prep, qa
        df["split_ok"] = prep.str.strip().ne("") & qa.str.strip().ne("")
    else:
        if "text" not in df.columns:
            raise ValueError("Need either a 'text' column or both 'prepared' and 'qa' columns.")
        parts = df["text"].apply(split_sections)
        df["prepared"] = parts.apply(lambda t: t[0])
        df["qa"] = parts.apply(lambda t: t[1])
        df["split_ok"] = parts.apply(lambda t: t[2])
    return df[["ticker", "date", "prepared", "qa", "split_ok"]].reset_index(drop=True)


def load_transcripts(path: str | Path) -> pd.DataFrame:
    """Load a CSV / JSONL / Parquet file with columns ticker, date, text
    (or ticker, date, prepared, qa)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".jsonl", ".json"):
        df = pd.read_json(path, lines=(suffix == ".jsonl"))
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    missing = {"ticker", "date"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return _normalise(df)


def load_hf(name: str, split: str = "train", ticker_col: str = "ticker",
            date_col: str = "date", text_col: str = "text",
            tickers: list[str] | None = None, max_companies: int | None = None,
            last_n: int | None = None) -> pd.DataFrame:
    """Load a transcript dataset from the HuggingFace Hub and map its columns.

    If last_n is given, the universe is chosen BEFORE the (slow) section split,
    so only the selected transcripts are processed.
    """
    from datasets import load_dataset  # lazy import

    ds = load_dataset(name, split=split)
    keep = [c for c in (ticker_col, date_col, text_col) if c in ds.column_names]
    raw = ds.select_columns(keep).to_pandas()
    for col in (ticker_col, date_col, text_col):
        if col not in raw.columns:
            raise ValueError(f"Column '{col}' not in dataset. Available: {ds.column_names}")
    df = raw.rename(columns={ticker_col: "ticker", date_col: "date", text_col: "text"})
    df = df[["ticker", "date", "text"]].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["ticker", "date"])
    df = df[df["text"].fillna("").astype(str).str.strip().ne("")]  # some transcripts are empty
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first")
    if last_n is not None:
        df = select_universe(df, tickers, max_companies or 20, last_n)
    return _normalise(df)

def select_universe(df: pd.DataFrame, tickers: list[str] | None, max_companies: int,
                    last_n: int) -> pd.DataFrame:
    """Pick the companies and keep each one's most recent `last_n` calls."""
    if tickers:
        wanted = [t.upper() for t in tickers]
        df = df[df["ticker"].isin(wanted)]
    else:
        top = df.groupby("ticker").size().sort_values(ascending=False).head(max_companies).index
        df = df[df["ticker"].isin(top)]
    df = df.sort_values(["ticker", "date"])
    return df.groupby("ticker", group_keys=False).tail(last_n).reset_index(drop=True)
