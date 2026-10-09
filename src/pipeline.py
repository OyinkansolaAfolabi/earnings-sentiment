"""End-to-end pipeline: transcripts -> sentiment -> returns -> results CSV + stats.

Example:
    python -m src.pipeline --transcripts data/transcripts.csv --max-companies 20 --last-n 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import build_features, correlation_table
from .prices import add_returns, fetch_closes
from .sentiment import FinBertScorer, LexiconScorer, score_text
from .transcripts import load_hf, load_transcripts, select_universe


def score_transcripts(df: pd.DataFrame, scorer) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        print(f"  [{i}/{len(df)}] {r.ticker} {r.date.date()}", flush=True)
        p = score_text(r.prepared, scorer) if r.split_ok else score_text("", scorer)
        q = score_text(r.qa, scorer) if r.split_ok else score_text("", scorer)
        rows.append({
            "ticker": r.ticker, "date": r.date, "split_ok": r.split_ok,
            "prepared_score": p["score"], "prepared_pos_frac": p["pos_frac"],
            "prepared_neg_frac": p["neg_frac"], "prepared_n_sent": p["n_sent"],
            "qa_score": q["score"], "qa_pos_frac": q["pos_frac"],
            "qa_neg_frac": q["neg_frac"], "qa_n_sent": q["n_sent"],
        })
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, scorer, closes: pd.DataFrame | None = None, benchmark: str = "SPY") -> pd.DataFrame:
    """Core logic, separated from I/O so it can be tested offline."""
    scored = score_transcripts(df, scorer)
    if closes is None:
        start = (scored["date"].min() - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        end = (scored["date"].max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        closes = fetch_closes(sorted(scored["ticker"].unique()), start, end, benchmark)
    scored = add_returns(scored, closes, benchmark)
    return build_features(scored)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--transcripts", help="CSV/JSONL/Parquet with ticker,date,text (or prepared,qa)")
    src.add_argument("--hf-dataset", help="HuggingFace dataset name (use with --*-col options)")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--ticker-col", default="ticker")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--tickers", nargs="*", help="Explicit universe; default = top N by transcript count")
    ap.add_argument("--max-companies", type=int, default=20)
    ap.add_argument("--last-n", type=int, default=4, help="Most recent calls kept per company")
    ap.add_argument("--backend", choices=["finbert", "lexicon"], default="finbert",
                    help="'lexicon' is an offline smoke-test scorer, NOT for real results")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()

    if args.transcripts:
        raw = load_transcripts(args.transcripts)
    else:
                raw = load_hf(args.hf_dataset, args.hf_split, args.ticker_col, args.date_col, args.text_col,
                      tickers=args.tickers, max_companies=args.max_companies, last_n=args.last_n)
    df = select_universe(raw, args.tickers, args.max_companies, args.last_n)
    print(f"{len(df)} transcripts / {df['ticker'].nunique()} companies; "
          f"{int((~df['split_ok']).sum())} without a detectable Q&A boundary")

    scorer = FinBertScorer() if args.backend == "finbert" else LexiconScorer()
    res = run(df, scorer, benchmark=args.benchmark)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nSaved {len(res)} rows -> {out}\n")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(correlation_table(res).to_string(index=False))


if __name__ == "__main__":
    main()
