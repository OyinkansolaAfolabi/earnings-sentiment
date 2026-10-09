# Earnings Call Sentiment & Signal Analyzer

Does the *tone* of an earnings call, and especially the gap between scripted prepared remarks and
unscripted Q&A, line up with how the stock reacts?

Pipeline: transcripts → split into **prepared remarks vs Q&A** → **FinBERT** sentence-level sentiment →
market-adjusted return around the call → correlation statistics → **Streamlit dashboard**.

## Quick start

```bash
pip install -r requirements.txt

# 1. Put a transcript file at data/transcripts.csv (columns: ticker, date, text; see data/README.md)
# 2. Run the pipeline (downloads FinBERT ~440 MB on first run, prices via yfinance)
python -m src.pipeline --transcripts data/transcripts.csv --max-companies 20 --last-n 4

# 3. Dashboard
streamlit run app.py

# Tests (offline, no model or network needed)
python -m unittest discover -s tests -v
```

Useful options: `--tickers AAPL MSFT ...` to pick the universe, `--last-n 12` for more history
(see "Sample size" below), `--benchmark SPY`.

## Where do the transcripts come from?

**Note: SEC EDGAR does not host earnings call transcripts.** It hosts 8-K filings, whose Exhibit 99.1 is
usually the earnings *press release*: management-written and one-way, with no Q&A, so the
prepared-vs-Q&A analysis can't be done on it. Full call transcripts come from vendors or aggregators
(Motley Fool, Seeking Alpha, FactSet, S&P Capital IQ, etc.) whose terms generally restrict scraping and
redistribution. Options:

1. A public transcript dataset (e.g. on the HuggingFace Hub or Kaggle): load with
   `--hf-dataset NAME --ticker-col ... --date-col ... --text-col ...`. Check the licence and that it has
   a call **date** and the **Q&A** section.
2. Your own licensed export, saved as CSV/JSONL/Parquet in the format in `data/README.md`.

`data/` is git-ignored so you don't accidentally republish licensed text.

## Method

| Step | Detail |
|---|---|
| Section split | Regex over common boundary phrases ("question-and-answer session", "our first question comes from", `[Operator Instructions]`, …). Matches in the first 10% of the text are ignored because the operator's intro usually only *mentions* the Q&A. Calls with no boundary found are flagged `split_ok = False` and get no Q&A score. |
| Sentiment | `ProsusAI/finbert` on each sentence (≥5 words, truncated at 128 tokens). Section score = mean of `P(positive) − P(negative)`, range [−1, 1]. Label order is read from the model config, not hard-coded. |
| Features | prepared score, Q&A score, overall (sentence-weighted), **Q&A − prepared gap**, and **quarter-over-quarter change** of each score (only between calls ≤135 days apart, so a missing quarter doesn't create a fake "delta"). |
| Return | Abnormal return = stock − SPY over **close(T−1) → close(T+1)**, T = first trading day on/after the call date. This window captures the reaction whether the call was before the open or after the close, since transcripts rarely carry reliable timestamps. |
| Statistics | Spearman and Pearson correlation with p-values; sign hit-rate (binomial test) for the signed features. The dashboard applies a Bonferroni threshold across the features tested. |

## Findings

> **Fill this in from your own run.** Don't copy numbers from anywhere else. Suggested structure:
>
> - Universe, date range, and number of calls with a valid Q&A split.
> - Table from `correlation_table` (the dashboard's *Signal summary* tab).
> - Which features, if any, survive multiple-testing correction.
> - Prepared remarks vs Q&A: is one more informative? (A common hypothesis is that Q&A is, because it's unscripted.)
> - A plain statement of what the result does and doesn't show. A weak or null result is a legitimate finding.

## Limitations (read before drawing conclusions)

- **Sample size.** 20 companies × 4 calls is ~80 observations and ~60 quarter-over-quarter changes. That
  gives little statistical power: only large effects would be detectable, and a null result there is weak
  evidence. Use `--last-n 12` or more if your data allows.
- **Multiple testing.** Seven features are examined; some will look "significant" by chance. Trust the
  corrected threshold.
- **Confounding.** Returns around earnings are dominated by the *surprise* (EPS/revenue vs consensus) and
  guidance, which this project doesn't model. Sentiment may just proxy the surprise. Adding consensus data would be the natural extension.
- **Q&A text includes analysts' questions** as well as management answers; speaker-level separation needs
  speaker-labelled transcripts.
- **Sentence-level FinBERT** ignores cross-sentence context and negation scope in long sentences; long
  sentences are truncated at 128 tokens.
- **Survivorship / selection**: a universe taken from whatever the dataset contains isn't a random S&P 500 sample.
- Daily close-to-close returns can't separate the call from other same-window news (e.g. the press release itself, often issued hours earlier).
- Event windows of nearby calls for different tickers overlap in time, so observations are not fully independent (market-wide moves are partly removed by subtracting SPY).
- Not investment advice.

## Deploy

- **Streamlit Community Cloud** (free): push the repo to GitHub, point it at `app.py`, and use
  `requirements-app.txt` (the dashboard doesn't need torch/transformers). Commit `results/results.csv`.
- **HuggingFace Spaces**: create a Space with the **Docker** SDK (README front-matter `sdk: docker`,
  `app_port: 7860`) and push this repo; the included `Dockerfile` serves the dashboard on port 7860.

## Layout

```
app.py                 Streamlit dashboard
src/transcripts.py     loading, Q&A split, universe selection
src/sentiment.py       FinBERT scorer, sentence handling, aggregation
src/prices.py          yfinance download, event-window returns
src/analysis.py        feature engineering, correlations
src/pipeline.py        CLI that runs everything
tests/test_core.py     offline tests (split, return maths, deltas, planted-signal recovery)
```
