"""Feature engineering and signal statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MAX_QUARTER_GAP_DAYS = 135  # a "quarter-over-quarter" delta is only valid between adjacent calls

LEVEL_FEATURES = ["prepared_score", "qa_score", "overall_score", "qa_minus_prepared"]
DELTA_FEATURES = ["prepared_delta", "qa_delta", "overall_delta"]
ALL_FEATURES = LEVEL_FEATURES + DELTA_FEATURES

FEATURE_LABELS = {
    "prepared_score": "Prepared remarks sentiment",
    "qa_score": "Q&A sentiment",
    "overall_score": "Overall sentiment",
    "qa_minus_prepared": "Q&A minus prepared (gap)",
    "prepared_delta": "Prepared sentiment, change vs prior quarter",
    "qa_delta": "Q&A sentiment, change vs prior quarter",
    "overall_delta": "Overall sentiment, change vs prior quarter",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add overall score, Q&A-vs-prepared gap and quarter-over-quarter changes.

    Everything here uses only information from the call itself and earlier calls,
    so no look-ahead is introduced.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()

    wp = df["prepared_n_sent"].fillna(0).where(df["prepared_score"].notna(), 0)
    wq = df["qa_n_sent"].fillna(0).where(df["qa_score"].notna(), 0)
    num = df["prepared_score"].fillna(0) * wp + df["qa_score"].fillna(0) * wq
    den = wp + wq
    df["overall_score"] = np.where(den > 0, num / den.replace(0, np.nan), np.nan)
    df["qa_minus_prepared"] = df["qa_score"] - df["prepared_score"]

    gap = df.groupby("ticker")["date"].diff().dt.days
    valid = gap.le(MAX_QUARTER_GAP_DAYS)
    for lvl, dlt in [("prepared_score", "prepared_delta"), ("qa_score", "qa_delta"),
                     ("overall_score", "overall_delta")]:
        df[dlt] = df.groupby("ticker")[lvl].diff().where(valid)
    return df


def _safe_corr(x: pd.Series, y: pd.Series):
    if len(x) < 5 or x.nunique() < 2 or y.nunique() < 2:
        return (np.nan,) * 4
    sr, sp = stats.spearmanr(x, y)
    pr, pp = stats.pearsonr(x, y)
    return float(sr), float(sp), float(pr), float(pp)


def correlation_table(df: pd.DataFrame, features: list[str] | None = None,
                      target: str = "abnormal_ret") -> pd.DataFrame:
    """Spearman / Pearson correlation of each feature with the target, plus the hit
    rate of sign(feature) vs sign(target) for signed features (the deltas and the gap)."""
    features = features or ALL_FEATURES
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        sub = df[[f, target]].dropna()
        sr, sp, pr, pp = _safe_corr(sub[f], sub[target])
        row = {"feature": f, "n": len(sub), "spearman_rho": sr, "spearman_p": sp,
               "pearson_r": pr, "pearson_p": pp, "hit_rate": np.nan, "hit_p": np.nan}
        if f in DELTA_FEATURES + ["qa_minus_prepared"]:
            nz = sub[(sub[f] != 0) & (sub[target] != 0)]
            if len(nz) >= 5:
                k = int((np.sign(nz[f]) == np.sign(nz[target])).sum())
                row["hit_rate"] = k / len(nz)
                row["hit_p"] = float(stats.binomtest(k, len(nz), 0.5).pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_line(x: pd.Series, y: pd.Series):
    """OLS slope/intercept for drawing a trend line. Returns (None, None) if undefined."""
    sub = pd.concat([x, y], axis=1).dropna()
    if len(sub) < 3 or sub.iloc[:, 0].nunique() < 2:
        return None, None
    slope, intercept = np.polyfit(sub.iloc[:, 0], sub.iloc[:, 1], 1)
    return float(slope), float(intercept)
