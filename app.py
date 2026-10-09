"""Streamlit dashboard.  Run:  streamlit run app.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from src.analysis import ALL_FEATURES, FEATURE_LABELS, correlation_table, fit_line  # noqa: E402

st.set_page_config(page_title="Earnings Call Sentiment Analyzer", layout="wide")
DEFAULT_PATH = Path(__file__).parent / "results" / "results.csv"


@st.cache_data
def load(src) -> pd.DataFrame:
    df = pd.read_csv(src, parse_dates=["date"])
    for c in ("ret", "bench_ret", "abnormal_ret"):
        df[c + "_pct"] = df[c] * 100
    return df


st.title("Earnings Call Sentiment & Signal Analyzer")
st.caption("FinBERT sentence-level sentiment, split into prepared remarks vs Q&A, "
           "compared with the stock's market-adjusted reaction around the call.")

upload = st.sidebar.file_uploader("Results CSV (from `python -m src.pipeline`)", type="csv")
if upload is not None:
    df = load(upload)
elif DEFAULT_PATH.exists():
    df = load(DEFAULT_PATH)
else:
    st.info("No results found. Run the pipeline first (see README) or upload a results CSV in the sidebar.")
    st.stop()

tickers = sorted(df["ticker"].unique())
sel = st.sidebar.multiselect("Companies", tickers, default=tickers)
df = df[df["ticker"].isin(sel)]
if df.empty:
    st.warning("Select at least one company.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Signal summary", "Explorer", "Company view", "Data"])

# ---------------------------------------------------------------- summary
with tab1:
    tbl = correlation_table(df)
    a, b, c = st.columns(3)
    a.metric("Calls analysed", len(df))
    b.metric("Companies", df["ticker"].nunique())
    b2 = tbl.dropna(subset=["spearman_rho"])
    if not b2.empty:
        best = b2.loc[b2["spearman_rho"].abs().idxmax()]
        c.metric("Strongest |Spearman ρ|", f"{abs(best['spearman_rho']):.2f}",
                 help=f"{FEATURE_LABELS[best['feature']]} (p = {best['spearman_p']:.3f}, n = {int(best['n'])})")

        m = len(b2)
        bonf = 0.05 / m
        if (b2["spearman_p"] < bonf).any():
            st.success(f"At least one feature clears the Bonferroni-corrected threshold (p < {bonf:.4f}, {m} tests).")
        elif (b2["spearman_p"] < 0.05).any():
            st.warning(f"Some features have p < 0.05 but none survive correction for {m} tests "
                       f"(p < {bonf:.4f}). Treat as suggestive at best.")
        else:
            st.info("No feature is statistically significant at the 5% level. "
                    "With a small sample this is weak evidence of no effect, not proof of none.")

        fig = go.Figure(go.Bar(
            x=b2["spearman_rho"], y=b2["feature"].map(FEATURE_LABELS), orientation="h",
            customdata=np.stack([b2["spearman_p"], b2["n"]], axis=-1),
            hovertemplate="ρ = %{x:.3f}<br>p = %{customdata[0]:.3f}<br>n = %{customdata[1]}<extra></extra>"))
        fig.update_layout(title="Spearman correlation with abnormal return", height=380,
                          xaxis_title="Spearman ρ", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

    show = tbl.copy()
    show["feature"] = show["feature"].map(FEATURE_LABELS)
    st.dataframe(show.rename(columns={"spearman_rho": "Spearman ρ", "spearman_p": "p (Spearman)",
                                      "pearson_r": "Pearson r", "pearson_p": "p (Pearson)",
                                      "hit_rate": "Sign hit-rate", "hit_p": "p (hit-rate)"}),
                 width="stretch", hide_index=True)
    st.caption("Abnormal return = stock return minus SPY return over close(T-1) → close(T+1), "
               "where T is the first trading day on/after the call date.")

# --------------------------------------------------------------- explorer
with tab2:
    c1, c2 = st.columns(2)
    feat = c1.selectbox("Sentiment feature (x)", ALL_FEATURES, format_func=FEATURE_LABELS.get)
    tgt = c2.selectbox("Return (y)", ["abnormal_ret_pct", "ret_pct"],
                       format_func=lambda t: "Abnormal return, %" if t == "abnormal_ret_pct" else "Raw return, %")
    sub = df[["ticker", "date", feat, tgt]].dropna()
    if len(sub) < 3:
        st.warning("Not enough data points for this feature.")
    else:
        fig = px.scatter(sub, x=feat, y=tgt, color="ticker", hover_data=["date"])
        slope, icpt = fit_line(sub[feat], sub[tgt])
        if slope is not None:
            xs = np.array([sub[feat].min(), sub[feat].max()])
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines",
                                     line=dict(color="black", dash="dash"), name="OLS fit"))
        fig.update_layout(height=520, xaxis_title=FEATURE_LABELS[feat], yaxis_title=tgt.replace("_pct", " (%)"))
        st.plotly_chart(fig, width="stretch")
        r = correlation_table(sub.rename(columns={tgt: "target"}), [feat], target="target")
        if not r.empty and pd.notna(r["spearman_rho"].iloc[0]):
            st.write(f"n = {int(r['n'].iloc[0])}, Spearman ρ = {r['spearman_rho'].iloc[0]:.3f} "
                     f"(p = {r['spearman_p'].iloc[0]:.3f}), Pearson r = {r['pearson_r'].iloc[0]:.3f} "
                     f"(p = {r['pearson_p'].iloc[0]:.3f})")

# --------------------------------------------------------- company drill-down
with tab3:
    tk = st.selectbox("Company", sorted(df["ticker"].unique()))
    d = df[df["ticker"] == tk].sort_values("date")
    labels = d["date"].dt.strftime("%Y-%m-%d")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=labels, y=d["prepared_score"], name="Prepared remarks"), secondary_y=False)
    fig.add_trace(go.Bar(x=labels, y=d["qa_score"], name="Q&A"), secondary_y=False)
    fig.add_trace(go.Scatter(x=labels, y=d["abnormal_ret_pct"], mode="lines+markers",
                             name="Abnormal return (%)", line=dict(color="black")), secondary_y=True)
    fig.update_layout(barmode="group", height=480, xaxis=dict(type="category"))
    fig.update_yaxes(title_text="Sentiment (P(pos) − P(neg))", secondary_y=False)
    fig.update_yaxes(title_text="Abnormal return (%)", secondary_y=True)
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------- data
with tab4:
    st.dataframe(df.drop(columns=[c for c in df.columns if c.endswith("_pct")]),
                 width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "results.csv", "text/csv")
