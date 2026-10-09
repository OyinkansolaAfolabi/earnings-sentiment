"""Offline tests. Run with:  python -m unittest discover -s tests -v   (or pytest)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import build_features, correlation_table
from src.pipeline import run
from src.prices import event_return
from src.sentiment import LexiconScorer, score_text, split_sentences
from src.transcripts import _normalise, select_universe, split_sections


class TestSplit(unittest.TestCase):
    def test_skips_early_intro_mention(self):
        intro = "Operator: After the prepared remarks we will hold a question-and-answer session. "
        prep = "We had a strong quarter with record growth across segments. " * 30
        qa = "Operator: Our first question comes from John at Bank. Thanks for taking my question here. " * 5
        p, q, ok = split_sections(intro + prep + qa)
        self.assertTrue(ok)
        self.assertIn("record growth", p)
        self.assertIn("our first question", q.lower()[:60])
        self.assertNotIn("record growth", q)

    def test_no_boundary(self):
        p, q, ok = split_sections("Just a monologue with no markers at all. " * 20)
        self.assertFalse(ok)
        self.assertEqual(q, "")

    def test_empty_and_nan(self):
        self.assertEqual(split_sections("")[2], False)
        self.assertEqual(split_sections(np.nan)[2], False)


class TestSentiment(unittest.TestCase):
    def test_sentence_filter(self):
        s = split_sentences("Thank you. Revenue increased strongly this quarter across all regions. Next question.")
        self.assertEqual(len(s), 1)

    def test_scores_sign(self):
        sc = LexiconScorer()
        pos = score_text("Revenue growth was strong and margins improved this quarter.", sc)
        neg = score_text("Demand was weak and we faced significant headwinds this quarter.", sc)
        self.assertGreater(pos["score"], 0)
        self.assertLess(neg["score"], 0)
        self.assertTrue(np.isnan(score_text("", sc)["score"]))


class TestReturns(unittest.TestCase):
    def setUp(self):
        idx = pd.bdate_range("2024-01-01", periods=10)  # Mon 1 Jan ... Fri 12 Jan
        self.s = pd.Series(np.arange(100.0, 110.0), index=idx)

    def test_window_math(self):
        # call on Wed 3 Jan (pos 2): close(Tue 2 Jan)=101 -> close(Thu 4 Jan)=103
        r = event_return(self.s, pd.Timestamp("2024-01-03"))
        self.assertAlmostEqual(r, 103 / 101 - 1)

    def test_weekend_rolls_forward(self):
        # Sat 6 Jan -> first trading day Mon 8 Jan (pos 5): 104 -> 106
        r = event_return(self.s, pd.Timestamp("2024-01-06"))
        self.assertAlmostEqual(r, 106 / 104 - 1)

    def test_edges_are_nan(self):
        self.assertTrue(np.isnan(event_return(self.s, pd.Timestamp("2024-01-01"))))  # no T-1
        self.assertTrue(np.isnan(event_return(self.s, pd.Timestamp("2024-01-12"))))  # no T+1
        self.assertTrue(np.isnan(event_return(self.s, pd.Timestamp("2024-03-01"))))  # beyond data


class TestFeatures(unittest.TestCase):
    def test_delta_respects_quarter_gap(self):
        df = pd.DataFrame({
            "ticker": ["A", "A", "A", "B"],
            "date": pd.to_datetime(["2024-01-30", "2024-04-30", "2025-04-30", "2024-04-30"]),
            "prepared_score": [0.1, 0.3, 0.5, 0.2], "prepared_n_sent": [50, 50, 50, 50],
            "qa_score": [0.0, 0.1, 0.2, 0.1], "qa_n_sent": [50, 50, 50, 50],
        })
        out = build_features(df).set_index(["ticker", "date"])
        self.assertTrue(np.isnan(out.loc[("A", pd.Timestamp("2024-01-30")), "prepared_delta"]))
        self.assertAlmostEqual(out.loc[("A", pd.Timestamp("2024-04-30")), "prepared_delta"], 0.2)
        self.assertTrue(np.isnan(out.loc[("A", pd.Timestamp("2025-04-30")), "prepared_delta"]))  # 1y gap
        self.assertTrue(np.isnan(out.loc[("B", pd.Timestamp("2024-04-30")), "prepared_delta"]))  # first call
        self.assertAlmostEqual(out.loc[("A", pd.Timestamp("2024-04-30")), "qa_minus_prepared"], -0.2)

    def test_overall_handles_missing_qa(self):
        df = pd.DataFrame({
            "ticker": ["A"], "date": pd.to_datetime(["2024-01-30"]),
            "prepared_score": [0.4], "prepared_n_sent": [40],
            "qa_score": [np.nan], "qa_n_sent": [0],
        })
        self.assertAlmostEqual(build_features(df)["overall_score"].iloc[0], 0.4)

    def test_correlation_degenerate_input(self):
        df = pd.DataFrame({"prepared_score": [0.1] * 10, "abnormal_ret": np.random.randn(10)})
        t = correlation_table(df, ["prepared_score"])
        self.assertTrue(np.isnan(t["spearman_rho"].iloc[0]))


def _synthetic(n_tickers=12, n_q=8, seed=0):
    """Transcripts whose tone drives the next-day abnormal return (planted signal)."""
    rng = np.random.default_rng(seed)
    pos = "Revenue growth was strong and margins improved across every segment this quarter. "
    neg = "Demand was weak and we faced significant headwinds and pressure on margins this quarter. "
    neu = "The company reported results for the quarter and discussed its operating segments today. "
    rows, closes = [], {}
    days = pd.bdate_range("2022-01-03", "2024-12-31")
    spy = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.005, len(days))), index=days)
    closes["SPY"] = spy
    for ti in range(n_tickers):
        tk = f"T{ti:02d}"
        px = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))
        px = pd.Series(px, index=days)
        for q in range(n_q):
            d = pd.Timestamp("2023-01-31") + pd.DateOffset(days=91 * q)
            d = days[days.searchsorted(d)]
            tone = rng.uniform(-1, 1)
            n_pos = int(round(20 * (1 + tone) / 2)); n_neg = 20 - n_pos
            prepared = (pos * n_pos + neg * n_neg + neu * 10)
            qa = pos * n_pos + neg * n_neg + neu * 10
            text = ("Operator: Welcome to the call. " + prepared +
                    " Operator: Our first question comes from Jane at Bank. " + qa)
            rows.append({"ticker": tk, "date": d, "text": text})
            pos_idx = days.get_loc(d)
            px.iloc[pos_idx + 1:] *= (1 + 0.04 * tone)  # reaction after T
        closes[tk] = px
    return pd.DataFrame(rows), pd.DataFrame(closes)


class TestEndToEnd(unittest.TestCase):
    def test_pipeline_recovers_planted_signal(self):
        raw, closes = _synthetic()
        df = select_universe(_normalise(raw), None, max_companies=20, last_n=8)
        self.assertEqual(len(df), 12 * 8)
        self.assertTrue(df["split_ok"].all())
        res = run(df, LexiconScorer(), closes=closes)
        self.assertEqual(len(res), 96)
        self.assertTrue(res["abnormal_ret"].notna().all())
        tbl = correlation_table(res).set_index("feature")
        self.assertGreater(tbl.loc["overall_score", "spearman_rho"], 0.5)
        self.assertLess(tbl.loc["overall_score", "spearman_p"], 0.001)

    def test_pipeline_finds_no_signal_in_noise(self):
        raw, closes = _synthetic(seed=1)
        rng = np.random.default_rng(5)
        closes = closes.copy()
        for c in closes.columns:
            closes[c] = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(closes)))  # reaction removed
        df = select_universe(_normalise(raw), None, 20, 8)
        res = run(df, LexiconScorer(), closes=closes)
        rho = correlation_table(res).set_index("feature").loc["overall_score", "spearman_rho"]
        self.assertLess(abs(rho), 0.35)


if __name__ == "__main__":
    unittest.main()
