import re
from datasets import load_dataset

ds = load_dataset("kurry/sp500_earnings_transcripts", split="train")
ds = ds.select_columns(["symbol", "date", "content"])
aapl = ds.filter(lambda s: s == "AAPL", input_columns=["symbol"])
aapl = aapl.sort("date")
row = aapl[len(aapl) - 1]
text = row["content"]
print("DATE:", row["date"], "| LENGTH:", len(text))
for m in list(re.finditer(r"question", text, re.IGNORECASE))[:12]:
    pos = m.start() / len(text) * 100
    snippet = text[max(0, m.start() - 100): m.end() + 100].replace("\n", " ")
    print(f"\n[{pos:.0f}% through] ...{snippet}...")