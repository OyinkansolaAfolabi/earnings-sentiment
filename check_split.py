from datasets import load_dataset
from src.transcripts import split_sections

ds = load_dataset("kurry/sp500_earnings_transcripts", split="train")
ds = ds.select_columns(["symbol", "date", "content"])
for tk in ["AAPL", "MSFT", "NVDA"]:
    sub = ds.filter(lambda s: s == tk, input_columns=["symbol"]).sort("date")
    for i in range(len(sub) - 4, len(sub)):
        r = sub[i]
        p, q, ok = split_sections(r["content"])
        print(f"\n=== {tk} {r['date'][:10]} ok={ok} prepared={len(p)} chars, qa={len(q)} chars")
        print("  END OF PREPARED:", p[-150:].replace("\n", " "))
        print("  START OF Q&A:   ", q[:150].replace("\n", " "))