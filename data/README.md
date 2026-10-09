# Transcript input format

Place your transcript file here (it is git-ignored because transcripts are usually licensed content).

Required columns:

| column   | meaning                                   |
|----------|-------------------------------------------|
| ticker   | e.g. `AAPL`                               |
| date     | call date, `YYYY-MM-DD`                   |
| text     | full transcript (prepared remarks + Q&A)  |

Alternatively supply `prepared` and `qa` columns instead of `text` if your source already separates them.
Formats: `.csv`, `.jsonl`, `.parquet`.
