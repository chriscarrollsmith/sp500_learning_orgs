# sp500_learning_orgs

Research on which S&P 500 companies yap the most about being a learning organization

## Datasets

GeminiLn/EarningsCall_Dataset: CEO-only transcripts (most-spoken executive isolated) for S&P 500, with matched audio recordings if you want vocal/tone features too. Zip file available here: https://drive.google.com/drive/folders/1BKCANORbcmUJKkOkBOghw6uNHPqS_az1.

Hugging Face — kurry/sp500_earnings_transcripts: Hugging Face Full S&P 500 transcripts from 2005–2025, speaker-segmented, MIT licensed. This is probably your best starting point. Download link: https://huggingface.co/datasets/kurry/sp500_earnings_transcripts/resolve/main/parquet_files/part-0.parquet.

**Data pipeline:**
1. Pull the HuggingFace dataset (`kurry/sp500_earnings_transcripts`)
2. Focus on the **management presentation** sections (prepared remarks), not Q&A, since that's where managers speak most deliberately

**Semantic analysis approach:**
- Pure keyword matching will miss a lot — a CEO talking about "blameless post-mortems" or "continuous improvement culture" won't match "psychological safety" literally
- Better to use **sentence embeddings** (e.g. `sentence-transformers/all-MiniLM-L6-v2`) and compute cosine similarity against your concept queries
- You'd define a few anchor phrases per concept cluster:
  - *Learning org*: "learning organization," "growth mindset," "continuous learning"
  - *Lean/OpEx*: "kaizen," "operational excellence", "andon cord"
  - *DevOps/Agile*: "DevOps," "test automation," "continuous deployment", "fast feedback loop"
  - *Psych safety*: "psychological safety," "speak up," "fail fast," "blame-free"
