# sp500_learning_orgs

Which S&P 500 companies talk the most about being a learning organization on earnings calls — and does it predict stock returns?

## Key Findings

**19,532 earnings call transcripts** scored across **629 companies** from **2005 to 2025**.

### The level of "learning org" language weakly predicts forward returns

Averaging the annual Q5-minus-Q1 return spread across all 20 years yields **+0.5% per 6-month period** in favor of companies with the highest learning-org scores. This average-of-annual-spreads methodology weights each year equally, avoiding the distortion that a naive pooled average introduces by overweighting recent years when stock prices (and therefore absolute return magnitudes) are higher. The composite score correlates with forward 6-month returns at r = +0.031. Small, but consistent: the signal is positive in 14 of 20 years measured.

All four concept clusters contribute positively, with psychological safety (+0.024) and learning org (+0.024) slightly ahead of lean/opex (+0.019) and devops/agile (+0.020).

### The change in score doesn't add signal

Computing a trailing 8-quarter slope ("inflection") per company yields near-zero correlation with forward returns (r = +0.002). The signal appears to be a **trait, not a state** — companies that score high tend to always score high, reflecting embedded culture rather than a strategic pivot. Sudden increases in learning-org buzzwords may even reflect cheap talk under pressure.

### No detectable signal decay

The signal is not weakening over time. If anything, the 2015-2024 period shows stronger correlations (avg r = +0.032) than 2006-2014 (avg r = +0.017). The negative years (2009, 2010, 2020, 2022) align with recessions and sharp drawdowns — it's **cyclical suppression, not alpha decay**. High-scoring companies tend to be higher-growth, higher-multiple names that underperform in risk-off environments.

### A secular rise in learning-org language

Average composite scores have risen steadily since ~2018, with a visible jump around COVID. Companies are increasingly adopting this vocabulary across the board.

### Top 10 companies by composite score

| Rank | Ticker | Company | Composite |
|------|--------|---------|-----------|
| 1 | PTC | PTC Inc. | 0.191 |
| 2 | PLTR | Palantir Technologies | 0.190 |
| 3 | JBL | Jabil Inc. | 0.187 |
| 4 | SW | Smurfit Westrock | 0.185 |
| 5 | DELL | Dell Technologies | 0.184 |
| 6 | NOW | ServiceNow | 0.181 |
| 7 | IEX | IDEX Corporation | 0.181 |
| 8 | FDS | FactSet Research Systems | 0.180 |
| 9 | WSM | Williams-Sonoma | 0.180 |
| 10 | FICO | Fair Isaac Corporation | 0.178 |

## Methodology

### Data sources

- **Earnings transcripts**: [kurry/sp500_earnings_transcripts](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts) on HuggingFace — full S&P 500 transcripts from 2005-2025, speaker-segmented, MIT licensed.
- **Stock prices**: [defeatbeta/yahoo-finance-data](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data) on HuggingFace — daily OHLCV prices, ODC-BY licensed.

### Scoring pipeline

1. **Extract prepared remarks** from each transcript's structured content — management presentation only, stopping at Q&A. Transcripts with fewer than 50 words of remarks are dropped.

2. **Sample 3 chunks** per transcript (beginning, middle, end; 200 words each) and embed them via **OpenAI `text-embedding-3-small`**.

3. **Score against 4 concept clusters** using cosine similarity between chunk embeddings and anchor phrase embeddings:
   - **Learning org**: "learning organization," "growth mindset," "continuous learning," "culture of learning," "knowledge sharing," "learning from mistakes," "organizational learning," "learn and adapt"
   - **Lean/OpEx**: "kaizen," "operational excellence," "andon cord," "continuous improvement," "lean manufacturing," "waste reduction," "six sigma," "process improvement," "total quality management"
   - **DevOps/Agile**: "DevOps," "test automation," "continuous deployment," "fast feedback loop," "agile development," "continuous integration," "iterative development," "rapid iteration," "ship fast and iterate"
   - **Psych safety**: "psychological safety," "speak up culture," "fail fast," "blame-free," "blameless post-mortem," "safe to fail," "open dialogue," "learning from failure," "trust and transparency"

4. **Hybrid scoring**: 70% embedding cosine similarity + 30% normalized keyword frequency. The keyword component catches exact phrase matches that embeddings might underweight.

5. **Composite score**: equal-weighted average of the four cluster scores.

### Returns analysis

For each earnings call, the **126-trading-day (~6 month) forward return** is computed from the first available closing price on or after the call date. Quintile analysis splits all transcript-return pairs by composite score and compares mean forward returns across quintiles.

## Reproducing

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI API key (for embeddings; ~$0.10 total cost)

### Setup and run

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up environment
echo "OPENAI_API_KEY=sk-..." > .env
uv sync

# Step 1: Score transcripts (~7 min, mostly API calls)
uv run python pipeline.py

# Step 2: Join with stock returns and generate plots (~30 sec)
uv run python returns_analysis.py

# Step 3 (optional): Inflection analysis
uv run python inflection_analysis.py
```

Both `pipeline.py` and `returns_analysis.py` auto-download their datasets from HuggingFace on first run. Intermediate results (extracted remarks, embeddings) are cached in `data/` to avoid redundant API calls.

### Output files

All outputs land in `data/` (gitignored):

| File | Description |
|------|-------------|
| `company_ranking.csv` | 450 companies ranked by composite + per-cluster scores |
| `quarterly_scores.csv` | 19,532 quarterly score time-series rows |
| `scores_with_returns.csv` | Scores joined with forward 6-month stock returns |
| `quintile_returns.png` | Bar chart: mean return by score quintile |
| `correlation_analysis.png` | Scatter plot + per-cluster correlation bars |
| `yearly_spread.png` | Q5-Q1 return spread by year |
| `score_timeseries.png` | Average composite score over time |
| `signal_decay.png` | Annual correlations and rolling windows over time |
| `inflection_analysis.png` | Slope quintile returns + level x slope interaction |

## Scripts

| Script | Purpose |
|--------|---------|
| `download_data.py` | Auto-downloads datasets from HuggingFace if missing |
| `pipeline.py` | Extracts remarks, embeds via OpenAI, scores 4 concept clusters |
| `returns_analysis.py` | Joins scores with forward 6-month stock returns, generates quintile analysis and plots |
| `inflection_analysis.py` | Measures trailing score slope per company, tests whether inflection predicts returns |

## Limitations

- **Semantic similarity is noisy.** A CEO saying "we're continuously improving our dividend" scores on "continuous improvement" even though it's unrelated to lean methodology. The hybrid keyword+embedding approach mitigates this but doesn't eliminate it.
- **Prepared remarks detection is heuristic.** The Q&A boundary is detected via keyword markers in the transcript text, which may misfire on some transcripts.
- **3 sampled chunks per transcript** is a cost-efficiency tradeoff. Embedding the full text would give better coverage but at ~10x the API cost.
- **Survivorship bias.** The dataset contains current and former S&P 500 constituents, but companies that were delisted or acquired may be underrepresented.
- **No risk adjustment.** Forward returns are raw, not adjusted for beta, sector, or market regime. The cyclical suppression pattern suggests that controlling for market conditions would sharpen the signal.
