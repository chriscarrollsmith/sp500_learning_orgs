"""
S&P 500 Learning Organization Analysis Pipeline

Scores earnings call transcripts on "learning org" concepts using:
  1. OpenAI embeddings API (text-embedding-3-small) for semantic similarity
  2. Keyword frequency boost for exact/near matches
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time
import gc
import os
from download_data import ensure_data

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Concept anchor phrases
# ---------------------------------------------------------------------------
CONCEPT_CLUSTERS = {
    "learning_org": [
        "learning organization",
        "growth mindset",
        "continuous learning",
        "culture of learning",
        "knowledge sharing",
        "learning from mistakes",
        "organizational learning",
        "learn and adapt",
    ],
    "lean_opex": [
        "kaizen",
        "operational excellence",
        "andon cord",
        "continuous improvement",
        "lean manufacturing",
        "waste reduction",
        "six sigma",
        "process improvement",
        "total quality management",
    ],
    "devops_agile": [
        "DevOps",
        "test automation",
        "continuous deployment",
        "fast feedback loop",
        "agile development",
        "continuous integration",
        "iterative development",
        "rapid iteration",
        "ship fast and iterate",
    ],
    "psych_safety": [
        "psychological safety",
        "speak up culture",
        "fail fast",
        "blame-free",
        "blameless post-mortem",
        "safe to fail",
        "open dialogue",
        "learning from failure",
        "trust and transparency",
    ],
}

# ---------------------------------------------------------------------------
# 2. Extract management presentation (prepared remarks before Q&A)
# ---------------------------------------------------------------------------
QA_MARKERS = [
    "q&a", "q & a", "question-and-answer", "question and answer",
    "questions and answers", "open the line", "open up the line",
    "first question", "operator instructions",
    "we'll now take questions", "we will now take questions",
    "open it up for questions",
]


def extract_prepared_remarks(structured_content):
    """Return concatenated text from management's prepared remarks only."""
    if structured_content is None or (hasattr(structured_content, '__len__') and len(structured_content) == 0):
        return ""

    remarks = []
    for seg in structured_content:
        speaker = (seg.get("speaker") or "").strip()
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        text_lower = text.lower()
        if any(marker in text_lower for marker in QA_MARKERS):
            if speaker.lower() in ("operator", ""):
                break
            if any(m in text_lower for m in [
                "q&a", "q & a", "question-and-answer", "question and answer",
                "open the line", "open it up for questions",
            ]):
                break

        if speaker.lower() == "operator":
            continue

        remarks.append(text)

    return " ".join(remarks)


# ---------------------------------------------------------------------------
# 3. OpenAI embeddings with batching and retry
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
# OpenAI allows up to 2048 texts per request; keep batches moderate
API_BATCH_SIZE = 500
EMBEDDINGS_CACHE = "data/embeddings.npy"
CHUNK_MAP_CACHE = "data/chunk_map.npy"
CHUNKS_META_CACHE = "data/chunks_meta.npz"


def embed_batch(client, texts, model=EMBEDDING_MODEL):
    """Embed a batch of texts via OpenAI API with retry."""
    for attempt in range(5):
        try:
            resp = client.embeddings.create(input=texts, model=model)
            return np.array([d.embedding for d in resp.data], dtype=np.float32)
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt
                print(f"  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def sample_chunks(text, n_samples=3, chunk_words=200):
    """Pick n evenly-spaced chunks from the text. Larger chunks for API (cheaper per token)."""
    words = text.split()
    total = len(words)
    if total <= chunk_words:
        return [text]

    chunks = []
    positions = np.linspace(0, max(0, total - chunk_words), n_samples, dtype=int)
    for pos in positions:
        chunks.append(" ".join(words[pos:pos + chunk_words]))
    return chunks


# ---------------------------------------------------------------------------
# 4. Keyword frequency scoring
# ---------------------------------------------------------------------------
def keyword_score(text_lower, phrases):
    """Count anchor phrase hits normalized by text length."""
    word_count = len(text_lower.split())
    if word_count == 0:
        return 0.0
    hits = sum(1 for phrase in phrases if phrase.lower() in text_lower)
    return hits / (word_count / 1000.0)


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
REMARKS_CACHE = "data/remarks.parquet"
SCORES_FILE = "data/quarterly_scores.csv"
RANKING_FILE = "data/company_ranking.csv"


def step1_extract_remarks():
    """Extract prepared remarks from all transcripts."""
    if os.path.exists(REMARKS_CACHE):
        print(f"Remarks cache exists, skipping extraction.")
        return

    ensure_data("transcripts.parquet")
    print("Step 1: Extracting prepared remarks...")
    pf = pq.ParquetFile("data/transcripts.parquet")
    total = pf.metadata.num_rows
    all_rows = []

    for batch in tqdm(pf.iter_batches(batch_size=500), total=(total // 500) + 1, desc="Extracting"):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            remarks = extract_prepared_remarks(row["structured_content"])
            if len(remarks.split()) < 50:
                continue
            all_rows.append({
                "symbol": row["symbol"],
                "company_name": row["company_name"],
                "year": int(row["year"]),
                "quarter": int(row["quarter"]),
                "date": row["date"],
                "remarks": remarks,
            })
        del df
        gc.collect()

    remarks_df = pd.DataFrame(all_rows)
    remarks_df.to_parquet(REMARKS_CACHE, index=False)
    print(f"  Saved {len(remarks_df)} transcripts to {REMARKS_CACHE}")


def step2_embed():
    """Embed all chunks via OpenAI API. Caches results to disk."""
    if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(CHUNK_MAP_CACHE):
        print("Embeddings cache exists, skipping API calls.")
        return

    print("Step 2: Embedding chunks via OpenAI API...")
    client = OpenAI()

    remarks_df = pd.read_parquet(REMARKS_CACHE)
    print(f"  {len(remarks_df)} transcripts")

    # Build chunks
    print("  Sampling chunks...")
    all_chunks = []
    chunk_map = []

    for i, row in remarks_df.iterrows():
        chunks = sample_chunks(row["remarks"])
        for c in chunks:
            all_chunks.append(c)
            chunk_map.append(i)

    chunk_map = np.array(chunk_map)
    print(f"  {len(all_chunks)} chunks to embed")

    # Embed in batches
    all_embeddings = np.zeros((len(all_chunks), EMBEDDING_DIMS), dtype=np.float32)
    for start in tqdm(range(0, len(all_chunks), API_BATCH_SIZE), desc="Embedding"):
        end = min(start + API_BATCH_SIZE, len(all_chunks))
        batch_embs = embed_batch(client, all_chunks[start:end])
        all_embeddings[start:end] = batch_embs

    # Save to disk
    np.save(EMBEDDINGS_CACHE, all_embeddings)
    np.save(CHUNK_MAP_CACHE, chunk_map)
    print(f"  Saved embeddings to {EMBEDDINGS_CACHE}")


def step3_score():
    """Score transcripts using cached embeddings + keyword hybrid."""
    print("Step 3: Scoring transcripts...")

    remarks_df = pd.read_parquet(REMARKS_CACHE)
    chunk_embs = np.load(EMBEDDINGS_CACHE)
    chunk_map = np.load(CHUNK_MAP_CACHE)

    client = OpenAI()
    n = len(remarks_df)

    # Embed anchor phrases
    print("  Embedding anchor phrases...")
    anchor_embs = {}
    for name, phrases in CONCEPT_CLUSTERS.items():
        anchor_embs[name] = embed_batch(client, phrases)

    # Normalize all embeddings for cosine similarity via dot product
    def normalize(arr):
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms

    chunk_embs = normalize(chunk_embs)
    for name in anchor_embs:
        anchor_embs[name] = normalize(anchor_embs[name])

    # Embedding scores per cluster
    print("  Computing similarity scores...")
    embedding_scores = {name: np.zeros(n) for name in CONCEPT_CLUSTERS}

    for cluster_name, anch_emb in anchor_embs.items():
        # Cosine similarity (already normalized)
        BLOCK = 20000
        max_sims = np.zeros(len(chunk_embs))
        for s in range(0, len(chunk_embs), BLOCK):
            e = min(s + BLOCK, len(chunk_embs))
            sims = chunk_embs[s:e] @ anch_emb.T
            max_sims[s:e] = sims.max(axis=1)

        # Aggregate per transcript
        for t_idx in range(n):
            mask = chunk_map == t_idx
            if mask.any():
                embedding_scores[cluster_name][t_idx] = max_sims[mask].mean()

    # Keyword scoring
    print("  Computing keyword scores...")
    kw_scores = {name: np.zeros(n) for name in CONCEPT_CLUSTERS}
    for i, row in tqdm(remarks_df.iterrows(), total=n, desc="Keywords"):
        text_lower = row["remarks"].lower()
        for cluster_name, phrases in CONCEPT_CLUSTERS.items():
            kw_scores[cluster_name][i] = keyword_score(text_lower, phrases)

    # Hybrid: 70% embedding + 30% keyword (normalized)
    print("  Computing hybrid scores...")
    for cluster_name in CONCEPT_CLUSTERS:
        emb = embedding_scores[cluster_name]
        kw = kw_scores[cluster_name]

        kw_min, kw_max = kw.min(), kw.max()
        if kw_max > kw_min:
            kw_normed = (kw - kw_min) / (kw_max - kw_min)
        else:
            kw_normed = np.zeros_like(kw)

        remarks_df[cluster_name] = 0.7 * emb + 0.3 * kw_normed

    remarks_df["composite"] = remarks_df[list(CONCEPT_CLUSTERS.keys())].mean(axis=1)
    remarks_df["date"] = pd.to_datetime(remarks_df["date"])

    # Save time-series
    timeseries = (
        remarks_df
        .sort_values(["symbol", "date"])
        .drop(columns=["remarks"])
    )
    timeseries.to_csv(SCORES_FILE, index=False)

    # Save ranking
    score_cols = list(CONCEPT_CLUSTERS.keys()) + ["composite"]
    ranking = (
        timeseries
        .groupby(["symbol", "company_name"])[score_cols]
        .mean()
        .sort_values("composite", ascending=False)
        .reset_index()
    )
    ranking.to_csv(RANKING_FILE, index=False)

    print(f"\n{'='*65}")
    print(f" TOP 25 COMPANIES BY LEARNING ORG COMPOSITE SCORE")
    print(f"{'='*65}")
    print(ranking.head(25).to_string(index=False, float_format="%.4f"))

    print(f"\n{'='*65}")
    print(f" TOP 10 PER CLUSTER")
    print(f"{'='*65}")
    for cluster in CONCEPT_CLUSTERS:
        top = ranking.nlargest(10, cluster)[["symbol", "company_name", cluster]]
        print(f"\n  {cluster}:")
        for _, r in top.iterrows():
            print(f"    {r['symbol']:6s} {r['company_name'][:35]:35s} {r[cluster]:.4f}")

    print(f"\nOutputs:")
    print(f"  {SCORES_FILE} ({len(timeseries)} rows) — quarterly scores per company")
    print(f"  {RANKING_FILE} ({len(ranking)} rows) — company rankings")
    print(f"\nDate range: {timeseries['date'].min()} to {timeseries['date'].max()}")
    print(f"Companies: {timeseries['symbol'].nunique()}")

    print("\nScore distributions:")
    for col in score_cols:
        s = timeseries[col]
        print(f"  {col:15s}  mean={s.mean():.4f}  std={s.std():.4f}  "
              f"p25={s.quantile(.25):.4f}  p75={s.quantile(.75):.4f}  max={s.max():.4f}")


def main():
    step1_extract_remarks()
    step2_embed()
    step3_score()
    print("\nDone!")


if __name__ == "__main__":
    main()
