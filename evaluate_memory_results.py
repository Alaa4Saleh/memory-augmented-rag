"""
evaluate_memory_results.py

Reads results from:
  results/gemini_memory_runs.jsonl

Computes per-probe and per-method metrics:
  - cosine similarity between gold_answer and model_answer
  - keyword_recall: fraction of gold keywords found in model_answer
  - composite memory_score = 0.6 * cosine + 0.4 * keyword_recall

Outputs:
  results/memory_eval_summary.csv

Requirements:
  pip install sentence-transformers numpy pandas
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

RESULTS_PATH = Path("results/gemini_memory_runs3.jsonl")
SUMMARY_PATH = Path("results/memory_eval_summary4.csv")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Load results
# -----------------------------

def load_results(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# -----------------------------
# Metrics
# -----------------------------

class TextEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            # zero vector
            return np.zeros(384, dtype=np.float32)
        return self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def keyword_recall(answer: str, keywords: List[str]) -> float:
    """
    Very simple metric:
      recall = (#keywords present as substrings in answer) / (#keywords)
    All lowercased.
    """
    answer_l = answer.lower()
    if not keywords:
        return 0.0
    hits = 0
    for kw in keywords:
        if kw and kw.lower() in answer_l:
            hits += 1
    return hits / len(keywords)


def compute_metrics(records: List[Dict[str, Any]]) -> pd.DataFrame:
    embedder = TextEmbedder()

    rows = []
    for rec in records:
        gold = rec.get("gold_answer", "") or ""
        pred = rec.get("model_answer", "") or ""
        keywords = rec.get("keywords", []) or []

        gold_emb = embedder.embed(gold)
        pred_emb = embedder.embed(pred)
        cos = cosine_similarity(gold_emb, pred_emb)
        kw_rec = keyword_recall(pred, keywords)
        memory_score = 0.6 * cos + 0.4 * kw_rec

        rows.append({
            "conversation_id": rec.get("conversation_id"),
            "memory_method": rec.get("memory_method"),
            "probe_turn": rec.get("probe_turn"),
            "probe_type": rec.get("probe_type"),
            "cosine_sim": cos,
            "keyword_recall": kw_rec,
            "memory_score": memory_score,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Compare to no-memory baseline
# -----------------------------

def add_improvements(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each probe (conversation_id + probe_turn + probe_type), we:
      - find the no_memory record
      - subtract its memory_score from that of other methods
    """
    # pivot baseline
    baseline = df[df["memory_method"] == "no_memory"].copy()
    baseline = baseline.set_index(["conversation_id", "probe_turn", "probe_type"])

    improvements = []
    for idx, row in df.iterrows():
        key = (row["conversation_id"], row["probe_turn"], row["probe_type"])
        if row["memory_method"] == "no_memory":
            delta = 0.0
        else:
            if key in baseline.index:
                base_score = baseline.loc[key, "memory_score"]
                delta = row["memory_score"] - base_score
            else:
                delta = 0.0
        r = row.to_dict()
        r["delta_vs_no_memory"] = delta
        improvements.append(r)

    return pd.DataFrame(improvements)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by memory_method and probe_type.
    """
    summary = (
        df.groupby(["memory_method", "probe_type"])
        .agg(
            mean_cosine=("cosine_sim", "mean"),
            mean_keyword_recall=("keyword_recall", "mean"),
            mean_memory_score=("memory_score", "mean"),
            mean_delta_vs_no_memory=("delta_vs_no_memory", "mean"),
            count=("memory_score", "count"),
        )
        .reset_index()
        .sort_values(["probe_type", "mean_memory_score"], ascending=[True, False])
    )
    return summary


# -----------------------------
# Main
# -----------------------------

def main():
    if not RESULTS_PATH.exists():
        print(f"❌ Results file not found: {RESULTS_PATH}")
        return

    print(f"📂 Loading model outputs from: {RESULTS_PATH}")
    records = load_results(RESULTS_PATH)
    print(f"Loaded {len(records)} probe answers.")

    print("🔢 Computing metrics...")
    df = compute_metrics(records)
    df_with_delta = add_improvements(df)
    summary = summarize(df_with_delta)

    print("\n📊 Summary by memory_method and probe_type:")
    print(summary)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\n💾 Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
