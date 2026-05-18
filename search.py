"""
AROsearch — query-time hybrid retrieval.

Loads the FAISS index, BM25 index, and metadata once. For each query:
  1. Encode query with Qwen3-Embedding, run FAISS top-K.
  2. Tokenize query, run BM25S top-K over the same corpus.
  3. Min-max normalize BM25 scores to [0,1] (FAISS cosine is already [-1,1]).
  4. Fuse with sigmoid-adaptive alpha: when lexical match is strong, weight
     it more; when weak, let semantic dominate.
  5. Return ranked, deduplicated hits with metadata.

The fusion follows VectorSage (Wijesekara et al. 2026, Bioinformatics Advances).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import bm25s
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Sigmoid-adaptive fusion parameters (from VectorSage paper, balanced default).
# alpha = 1 / (1 + exp(-A * (bm25_norm - B)))
# A controls sharpness; B is the midpoint where lexical and semantic contribute equally.
SIGMOID_A = 10.0
SIGMOID_B = 0.5

# How many candidates to pull from each retriever before fusion.
CANDIDATE_K = 50


@dataclass
class Hit:
    aro_id: str
    name: str
    description: str
    drug_classes: str
    mechanisms: str
    families: str
    score: float
    bm25_norm: float
    cosim: float
    alpha: float


class AROSearcher:
    def __init__(self, data_dir: Path = DATA_DIR):
        # Friendly error if artifacts haven't been built/committed yet
        required = ["aro_index.faiss", "aro_bm25.pkl", "aro_meta.parquet"]
        missing = [f for f in required if not (data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing index artifacts: {missing}. "
                f"Build them by running build_index_colab.ipynb on Colab, "
                f"download the zip, and place contents in {data_dir}/"
            )

        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(str(data_dir / "aro_index.faiss"))

        print("Loading BM25 index...")
        with (data_dir / "aro_bm25.pkl").open("rb") as f:
            self.bm25 = pickle.load(f)

        print("Loading metadata...")
        self.meta = pd.read_parquet(data_dir / "aro_meta.parquet")

        print(f"Loading embedding model {EMBED_MODEL} (fp16)...")
        self.encoder = SentenceTransformer(EMBED_MODEL, trust_remote_code=True).half()

        print(f"Ready. {len(self.meta)} ARO entries loaded.")

    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        return vec

    def _dense_topk(self, query: str, k: int) -> dict[int, float]:
        vec = self._encode_query(query)
        scores, idx = self.faiss_index.search(vec, k)
        # FAISS returns inner-product on L2-normalized vectors == cosine in [-1, 1].
        return {int(i): float(s) for i, s in zip(idx[0], scores[0]) if i >= 0}

    def _sparse_topk(self, query: str, k: int) -> dict[int, float]:
        tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = self.bm25.retrieve(tokens, k=k)
        return {int(i): float(s) for i, s in zip(results[0], scores[0])}

    @staticmethod
    def _normalize_bm25(scores: dict[int, float]) -> dict[int, float]:
        """Min-max normalize BM25 scores to [0,1] over the candidate set.

        Edge cases:
          - Empty input: return {}.
          - All scores zero (BM25 had no real lexical match): return {} so
            these docs aren't spuriously promoted to score=1.0. The dense
            retriever's results will dominate via cosine.
          - All scores equal but nonzero (genuine tie): return all 1.0.
        """
        if not scores:
            return {}
        values = np.array(list(scores.values()))
        # No actual lexical match — BM25 returned default-zero scores.
        # This happens for queries like "pikachu" or anything with no in-vocab terms.
        if values.max() <= 1e-9:
            return {}
        lo, hi = values.min(), values.max()
        if hi - lo < 1e-9:
            return {k: 1.0 for k in scores}  # genuine tie at a nonzero score
        return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}

    @staticmethod
    def _fuse(bm25_norm: float, cosim: float) -> tuple[float, float]:
        """Sigmoid-adaptive fusion. Returns (final_score, alpha).

        alpha rises with bm25_norm: when the query has strong lexical match,
        alpha approaches 1 and BM25 dominates. Weak lexical match means alpha
        near 0 and cosine similarity dominates. Cosine is clamped at 0 below
        so a strongly negative cosine doesn't drag positive lexical matches down.
        """
        cosim_clamped = max(0.0, cosim)
        alpha = 1.0 / (1.0 + np.exp(-SIGMOID_A * (bm25_norm - SIGMOID_B)))
        score = alpha * bm25_norm + (1.0 - alpha) * cosim_clamped
        return float(score), float(alpha)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        if not query.strip():
            return []

        dense = self._dense_topk(query, CANDIDATE_K)
        sparse_raw = self._sparse_topk(query, CANDIDATE_K)
        sparse = self._normalize_bm25(sparse_raw)

        # Union of candidate IDs from both retrievers.
        candidates = set(dense) | set(sparse)

        hits: list[Hit] = []
        for idx in candidates:
            cosim = dense.get(idx, 0.0)
            bm25_norm = sparse.get(idx, 0.0)
            score, alpha = self._fuse(bm25_norm, cosim)
            row = self.meta.iloc[idx]
            hits.append(Hit(
                aro_id=row["aro_id"],
                name=row["name"],
                description=row["description"],
                drug_classes=row["drug_classes"],
                mechanisms=row["mechanisms"],
                families=row["families"],
                score=score,
                bm25_norm=bm25_norm,
                cosim=cosim,
                alpha=alpha,
            ))

        # Sort by fused score; tie-break by higher BM25 (preserve exact-match precision).
        hits.sort(key=lambda h: (h.score, h.bm25_norm), reverse=True)
        return hits[:top_k]


if __name__ == "__main__":
    # Smoke test
    s = AROSearcher()
    for q in [
        "enzymes that hydrolyze carbapenems in Klebsiella",
        "KPC-2",
        "efflux pump conferring resistance to fluoroquinolones",
    ]:
        print(f"\n=== {q} ===")
        for h in s.search(q, top_k=5):
            print(f"  {h.aro_id:14}  score={h.score:.3f}  α={h.alpha:.2f}  "
                  f"cos={h.cosim:.2f}  bm25={h.bm25_norm:.2f}  {h.name}")
