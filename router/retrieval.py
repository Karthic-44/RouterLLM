import json
from functools import lru_cache

import numpy as np

from router.config import CORPUS


@lru_cache(maxsize=1)
def get_embedder(name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(embedder, texts, batch_size=64, show=False):
    vecs = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show,
    )
    return np.asarray(vecs, dtype="float32")


def passage_block(passages):
    lines = []
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}] {p['title']}: {p['text']}")
    return "\n".join(lines)


class Library:
    def __init__(self):
        self.passages = []
        with open(CORPUS / "passages.jsonl", encoding="utf-8") as f:
            for line in f:
                self.passages.append(json.loads(line))
        self.emb = np.load(CORPUS / "embeddings.npy")
        if len(self.passages) != len(self.emb):
            raise ValueError("passages.jsonl and embeddings.npy do not match, run build_index.py again")

    def search_many(self, vecs, k, chunk=256):
        k = min(k, len(self.emb))
        all_idx, all_scores = [], []
        for start in range(0, len(vecs), chunk):
            sims = vecs[start:start + chunk] @ self.emb.T
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(sims, idx, axis=1)
            order = np.argsort(-vals, axis=1)
            all_idx.append(np.take_along_axis(idx, order, axis=1))
            all_scores.append(np.take_along_axis(vals, order, axis=1))
        return np.vstack(all_idx), np.vstack(all_scores)
