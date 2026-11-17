import re

import numpy as np

from router.retrieval import embed

WH = re.compile(r"^(who|what|which|when|where|whose|whom)\b", re.IGNORECASE)
OP = re.compile(r"\d\s*[-+*/=x]\s*\d")
OPTIONS = re.compile(r"\n[A-E]\)")


def simple(text):
    words = text.split()
    caps = sum(1 for w in words[1:] if w[:1].isupper())
    digits = sum(c.isdigit() for c in text)
    return [
        len(words),
        len(text),
        digits,
        caps,
        1 if OP.search(text) else 0,
        1 if WH.match(text.strip()) else 0,
        1 if OPTIONS.search(text) else 0,
    ]


def retrieval_stats(scores):
    s = np.asarray(scores)
    top1 = s[:, 0]
    top2 = s[:, 1]
    return np.stack([top1, top2, top1 - top2, s.mean(axis=1)], axis=1)


def build(texts, embedder, library, k):
    texts = list(texts)
    vecs = embed(embedder, texts)
    idx, scores = library.search_many(vecs, k)
    simp = np.array([simple(t) for t in texts], dtype="float32")
    X = np.hstack([vecs, simp, retrieval_stats(scores)]).astype("float32")
    return X, idx, scores
