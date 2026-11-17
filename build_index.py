import json

import numpy as np

from router.config import CORPUS, load
from router.retrieval import embed, get_embedder


def main():
    cfg = load()
    embedder = get_embedder(cfg["exp"]["embedding_model"])
    texts = []
    with open(CORPUS / "passages.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            texts.append(p["title"] + ". " + p["text"])
    print(len(texts), "passages to embed")
    vecs = embed(embedder, texts, batch_size=128, show=True)
    np.save(CORPUS / "embeddings.npy", vecs)
    print("saved", vecs.shape)


if __name__ == "__main__":
    main()
