import json

import joblib

from router import actions, features
from router.clients import make_client
from router.config import ACTIONS, MODELS, RESULTS, load
from router.grading import final_part
from router.policy import choose, prob_true
from router.retrieval import Library, get_embedder


class Service:
    def __init__(self):
        self.cfg = load()
        art = joblib.load(MODELS / "router.joblib")
        self.models = art["models"]
        self.order = art["order"]
        self.costs = art["costs"]
        self.k = art["top_k"]
        self.embedder = get_embedder(art["embedding_model"])
        self.library = Library()
        self.clients = {key: make_client(self.cfg["models"][key]) for key in ("weak", "strong")}
        self.max_tokens = self.cfg["exp"]["max_tokens"]["open"]

        self.default_threshold = 0.5
        curve_path = RESULTS / "curve.json"
        if curve_path.exists():
            with open(curve_path, encoding="utf-8") as f:
                self.default_threshold = json.load(f)["default_threshold"]

    def decide(self, question, threshold=None, force=None):
        t = self.default_threshold if threshold is None else threshold
        X, idx, scores = features.build([question], self.embedder, self.library, self.k)
        probs = {a: float(prob_true(self.models[a], X)[0]) for a in ACTIONS}
        action = force or choose(probs, self.order, t)
        passages = []
        for i, s in zip(idx[0], scores[0]):
            p = dict(self.library.passages[i])
            p["score"] = float(s)
            passages.append(p)
        return {"action": action, "threshold": t, "probs": probs, "passages": passages}

    def answer(self, question, threshold=None, force=None):
        d = self.decide(question, threshold, force)
        key, rag = actions.split_action(d["action"])
        used = d["passages"] if rag else []
        out = actions.run(
            d["action"],
            "open",
            question,
            used,
            self.clients,
            self.cfg["models"],
            self.max_tokens,
        )
        return {
            "text": out["text"],
            "answer": final_part(out["text"]),
            "usage": {
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
            },
            "router": {
                "action": d["action"],
                "model": self.cfg["models"][key]["name"],
                "retrieval": rag,
                "threshold": d["threshold"],
                "probs": d["probs"],
                "costs": self.costs,
                "cost": out["cost"],
                "seconds": round(out["seconds"], 3),
                "passages": used,
            },
        }
