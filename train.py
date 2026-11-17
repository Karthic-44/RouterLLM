import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from router import features
from router.config import ACTIONS, DATA, MODELS, RESPONSES, ensure_dirs, load
from router.policy import cost_order, prob_true
from router.retrieval import Library, get_embedder


def read_responses(action):
    rows = []
    with open(RESPONSES / f"{action}.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows).set_index("id")


def candidates():
    return {
        "logreg_c0.01": make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)),
        "logreg_c0.1": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
        "boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0),
    }


def fit_best(Xtr, ytr, Xva, yva):
    if len(set(ytr)) < 2:
        return DummyClassifier(strategy="prior").fit(Xtr, ytr), "constant"
    if len(yva) == 0:
        Xva, yva = Xtr, ytr
    best, best_name, best_loss = None, "", 1e9
    for name, model in candidates().items():
        model.fit(Xtr, ytr)
        p = np.clip(prob_true(model, Xva), 1e-4, 1 - 1e-4)
        loss = log_loss(yva, p, labels=[0, 1])
        print("  ", name, "val log loss", round(loss, 4))
        if loss < best_loss:
            best, best_name, best_loss = model, name, loss
    return best, best_name


def main():
    ensure_dirs()
    cfg = load()
    exp = cfg["exp"]

    df = pd.read_parquet(DATA / "queries.parquet")
    resp = {a: read_responses(a) for a in ACTIONS}
    have = set(df["id"])
    for a in ACTIONS:
        have &= set(resp[a].index)
    df = df[df["id"].isin(have)].reset_index(drop=True)
    print(len(df), "questions have answers from all four pipelines")

    embedder = get_embedder(exp["embedding_model"])
    library = Library()
    X, idx, scores = features.build(df["question"], embedder, library, exp["top_k"])

    train = (df["split"] == "train").to_numpy()
    val = (df["split"] == "val").to_numpy()
    ids = df["id"]

    Y = {a: resp[a].loc[ids, "correct"].astype(int).to_numpy() for a in ACTIONS}
    models, probs = {}, {}
    for a in ACTIONS:
        print(a)
        models[a], name = fit_best(X[train], Y[a][train], X[val], Y[a][val])
        probs[a] = prob_true(models[a], X)
        if val.any() and len(set(Y[a][val])) == 2:
            auc = roc_auc_score(Y[a][val], probs[a][val])
            print("  chose", name, "- val auc", round(auc, 3))

    costs = {a: float(resp[a].loc[ids[train], "cost"].mean()) for a in ACTIONS}
    print("mean cost per question on train:", {a: round(c, 8) for a, c in costs.items()})

    out = df[["id", "split", "dataset"]].copy()
    for a in ACTIONS:
        out["p_" + a] = probs[a]
    out["top1"] = X[:, -4]
    out["gap"] = X[:, -2]
    out.to_parquet(DATA / "predictions.parquet", index=False)

    joblib.dump(
        {
            "actions": ACTIONS,
            "models": models,
            "costs": costs,
            "order": cost_order(costs),
            "embedding_model": exp["embedding_model"],
            "top_k": exp["top_k"],
        },
        MODELS / "router.joblib",
    )
    print("saved models/router.joblib and data/predictions.parquet")


if __name__ == "__main__":
    main()
