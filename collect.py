import argparse
import json

import pandas as pd
from tqdm import tqdm

from router import actions
from router.clients import make_client
from router.config import ACTIONS, DATA, RESPONSES, ensure_dirs, load
from router.data import gold_of
from router.grading import grade
from router.retrieval import Library, embed, get_embedder


def done_ids(path):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f}


def collect(action, df, cfg, shared):
    exp = cfg["exp"]
    path = RESPONSES / f"{action}.jsonl"
    todo = df[~df["id"].isin(done_ids(path))]
    print(action, "-", len(todo), "questions left")
    if todo.empty:
        return

    key, rag = actions.split_action(action)
    clients = {key: make_client(cfg["models"][key])}

    idx = scores = None
    if rag:
        if "library" not in shared:
            shared["embedder"] = get_embedder(exp["embedding_model"])
            shared["library"] = Library()
        vecs = embed(shared["embedder"], list(todo["question"]), show=True)
        idx, scores = shared["library"].search_many(vecs, exp["top_k"])

    rows = todo.to_dict("records")
    with open(path, "a", encoding="utf-8") as f:
        for n, row in enumerate(tqdm(rows)):
            passages = None
            if rag:
                passages = [shared["library"].passages[i] for i in idx[n]]
            res = actions.run(
                action,
                row["kind"],
                row["question"],
                passages,
                clients,
                cfg["models"],
                exp["max_tokens"][row["kind"]],
            )
            rec = {
                "id": row["id"],
                "action": action,
                "text": res["text"],
                "correct": bool(grade(row["kind"], res["text"], gold_of(row))),
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "seconds": round(res["seconds"], 3),
                "cost": res["cost"],
            }
            if passages:
                rec["passages"] = [p["id"] for p in passages]
                rec["top_score"] = float(scores[n][0])
            f.write(json.dumps(rec) + "\n")
            f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=ACTIONS + ["all"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    cfg = load()
    df = pd.read_parquet(DATA / "queries.parquet")
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0)

    shared = {}
    for action in (ACTIONS if args.action == "all" else [args.action]):
        collect(action, df, cfg, shared)


if __name__ == "__main__":
    main()
