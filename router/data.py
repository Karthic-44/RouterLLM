import json
import random

import pandas as pd

from router.config import DATA


def gold_of(row):
    return json.loads(row["gold"])


def pick(rows, n, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_gsm8k(n, seed):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i, r in enumerate(ds):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({
            "id": f"gsm8k-{i}",
            "dataset": "gsm8k",
            "kind": "math",
            "question": r["question"].strip(),
            "gold": json.dumps([gold]),
            "pop": float("nan"),
            "wiki_title": "",
        })
    return pick(rows, n, seed)


def load_arc(n, seed):
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for i, r in enumerate(ds):
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        letters = "ABCDE"[:len(texts)]
        options = [f"{l}) {t}" for l, t in zip(letters, texts)]
        gold = letters[labels.index(r["answerKey"])]
        rows.append({
            "id": f"arc-{i}",
            "dataset": "arc",
            "kind": "mc",
            "question": r["question"].strip() + "\n" + "\n".join(options),
            "gold": json.dumps([gold]),
            "pop": float("nan"),
            "wiki_title": "",
        })
    return pick(rows, n, seed)


def parse_answers(raw):
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        import ast

        return list(ast.literal_eval(raw))


def load_popqa(n, seed):
    local = DATA / "raw" / "popQA.tsv"
    if local.exists():
        records = pd.read_csv(local, sep="\t").to_dict("records")
    else:
        from datasets import load_dataset

        records = list(load_dataset("akariasai/PopQA", split="test"))
    rows = []
    for i, r in enumerate(records):
        title = r.get("s_wiki_title")
        if not isinstance(title, str) or not title:
            continue
        rows.append({
            "id": f"popqa-{i}",
            "dataset": "popqa",
            "kind": "fact",
            "question": r["question"].strip(),
            "gold": json.dumps(parse_answers(r["possible_answers"])),
            "pop": float(r.get("s_pop") or 0),
            "wiki_title": title,
        })
    return pick(rows, n, seed)


LOADERS = {"gsm8k": load_gsm8k, "arc": load_arc, "popqa": load_popqa}


def assign_splits(df, exp):
    seed = exp["seed"]
    holdout = exp.get("holdout")
    df = df.copy()
    df["split"] = ""
    for name, part in df.groupby("dataset"):
        idx = list(part.index)
        if name == holdout:
            df.loc[idx, "split"] = "ood"
            continue
        random.Random(seed).shuffle(idx)
        n_train = int(len(idx) * exp["split"]["train"])
        n_val = int(len(idx) * exp["split"]["val"])
        df.loc[idx[:n_train], "split"] = "train"
        df.loc[idx[n_train:n_train + n_val], "split"] = "val"
        df.loc[idx[n_train + n_val:], "split"] = "test"
    return df


def build_queries(exp):
    rows = []
    for name, n in exp["datasets"].items():
        print("loading", name)
        rows += LOADERS[name](n, exp["seed"])
    df = pd.DataFrame(rows)
    df = assign_splits(df, exp)
    return df.reset_index(drop=True)
