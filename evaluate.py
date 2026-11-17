import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from router.config import ACTIONS, DATA, MODELS, RESPONSES, RESULTS, ensure_dirs
from router.policy import choose_many

COLORS = {
    "weak": "#2b7bb9",
    "weak_rag": "#1f7a6e",
    "strong": "#e0a526",
    "strong_rag": "#b04632",
}
NAMES = {
    "weak": "small",
    "weak_rag": "small + lookup",
    "strong": "large",
    "strong_rag": "large + lookup",
}


def load_all():
    df = pd.read_parquet(DATA / "queries.parquet")
    pred = pd.read_parquet(DATA / "predictions.parquet")
    keep = ["id", "top1", "gap"] + [c for c in pred.columns if c.startswith("p_")]
    df = df.merge(pred[keep], on="id")
    for a in ACTIONS:
        rows = []
        with open(RESPONSES / f"{a}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append({"id": r["id"], "c_" + a: int(r["correct"]), "k_" + a: r["cost"]})
        df = df.merge(pd.DataFrame(rows), on="id")
    return df


def mats(g):
    P = g[["p_" + a for a in ACTIONS]].to_numpy()
    C = g[["c_" + a for a in ACTIONS]].to_numpy()
    K = g[["k_" + a for a in ACTIONS]].to_numpy() * 1000
    return P, C, K


def curve(P, C, K, order, thresholds):
    rows = np.arange(len(P))
    out = []
    for t in thresholds:
        pick = choose_many(P, ACTIONS, order, t)
        share = np.bincount(pick, minlength=len(ACTIONS)) / len(P)
        out.append({
            "threshold": round(float(t), 3),
            "accuracy": float(C[rows, pick].mean()),
            "cost": float(K[rows, pick].mean()),
            "share": {a: float(s) for a, s in zip(ACTIONS, share)},
        })
    return out


def fixed(C, K):
    return {
        a: {"accuracy": float(C[:, i].mean()), "cost": float(K[:, i].mean())}
        for i, a in enumerate(ACTIONS)
    }


def oracle(C, K):
    masked = np.where(C == 1, K, np.inf)
    pick = masked.argmin(axis=1)
    none_right = ~np.isfinite(masked.min(axis=1))
    pick[none_right] = K[none_right].argmin(axis=1)
    rows = np.arange(len(C))
    return {"accuracy": float(C[rows, pick].mean()), "cost": float(K[rows, pick].mean())}


def mix(C, K):
    i = int(K.mean(axis=0).argmin())
    j = best_fixed(C)
    a_lo, a_hi = C[:, i].mean(), C[:, j].mean()
    k_lo, k_hi = K[:, i].mean(), K[:, j].mean()
    out = []
    for p in np.linspace(0, 1, 21):
        out.append({
            "share_hi": float(p),
            "accuracy": float(a_lo + p * (a_hi - a_lo)),
            "cost": float(k_lo + p * (k_hi - k_lo)),
        })
    return out


def cheapest_at(points, target):
    ok = [p for p in points if p["accuracy"] >= target]
    if not ok:
        return None
    return min(ok, key=lambda p: p["cost"])


def best_fixed(C):
    return max(range(len(ACTIONS)), key=lambda i: C[:, i].mean())


def pick_threshold(g, order, thresholds):
    P, C, K = mats(g)
    pts = curve(P, C, K, order, thresholds)
    target = 0.95 * C[:, best_fixed(C)].mean()
    point = cheapest_at(pts, target)
    if point is None:
        point = max(pts, key=lambda p: p["accuracy"])
    return point["threshold"]


def summarize(name, group, point, ref):
    row = {"group": group, "name": name, "accuracy": None, "cost_per_1k": None, "saved_vs_best_fixed_pct": None}
    if point is not None:
        row["accuracy"] = round(point["accuracy"], 4)
        row["cost_per_1k"] = round(point["cost"], 3)
        if ref["cost"] > 0:
            row["saved_vs_best_fixed_pct"] = round(100 * (1 - point["cost"] / ref["cost"]), 1)
    return row


def plot_curves(data, path):
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.6), squeeze=False)
    for ax, (name, d) in zip(axes[0], data.items()):
        ax.plot([p["cost"] for p in d["router"]], [p["accuracy"] for p in d["router"]],
                color="#1f2a37", lw=2, label="router, 4 pipelines")
        ax.plot([p["cost"] for p in d["router_v1"]], [p["accuracy"] for p in d["router_v1"]],
                color="#8a94a1", lw=1.5, label="router, 2 models only")
        ax.plot([p["cost"] for p in d["random"]], [p["accuracy"] for p in d["random"]],
                color="#b8c0ca", lw=1.2, ls="--", label="random mix")
        for a in ACTIONS:
            ax.scatter(d["fixed"][a]["cost"], d["fixed"][a]["accuracy"],
                       color=COLORS[a], s=60, zorder=3, label="always " + NAMES[a])
        ax.scatter(d["oracle"]["cost"], d["oracle"]["accuracy"], marker="*", s=140,
                   color="black", zorder=4, label="oracle")
        ax.set_title(name)
        ax.set_xlabel("cost per 1,000 questions (notional)")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_popularity(df, path):
    pq = df[(df["dataset"] == "popqa") & df["pop"].notna()].copy()
    if len(pq) < 20:
        print("not enough popqa rows for the popularity plot")
        return
    pq["lp"] = np.log10(pq["pop"].astype(float) + 1)
    pq["bin"] = pd.qcut(pq["lp"], 5, duplicates="drop")
    labels = [f"{10 ** b.left:.0f} to {10 ** b.right:.0f}" for b in pq["bin"].cat.categories]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for a in ACTIONS:
        acc = pq.groupby("bin", observed=True)["c_" + a].mean().to_numpy()
        ax.plot(range(len(acc)), acc, marker="o", color=COLORS[a], label=NAMES[a])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_xlabel("entity popularity (Wikipedia page views), rare to popular")
    ax.set_ylabel("accuracy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ensure_dirs()
    df = load_all()
    art = joblib.load(MODELS / "router.joblib")
    order = art["order"]
    order_v1 = [a for a in order if a in ("weak", "strong")]
    thresholds = np.linspace(0, 1, 51)

    groups = {}
    for name in ("test", "ood"):
        part = df[df["split"] == name]
        if len(part):
            groups[name] = part
    if not groups:
        raise SystemExit("no test or ood rows found, run train.py first")

    val = df[df["split"] == "val"]
    base = val if len(val) else next(iter(groups.values()))
    default_t = pick_threshold(base, order, thresholds)
    print("threshold picked on validation rows:", default_t)

    summary = []
    plot_data = {}
    per_dataset = []
    for gname, g in groups.items():
        P, C, K = mats(g)
        fx = fixed(C, K)
        ref = fx[ACTIONS[best_fixed(C)]]
        target = 0.95 * ref["accuracy"]
        v2 = curve(P, C, K, order, thresholds)
        v1 = curve(P, C, K, order_v1, thresholds)
        orc = oracle(C, K)
        rnd = mix(C, K)

        for a in ACTIONS:
            summary.append(summarize("always " + NAMES[a], gname, fx[a], ref))
        summary.append(summarize("router, 4 pipelines, cheapest point at 95% of best fixed", gname, cheapest_at(v2, target), ref))
        summary.append(summarize("router, 2 models only, cheapest point at 95% of best fixed", gname, cheapest_at(v1, target), ref))
        summary.append(summarize("random mix, cheapest point at 95% of best fixed", gname, cheapest_at(rnd, target), ref))
        at_default = curve(P, C, K, order, [default_t])[0]
        summary.append(summarize(f"router, 4 pipelines, threshold {default_t} (chosen on validation)", gname, at_default, ref))
        summary.append(summarize("oracle", gname, orc, ref))

        plot_data[gname] = {"router": v2, "router_v1": v1, "random": rnd, "fixed": fx, "oracle": orc}

        for ds, part in g.groupby("dataset"):
            Pd, Cd, Kd = mats(part)
            pick = choose_many(Pd, ACTIONS, order, default_t)
            r = np.arange(len(part))
            row = {"group": gname, "dataset": ds, "n": len(part)}
            for i, a in enumerate(ACTIONS):
                row["acc_" + a] = round(float(Cd[:, i].mean()), 3)
            row["acc_router"] = round(float(Cd[r, pick].mean()), 3)
            row["cost_router"] = round(float(Kd[r, pick].mean()), 3)
            row["cost_strong_rag"] = round(float(Kd[:, ACTIONS.index("strong_rag")].mean()), 3)
            per_dataset.append(row)

    table = pd.DataFrame(summary)
    table.to_csv(RESULTS / "summary.csv", index=False)
    pd.DataFrame(per_dataset).to_csv(RESULTS / "by_dataset.csv", index=False)
    plot_curves(plot_data, RESULTS / "cost_quality_curve.png")
    plot_popularity(df, RESULTS / "retrieval_gain_by_popularity.png")

    first = next(iter(plot_data))
    out = dict(plot_data[first])
    out["group"] = first
    out["default_threshold"] = default_t
    out["unit"] = "notional cost per 1,000 questions"
    with open(RESULTS / "curve.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(table.to_string(index=False))
    print(pd.DataFrame(per_dataset).to_string(index=False))
    print("wrote results/summary.csv, by_dataset.csv, curve.json and two figures")


if __name__ == "__main__":
    main()
