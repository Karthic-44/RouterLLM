from router.config import DATA, ensure_dirs, load
from router.data import build_queries


def main():
    ensure_dirs()
    cfg = load()
    df = build_queries(cfg["exp"])
    df.to_parquet(DATA / "queries.parquet", index=False)
    print(df.groupby(["dataset", "split"]).size())
    print("saved", len(df), "questions")


if __name__ == "__main__":
    main()
