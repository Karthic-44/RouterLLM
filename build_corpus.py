import argparse
import json
import os
import time

import pandas as pd
import requests

from router.config import CORPUS, DATA, ensure_dirs, load

API = "https://en.wikipedia.org/w/api.php"
CONTACT = os.environ.get("WIKI_CONTACT") or "https://github.com/Karthic-44"
USER_AGENT = f"llm-router/0.1 ({CONTACT}) python-requests/{requests.__version__}"
PAGES = DATA / "raw" / "wiki_pages.jsonl"
ASKED = DATA / "raw" / "wiki_asked.txt"
PAUSE = 2.0
BLOCKED = (
    "Wikipedia is limiting this connection. Progress is saved, so wait about ten "
    "minutes and run the same command again."
)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def call(params):
    wait = 5
    for attempt in range(6):
        try:
            r = session.get(API, params=params, timeout=30)
        except requests.RequestException as e:
            print("wikipedia request failed:", type(e).__name__)
            time.sleep(wait)
            wait = min(wait * 2, 120)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 503):
            header = r.headers.get("Retry-After", "")
            pause = int(header) if header.isdigit() else wait
            pause = min(max(pause, 5), 300)
            print(f"wikipedia answered {r.status_code}, waiting {pause} seconds")
            time.sleep(pause)
            wait = min(wait * 2, 120)
            continue
        print("wikipedia answered", r.status_code)
        return None
    return None


def pages_from(data):
    out = []
    for p in data.get("query", {}).get("pages", {}).values():
        text = p.get("extract", "")
        if text and len(text.split()) >= 30:
            out.append({"title": p["title"], "text": text})
    return out


def read_pages():
    rows = []
    if PAGES.exists():
        with open(PAGES, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
    return rows


def read_asked():
    if not ASKED.exists():
        return set()
    with open(ASKED, encoding="utf-8") as f:
        return set(line.rstrip("\n") for line in f)


def save_pages(pages, src):
    with open(PAGES, "a", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps({"title": p["title"], "text": p["text"], "src": src}) + "\n")


def get_subjects(titles, have):
    asked = read_asked()
    todo = [t for t in titles if t not in asked]
    print(len(todo), "subject pages to fetch")
    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        data = call({
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exlimit": "max",
            "redirects": 1,
            "titles": "|".join(batch),
        })
        if data is None:
            return False
        fresh = [p for p in pages_from(data) if p["title"] not in have]
        save_pages(fresh, "subject")
        have.update(p["title"] for p in fresh)
        with open(ASKED, "a", encoding="utf-8") as f:
            f.write("\n".join(batch) + "\n")
        if (i // 20) % 10 == 9:
            print(i + 20, "of", len(todo), "titles done")
        time.sleep(PAUSE)
    return True


def get_random(n_needed, have):
    got = 0
    misses = 0
    while got < n_needed and misses < 40:
        data = call({
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": 20,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exlimit": "max",
        })
        if data is None:
            return False
        fresh = [p for p in pages_from(data) if p["title"] not in have]
        if not fresh:
            misses += 1
            time.sleep(PAUSE)
            continue
        fresh = fresh[:n_needed - got]
        save_pages(fresh, "random")
        have.update(p["title"] for p in fresh)
        before = got
        got += len(fresh)
        if got // 500 > before // 500:
            print(got, "random pages so far")
        time.sleep(PAUSE)
    return True


def chunk(text, size, limit):
    words = text.split()[:limit]
    pieces = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    if len(pieces) > 1 and len(pieces[-1].split()) < 20:
        pieces[-2] += " " + pieces[-1]
        pieces.pop()
    return pieces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    cfg = load()
    ccfg = cfg["exp"]["corpus"]
    n_random = ccfg["distractor_pages"] if args.random is None else args.random

    df = pd.read_parquet(DATA / "queries.parquet")
    wanted = sorted(set(t for t in df["wiki_title"] if t))

    rows = read_pages()
    have = set(r["title"] for r in rows)
    if not get_subjects(wanted, have):
        raise SystemExit(BLOCKED)

    rows = read_pages()
    have = set(r["title"] for r in rows)
    have_random = sum(1 for r in rows if r["src"] == "random")
    if have_random < n_random:
        if not get_random(n_random - have_random, have):
            raise SystemExit(BLOCKED)

    rows = read_pages()
    seen = set()
    count = 0
    with open(CORPUS / "passages.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["title"] in seen:
                continue
            seen.add(r["title"])
            for piece in chunk(r["text"], ccfg["words_per_passage"], ccfg["max_words_per_page"]):
                f.write(json.dumps({"id": count, "title": r["title"], "text": piece}) + "\n")
                count += 1
    subjects = sum(1 for r in rows if r["src"] == "subject")
    print(count, "passages from", len(seen), "pages;", subjects, "of", len(wanted), "subject pages found")


if __name__ == "__main__":
    main()