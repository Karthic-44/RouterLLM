import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
RESPONSES = DATA / "responses"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

ACTIONS = ["weak", "weak_rag", "strong", "strong_rag"]

load_dotenv(ROOT / ".env")


def read_yaml(name):
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load():
    models = read_yaml("models.yaml")
    exp = read_yaml("experiment.yaml")
    for m in models.values():
        env_name = m.get("api_key_env")
        if env_name:
            m["api_key"] = os.environ.get(env_name, "")
    return {"models": models, "exp": exp}


def ensure_dirs():
    for p in (DATA, DATA / "raw", CORPUS, RESPONSES, MODELS, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
