import json
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import ChatRequest, RouteRequest
from backend.service import Service
from router.config import ACTIONS, DATA, RESULTS


@asynccontextmanager
async def lifespan(app):
    app.state.service = Service()
    yield


app = FastAPI(title="LLM router", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


def svc():
    return app.state.service


@app.get("/health")
def health():
    s = svc()
    return {
        "status": "ok",
        "passages": len(s.library.passages),
        "default_threshold": s.default_threshold,
    }


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    users = [m.content for m in req.messages if m.role == "user"]
    if not users:
        raise HTTPException(status_code=400, detail="messages needs at least one user message")
    force = req.model if req.model in ACTIONS else None
    res = svc().answer(users[-1], req.threshold, force)
    usage = res["usage"]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": res["router"]["action"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": res["text"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
        },
        "answer": res["answer"],
        "router": res["router"],
    }


@app.post("/route")
def route(req: RouteRequest):
    d = svc().decide(req.question, req.threshold)
    return {"action": d["action"], "threshold": d["threshold"], "probs": d["probs"]}


@app.get("/curve")
def curve():
    path = RESULTS / "curve.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="results/curve.json is missing, run evaluate.py first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/examples")
def examples():
    path = DATA / "queries.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    pool = df[df["split"] == "test"]
    out = []
    for _, g in pool.groupby("dataset"):
        for _, row in g.sample(min(2, len(g)), random_state=1).iterrows():
            out.append({"question": row["question"], "dataset": row["dataset"]})
    return out
