# LLM router

Not every question needs the biggest model, and not every question needs a Wikipedia lookup. This project decides, for each question, which model should answer it and whether that model should read a few Wikipedia passages first (retrieval-augmented generation, or RAG), so the answers stay good and the bill stays small.

On 400 questions it had never seen, it cost about a third less than always using the large model with lookup, and its accuracy was within a point of that setup (68.3% against 69.5%). It also fails in a way I think is worth showing, and I cover that under Limits.

Everything runs locally with Ollama.


## How it works

A question can take one of four routes:

| | From memory | With lookup |
| --- | --- | --- |
| **Small model** | `weak` | `weak_rag` |
| **Large model** | `strong` | `strong_rag` |

1. I sampled 2,500 questions from three datasets: GSM8K (maths), ARC-Challenge (science multiple choice) and PopQA (facts about obscure people, places and films).
2. I ran every question down all four routes and graded each answer. That is 10,000 graded answers. The small model is `llama3.2:1b` and the large one is `llama3.1:8b`, both through Ollama.
3. For the lookup route I saved about 21,000 Wikipedia pages as passages of roughly 100 words and search them with sentence embeddings (`all-MiniLM-L6-v2`).
4. I trained one small scikit-learn classifier per route to predict whether that route would get a question right. The inputs are the question's embedding, a few plain features (length, digits, capital letters) and how well the best passages match the question.
5. At run time the router goes through the routes from cheapest to most expensive and picks the first one whose predicted chance of being right clears a confidence bar. That bar is the slider in the demo.

The router never writes an answer. It only decides who does.

## Results

400 held-out questions (100 GSM8K, 300 PopQA). Cost is per 1,000 questions and uses made-up prices, so read it as relative, not as dollars.

| Always use | Accuracy | Cost |
| --- | --- | --- |
| small model | 22.5% | 0.007 |
| small model + lookup | 56.0% | 0.024 |
| large model | 40.0% | 0.028 |
| large model + lookup | 69.5% | 0.095 |
| **the router** | **68.3%** | **0.063** |

By dataset, with the router at the same setting:

| | Small | Small + lookup | Large | Large + lookup | Router |
| --- | --- | --- | --- | --- | --- |
| GSM8K (100) | 48% | 44% | 80% | 78% | 77% |
| PopQA (300) | 14% | 60% | 27% | 67% | 65% |
| ARC (500, never trained on) | 49% | 52% | 87% | 83% | 84% |

What stood out:

- **Lookup is worth paying for on rare facts and almost nowhere else.** On PopQA it took the small model from 14% to 60% and the large model from 27% to 67%. On maths it did nothing, and on ARC it made the large model slightly worse.
- **A small model with lookup beat a large model without it,** 56% against 40%, and cost less.
- **The router saves about a third.** It cut cost by 33% against the best fixed setup for about a point of accuracy. A random mix of small and large that keeps the same accuracy saved under 5%. An oracle that always picked the cheapest correct route would save about 76%, so there is room left.
- I picked the confidence bar on a separate validation set. Picking it by looking at the test answers gives a prettier 44% saving, but that is cheating a little, so I quote the 33%.

## Running it

You need Python 3.10 or newer, [Ollama](https://ollama.com), and Node 20.19+ or 22.12+ for the web page. I built it on Windows with Python 3.14.

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2:1b
ollama pull llama3.1:8b
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

Then run the steps in order:

```
python prepare_data.py
python build_corpus.py
python build_index.py
python collect.py --action all
python train.py
python evaluate.py
```

Start the backend, then the page in a second terminal:

```
uvicorn backend.main:app --port 8000
```

```
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Layout

```
router/      shared code: data, model calls, grading, retrieval, features, policy
backend/     FastAPI app
frontend/    React page built with Vite
configs/     model names, prices, dataset sizes
results/     tables and figures from evaluate.py
```
