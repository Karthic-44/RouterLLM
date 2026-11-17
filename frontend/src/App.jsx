import { useEffect, useState } from "react";
import AskBox from "./components/AskBox";
import AnswerCard from "./components/AnswerCard";
import CurveChart from "./components/CurveChart";
import ThresholdSlider from "./components/ThresholdSlider";
import { ask, getCurve, getExamples } from "./api";

function friendly(err) {
  if (err instanceof TypeError) {
    return "Can't reach the backend. Start it with: uvicorn backend.main:app --port 8000";
  }
  return err.message;
}

function nearest(points, threshold) {
  let best = points[0];
  for (const p of points) {
    if (Math.abs(p.threshold - threshold) < Math.abs(best.threshold - threshold)) {
      best = p;
    }
  }
  return best;
}

export default function App() {
  const [curve, setCurve] = useState(null);
  const [examples, setExamples] = useState([]);
  const [threshold, setThreshold] = useState(0.5);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getCurve()
      .then((c) => {
        setCurve(c);
        setThreshold(c.default_threshold);
      })
      .catch((e) => setError(friendly(e)));
    getExamples()
      .then(setExamples)
      .catch(() => {});
  }, []);

  async function handleAsk(question, pipeline) {
    setBusy(true);
    setError("");
    try {
      const res = await ask(question, threshold, pipeline);
      setResult(res);
    } catch (e) {
      setError(friendly(e));
    } finally {
      setBusy(false);
    }
  }

  const point = curve ? nearest(curve.router, threshold) : null;

  return (
    <div className="page">
      <header>
        <h1>LLM router</h1>
        <p className="lead">
          Each question goes to the cheapest pipeline the router expects to answer correctly.
        </p>
      </header>
      <main>
        <AskBox examples={examples} busy={busy} onAsk={handleAsk} />
        <AnswerCard result={result} busy={busy} error={error} />
        <section className="tradeoff">
          <h2>Cost and accuracy</h2>
          {curve && (
            <p className="muted">
              Each circle is what you get by always using one pipeline. The line is the router as
              you move the confidence bar, and the blue ring is where it is now.
            </p>
          )}
          <CurveChart curve={curve} point={point} />
          {curve && <ThresholdSlider value={threshold} onChange={setThreshold} point={point} />}
        </section>
      </main>
    </div>
  );
}
