import { useState } from "react";
import { PIPELINES } from "../pipelines";

export default function AskBox({ examples, busy, onAsk }) {
  const [text, setText] = useState("");
  const [pipeline, setPipeline] = useState("auto");

  function submit(e) {
    e.preventDefault();
    if (text.trim() && !busy) {
      onAsk(text.trim(), pipeline);
    }
  }

  return (
    <form className="ask" onSubmit={submit}>
      <label htmlFor="question">Question</label>
      <textarea
        id="question"
        rows={3}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="ask-row">
        <select
          aria-label="Pipeline"
          value={pipeline}
          onChange={(e) => setPipeline(e.target.value)}
        >
          <option value="auto">Router decides</option>
          {PIPELINES.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
        <button type="submit" disabled={busy || !text.trim()}>
          {busy ? "Asking" : "Ask"}
        </button>
      </div>
      {examples.length > 0 && (
        <div className="examples">
          <p className="muted">Try one of these</p>
          <ul>
            {examples.map((ex, i) => (
              <li key={i}>
                <button type="button" title={ex.question} onClick={() => setText(ex.question)}>
                  {ex.question.split("\n")[0]}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </form>
  );
}
