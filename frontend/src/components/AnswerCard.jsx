const ROWS = [
  { label: "Small model", ids: ["weak", "weak_rag"] },
  { label: "Large model", ids: ["strong", "strong_rag"] },
];

function percent(x) {
  return Math.round(x * 100) + "%";
}

export default function AnswerCard({ result, busy, error }) {
  const info = result ? result.router : null;
  const cheapest = info ? Math.min(...Object.values(info.costs)) : 1;

  return (
    <section className="answer" aria-live="polite" aria-busy={busy}>
      <table className="grid">
        <thead>
          <tr>
            <td />
            <th scope="col">From memory</th>
            <th scope="col">With lookup</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map((row) => (
            <tr key={row.label}>
              <th scope="row">{row.label}</th>
              {row.ids.map((id) => {
                const chosen = info && info.action === id;
                return (
                  <td key={id} className={chosen ? "chosen" : ""}>
                    {info ? (
                      <>
                        <div>{percent(info.probs[id])} likely right</div>
                        <div className="muted">{(info.costs[id] / cheapest).toFixed(1)}x cost</div>
                        {chosen && <div className="picked">chosen</div>}
                      </>
                    ) : (
                      <div className="muted">-</div>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {!info && !error && <p className="muted">Ask a question to see which pipeline gets picked.</p>}

      {info && (
        <div className="reply">
          <h2>Answer</h2>
          <p className="final">{result.answer}</p>
          <p className="muted">
            {info.model}, {info.seconds} s, {result.usage.total_tokens} tokens, cost{" "}
            {(info.cost * 1000).toFixed(4)} per 1,000 asks
          </p>
          {info.retrieval && info.passages.length > 0 && (
            <>
              <h2>Passages it read</h2>
              <ol className="passages">
                {info.passages.map((p) => (
                  <li key={p.id}>
                    <strong>{p.title}</strong> <span className="muted">(match {p.score.toFixed(2)})</span>
                    <div className="muted">{p.text}</div>
                  </li>
                ))}
              </ol>
            </>
          )}
          <details>
            <summary>Full response</summary>
            <pre>{result.choices[0].message.content}</pre>
          </details>
        </div>
      )}
    </section>
  );
}
