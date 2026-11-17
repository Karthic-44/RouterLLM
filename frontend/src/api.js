const base = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(base + path, options);
  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = body.detail || message;
    } catch {
      message = res.statusText;
    }
    throw new Error(message);
  }
  return res.json();
}

export function getCurve() {
  return request("/curve");
}

export function getExamples() {
  return request("/examples");
}

export function ask(question, threshold, pipeline) {
  return request("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: question }],
      threshold,
      model: pipeline,
    }),
  });
}
