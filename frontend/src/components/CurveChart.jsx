import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PIPELINES } from "../pipelines";

const SIDES = {
  weak: { dx: 9, dy: 4, anchor: "start" },
  weak_rag: { dx: 0, dy: -11, anchor: "middle" },
  strong: { dx: 9, dy: 4, anchor: "start" },
  strong_rag: { dx: 4, dy: -11, anchor: "end" },
};

function Pipeline({ cx, cy, payload }) {
  const side = SIDES[payload.id];
  return (
    <g>
      <circle cx={cx} cy={cy} r={4.5} fill="#fff" stroke="#1a1a1a" strokeWidth={1.5} />
      <text x={cx + side.dx} y={cy + side.dy} textAnchor={side.anchor} fontSize={12} fill="#666">
        {payload.label}
      </text>
    </g>
  );
}

function Ring({ cx, cy }) {
  return <circle cx={cx} cy={cy} r={8} fill="none" stroke="#2a4ec9" strokeWidth={2} />;
}

function Hidden() {
  return null;
}

export default function CurveChart({ curve, point }) {
  if (!curve) {
    return (
      <p className="muted">
        No evaluation results yet. Run evaluate.py, then reload, to see the cost and accuracy curve.
      </p>
    );
  }

  const line = curve.router.map((p) => ({ cost: p.cost, accuracy: p.accuracy }));
  const fixed = PIPELINES.map((p) => ({
    id: p.id,
    label: p.short,
    cost: curve.fixed[p.id].cost,
    accuracy: curve.fixed[p.id].accuracy,
  }));
  const now = point ? [{ cost: point.cost, accuracy: point.accuracy }] : [];
  const tick = { fontSize: 12, fill: "#666" };

  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={300} initialDimension={{ width: 640, height: 300 }}>
        <ScatterChart margin={{ top: 12, right: 16, bottom: 28, left: 0 }}>
          <CartesianGrid vertical={false} stroke="#ececec" />
          <XAxis
            type="number"
            dataKey="cost"
            name="cost"
            domain={["auto", "auto"]}
            tick={tick}
            stroke="#bbb"
            tickFormatter={(v) => v.toFixed(3)}
            label={{ value: "cost per 1,000 questions", position: "insideBottom", offset: -16, fontSize: 12, fill: "#666" }}
          />
          <YAxis
            type="number"
            dataKey="accuracy"
            name="accuracy"
            domain={[0, 1]}
            tick={tick}
            stroke="#bbb"
            tickFormatter={(v) => Math.round(v * 100) + "%"}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{ border: "1px solid #d9d9d9", borderRadius: 2, boxShadow: "none", fontSize: 12 }}
            formatter={(v, name) => (name === "accuracy" ? Math.round(v * 100) + "%" : v.toFixed(4))}
          />
          <Scatter
            name="router"
            data={line}
            line={{ stroke: "#1a1a1a", strokeWidth: 1.5 }}
            shape={Hidden}
            isAnimationActive={false}
          />
          <Scatter name="always one pipeline" data={fixed} shape={Pipeline} isAnimationActive={false} />
          <Scatter name="current setting" data={now} shape={Ring} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
