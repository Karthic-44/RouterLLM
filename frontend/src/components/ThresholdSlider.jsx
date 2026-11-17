export default function ThresholdSlider({ value, onChange, point }) {
  return (
    <div className="dial">
      <label htmlFor="threshold">Confidence bar: {value.toFixed(2)}</label>
      <input
        id="threshold"
        type="range"
        min="0"
        max="1"
        step="0.01"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <div className="dial-ends muted">
        <span>cheaper</span>
        <span>more accurate</span>
      </div>
      {point && (
        <p className="muted">
          On held-out test questions this setting got {Math.round(point.accuracy * 100)}% right at a
          cost of {point.cost.toFixed(3)} per 1,000 questions.
        </p>
      )}
    </div>
  );
}
