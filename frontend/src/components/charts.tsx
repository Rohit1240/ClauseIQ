// Minimal SVG charts with hover tooltips. Data always comes from the API's saved metrics.
import { useState, type ReactNode } from "react";

function useTip() {
  const [tip, setTip] = useState<{ x: number; y: number; body: ReactNode } | null>(null);
  const node = tip ? <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 12 }}>{tip.body}</div> : null;
  return { show: (e: React.MouseEvent, body: ReactNode) => setTip({ x: e.clientX, y: e.clientY, body }), hide: () => setTip(null), node };
}

export interface Series { name: string; color: string; values: { x: number; y: number }[] }

export function LineChart({ series, height = 220, yLabel, yMax }: { series: Series[]; height?: number; yLabel?: string; yMax?: number }) {
  const t = useTip();
  const W = 560, H = height, L = 44, R = 12, T = 12, B = 30;
  const xs = series.flatMap((s) => s.values.map((v) => v.x));
  const ys = series.flatMap((s) => s.values.map((v) => v.y));
  if (!xs.length) return <p className="muted small">No training history recorded.</p>;
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  const y1 = yMax ?? Math.max(...ys) * 1.08, y0 = 0;
  const sx = (x: number) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const sy = (y: number) => T + (1 - (y - y0) / (y1 - y0 || 1)) * (H - T - B);
  const ticks = Array.from({ length: 5 }, (_, i) => y0 + ((y1 - y0) * i) / 4);
  const xticks = Array.from(new Set(xs)).sort((a, b) => a - b);
  return (
    <div style={{ position: "relative" }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={yLabel}>
        <g className="grid">{ticks.map((v) => <line key={v} x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} />)}</g>
        {ticks.map((v) => <text key={v} x={L - 6} y={sy(v) + 4} textAnchor="end">{v.toFixed(y1 < 2 ? 2 : 0)}</text>)}
        {xticks.map((v) => <text key={v} x={sx(v)} y={H - 10} textAnchor="middle">{v}</text>)}
        {series.map((s) => (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color} strokeWidth={2} points={s.values.map((v) => `${sx(v.x)},${sy(v.y)}`).join(" ")} />
            {s.values.map((v) => (
              <g key={v.x}>
                <circle cx={sx(v.x)} cy={sy(v.y)} r={4} fill={s.color} stroke="#fff" strokeWidth={2} />
                <circle cx={sx(v.x)} cy={sy(v.y)} r={10} fill="transparent" onMouseMove={(e) => t.show(e, <><b>{s.name}</b><br />epoch {v.x}: {v.y.toFixed(4)}</>)} onMouseLeave={t.hide} />
              </g>
            ))}
          </g>
        ))}
      </svg>
      {t.node}
    </div>
  );
}

export function HBars({ rows, color, max = 1, format = (v: number) => v.toFixed(2) }: {
  rows: { label: string; value: number; note?: string }[]; color: string; max?: number; format?: (v: number) => string;
}) {
  const t = useTip();
  const rowH = 20, W = 560, L = 230, R = 46;
  const H = rows.length * rowH + 8;
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img">
        {rows.map((r, i) => {
          const w = Math.max(0, (r.value / max) * (W - L - R));
          const y = 4 + i * rowH;
          return (
            <g key={r.label} onMouseMove={(e) => t.show(e, <><b>{r.label}</b><br />{format(r.value)}{r.note ? <><br />{r.note}</> : null}</>)} onMouseLeave={t.hide}>
              <rect x={0} y={y} width={W} height={rowH} fill="transparent" />
              <text x={L - 8} y={y + 14} textAnchor="end">{r.label.length > 34 ? r.label.slice(0, 33) + "…" : r.label}</text>
              <rect x={L} y={y + 4} width={W - L - R} height={11} rx={2} fill="#f0f2f5" />
              <rect x={L} y={y + 4} width={w} height={11} rx={2} fill={color} />
              <text x={L + w + 6} y={y + 14}>{format(r.value)}</text>
            </g>
          );
        })}
      </svg>
      {t.node}
    </div>
  );
}

export function Heatmap({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const t = useTip();
  const n = labels.length, cell = 13, L = 170, T = 170;
  const W = L + n * cell + 4, H = T + n * cell + 4;
  const max = Math.max(1, ...matrix.flat());
  const shade = (v: number) => {
    if (v === 0) return "#ffffff";
    const a = Math.log(v + 1) / Math.log(max + 1);
    const stops = [[205, 226, 251], [134, 182, 239], [57, 135, 229], [28, 92, 171], [13, 54, 107]];
    const p = a * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), f = p - i;
    const c = stops[i].map((s, k) => Math.round(s + (stops[i + 1][k] - s) * f));
    return `rgb(${c.join(",")})`;
  };
  return (
    <div style={{ overflow: "auto" }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Confusion matrix">
        {labels.map((l, i) => (
          <g key={l}>
            <text x={L - 4} y={T + i * cell + 10} textAnchor="end" style={{ fontSize: 9 }}>{l.length > 26 ? l.slice(0, 25) + "…" : l}</text>
            <text transform={`translate(${L + i * cell + 9}, ${T - 4}) rotate(-60)`} style={{ fontSize: 9 }}>{l.length > 26 ? l.slice(0, 25) + "…" : l}</text>
          </g>
        ))}
        {matrix.map((row, i) => row.map((v, j) => (
          <rect key={`${i}-${j}`} x={L + j * cell} y={T + i * cell} width={cell - 1} height={cell - 1} fill={shade(v)}
            stroke={i === j ? "#c9ced8" : "#eef0f4"} strokeWidth={0.5}
            onMouseMove={(e) => t.show(e, <>gold <b>{labels[i]}</b><br />predicted <b>{labels[j]}</b><br />{v.toLocaleString()} segment{v === 1 ? "" : "s"}</>)} onMouseLeave={t.hide} />
        )))}
      </svg>
      {t.node}
    </div>
  );
}
