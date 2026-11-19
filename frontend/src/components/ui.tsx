import type { ReactNode } from "react";
import { MODEL_COLOR, MODEL_LABEL } from "../lib/format";
import type { ModelName } from "../lib/types";
import { IconAlert } from "./icons";

export function PageHead({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="page-head">
      <div>
        <h1>{title}</h1>
        {sub && <p>{sub}</p>}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </header>
  );
}

export function Panel({ title, sub, right, children, tight }: {
  title?: ReactNode; sub?: ReactNode; right?: ReactNode; children: ReactNode; tight?: boolean;
}) {
  return (
    <section className="panel">
      {(title || right) && (
        <div className="panel-head">
          <div>
            {title && <h2>{title}</h2>}
            {sub && <p>{sub}</p>}
          </div>
          {right}
        </div>
      )}
      <div className={`panel-body${tight ? " tight" : ""}`}>{children}</div>
    </section>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return <div className="loading" role="status"><span className="spinner" />{label}</div>;
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="alert error" role="alert">
      <IconAlert />
      <div style={{ flex: 1 }}>{message}</div>
      {onRetry && <button className="btn small" onClick={onRetry}>Try again</button>}
    </div>
  );
}

export function ModelTag({ model, short }: { model: ModelName; short?: boolean }) {
  return (
    <span className="nowrap" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span className="swatch" style={{ background: MODEL_COLOR[model] }} />
      {short ? MODEL_LABEL[model].replace(" + Attention", "+Attn") : MODEL_LABEL[model]}
    </span>
  );
}

export function MetricBar({ value, color, max = 1 }: { value: number | null | undefined; color: string; max?: number }) {
  const v = value ?? 0;
  return (
    <div className="cell-bar">
      <span>{value == null ? "—" : v.toFixed(3)}</span>
      <div className="bar"><span style={{ width: `${Math.max(0, Math.min(1, v / max)) * 100}%`, background: color }} /></div>
    </div>
  );
}

export const MODEL_INITIAL: Record<ModelName, string> = { tfidf_lr: "T", cnn: "C", bilstm: "L", bilstm_attn: "A" };
