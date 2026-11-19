import type { ModelName } from "./types";

export const MODEL_ORDER: ModelName[] = ["tfidf_lr", "cnn", "bilstm", "bilstm_attn"];

export const MODEL_LABEL: Record<ModelName, string> = {
  tfidf_lr: "TF-IDF + LR",
  cnn: "Dilated CNN",
  bilstm: "BiLSTM",
  bilstm_attn: "BiLSTM + Attention",
};

/** Fixed colour per model (validated categorical slots 1–4). Colour follows the model everywhere. */
export const MODEL_COLOR: Record<ModelName, string> = {
  tfidf_lr: "#2a78d6",
  cnn: "#eb6834",
  bilstm: "#1baf7a",
  bilstm_attn: "#eda100",
};

export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 3) =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const int = (v: number | null | undefined) => (v == null ? "—" : v.toLocaleString("en-US"));

export function duration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m} min ${s} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function when(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

export function pageLabel(a: number | null, b: number | null): string {
  if (a == null) return "";
  return b != null && b !== a ? `pp. ${a}–${b}` : `p. ${a}`;
}
