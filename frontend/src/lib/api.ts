import type {
  Analysis, Category, Dashboard, DatasetInfo, DocumentInfo, Health, ModelDetail, ModelName,
  ModelSummary, RecentAnalysis,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") || "";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError("Cannot reach the ClauseIQ API. Check that the backend is running.", 0);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* non-JSON error body */ }
    throw new ApiError(detail, res.status);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  dashboard: () => request<Dashboard>("/api/dashboard"),
  dataset: () => request<DatasetInfo>("/api/dataset"),
  categories: () => request<Category[]>("/api/categories"),
  models: () => request<ModelSummary[]>("/api/models"),
  model: (name: ModelName) => request<ModelDetail>(`/api/models/${name}`),
  analyses: (limit = 50) => request<RecentAnalysis[]>(`/api/analyses?limit=${limit}`),
  analysis: (id: string) => request<Analysis>(`/api/analyses/${encodeURIComponent(id)}`),
  deleteAnalysis: (id: string) => request<void>(`/api/analyses/${encodeURIComponent(id)}`, { method: "DELETE" }),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<DocumentInfo & { text: string }>("/api/documents", { method: "POST", body: fd });
  },
  analyse: (document_id: string, models: ModelName[], categories: string[] | null) =>
    request<Analysis>("/api/analyses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id, models, categories }),
    }),
  reportUrl: (id: string, format: "html" | "csv" | "json") =>
    `${BASE}/api/analyses/${encodeURIComponent(id)}/report?format=${format}`,
};
