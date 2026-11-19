export type ModelName = "tfidf_lr" | "cnn" | "bilstm" | "bilstm_attn";

export interface SpanMetrics {
  n_gold: number;
  n_pred: number;
  span_precision: number;
  span_recall: number;
  span_f1: number;
  span_macro_f1: number;
  exact_match_precision: number;
  exact_match_recall: number;
  exact_match_f1: number;
  exact_match_rate: number;
  mean_best_iou: number;
  char_iou: number;
  char_iou_macro: number;
  char_precision: number;
  char_recall: number;
  char_f1: number;
}

export interface SegmentMetrics {
  n_segments: number;
  accuracy: number;
  accuracy_on_labelled_segments: number;
  micro_precision: number;
  micro_recall: number;
  micro_f1: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  weighted_precision: number;
  weighted_recall: number;
  weighted_f1: number;
}

export interface ContractMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  macro_f1: number;
  confusion: { tp: number; fp: number; fn: number; tn: number };
}

export interface HistoryEntry {
  epoch: number;
  train_loss: number;
  "val_span_f1": number;
  val_threshold?: number;
  "val_span_macro_f1": number;
  "val_char_iou": number;
  lr: number;
  epoch_seconds: number;
}

export interface ModelSummary {
  name: ModelName;
  display_name: string;
  status: "loaded" | "available" | "missing" | "error";
  kind: "segment" | "token";
  evaluated: boolean;
  span_extraction: SpanMetrics | null;
  segment_classification: SegmentMetrics | null;
  contract_detection: ContractMetrics | null;
  timing: {
    train_seconds: number | null;
    inference_seconds_total: number;
    inference_seconds_per_contract: number;
    chars_per_second: number;
    hardware: string;
  } | null;
  training: {
    config?: Record<string, unknown>;
    history?: HistoryEntry[] | Record<string, unknown>[];
    best_epoch?: number;
    n_parameters?: number;
    n_features?: number;
    n_train_segments?: number;
    selected?: Record<string, unknown>;
    train_seconds?: number;
    device?: string;
  };
}

export interface CategorySpanPerf {
  n_gold: number;
  n_pred: number;
  span_precision: number;
  span_recall: number;
  span_f1: number;
  exact_match_f1: number;
  mean_best_iou: number | null;
  char_iou: number | null;
}

export interface ModelDetail extends ModelSummary {
  span_per_category?: Record<string, CategorySpanPerf>;
  confusion_matrix?: { labels: string[]; matrix: number[][]; note: string };
  contract_per_category?: Record<string, { tp: number; fp: number; fn: number; tn: number; f1: number; precision: number; recall: number }>;
  thresholds?: Record<string, number>;
}

export interface CategoryStats {
  name: string;
  n_spans: number;
  n_contracts_with_clause: number;
  n_contracts_without_clause: number;
  pct_contracts_with_clause: number;
  span_chars: { median?: number; mean?: number; max?: number };
  spans_by_split: { train: number; val: number; test: number };
}

export interface Category {
  key: string;
  name: string;
  description: string;
  answer_format: string;
  group: string;
  index: number;
  stats: CategoryStats;
  performance: Partial<Record<ModelName, {
    span_f1: number; span_precision: number; span_recall: number; exact_match_f1: number;
    char_iou: number | null; test_gold_spans: number; contract_f1: number; segment_f1: number;
  }>>;
}

export interface PageRange { page: number; start: number; end: number }

export interface DocumentInfo {
  id: string;
  filename: string;
  file_type: string;
  n_chars: number;
  n_pages: number;
  pages: PageRange[];
  warnings: string[];
  created_at: string;
  text?: string;
}

export interface AttentionToken { token: string; weight: number; start: number; end: number }

export interface Clause {
  id: string;
  model: ModelName;
  category: string;
  category_name: string;
  start: number;
  end: number;
  text: string;
  score: number;
  threshold: number;
  page_start: number | null;
  page_end: number | null;
  attention?: AttentionToken[];
}

export interface Run {
  model: ModelName;
  display_name: string;
  kind: string;
  seconds: number;
  units: number;
  n_clauses: number;
  score_description: string;
}

export interface Finding {
  category: string;
  category_name: string;
  start: number;
  end: number;
  page_start: number | null;
  models: ModelName[];
  missing_models: ModelName[];
  agreement: number;
  status: "all" | "partial" | "single";
  spans: { model: ModelName; start: number; end: number; score: number; id: string }[];
}

export interface Comparison {
  models: ModelName[];
  summary: { findings: number; unanimous: number; partial: number; single_model: number };
  findings: Finding[];
  pairwise_char_jaccard: Record<string, number | null>;
  categories: { category: string; category_name: string; by_model: Record<ModelName, number>; findings: number; unanimous: number }[];
}

export interface Analysis {
  id: string;
  created_at: string;
  seconds: number;
  models: ModelName[];
  categories: string[];
  n_clauses: number;
  runs: Record<ModelName, Run>;
  clauses: Clause[];
  comparison: Comparison;
  document: DocumentInfo & { text: string };
}

export interface RecentAnalysis {
  id: string;
  document_id: string;
  models: ModelName[];
  n_categories: number;
  n_clauses: number;
  seconds: number;
  created_at: string;
  filename: string;
  n_pages: number;
  n_chars: number;
}

export interface Dashboard {
  totals: { analyses: number; contracts_analysed: number; clauses_extracted: number };
  n_categories: number;
  models: ModelSummary[];
  recent: RecentAnalysis[];
  dataset: { n_contracts: number; n_spans: number; split: { train: number; val: number; test: number } };
}

export interface DatasetInfo {
  stats: {
    n_contracts: number;
    n_categories: number;
    n_spans: number;
    split: { train: number; val: number; test: number };
    split_source: string;
    spans_by_split: { train: number; val: number; test: number };
    contract_chars: { min: number; max: number; mean: number; median: number };
    total_chars: number;
    annotated_char_fraction: number;
    missing_annotations: { contract_category_pairs: number; pairs_without_clause: number; pct_pairs_without_clause: number; note: string };
    class_imbalance: { max_spans: number; min_spans: number; imbalance_ratio: number; gini: number; most_common: [string, number][]; rarest: [string, number][] };
  };
  features: Record<string, unknown> & { vocab_size?: number };
  attribution: string;
}

export interface Health {
  status: string;
  version: string;
  models: Record<ModelName, string>;
  models_available: number;
  metrics_available: ModelName[];
}
