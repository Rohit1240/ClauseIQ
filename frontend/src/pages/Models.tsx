import { Fragment, useMemo, useState } from "react";
import { HBars, Heatmap, LineChart } from "../components/charts";
import { ErrorBox, Loading, ModelTag, PageHead, Panel } from "../components/ui";
import { api } from "../lib/api";
import { MODEL_COLOR, MODEL_LABEL, MODEL_ORDER, duration, int } from "../lib/format";
import type { HistoryEntry, ModelName, ModelSummary } from "../lib/types";
import { useAsync } from "../lib/useAsync";

const fmt = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(3));

function AllModelsTable({ models }: { models: ModelSummary[] }) {
  const groups: [string, [string, (m: ModelSummary) => number | null | undefined][]][] = [
    ["Span extraction", [
      ["Span precision (IoU ≥ 0.5)", (m) => m.span_extraction?.span_precision],
      ["Span recall (IoU ≥ 0.5)", (m) => m.span_extraction?.span_recall],
      ["Span F1 (IoU ≥ 0.5)", (m) => m.span_extraction?.span_f1],
      ["Span macro-F1 over categories", (m) => m.span_extraction?.span_macro_f1],
      ["Exact match precision", (m) => m.span_extraction?.exact_match_precision],
      ["Exact match recall (rate)", (m) => m.span_extraction?.exact_match_recall],
      ["Exact match F1", (m) => m.span_extraction?.exact_match_f1],
      ["Character IoU (micro)", (m) => m.span_extraction?.char_iou],
      ["Character IoU (macro)", (m) => m.span_extraction?.char_iou_macro],
      ["Mean best IoU per gold clause", (m) => m.span_extraction?.mean_best_iou],
    ]],
    ["Segment classification (multi-label)", [
      ["Accuracy (exact label-set match)", (m) => m.segment_classification?.accuracy],
      ["Accuracy on segments with a gold label", (m) => m.segment_classification?.accuracy_on_labelled_segments],
      ["Precision (micro)", (m) => m.segment_classification?.micro_precision],
      ["Recall (micro)", (m) => m.segment_classification?.micro_recall],
      ["Macro F1", (m) => m.segment_classification?.macro_f1],
      ["Weighted F1", (m) => m.segment_classification?.weighted_f1],
    ]],
    ["Clause detection per contract", [
      ["Accuracy", (m) => m.contract_detection?.accuracy],
      ["Precision", (m) => m.contract_detection?.precision],
      ["Recall", (m) => m.contract_detection?.recall],
      ["F1 (micro)", (m) => m.contract_detection?.f1],
      ["Macro F1", (m) => m.contract_detection?.macro_f1],
    ]],
  ];
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Metric</th>{models.map((m) => <th key={m.name} className="num"><ModelTag model={m.name} short /></th>)}</tr></thead>
        <tbody>
          {groups.map(([g, rows]) => (
            <Fragment key={g}>
              <tr><td colSpan={models.length + 1} style={{ background: "var(--paper-2)", fontWeight: 600, fontSize: 12.5 }}>{g}</td></tr>
              {rows.map(([label, f]) => {
                const vals = models.map(f);
                const best = Math.max(...vals.map((v) => v ?? -1));
                return <tr key={label}><td>{label}</td>{vals.map((v, i) => <td key={i} className={`num${v === best ? " best" : ""}`}>{fmt(v)}</td>)}</tr>;
              })}
            </Fragment>
          ))}
          <tr><td colSpan={models.length + 1} style={{ background: "var(--paper-2)", fontWeight: 600, fontSize: 12.5 }}>Cost (CPU, 2 threads)</td></tr>
          <tr><td>Training time</td>{models.map((m) => <td key={m.name} className="num">{duration(m.timing?.train_seconds ?? m.training.train_seconds)}</td>)}</tr>
          <tr><td>Inference time per test contract</td>{models.map((m) => <td key={m.name} className="num">{duration(m.timing?.inference_seconds_per_contract)}</td>)}</tr>
          <tr><td>Throughput (characters / second)</td>{models.map((m) => <td key={m.name} className="num">{int(m.timing?.chars_per_second)}</td>)}</tr>
          <tr><td>Trainable parameters / features</td>{models.map((m) => <td key={m.name} className="num">{m.training.n_parameters ? int(m.training.n_parameters) : m.training.n_features ? `${int(m.training.n_features)} feat.` : "—"}</td>)}</tr>
        </tbody>
      </table>
    </div>
  );
}

function ModelDetailView({ name }: { name: ModelName }) {
  const { data, error, loading, reload } = useAsync(() => api.model(name), [name]);
  const cats = useAsync(() => api.categories());
  const nameOf = useMemo(() => new Map((cats.data ?? []).map((c) => [c.key, c.name])), [cats.data]);
  if (loading && !data) return <Loading />;
  if (error || !data) return <ErrorBox message={error ?? "No data"} onRetry={reload} />;
  if (!data.span_per_category) return <div className="alert info">This model has not been evaluated yet.</div>;

  const perCat = Object.entries(data.span_per_category)
    .filter(([, v]) => v.n_gold > 0)
    .map(([k, v]) => ({ label: nameOf.get(k) ?? k, value: v.span_f1, note: `P ${v.span_precision.toFixed(2)}, R ${v.span_recall.toFixed(2)}, ${v.n_gold} gold / ${v.n_pred} predicted` }))
    .sort((a, b) => b.value - a.value);
  const hist = (data.training.history ?? []) as HistoryEntry[];
  const hasCurves = hist.length > 0 && "epoch" in (hist[0] ?? {});
  const cm = data.confusion_matrix;
  const cmLabels = cm ? cm.labels.map((k) => (k === "None" ? "None" : nameOf.get(k) ?? k)) : [];
  const conf = data.contract_detection?.confusion;

  return (
    <>
      <div className="grid-even">
        <Panel title="Per-category span F1" sub="Test split, categories with at least one gold clause. Hover for precision, recall and counts.">
          <HBars rows={perCat} color={MODEL_COLOR[name]} />
        </Panel>
        <div>
          {hasCurves ? (
            <Panel title="Training history" sub={`Validation span F1 per epoch (best of four global thresholds); best epoch ${data.training.best_epoch} restored at the end of training.`}>
              <LineChart yMax={Math.max(0.1, ...hist.map((h) => h.val_span_f1)) * 1.15} series={[{ name: "Validation span F1", color: MODEL_COLOR[name], values: hist.map((h) => ({ x: h.epoch, y: h.val_span_f1 })) }]} />
              <LineChart series={[{ name: "Training loss", color: "#7c8594", values: hist.map((h) => ({ x: h.epoch, y: h.train_loss })) }]} height={160} />
            </Panel>
          ) : (
            <Panel title="Training" sub="Regularisation strength selected on validation span F1.">
              <table className="table">
                <thead><tr><th>C</th><th className="num">Fit time</th><th className="num">Val span F1</th><th className="num">Val char IoU</th></tr></thead>
                <tbody>
                  {(data.training.history as Record<string, number>[] | undefined ?? []).map((h) => (
                    <tr key={h.C}><td>{h.C}{h.C === (data.training.selected as Record<string, number> | undefined)?.C ? " (selected)" : ""}</td><td className="num">{duration(h.fit_seconds)}</td><td className="num">{h.val_span_f1.toFixed(3)}</td><td className="num">{h.val_char_iou.toFixed(3)}</td></tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          )}
          <div style={{ height: 20 }} />
          <Panel title="Clause detection confusion" sub="Every (contract, category) pair in the test split: does the contract contain the clause?">
            {conf && (
              <table className="table" style={{ maxWidth: 380 }}>
                <thead><tr><th /><th className="num">Predicted present</th><th className="num">Predicted absent</th></tr></thead>
                <tbody>
                  <tr><td>Gold present</td><td className="num best">{int(conf.tp)}</td><td className="num">{int(conf.fn)}</td></tr>
                  <tr><td>Gold absent</td><td className="num">{int(conf.fp)}</td><td className="num best">{int(conf.tn)}</td></tr>
                </tbody>
              </table>
            )}
          </Panel>
          <div style={{ height: 20 }} />
          <Panel title="Configuration">
            <pre className="arch" style={{ margin: 0, maxHeight: 300 }}>{JSON.stringify({ training: data.training.config, decoding_tuned_on_validation: (data.training as Record<string, unknown>).decoding ?? "per-category thresholds on segments" }, null, 2)}</pre>
          </Panel>
        </div>
      </div>
      {cm && (
        <>
          <div style={{ height: 20 }} />
          <Panel title="Segment confusion matrix" sub={`Primary label per test segment (${cm.note}). Log colour scale; the diagonal holds correct segments, the last row/column is "no clause". Hover any cell for its count.`}>
            <Heatmap labels={cmLabels} matrix={cm.matrix} />
          </Panel>
        </>
      )}
    </>
  );
}

export default function ModelsPage() {
  const { data, error, loading, reload } = useAsync(() => api.models());
  const [sel, setSel] = useState<ModelName>("bilstm_attn");
  const ordered = (data ?? []).slice().sort((a, b) => MODEL_ORDER.indexOf(a.name) - MODEL_ORDER.indexOf(b.name));
  return (
    <div className="page">
      <PageHead title="Model performance" sub="Everything on this page was measured on the 102-contract CUAD test split. Span metrics evaluate extracted text; classification metrics evaluate labels only." />
      {error && <ErrorBox message={error} onRetry={reload} />}
      {loading && !data ? <Loading /> : (
        <>
          <Panel title="All metrics" tight><AllModelsTable models={ordered} /></Panel>
          <div style={{ height: 28 }} />
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 14, flexWrap: "wrap" }}>
            <h2>{MODEL_LABEL[sel]} in detail</h2>
            <div className="seg" role="tablist">
              {MODEL_ORDER.map((m) => <button key={m} role="tab" aria-selected={sel === m} className={sel === m ? "on" : ""} onClick={() => setSel(m)}><span className="swatch" style={{ background: MODEL_COLOR[m] }} />{MODEL_LABEL[m]}</button>)}
            </div>
          </div>
          <ModelDetailView name={sel} />
        </>
      )}
    </div>
  );
}
