import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ErrorBox, Loading, MODEL_INITIAL, MetricBar, ModelTag, PageHead, Panel } from "../components/ui";
import { api } from "../lib/api";
import { MODEL_COLOR, MODEL_LABEL, MODEL_ORDER, int, pageLabel, when } from "../lib/format";
import type { Analysis, Finding, ModelName, ModelSummary } from "../lib/types";
import { useAsync } from "../lib/useAsync";

function TestMetricsTable({ models }: { models: ModelSummary[] }) {
  const rows: [string, (m: ModelSummary) => number | undefined][] = [
    ["Span F1 (IoU ≥ 0.5)", (m) => m.span_extraction?.span_f1],
    ["Span precision", (m) => m.span_extraction?.span_precision],
    ["Span recall", (m) => m.span_extraction?.span_recall],
    ["Span macro-F1 (per category)", (m) => m.span_extraction?.span_macro_f1],
    ["Exact match F1", (m) => m.span_extraction?.exact_match_f1],
    ["Character IoU", (m) => m.span_extraction?.char_iou],
    ["Mean best IoU per gold clause", (m) => m.span_extraction?.mean_best_iou],
    ["Segment macro-F1", (m) => m.segment_classification?.macro_f1],
    ["Segment weighted-F1", (m) => m.segment_classification?.weighted_f1],
    ["Clause detection F1 (per contract)", (m) => m.contract_detection?.f1],
  ];
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Metric (CUAD test split)</th>{models.map((m) => <th key={m.name} className="num"><ModelTag model={m.name} short /></th>)}</tr></thead>
        <tbody>
          {rows.map(([label, f]) => {
            const vals = models.map(f);
            const best = Math.max(...vals.map((v) => v ?? -1));
            return (
              <tr key={label}>
                <td>{label}</td>
                {vals.map((v, i) => <td key={i} className={`num${v === best ? " best" : ""}`}>{v == null ? "—" : v.toFixed(3)}</td>)}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Excerpt({ text, f }: { text: string; f: Finding }) {
  const pad = 70;
  const s = Math.max(0, f.start - pad), e = Math.min(text.length, f.end + pad);
  const inner = text.slice(f.start, f.end);
  const short = inner.length > 420 ? inner.slice(0, 260) + " … " + inner.slice(-120) : inner;
  return (
    <div className="excerpt">
      {s > 0 && "…"}{text.slice(s, f.start)}<mark>{short}</mark>{text.slice(f.end, e)}{e < text.length && "…"}
    </div>
  );
}

function AnalysisComparison({ a }: { a: Analysis }) {
  const [filter, setFilter] = useState<"all" | "disagree" | "unanimous">("disagree");
  const [outModel, setOutModel] = useState<ModelName>(a.models[0]);
  const c = a.comparison;
  const findings = useMemo(() => c.findings.filter((f) => filter === "all" || (filter === "unanimous" ? f.status === "all" : f.status !== "all")), [c, filter]);
  const pairs = a.models.flatMap((x, i) => a.models.slice(i + 1).map((y) => [x, y] as const));
  const modelClauses = a.clauses.filter((x) => x.model === outModel);

  return (
    <>
      <div className="strip">
        <div><div className="v">{c.summary.findings}</div><div className="k">Distinct passages found</div></div>
        <div><div className="v">{c.summary.unanimous}</div><div className="k">Found by all {a.models.length} models</div></div>
        <div><div className="v">{c.summary.partial}</div><div className="k">Found by some models</div></div>
        <div><div className="v">{c.summary.single_model}</div><div className="k">Found by one model only</div></div>
      </div>
      <div style={{ height: 20 }} />
      <div className="grid-2">
        <Panel title="Clauses per category and model" sub="Count of extracted spans. Rows sorted by name; agreement is the share of passages every model found." tight>
          <div className="table-wrap" style={{ maxHeight: 480 }}>
            <table className="table">
              <thead>
                <tr><th>Category</th>{a.models.map((m) => <th key={m} className="num" title={MODEL_LABEL[m]}><span className="swatch" style={{ background: MODEL_COLOR[m] }} /> {MODEL_INITIAL[m]}</th>)}<th className="num">Agreement</th></tr>
              </thead>
              <tbody>
                {c.categories.map((row) => {
                  const counts = a.models.map((m) => row.by_model[m] ?? 0);
                  const disagree = counts.some((x) => x === 0);
                  return (
                    <tr key={row.category}>
                      <td>{row.category_name}</td>
                      {counts.map((n, i) => (
                        <td key={i} className="num">
                          <span className="dotcell" style={{ background: n ? `${MODEL_COLOR[a.models[i]]}22` : "transparent", color: n ? "var(--ink)" : "var(--ink-3)", border: n ? "none" : "1px dashed var(--rule)" }}>{n || "0"}</span>
                        </td>
                      ))}
                      <td className="num">{disagree ? <span className="pill warn">{row.unanimous}/{row.findings}</span> : <span className="pill good">{row.unanimous}/{row.findings}</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel title="Pairwise overlap" sub="Character-level Jaccard similarity of two models' extractions (same category). 1.00 = identical highlights." tight>
          <table className="table">
            <tbody>
              {pairs.map(([x, y]) => {
                const v = c.pairwise_char_jaccard[`${x}|${y}`];
                return (
                  <tr key={`${x}${y}`}>
                    <td><ModelTag model={x} short /> <span className="faint">vs</span> <ModelTag model={y} short /></td>
                    <td style={{ width: 150 }}><MetricBar value={v} color="var(--accent)" /></td>
                  </tr>
                );
              })}
              {pairs.length === 0 && <tr><td className="muted">Run at least two models to compare them.</td></tr>}
            </tbody>
          </table>
        </Panel>
      </div>

      <div style={{ height: 20 }} />
      <Panel
        title="Passages and which models found them"
        right={
          <div className="seg">
            <button className={filter === "disagree" ? "on" : ""} onClick={() => setFilter("disagree")}>Disagreements ({c.summary.partial + c.summary.single_model})</button>
            <button className={filter === "unanimous" ? "on" : ""} onClick={() => setFilter("unanimous")}>Unanimous ({c.summary.unanimous})</button>
            <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")}>All ({c.summary.findings})</button>
          </div>
        }
        tight
      >
        <div style={{ maxHeight: 620, overflow: "auto" }}>
          {findings.length === 0 && <div className="empty"><p>No passages in this view.</p></div>}
          {findings.slice(0, 300).map((f, i) => (
            <div className="finding" key={i}>
              <div>
                <div className="small faint">{pageLabel(f.page_start, null)}</div>
                <div className="models" aria-label={`Found by ${f.models.map((m) => MODEL_LABEL[m]).join(", ")}`}>
                  {a.models.map((m) => (
                    <span key={m} className={`mchip${f.models.includes(m) ? "" : " off"}`} style={{ background: MODEL_COLOR[m] }} title={`${MODEL_LABEL[m]}: ${f.models.includes(m) ? "found" : "not found"}`}>{MODEL_INITIAL[m]}</span>
                  ))}
                </div>
              </div>
              <div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <strong style={{ fontWeight: 600 }}>{f.category_name}</strong>
                  {f.status === "all" ? <span className="pill good">All models</span> : f.status === "single" ? <span className="pill">{MODEL_LABEL[f.models[0]]} only</span> : <span className="pill warn">{f.models.length} of {a.models.length}</span>}
                </div>
                <Excerpt text={a.document.text} f={f} />
              </div>
            </div>
          ))}
          {findings.length > 300 && <p className="faint small" style={{ padding: 14 }}>Showing the first 300 of {findings.length}.</p>}
        </div>
      </Panel>

      <div style={{ height: 20 }} />
      <Panel
        title="Outputs by model"
        right={
          <div className="seg">
            {a.models.map((m) => <button key={m} className={outModel === m ? "on" : ""} onClick={() => setOutModel(m)}><span className="swatch" style={{ background: MODEL_COLOR[m] }} />{MODEL_LABEL[m]}</button>)}
          </div>
        }
        tight
      >
        <div className="table-wrap" style={{ maxHeight: 480 }}>
          <table className="table">
            <thead><tr><th>Category</th><th>Extracted text</th><th className="num">Page</th><th className="num">Score</th></tr></thead>
            <tbody>
              {modelClauses.map((x) => (
                <tr key={x.id}>
                  <td className="nowrap">{x.category_name}</td>
                  <td style={{ fontFamily: "var(--serif)", fontSize: 13.5 }}>{x.text.length > 320 ? x.text.slice(0, 320) + "…" : x.text}</td>
                  <td className="num">{x.page_start ?? "—"}</td>
                  <td className="num">{x.score.toFixed(2)}</td>
                </tr>
              ))}
              {modelClauses.length === 0 && <tr><td colSpan={4} className="muted">{MODEL_LABEL[outModel]} extracted no clauses from this contract.</td></tr>}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}

export default function ComparePage() {
  const { id } = useParams();
  const nav = useNavigate();
  const models = useAsync(() => api.models());
  const list = useAsync(() => api.analyses(50));
  const analysis = useAsync(() => (id ? api.analysis(id) : Promise.resolve(null)), [id]);
  const multi = (list.data ?? []).filter((r) => r.models.length > 1);
  const ordered = (models.data ?? []).slice().sort((x, y) => MODEL_ORDER.indexOf(x.name) - MODEL_ORDER.indexOf(y.name));

  return (
    <div className="page">
      <PageHead
        title="Model comparison"
        sub="Measured test-set performance of the four models, and how their predictions agree on a contract you analysed."
        actions={
          <select className="input" value={id ?? ""} onChange={(e) => nav(e.target.value ? `/compare/${e.target.value}` : "/compare")} aria-label="Choose an analysis" style={{ minWidth: 280 }}>
            <option value="">Choose an analysed contract…</option>
            {multi.map((r) => <option key={r.id} value={r.id}>{r.filename} — {r.models.length} models, {when(r.created_at)}</option>)}
          </select>
        }
      />
      {models.error && <ErrorBox message={models.error} onRetry={models.reload} />}
      <Panel title="Test-set performance" sub="102 held-out CUAD contracts never seen in training or threshold tuning." right={<Link className="small" to="/models">Per-category detail</Link>} tight>
        {models.loading && !models.data ? <Loading /> : <TestMetricsTable models={ordered} />}
      </Panel>
      <div style={{ height: 24 }} />
      {!id && (
        <Panel>
          <div className="empty">
            <h3>Compare predictions on a contract</h3>
            <p>{multi.length ? "Choose an analysis above that ran two or more models." : "Analyse a contract with two or more models to see where they agree and disagree."}</p>
            {!multi.length && <div style={{ marginTop: 14 }}><Link className="btn primary" to="/analyze">Analyse a contract</Link></div>}
          </div>
        </Panel>
      )}
      {id && analysis.loading && !analysis.data && <Loading label="Loading analysis…" />}
      {id && analysis.error && <ErrorBox message={analysis.error} onRetry={analysis.reload} />}
      {id && analysis.data && (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12, gap: 12 }}>
            <h2>{analysis.data.document.filename}</h2>
            <span className="muted small">{int(analysis.data.n_clauses)} spans from {analysis.data.models.length} models · <Link to={`/analyses/${analysis.data.id}`}>Open in reader</Link></span>
          </div>
          {analysis.data.models.length < 2
            ? <div className="alert info">This analysis ran a single model. Run two or more models to compare them.</div>
            : <AnalysisComparison a={analysis.data} />}
        </>
      )}
    </div>
  );
}
