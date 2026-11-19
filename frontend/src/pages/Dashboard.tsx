import { Link, useNavigate } from "react-router-dom";
import { ErrorBox, Loading, MetricBar, ModelTag, PageHead, Panel } from "../components/ui";
import { IconUpload } from "../components/icons";
import { api } from "../lib/api";
import { MODEL_COLOR, duration, int, when } from "../lib/format";
import { useAsync } from "../lib/useAsync";

export default function DashboardPage() {
  const { data, error, loading, reload } = useAsync(() => api.dashboard());
  const nav = useNavigate();
  if (loading && !data) return <div className="page"><Loading /></div>;
  if (error || !data) return <div className="page"><ErrorBox message={error ?? "No data"} onRetry={reload} /></div>;

  const ready = data.models.filter((m) => m.status === "loaded" || m.status === "available").length;
  const evaluated = data.models.filter((m) => m.span_extraction);
  const best = (f: (m: (typeof evaluated)[number]) => number) => Math.max(...evaluated.map(f));
  const bestSpan = evaluated.length ? best((m) => m.span_extraction!.span_f1) : 0;

  return (
    <div className="page">
      <PageHead
        title="Dashboard"
        sub={<>Clause extraction with four models trained from scratch on {int(data.dataset.n_contracts)} CUAD contracts. Scores below are measured on the {data.dataset.split.test} held-out test contracts.</>}
        actions={<Link className="btn primary" to="/analyze"><IconUpload />Analyse a contract</Link>}
      />

      <div className="strip" aria-label="Totals">
        <div><div className="v">{int(data.totals.contracts_analysed)}</div><div className="k">Contracts analysed</div></div>
        <div><div className="v">{int(data.totals.clauses_extracted)}</div><div className="k">Clauses extracted</div></div>
        <div><div className="v">{data.n_categories}</div><div className="k">Clause categories</div></div>
        <div><div className="v">{ready} / 4</div><div className="k">Models available</div></div>
        <div><div className="v">{int(data.dataset.n_spans)}</div><div className="k">Expert annotations in CUAD</div></div>
      </div>

      <div style={{ height: 20 }} />
        <Panel
          title="Model performance"
          sub="Test split. Span F1 counts a prediction as correct when it overlaps a gold clause of the same category with IoU ≥ 0.5."
          right={<Link to="/models" className="small nowrap">Full metrics</Link>}
          tight
        >
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Span F1</th>
                  <th className="num">Exact match F1</th>
                  <th className="num">Char IoU</th>
                  <th className="num">Clause detection F1</th>
                  <th className="num">Inference / contract</th>
                </tr>
              </thead>
              <tbody>
                {data.models.map((m) => (
                  <tr key={m.name}>
                    <td><ModelTag model={m.name} /><div className="faint small">{m.kind === "segment" ? "segment classifier" : "token tagger"} · {m.status}</div></td>
                    {m.span_extraction ? (
                      <>
                        <td style={{ minWidth: 150 }} className={m.span_extraction.span_f1 === bestSpan ? "best" : ""}>
                          <MetricBar value={m.span_extraction.span_f1} color={MODEL_COLOR[m.name]} />
                        </td>
                        <td className="num">{m.span_extraction.exact_match_f1.toFixed(3)}</td>
                        <td className="num">{m.span_extraction.char_iou.toFixed(3)}</td>
                        <td className="num">{m.contract_detection!.f1.toFixed(3)}</td>
                        <td className="num">{duration(m.timing!.inference_seconds_per_contract)}</td>
                      </>
                    ) : (
                      <td colSpan={5} className="muted">Not evaluated yet — run <code>python scripts/evaluate.py --model {m.name}</code></td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

      <div style={{ height: 20 }} />
      <div className="grid-2">
      <Panel title="How the four models differ" tight>
        <table className="table">
          <thead><tr><th>Model</th><th>What it learns</th><th>Extraction unit</th><th className="num">Training time (CPU)</th></tr></thead>
          <tbody>
            {data.models.map((m) => (
              <tr key={m.name}>
                <td><ModelTag model={m.name} /></td>
                <td className="muted">{DESCR[m.name]}</td>
                <td className="muted">{m.kind === "segment" ? "Sentence-like segments" : "Tokens or segments, chosen per category on validation"}</td>
                <td className="num">{duration(m.training.train_seconds ?? m.timing?.train_seconds ?? null)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
        <Panel title="Recent activity" right={<Link to="/analyses" className="small">All analyses</Link>} tight>
          {data.recent.length === 0 ? (
            <div className="empty">
              <h3>No contracts analysed yet</h3>
              <p>Upload a PDF or TXT contract to extract its clauses.</p>
              <div style={{ marginTop: 14 }}><Link className="btn primary" to="/analyze">Upload a contract</Link></div>
            </div>
          ) : (
            <table className="table">
              <tbody>
                {data.recent.map((r) => (
                  <tr key={r.id} className="click" onClick={() => nav(`/analyses/${r.id}`)}>
                    <td>
                      <Link to={`/analyses/${r.id}`} onClick={(e) => e.stopPropagation()} style={{ fontWeight: 500, color: "var(--ink)" }}>{r.filename}</Link>
                      <div className="faint small">{r.n_pages} page{r.n_pages === 1 ? "" : "s"}, {r.models.length} model{r.models.length === 1 ? "" : "s"}, {r.n_categories} categories</div>
                    </td>
                    <td className="num"><div>{int(r.n_clauses)} clauses</div><div className="faint small">{when(r.created_at)}</div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
      </div>
    </div>
  );
}

const DESCR = {
  tfidf_lr: "Weighted word n-grams of each sentence and its neighbours; one logistic regression per category.",
  cnn: "Local phrase patterns through stacked dilated convolutions (≈65-token receptive field).",
  bilstm: "Left and right context of every token through a two-layer bidirectional LSTM.",
  bilstm_attn: "BiLSTM context plus multi-head self-attention that weighs the most relevant words.",
} as const;
