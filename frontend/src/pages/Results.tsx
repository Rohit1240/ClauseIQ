import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { DocumentReader } from "../components/DocumentReader";
import { IconChevron, IconCompare, IconDownload } from "../components/icons";
import { ErrorBox, Loading, ModelTag } from "../components/ui";
import { api } from "../lib/api";
import { MODEL_COLOR, MODEL_LABEL, duration, int, pageLabel } from "../lib/format";
import type { Clause, ModelName } from "../lib/types";
import { useAsync } from "../lib/useAsync";

export default function ResultsPage() {
  const { id = "" } = useParams();
  const { data: a, error, loading, reload } = useAsync(() => api.analysis(id), [id]);
  const [mode, setMode] = useState<"agreement" | ModelName>("agreement");
  const [cat, setCat] = useState<string>("");
  const [activeId, setActiveId] = useState<string | null>(null);
  const [flashId, setFlashId] = useState<string | null>(null);
  const [open, setOpen] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (a && a.models.length === 1) setMode(a.models[0]);
  }, [a]);

  const visible: Clause[] = useMemo(() => {
    if (!a) return [];
    return a.clauses.filter((c) => (mode === "agreement" || c.model === mode) && (!cat || c.category === cat));
  }, [a, mode, cat]);

  const groups = useMemo(() => {
    const m = new Map<string, Clause[]>();
    visible.forEach((c) => { if (!m.has(c.category_name)) m.set(c.category_name, []); m.get(c.category_name)!.push(c); });
    return [...m.entries()].sort((x, y) => x[0].localeCompare(y[0]));
  }, [visible]);

  useEffect(() => { setOpen(new Set(groups.slice(0, 3).map(([k]) => k))); }, [a?.id, mode, cat]); // eslint-disable-line

  const jumpTo = useCallback((c: Clause) => {
    setActiveId(c.id);
    setFlashId(null);
    window.requestAnimationFrame(() => {
      const el = document.getElementById(`clause-${c.id}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        setFlashId(c.id);
      }
    });
  }, []);

  const onSelectInDoc = useCallback((ids: string[]) => {
    if (!a) return;
    const first = a.clauses.find((c) => ids.includes(c.id));
    if (!first) return;
    setActiveId(first.id);
    setOpen((o) => new Set(o).add(first.category_name));
    window.requestAnimationFrame(() => document.getElementById(`side-${first.id}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
  }, [a]);

  if (loading && !a) return <div className="page"><Loading label="Loading analysis…" /></div>;
  if (error || !a) return <div className="page"><ErrorBox message={error ?? "Analysis not found"} onRetry={reload} /></div>;

  const catsPresent = [...new Map(a.clauses.map((c) => [c.category, c.category_name])).entries()].sort((x, y) => x[1].localeCompare(y[1]));

  return (
    <div className="reader-layout">
      <div className="reader-col">
        <div className="reader-toolbar">
          <Link to="/analyses" className="small">Analyses</Link>
          <span className="faint">/</span>
          <strong style={{ fontWeight: 600, maxWidth: 320, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={a.document.filename}>{a.document.filename}</strong>
          <div style={{ flex: 1 }} />
          <div className="seg" role="tablist" aria-label="Show predictions from">
            {a.models.length > 1 && (
              <button className={mode === "agreement" ? "on" : ""} onClick={() => setMode("agreement")} role="tab" aria-selected={mode === "agreement"}>All models</button>
            )}
            {a.models.map((m) => (
              <button key={m} className={mode === m ? "on" : ""} onClick={() => setMode(m)} role="tab" aria-selected={mode === m}>
                <span className="swatch" style={{ background: MODEL_COLOR[m] }} />{MODEL_LABEL[m]}
              </button>
            ))}
          </div>
          <select className="input" value={cat} onChange={(e) => setCat(e.target.value)} aria-label="Filter by category" style={{ maxWidth: 220 }}>
            <option value="">All categories ({catsPresent.length})</option>
            {catsPresent.map(([k, n]) => <option key={k} value={k}>{n}</option>)}
          </select>
        </div>
        <div style={{ padding: "12px 28px 0" }} className="legend">
          {mode === "agreement" ? (
            <>
              <span>Highlight strength = number of models that extracted the passage:</span>
              {[1, 2, 3, 4].slice(0, a.models.length).map((n) => (
                <span key={n}><span className="swatch" style={{ background: `rgba(245,197,66,${[0, 0.22, 0.36, 0.5, 0.64][n]})`, width: 18 }} />{n}</span>
              ))}
            </>
          ) : (
            <span><span className="swatch" style={{ background: MODEL_COLOR[mode], opacity: 0.6 }} />Passages extracted by {MODEL_LABEL[mode]}</span>
          )}
          <span className="faint">Click a highlight to find its clause.</span>
        </div>
        <article className="doc-sheet" aria-label="Contract text">
          <DocumentReader text={a.document.text} pages={a.document.pages} clauses={visible} mode={mode}
            activeId={activeId} flashId={flashId} onSelect={onSelectInDoc} />
        </article>
      </div>

      <aside className="side-col" aria-label="Extracted clauses">
        <div className="side-head">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}>
            <h2>{int(visible.length)} extracted clause{visible.length === 1 ? "" : "s"}</h2>
            <span className="faint small">{a.document.n_pages} pp., {duration(a.seconds)}</span>
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 10, flexWrap: "wrap" }}>
            <a className="btn small" href={api.reportUrl(a.id, "html")} download><IconDownload />Report</a>
            <a className="btn small" href={api.reportUrl(a.id, "csv")} download>CSV</a>
            <a className="btn small" href={api.reportUrl(a.id, "json")} download>JSON</a>
            {a.models.length > 1 && <Link className="btn small" to={`/compare/${a.id}`}><IconCompare />Compare</Link>}
          </div>
          <table className="table" style={{ marginTop: 10, fontSize: 12.5 }}>
            <tbody>
              {a.models.map((m) => (
                <tr key={m}>
                  <td style={{ padding: "4px 0" }}><ModelTag model={m} /></td>
                  <td className="num" style={{ padding: "4px 0" }}>{a.runs[m].n_clauses} spans</td>
                  <td className="num faint" style={{ padding: "4px 0 4px 10px" }}>{duration(a.runs[m].seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {a.document.warnings.length > 0 && <div className="alert warn small" style={{ marginTop: 8 }}>{a.document.warnings.join(" ")}</div>}
        </div>
        <div className="side-scroll">
          {groups.length === 0 && (
            <div className="empty">
              <h3>No clauses found</h3>
              <p>{mode === "agreement" ? "None of the selected models" : MODEL_LABEL[mode] + " did not"} extract clauses{cat ? " for this category" : ""} above the validation-tuned thresholds.</p>
            </div>
          )}
          {groups.map(([name, items]) => {
            const isOpen = open.has(name);
            return (
              <div className="cat-group" key={name}>
                <button onClick={() => setOpen((o) => { const s = new Set(o); if (s.has(name)) s.delete(name); else s.add(name); return s; })} aria-expanded={isOpen}>
                  <span style={{ display: "flex", alignItems: "center", gap: 6 }}><IconChevron open={isOpen} width={14} height={14} />{name}</span>
                  <span className="pill">{items.length}</span>
                </button>
                {isOpen && items.map((c) => (
                  <div key={c.id} id={`side-${c.id}`} className={`clause${activeId === c.id ? " active" : ""}`} onClick={() => jumpTo(c)}
                    role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === "Enter") jumpTo(c); }}>
                    <div className="clause-meta">
                      <ModelTag model={c.model} short />
                      <span>{pageLabel(c.page_start, c.page_end)}</span>
                      <span title={a.runs[c.model]?.score_description}>score {c.score.toFixed(2)} <span className="faint">/ thr {c.threshold.toFixed(2)}</span></span>
                    </div>
                    <div className="clause-text">{c.text}</div>
                    <div className="faint small" style={{ marginTop: 3 }}>chars {int(c.start)}–{int(c.end)}</div>
                    {c.attention && c.attention.length > 0 && (
                      <div className="attn" title="Words (3+ letters) that received the most self-attention from this span's tokens, averaged over heads">
                        Attention focus:{c.attention.map((t) => <span key={t.start} className="tok">{t.token}</span>)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            );
          })}
          <p className="faint small" style={{ padding: "14px 18px" }}>
            Scores are model probabilities compared with per-category thresholds tuned on the validation split. They are not calibrated likelihoods; review each clause against the source.
          </p>
        </div>
      </aside>
    </div>
  );
}
