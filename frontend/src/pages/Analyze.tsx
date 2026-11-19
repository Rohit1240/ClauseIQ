import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ErrorBox, ModelTag, PageHead, Panel } from "../components/ui";
import { IconAlert, IconUpload, IconX } from "../components/icons";
import { api } from "../lib/api";
import { MODEL_ORDER, int } from "../lib/format";
import type { Category, DocumentInfo, ModelName, ModelSummary } from "../lib/types";

type Phase = "idle" | "uploading" | "ready" | "running";
const MAX_MB = 15;

export default function AnalyzePage() {
  const nav = useNavigate();
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [doc, setDoc] = useState<(DocumentInfo & { text: string }) | null>(null);
  const [over, setOver] = useState(false);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [cats, setCats] = useState<Category[]>([]);
  const [selModels, setSelModels] = useState<Set<ModelName>>(new Set(MODEL_ORDER));
  const [allCats, setAllCats] = useState(true);
  const [selCats, setSelCats] = useState<Set<string>>(new Set());
  const [catQuery, setCatQuery] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.models().then((ms) => {
      setModels(ms);
      setSelModels(new Set(ms.filter((m) => m.status !== "missing" && m.status !== "error").map((m) => m.name)));
    }).catch((e) => setError(e.message));
    api.categories().then(setCats).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (phase !== "running") return;
    const t0 = Date.now();
    const id = window.setInterval(() => setElapsed((Date.now() - t0) / 1000), 200);
    return () => window.clearInterval(id);
  }, [phase]);

  async function handleFile(f: File | undefined) {
    if (!f) return;
    setError(null);
    const ext = f.name.toLowerCase().split(".").pop();
    if (ext !== "pdf" && ext !== "txt") { setError("Only .pdf and .txt files are supported."); return; }
    if (f.size > MAX_MB * 1024 * 1024) { setError(`The file is ${(f.size / 1048576).toFixed(1)} MB; the limit is ${MAX_MB} MB.`); return; }
    if (f.size === 0) { setError("The selected file is empty."); return; }
    setPhase("uploading");
    try {
      const d = await api.upload(f);
      setDoc(d);
      setPhase("ready");
    } catch (e) {
      setError((e as Error).message);
      setPhase("idle");
    }
  }

  async function run() {
    if (!doc) return;
    if (selModels.size === 0) { setError("Select at least one model."); return; }
    if (!allCats && selCats.size === 0) { setError("Select at least one clause category, or choose all 41."); return; }
    setError(null);
    setPhase("running");
    setElapsed(0);
    try {
      const order = MODEL_ORDER.filter((m) => selModels.has(m));
      const a = await api.analyse(doc.id, order, allCats ? null : [...selCats]);
      nav(`/analyses/${a.id}`);
    } catch (e) {
      setError((e as Error).message);
      setPhase("ready");
    }
  }

  const filteredCats = useMemo(() => {
    const q = catQuery.trim().toLowerCase();
    return q ? cats.filter((c) => c.name.toLowerCase().includes(q) || c.description.toLowerCase().includes(q)) : cats;
  }, [cats, catQuery]);

  const reset = () => { setDoc(null); setPhase("idle"); setError(null); };

  return (
    <div className="page">
      <PageHead title="Analyse a contract" sub="Upload a commercial contract, choose models and clause categories, and ClauseIQ extracts the matching clause text with its location." />
      {error && <div style={{ marginBottom: 16 }}><ErrorBox message={error} /></div>}

      {!doc && (
        <div
          className={`drop${over ? " over" : ""}`}
          role="button"
          tabIndex={0}
          aria-label="Upload a contract"
          onClick={() => phase === "idle" && input.current?.click()}
          onKeyDown={(e) => { if ((e.key === "Enter" || e.key === " ") && phase === "idle") input.current?.click(); }}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); if (phase === "idle") handleFile(e.dataTransfer.files?.[0]); }}
        >
          {phase === "uploading" ? (
            <><span className="spinner" style={{ width: 24, height: 24, display: "inline-block" }} /><h3>Extracting text…</h3><p>Reading pages and preserving character offsets.</p></>
          ) : (
            <>
              <IconUpload />
              <h3>Drop a contract here, or click to choose a file</h3>
              <p>PDF with selectable text, or plain text (.txt). Up to {MAX_MB} MB. Scanned PDFs are not supported because ClauseIQ does not run OCR.</p>
            </>
          )}
          <input ref={input} type="file" accept=".pdf,.txt,application/pdf,text/plain" hidden onChange={(e) => { handleFile(e.target.files?.[0]); e.target.value = ""; }} />
        </div>
      )}

      {doc && (
        <div className="grid-2">
          <div>
            <Panel
              title={<span className="file-row"><span className="file-icon">{doc.file_type.toUpperCase()}</span><span>{doc.filename}<div className="faint small" style={{ fontWeight: 400 }}>{doc.n_pages} page{doc.n_pages === 1 ? "" : "s"}, {int(doc.n_chars)} characters extracted</div></span></span>}
              right={phase !== "running" && <button className="btn small ghost" onClick={reset}><IconX />Remove</button>}
              tight
            >
              {doc.warnings.length > 0 && (
                <div className="alert warn" style={{ margin: 12 }}><IconAlert /><div>{doc.warnings.join(" ")}</div></div>
              )}
              <div className="preview" aria-label="Contract preview">{doc.text.slice(0, 12000)}{doc.text.length > 12000 && <span className="faint">{"\n\n"}… preview truncated; the full text ({int(doc.n_chars)} characters) is analysed.</span>}</div>
            </Panel>
          </div>

          <div>
            <Panel title="Models" sub="Run one model or compare all four on the same contract.">
              <div style={{ display: "grid", gap: 8 }}>
                {MODEL_ORDER.map((name) => {
                  const m = models.find((x) => x.name === name);
                  const usable = m && m.status !== "missing" && m.status !== "error";
                  const on = selModels.has(name);
                  return (
                    <label key={name} className={`check${on ? " on" : ""}`} style={{ opacity: usable ? 1 : 0.55 }}>
                      <input type="checkbox" checked={on} disabled={!usable || phase === "running"}
                        onChange={() => { const s = new Set(selModels); if (on) s.delete(name); else s.add(name); setSelModels(s); }} />
                      <div style={{ flex: 1 }}>
                        <div className="t"><ModelTag model={name} /></div>
                        <div className="d">
                          {!m ? "Loading…" : !usable ? "No trained checkpoint available" :
                            m.span_extraction ? `Test span F1 ${m.span_extraction.span_f1.toFixed(3)}, clause detection F1 ${m.contract_detection!.f1.toFixed(3)}` : "Checkpoint available, not yet evaluated"}
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
              <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
                <button className="btn small" disabled={phase === "running"} onClick={() => setSelModels(new Set(models.filter((m) => m.status !== "missing" && m.status !== "error").map((m) => m.name)))}>Select all four</button>
              </div>
            </Panel>

            <div style={{ height: 16 }} />
            <Panel title="Clause categories">
              <div className="seg" role="radiogroup" aria-label="Category scope">
                <button className={allCats ? "on" : ""} onClick={() => setAllCats(true)} role="radio" aria-checked={allCats} disabled={phase === "running"}>All {cats.length || 41}</button>
                <button className={!allCats ? "on" : ""} onClick={() => setAllCats(false)} role="radio" aria-checked={!allCats} disabled={phase === "running"}>Choose categories{!allCats && selCats.size ? ` (${selCats.size})` : ""}</button>
              </div>
              {!allCats && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                    <input className="input" style={{ flex: 1 }} placeholder="Filter categories" value={catQuery} onChange={(e) => setCatQuery(e.target.value)} aria-label="Filter categories" />
                    <button className="btn small" onClick={() => setSelCats(new Set(filteredCats.map((c) => c.key)))}>Select shown</button>
                    <button className="btn small ghost" onClick={() => setSelCats(new Set())}>Clear</button>
                  </div>
                  <div className="cat-picker">
                    {filteredCats.map((c) => (
                      <label key={c.key} title={c.description}>
                        <input type="checkbox" checked={selCats.has(c.key)} onChange={() => { const s = new Set(selCats); if (s.has(c.key)) s.delete(c.key); else s.add(c.key); setSelCats(s); }} />
                        {c.name}
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </Panel>

            <div style={{ height: 16 }} />
            {phase === "running" ? (
              <Panel title="Analysing">
                <div className="progress" aria-live="polite">
                  {MODEL_ORDER.filter((m) => selModels.has(m)).map((m) => (
                    <div className="progress-row" key={m}><span className="spinner" /><ModelTag model={m} /></div>
                  ))}
                  <div className="faint small">{elapsed.toFixed(1)} s elapsed. Long contracts are processed in overlapping 256-token windows; the recurrent models take longest on CPU.</div>
                </div>
              </Panel>
            ) : (
              <button className="btn primary" style={{ width: "100%", height: 40 }} onClick={run} disabled={selModels.size === 0}>
                Extract clauses with {selModels.size === 4 ? "all four models" : `${selModels.size} model${selModels.size === 1 ? "" : "s"}`}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
