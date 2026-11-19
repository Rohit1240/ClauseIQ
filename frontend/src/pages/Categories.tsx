import { Fragment, useMemo, useState } from "react";
import { ErrorBox, Loading, ModelTag, PageHead, Panel } from "../components/ui";
import { IconChevron } from "../components/icons";
import { api } from "../lib/api";
import { MODEL_COLOR, MODEL_ORDER, int, pct } from "../lib/format";
import type { Category, ModelName } from "../lib/types";
import { useAsync } from "../lib/useAsync";

type SortKey = "name" | "spans" | "contracts" | ModelName;

export default function CategoriesPage() {
  const { data, error, loading, reload } = useAsync(() => api.categories());
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<SortKey>("name");
  const [open, setOpen] = useState<string | null>(null);

  const rows = useMemo(() => {
    if (!data) return [];
    const ql = q.trim().toLowerCase();
    const f = data.filter((c) => !ql || c.name.toLowerCase().includes(ql) || c.description.toLowerCase().includes(ql));
    const val = (c: Category): number | string => {
      if (sort === "name") return c.index;
      if (sort === "spans") return -c.stats.n_spans;
      if (sort === "contracts") return -c.stats.n_contracts_with_clause;
      return -(c.performance[sort]?.span_f1 ?? -1);
    };
    return [...f].sort((a, b) => (val(a) < val(b) ? -1 : val(a) > val(b) ? 1 : 0));
  }, [data, q, sort]);

  if (loading && !data) return <div className="page"><Loading /></div>;
  if (error || !data) return <div className="page"><ErrorBox message={error ?? "No data"} onRetry={reload} /></div>;
  const models = MODEL_ORDER.filter((m) => data[0]?.performance[m]);
  const th = (key: SortKey, label: React.ReactNode, cls = "num") => (
    <th className={cls}>
      <button onClick={() => setSort(key)} style={{ all: "unset", cursor: "pointer", fontWeight: sort === key ? 600 : 500, color: sort === key ? "var(--ink)" : undefined }} aria-pressed={sort === key}>{label}</button>
    </th>
  );

  return (
    <div className="page">
      <PageHead
        title="Clause categories"
        sub="The 41 CUAD categories with their official definitions, annotation counts across all 510 contracts, and each model's span F1 on the test split."
        actions={<input className="input" placeholder="Search categories or definitions" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 280 }} aria-label="Search categories" />}
      />
      <Panel tight>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                {th("name", "Category", "")}
                {th("spans", "Annotations")}
                {th("contracts", "Contracts with clause")}
                {models.map((m) => <Fragment key={m}>{th(m, <ModelTag model={m} short />)}</Fragment>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => {
                const isOpen = open === c.key;
                const best = Math.max(...models.map((m) => c.performance[m]?.span_f1 ?? -1));
                return (
                  <Fragment key={c.key}>
                    <tr className="click" onClick={() => setOpen(isOpen ? null : c.key)} aria-expanded={isOpen}>
                      <td style={{ minWidth: 260 }}>
                        <div style={{ display: "flex", gap: 6, alignItems: "center", fontWeight: 500 }}><IconChevron open={isOpen} width={14} height={14} />{c.name}</div>
                        <div className="faint small" style={{ marginLeft: 20 }}>{c.description.length > 110 && !isOpen ? c.description.slice(0, 110) + "…" : !isOpen ? c.description : ""}</div>
                      </td>
                      <td className="num">{int(c.stats.n_spans)}</td>
                      <td className="num">{c.stats.n_contracts_with_clause} <span className="faint">({pct(c.stats.pct_contracts_with_clause / 100, 0)})</span></td>
                      {models.map((m) => {
                        const v = c.performance[m]?.span_f1;
                        return (
                          <td key={m} className={`num${v === best && v! > 0 ? " best" : ""}`}>
                            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                              {v == null ? "—" : v.toFixed(2)}
                              <span className="bar" style={{ minWidth: 36, width: 36 }}><span style={{ width: `${(v ?? 0) * 100}%`, background: MODEL_COLOR[m] }} /></span>
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                    {isOpen && (
                      <tr>
                        <td colSpan={3 + models.length} style={{ background: "var(--paper-2)", padding: "14px 20px 18px" }}>
                          <div className="grid-even">
                            <div>
                              <h3 style={{ marginBottom: 6 }}>Definition</h3>
                              <p style={{ fontFamily: "var(--serif)", fontSize: 14.5, lineHeight: 1.6 }}>{c.description}</p>
                              <dl className="kv" style={{ marginTop: 12 }}>
                                <dt>Answer format</dt><dd>{c.answer_format || "—"}</dd>
                                <dt>CUAD group</dt><dd>{c.group === "-" ? "None" : c.group}</dd>
                                <dt>Annotations by split</dt><dd>train {c.stats.spans_by_split.train}, validation {c.stats.spans_by_split.val}, test {c.stats.spans_by_split.test}</dd>
                                <dt>Median clause length</dt><dd>{int(c.stats.span_chars.median)} characters</dd>
                                <dt>Contracts without clause</dt><dd>{c.stats.n_contracts_without_clause} of 510</dd>
                              </dl>
                            </div>
                            <div>
                              <h3 style={{ marginBottom: 6 }}>Test-set performance</h3>
                              <table className="table" style={{ background: "var(--paper)", border: "1px solid var(--rule)" }}>
                                <thead><tr><th>Model</th><th className="num">Span P</th><th className="num">Span R</th><th className="num">Span F1</th><th className="num">Exact F1</th><th className="num">Detect F1</th></tr></thead>
                                <tbody>
                                  {models.map((m) => {
                                    const p = c.performance[m]!;
                                    return (
                                      <tr key={m}>
                                        <td><ModelTag model={m} short /></td>
                                        <td className="num">{p.span_precision.toFixed(2)}</td>
                                        <td className="num">{p.span_recall.toFixed(2)}</td>
                                        <td className="num">{p.span_f1.toFixed(2)}</td>
                                        <td className="num">{p.exact_match_f1.toFixed(2)}</td>
                                        <td className="num">{p.contract_f1.toFixed(2)}</td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                              <p className="faint small" style={{ marginTop: 6 }}>{c.performance[models[0]]?.test_gold_spans ?? 0} gold spans in the test split (after merging overlaps). Detect F1 = clause presence per contract.</p>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
      <p className="faint small" style={{ marginTop: 10 }}>Category names and definitions are reproduced from CUAD v1 (The Atticus Project, CC BY 4.0).</p>
    </div>
  );
}
