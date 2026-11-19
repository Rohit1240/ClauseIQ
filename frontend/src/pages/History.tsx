import { Link, useNavigate } from "react-router-dom";
import { ErrorBox, Loading, ModelTag, PageHead, Panel } from "../components/ui";
import { api } from "../lib/api";
import { duration, int, when } from "../lib/format";
import { useAsync } from "../lib/useAsync";

export default function HistoryPage() {
  const { data, error, loading, reload } = useAsync(() => api.analyses(100));
  const nav = useNavigate();

  async function remove(id: string, name: string) {
    if (!window.confirm(`Delete the analysis of “${name}”? This cannot be undone.`)) return;
    try {
      await api.deleteAnalysis(id);
      reload();
    } catch (e) {
      window.alert((e as Error).message);
    }
  }

  return (
    <div className="page">
      <PageHead title="Analyses" sub="Every contract analysed in this workspace, newest first." actions={<Link className="btn primary" to="/analyze">Analyse a contract</Link>} />
      {error && <ErrorBox message={error} onRetry={reload} />}
      {loading && !data ? <Loading /> : data && (
        <Panel tight>
          {data.length === 0 ? (
            <div className="empty"><h3>No analyses yet</h3><p>Upload a contract to get started.</p></div>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Contract</th><th>Models</th><th className="num">Categories</th><th className="num">Clauses</th><th className="num">Run time</th><th className="num">Analysed</th><th /></tr></thead>
                <tbody>
                  {data.map((r) => (
                    <tr key={r.id} className="click" onClick={() => nav(`/analyses/${r.id}`)}>
                      <td><span style={{ fontWeight: 500 }}>{r.filename}</span><div className="faint small">{r.n_pages} page{r.n_pages === 1 ? "" : "s"}, {int(r.n_chars)} characters</div></td>
                      <td><div style={{ display: "flex", flexDirection: "column", gap: 2 }}>{r.models.map((m) => <ModelTag key={m} model={m} short />)}</div></td>
                      <td className="num">{r.n_categories}</td>
                      <td className="num">{int(r.n_clauses)}</td>
                      <td className="num">{duration(r.seconds)}</td>
                      <td className="num">{when(r.created_at)}</td>
                      <td className="num" onClick={(e) => e.stopPropagation()}>
                        {r.models.length > 1 && <Link className="btn small ghost" to={`/compare/${r.id}`}>Compare</Link>}
                        <button className="btn small ghost" onClick={() => remove(r.id, r.filename)} aria-label={`Delete analysis of ${r.filename}`}>Delete</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      )}
    </div>
  );
}
