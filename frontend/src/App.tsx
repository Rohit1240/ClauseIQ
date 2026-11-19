import { useEffect, useState } from "react";
import { NavLink, Link, Route, Routes } from "react-router-dom";
import { BrandMark, IconBook, IconChart, IconCompare, IconHome, IconTag, IconUpload, IconDoc } from "./components/icons";
import { api } from "./lib/api";
import type { Health } from "./lib/types";
import DashboardPage from "./pages/Dashboard";
import AnalyzePage from "./pages/Analyze";
import ResultsPage from "./pages/Results";
import ComparePage from "./pages/Compare";
import CategoriesPage from "./pages/Categories";
import ModelsPage from "./pages/Models";
import DocsPage from "./pages/Docs";
import HistoryPage from "./pages/History";

function Sidebar() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);
  useEffect(() => {
    let t: number;
    const tick = () => api.health().then((h) => { setHealth(h); setDown(false); }).catch(() => setDown(true))
      .finally(() => { t = window.setTimeout(tick, 20000); });
    tick();
    return () => window.clearTimeout(t);
  }, []);
  const ready = health ? Object.values(health.models).filter((s) => s === "loaded" || s === "available").length : 0;
  return (
    <aside className="sidebar">
      <Link to="/" className="brand" aria-label="ClauseIQ home">
        <BrandMark />
        <span className="brand-name">Clause<span>IQ</span></span>
      </Link>
      <nav className="nav" aria-label="Main">
        <NavLink to="/" end><IconHome />Dashboard</NavLink>
        <NavLink to="/analyze"><IconUpload />Analyse a contract</NavLink>
        <NavLink to="/analyses"><IconDoc />Analyses</NavLink>
        <NavLink to="/compare"><IconCompare />Model comparison</NavLink>
        <div className="nav-sep" />
        <NavLink to="/categories"><IconTag />Clause categories</NavLink>
        <NavLink to="/models"><IconChart />Model performance</NavLink>
        <NavLink to="/docs"><IconBook />Documentation</NavLink>
      </nav>
      <div className="sidebar-foot">
        <div className="status-line">
          <span className={`dot ${down ? "bad" : health ? "ok" : ""}`} />
          {down ? "API unreachable" : health ? `API online · ${ready}/4 models ready` : "Connecting…"}
        </div>
        <div style={{ marginTop: 6 }}>Trained on CUAD v1 (CC BY 4.0)</div>
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <div className="shell">
      <Sidebar />
      <main className="main">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/analyses" element={<HistoryPage />} />
          <Route path="/analyses/:id" element={<ResultsPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/compare/:id" element={<ComparePage />} />
          <Route path="/categories" element={<CategoriesPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/docs" element={<DocsPage />} />
          <Route path="*" element={<div className="page"><div className="empty"><h3>Page not found</h3><p><Link to="/">Go to the dashboard</Link></p></div></div>} />
        </Routes>
      </main>
    </div>
  );
}
