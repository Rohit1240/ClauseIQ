import { memo, useMemo } from "react";
import { MODEL_COLOR } from "../lib/format";
import type { Clause, ModelName, PageRange } from "../lib/types";

interface Piece { start: number; end: number; clauses: number[] }

function hexToRgba(hex: string, a: number) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** Splits the document at every clause boundary (and page start) so overlapping clauses from
 *  several models/categories can be highlighted precisely by character offset. */
function buildPieces(textLen: number, clauses: Clause[], pages: PageRange[]): Piece[] {
  const cuts = new Set<number>([0, textLen]);
  clauses.forEach((c) => { cuts.add(c.start); cuts.add(c.end); });
  pages.forEach((p) => cuts.add(p.start));
  const pts = [...cuts].filter((x) => x >= 0 && x <= textLen).sort((a, b) => a - b);
  const starts = clauses.map((c, i) => [c.start, c.end, i] as const).sort((a, b) => a[0] - b[0]);
  const pieces: Piece[] = [];
  let k = 0;
  const active: (readonly [number, number, number])[] = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const s = pts[i], e = pts[i + 1];
    while (k < starts.length && starts[k][0] <= s) active.push(starts[k++]);
    for (let j = active.length - 1; j >= 0; j--) if (active[j][1] <= s) active.splice(j, 1);
    pieces.push({ start: s, end: e, clauses: active.filter((a) => a[0] < e && a[1] > s).map((a) => a[2]) });
  }
  return pieces;
}

interface Props {
  text: string;
  pages: PageRange[];
  clauses: Clause[];
  mode: "agreement" | ModelName;
  activeId: string | null;
  flashId: string | null;
  onSelect: (clauseIds: string[]) => void;
}

function DocumentReaderInner({ text, pages, clauses, mode, activeId, flashId, onSelect }: Props) {
  const pieces = useMemo(() => buildPieces(text.length, clauses, pages), [text.length, clauses, pages]);
  const pageStarts = useMemo(() => new Map(pages.filter((p) => p.page > 1).map((p) => [p.start, p.page])), [pages]);
  const firstPieceOf = useMemo(() => {
    const m = new Map<number, string>();
    pieces.forEach((p, i) => p.clauses.forEach((ci) => { if (!m.has(ci)) m.set(ci, `piece-${i}`); }));
    return m;
  }, [pieces]);
  const anchor = useMemo(() => {
    const m = new Map<string, string>();
    clauses.forEach((c, i) => { const id = firstPieceOf.get(i); if (id) m.set(id, c.id); });
    return m;
  }, [clauses, firstPieceOf]);

  return (
    <div className="doc-text">
      {pieces.map((p, i) => {
        const pageNo = pageStarts.get(p.start);
        const pb = pageNo ? <span className="page-break" aria-label={`Page ${pageNo}`}><span>Page {pageNo}</span></span> : null;
        const slice = text.slice(p.start, p.end);
        if (!p.clauses.length) return <span key={i}>{pb}{slice}</span>;
        const cs = p.clauses.map((ci) => clauses[ci]);
        const models = [...new Set(cs.map((c) => c.model))];
        let bg: string;
        if (mode === "agreement") {
          bg = `rgba(245, 197, 66, ${[0, 0.22, 0.36, 0.5, 0.64][Math.min(4, models.length)]})`;
        } else {
          bg = hexToRgba(MODEL_COLOR[mode], 0.26);
        }
        const ids = cs.map((c) => c.id);
        const isActive = activeId != null && ids.includes(activeId);
        const isFlash = flashId != null && ids.includes(flashId);
        const anchorId = anchor.get(`piece-${i}`);
        const title = cs.map((c) => `${c.category_name} — ${c.model}`).join("\n");
        return (
          <span key={i}>
            {pb}
            <mark
              id={anchorId ? `clause-${anchorId}` : undefined}
              className={`hl${isActive ? " active" : ""}${isFlash ? " flash" : ""}`}
              style={{ background: bg, color: "inherit" }}
              title={title}
              onClick={() => onSelect(ids)}
            >
              {slice}
            </mark>
          </span>
        );
      })}
    </div>
  );
}

export const DocumentReader = memo(DocumentReaderInner);
