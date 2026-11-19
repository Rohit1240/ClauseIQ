"""Agreement / disagreement between models for one analysed contract."""
from __future__ import annotations

from itertools import combinations


def _overlap(a, b) -> int:
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def compare(clauses: list[dict], models: list[str]) -> dict:
    """Group same-category spans from different models that overlap (any shared characters)
    into 'findings'. Each finding lists which models found it; agreement = share of the
    selected models that predicted an overlapping span of that category."""
    by_cat: dict[str, list[dict]] = {}
    for c in clauses:
        by_cat.setdefault(c["category"], []).append(c)
    findings = []
    for cat, items in by_cat.items():
        items = sorted(items, key=lambda c: c["start"])
        groups: list[list[dict]] = []
        for it in items:
            for g in groups:
                if any(_overlap(it, o) > 0 for o in g):
                    g.append(it)
                    break
            else:
                groups.append([it])
        for g in groups:
            ms = sorted({c["model"] for c in g}, key=models.index)
            start, end = min(c["start"] for c in g), max(c["end"] for c in g)
            findings.append({
                "category": cat, "category_name": g[0]["category_name"],
                "start": start, "end": end, "page_start": min(c["page_start"] or 0 for c in g) or None,
                "models": ms, "missing_models": [m for m in models if m not in ms],
                "agreement": len(ms) / len(models) if models else 0,
                "status": "all" if len(ms) == len(models) else ("single" if len(ms) == 1 else "partial"),
                "spans": [{"model": c["model"], "start": c["start"], "end": c["end"], "score": c["score"],
                           "id": c["id"]} for c in g],
            })
    findings.sort(key=lambda f: (f["start"], f["category"]))

    # pairwise character-level agreement (Jaccard of predicted (category, char) sets)
    def charset(m):
        out = {}
        for c in clauses:
            if c["model"] == m:
                out.setdefault(c["category"], []).append((c["start"], c["end"]))
        return out

    sets = {m: charset(m) for m in models}
    pair = {}
    for a, b in combinations(models, 2):
        inter = union = 0
        for cat in set(sets[a]) | set(sets[b]):
            ra = _ranges(sets[a].get(cat, []))
            rb = _ranges(sets[b].get(cat, []))
            i = _inter(ra, rb)
            inter += i
            union += _len(ra) + _len(rb) - i
        pair[f"{a}|{b}"] = round(inter / union, 4) if union else None

    cats = {}
    for f in findings:
        d = cats.setdefault(f["category"], {"category": f["category"], "category_name": f["category_name"],
                                            "by_model": {m: 0 for m in models}, "findings": 0, "unanimous": 0})
        d["findings"] += 1
        d["unanimous"] += f["status"] == "all"
        for s in f["spans"]:
            d["by_model"][s["model"]] += 1
    summary = {
        "findings": len(findings),
        "unanimous": sum(f["status"] == "all" for f in findings),
        "partial": sum(f["status"] == "partial" for f in findings),
        "single_model": sum(f["status"] == "single" for f in findings),
    }
    return {"models": models, "summary": summary, "findings": findings,
            "pairwise_char_jaccard": pair, "categories": sorted(cats.values(), key=lambda d: d["category_name"])}


def _ranges(xs):
    out = []
    for s, e in sorted(xs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _len(r):
    return sum(e - s for s, e in r)


def _inter(a, b):
    i = j = t = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            t += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t
