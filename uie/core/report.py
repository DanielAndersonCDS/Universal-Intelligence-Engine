from __future__ import annotations
import csv, io, json
from . import provenance
from .util import human_bytes, human_count

LABEL = {"trend": "TREND", "changepoint": "SHIFT", "correlation": "ASSOCIATION", "near_duplicate": "REDUNDANCY",
         "outliers": "OUTLIERS", "concentration": "CONCENTRATION", "trend_caveat": "TREND CAVEAT", "group_contrast": "GROUP CONTRAST",
         "volatility": "VOLATILITY", "vol_shift": "VOL SHIFT", "return_correlation": "RETURN ASSOC", "spurious_correlation": "SPURIOUS",
         "benford": "BENFORD", "round_numbers": "ROUND NUMBERS", "threshold_bunching": "THRESHOLD", "near_duplicate_tx": "NEAR-DUP TX",
         "variance_driver": "VARIANCE", "time_gap": "TIME GAP", "target_leakage": "LEAKAGE", "leakage_missingness": "LEAKAGE",
         "class_imbalance": "IMBALANCE", "near_constant": "NEAR-CONSTANT", "high_cardinality": "HIGH-CARD"}


def template_summary(sheet) -> str:
    top = sheet.ranked()[:3]
    s = (f"Analyzed {sheet.rows:,} records across {sheet.cols} columns. "
         f"Detected {sheet.patterns()} patterns and {sheet.anomalies()} anomalies.")
    if top:
        s += " Most notable: " + " ".join(f.narrate() for f in top)
    return s


def _lab(f):
    return LABEL.get(f.type, f.type.replace("quality_", "").upper())


def render_markdown(sheet, mode: str = "technical") -> str:
    L = [f"# Intelligence Report: {sheet.dataset}", "",
         f"*Generated {sheet.created}. Narration: {sheet.narrator}.*", "",
         "## Dataset overview", "", "| Metric | Value |", "|---|---|",
         f"| Records analyzed | {sheet.rows:,} ({human_count(sheet.rows)}) |",
         f"| Columns | {sheet.cols} |", f"| Original size | {human_bytes(sheet.size_bytes)} |",
         f"| Patterns detected | {sheet.patterns()} |", f"| Notable anomalies | {sheet.anomalies()} |",
         f"| Time column | {sheet.time_col or 'none'} |", f"| Analysis time | {sheet.seconds:.1f}s |"]
    if sheet.group_by:
        L.append(f"| Grouped by | {sheet.group_by} |")
    if sheet.label:
        L.append(f"| Label column | {sheet.label} |")
    if sheet.triaged():
        L.append(f"| Reviewed findings (hidden) | {len(sheet.triaged())} |")
    L += ["",
         "## Summary", "", sheet.narrative, ""]
    ranked = sheet.ranked()
    shown = ranked[:5] if mode in ("executive", "key") else ranked
    L += ["## Key findings", ""]
    L += [f"- **{f.id}** [{_lab(f)}] {f.narrate()}" for f in shown] or ["- No significant patterns detected."]
    L.append("")
    quality = sheet.quality()
    if mode != "key" and quality:
        if mode == "executive":
            quality = [f for f in quality if f.severity != "info"]
        L += ["## Data quality", ""] + [f"- **{f.id}** {f.narrate()}" for f in quality] + [""]
    if mode != "key" and sheet.triaged():
        L += ["## Reviewed findings", "", "These were accepted or dismissed in an earlier review and are excluded from the counts above.", ""]
        L += [f"- **{f.id}** [{f.status.upper()}] {f.narrate()}" + (f" *({f.status_note})*" if f.status_note else "") for f in sheet.triaged()]
        L.append("")
    if mode == "technical":
        L += ["## Evidence appendix", ""]
        for f in sheet.active():
            L += [f"### {f.id} [{_lab(f)}] {f.title}", "", f.narrate(), "",
                  f"- Confidence: `{f.confidence}`", f"- Columns: {', '.join('`' + c + '`' for c in f.columns) or 'n/a'}",
                  f"- Metrics: `{json.dumps(f.metrics, default=str)}`", "", "```sql", f.evidence_sql, "```", ""]
    if sheet.warnings:
        L += ["## Warnings", ""] + [f"- {w}" for w in sheet.warnings] + [""]
    L += provenance.render_markdown(sheet.provenance)
    L += ["---", "*Every figure above was computed deterministically from the data. Associations are reported as "
           "associations; this report makes no causal claims.*"]
    return "\n".join(L)


def facts_csv(sheet) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "status", "severity", "type", "title", "text", "confidence", "columns", "group", "fingerprint", "evidence_sql"])
    for f in sheet.facts:
        w.writerow([f.id, f.status, f.severity, f.type, f.title, f.narrate(), f.confidence, ";".join(f.columns),
                    f"{f.group['column']}={f.group['value']}" if f.group else "", f.fingerprint, f.evidence_sql])
    return buf.getvalue()


# ---- shared export helpers (used by the GUI and the CLI) ----
def markdown_to_html(md: str, title: str = "UIE report") -> str:
    """Tiny converter for the markdown subset UIE emits: headings, tables, bullets, code fences, bold/italic/code."""
    import html as h, re

    def inline(t):
        t = h.escape(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        return re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)

    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + h.escape("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        m = re.match(r"(#{1,3}) (.*)", ln)
        if m:
            out.append(f"<h{len(m[1])}>{inline(m[2])}</h{len(m[1])}>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[-| ]+\|$", lines[i]):
                    rows.append([c.strip() for c in re.split(r"(?<!\\)\|", lines[i].strip().strip("|"))])
                i += 1
            t = "<table>" + "".join(f"<tr>{''.join(('<th>' if k == 0 else '<td>') + inline(c.replace(chr(92) + '|', '|')) + ('</th>' if k == 0 else '</td>') for c in r)}</tr>" for k, r in enumerate(rows)) + "</table>"
            out.append(t)
            continue
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif ln.strip() == "---":
            out.append("<hr>")
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}"
           "table{border-collapse:collapse;margin:1em 0}td,th{border:1px solid #bbb;padding:3px 8px;text-align:left}th{background:#eee}"
           "pre{background:#f4f4f4;padding:8px;overflow:auto}code{background:#f4f4f4;padding:0 3px}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{h.escape(title)}</title><style>{css}</style></head><body>{''.join(out)}</body></html>"


def sheet_json(sheet) -> str:
    return json.dumps(sheet.to_dict(), indent=2, default=str)
