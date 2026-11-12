from __future__ import annotations
import json
from pathlib import Path
from PySide6.QtGui import QPageSize, QPdfWriter, QTextDocument
from ..core.report import markdown_to_html


def write_export(fmt: str, path: str, md: str, json_text: str, csv_text: str, provenance: dict):
    """Write one export. Every format carries the provenance manifest: MD/HTML/PDF embed a Provenance section,
    JSON includes it, and CSV gets a sidecar <file>.manifest.json."""
    p = Path(path)
    fmt = fmt.lower()
    if fmt == "md":
        p.write_text(md, encoding="utf-8")
    elif fmt == "json":
        p.write_text(json_text, encoding="utf-8")
    elif fmt == "csv":
        p.write_text(csv_text, encoding="utf-8", newline="")
        p.with_name(p.name + ".manifest.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    elif fmt == "html":
        p.write_text(markdown_to_html(md, p.stem), encoding="utf-8")
    elif fmt == "pdf":
        doc = QTextDocument()
        doc.setMarkdown(md)
        w = QPdfWriter(str(p))
        w.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        doc.print_(w)
    else:
        raise ValueError(f"unknown export format: {fmt}")
