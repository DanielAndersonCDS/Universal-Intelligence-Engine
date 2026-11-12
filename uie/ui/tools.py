"""Result viewer shared by the COMPARE, RECONCILE, SPLITS and EXPECT tabs."""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)
from ..core import subreport
from ..core.util import fmt_num
from .exporter import write_export


def _table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return fmt_num(v) if abs(v) >= 0.0001 or v == 0 else f"{v:.2e}"
    return str(v)


class ReportView(QWidget):
    """Controls go in `self.controls` (a layout); call show_report() with a SubReport."""

    def __init__(self, on_status, colors):
        super().__init__()
        self.report = None
        self.on_status = on_status
        self.colors = colors
        lay = QVBoxLayout(self)
        self.controls = QVBoxLayout()
        lay.addLayout(self.controls)
        self.summary = QLabel("")
        self.summary.setObjectName("Dim")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.find_tbl = _table(["ID", "SEV", "FINDING"])
        self.find_tbl.setColumnWidth(0, 60)
        self.find_tbl.setColumnWidth(1, 80)
        self.main_tbl = _table([""])
        self.table_name = QLabel("")
        self.table_name.setObjectName("Dim")
        self.detail = QTextBrowser()
        self.find_tbl.itemSelectionChanged.connect(self._on_find)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.table_name)
        bl.addWidget(self.main_tbl)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.find_tbl)
        top.addWidget(self.detail)
        top.setSizes([700, 500])
        sp = QSplitter(Qt.Orientation.Vertical)
        sp.addWidget(top)
        sp.addWidget(bottom)
        sp.setSizes([260, 360])
        lay.addWidget(sp, 1)
        bar = QHBoxLayout()
        bar.addStretch(1)
        for fmt in ("md", "html", "pdf", "json", "csv"):
            b = QPushButton(fmt.upper())
            b.clicked.connect(lambda _=False, f=fmt: self.export(f))
            bar.addWidget(b)
        lay.addLayout(bar)

    def show_report(self, r: subreport.SubReport):
        self.report = r
        self.summary.setText("   |   ".join(f"{k}: {_fmt(v)}" for k, v in list(r.summary.items())[:9]))
        sev = {"warning": self.colors["warning"], "notable": self.colors["accent"], "info": self.colors["text_dim"]}
        self.find_tbl.setRowCount(0)
        for f in r.facts:
            i = self.find_tbl.rowCount()
            self.find_tbl.insertRow(i)
            for j, (txt, col) in enumerate(((f.id, self.colors["accent"]), (f.severity.upper(), sev[f.severity]), (f.narrate().replace("`", ""), None))):
                it = QTableWidgetItem(txt)
                if col:
                    it.setForeground(QColor(col))
                self.find_tbl.setItem(i, j, it)
        name, (cols, rows) = next(iter(r.tables.items()), ("", ([], [])))
        self._fill(name, cols, rows)
        self.detail.setHtml("<p style='color:%s'>Select a finding to see its evidence SQL.</p>" % self.colors["text_dim"])
        for w in r.warnings:
            self.on_status("WARNING: " + w)

    def _fill(self, name, cols, rows):
        t = self.main_tbl
        t.clear()
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        shown = rows[:5000]
        t.setRowCount(len(shown))
        for i, row in enumerate(shown):
            for j, v in enumerate(row):
                it = QTableWidgetItem(_fmt(v))
                if isinstance(v, (int, float)):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(i, j, it)
        t.resizeColumnsToContents()
        self.table_name.setText(f"{name.upper()}  ({len(rows):,} rows" + (", first 5,000 shown; export CSV for all)" if len(rows) > 5000 else ")"))

    def _on_find(self):
        r = self.find_tbl.currentRow()
        if not self.report or r < 0 or r >= len(self.report.facts):
            return
        import html
        f = self.report.facts[r]
        setup = "\n".join(self.report.setup_sql)
        self.detail.setHtml(
            f"<h3 style='color:{self.colors['accent']}'>{f.id} &middot; {f.type.upper()}</h3><p>{html.escape(f.narrate())}</p>"
            f"<p style='color:{self.colors['accent']}'><b>EVIDENCE SQL</b></p><pre style='color:{self.colors['info']}; white-space: pre-wrap'>{html.escape(f.evidence_sql or '(none)')}</pre>"
            f"<p style='color:{self.colors['accent']}'><b>SETUP SQL</b></p><pre style='color:{self.colors['text_dim']}; white-space: pre-wrap'>{html.escape(setup)}</pre>")

    def export(self, fmt):
        if not self.report:
            return self.on_status("NOTHING TO EXPORT. Run the check first.")
        r = self.report
        stem = "".join(ch if ch.isalnum() else "_" for ch in r.kind + "_" + r.title.split(":")[-1].strip())[:40]
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}", f"{stem}.{fmt}")
        if not path:
            return
        table = next(iter(r.tables), None)
        try:
            write_export(fmt, path, subreport.render_markdown(r), subreport.to_json(r), subreport.tables_csv(r, table), r.provenance)
            self.on_status(f"EXPORTED {path}" + ("  (+ .manifest.json)" if fmt == "csv" else ""))
        except Exception as e:
            self.on_status("EXPORT FAILED: " + str(e))
