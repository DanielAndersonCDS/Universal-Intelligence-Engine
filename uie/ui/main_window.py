from __future__ import annotations
import html
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QCompleter, QDoubleSpinBox, QFileDialog, QListWidget, QListWidgetItem,
                               QFrame, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)
from ..core import expectations, narrator, ollama, qa
from ..core.curvefit import fit_curves
from ..core.drift import drift as run_drift
from ..core.regression import regress as run_regress
from ..core.engine import Engine
from ..core.reconcile import reconcile as run_reconcile
from ..core.report import facts_csv, render_markdown, sheet_json
from ..core.splitcheck import split_check as run_split_check
from ..core.util import fmt_num, human_bytes, human_count, pct
from .charts import EvidenceChart
from .exporter import write_export
from .plotbuilder import PlotBuilder
from .quanttab import QuantTab
from .tools import ReportView
from .widgets import Panel, StatTile
from .worker import Worker

FILTER = "Data (*.csv *.tsv *.txt *.parquet *.json *.jsonl *.ndjson *.xlsx *.duckdb *.db *.sqlite *.sqlite3);;All files (*)"
SEM_FILTER = "Semantics (*.yaml *.yml *.json);;All files (*)"
COMMANDS = ["OPEN", "RUN", "FINDINGS", "REPORT", "ASK", "PROFILE", "COMPARE", "RECONCILE", "SPLITS", "EXPECT", "MODEL", "CHARTS", "QUANT", "SEMANTICS", "GROUP",
            "LABEL", "DISMISS", "ACCEPT", "RESTORE", "THEME", "EXPORT", "HELP"]
HELP = ("OPEN [path]        load a dataset\n"
        "RUN                run analysis\n"
        "FINDINGS / REPORT / PROFILE   switch tab\n"
        "COMPARE            drift vs another dataset (PSI, KS, chi-square)\n"
        "RECONCILE / SPLITS / EXPECT / MODEL / CHARTS / QUANT   switch tab\n"
        "SEMANTICS [path]   load a column semantics file (SEMANTICS none clears it)\n"
        "GROUP <col|none>   run detectors inside each group\n"
        "LABEL <col|none>   ML label column\n"
        "DISMISS|ACCEPT|RESTORE <id> [note]   review a finding\n"
        "ASK <question>     question or SQL (head, schema, count; 'raw' is the uncleaned data)\n"
        "THEME <n>          switch theme\n"
        "EXPORT MD|HTML|PDF|JSON|CSV")


def table(headers, stretch_last=True):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setStretchLastSection(stretch_last)
    return t


def cell(text, align_right=False, color=None):
    it = QTableWidgetItem(str(text))
    if align_right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if color:
        from PySide6.QtGui import QColor
        it.setForeground(QColor(color))
    return it


class MainWindow(QMainWindow):
    def __init__(self, themes):
        super().__init__()
        self.themes, self.engine = themes, Engine()
        self.sheet = self.cmp_sheet = None
        self.settings = QSettings("UIE", "UniversalIntelligenceEngine")
        self._workers, self._chart_token = set(), 0
        self.recon_right = self.split_test = None
        self.tool_views = []
        self._row_facts = []
        self.setWindowTitle("UIE  |  Universal Intelligence Engine")
        self.resize(1500, 920)
        self.setAcceptDrops(True)
        self._build_menu()
        self._build_ui()
        themes.changed.connect(self._on_theme)
        self._on_theme(themes.current)

    # ---------- UI construction ----------
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, fn in (("Open dataset...", self.open_dataset), ("Run analysis", self.run_analysis)):
            a = QAction(text, self); a.triggered.connect(lambda _=False, f=fn: f()); m.addAction(a)
        m.addSeparator()
        for fmt in ("md", "html", "pdf", "json", "csv"):
            a = QAction(f"Export {fmt.upper()}...", self); a.triggered.connect(lambda _=False, f=fmt: self.export(f)); m.addAction(a)
        m.addSeparator()
        q = QAction("Quit", self); q.triggered.connect(self.close); m.addAction(q)
        v = self.menuBar().addMenu("&View")
        r = QAction("Reload themes", self); r.triggered.connect(self._reload_themes); v.addAction(r)
        h = self.menuBar().addMenu("&Help")
        a = QAction("Commands", self); a.triggered.connect(lambda: QMessageBox.information(self, "Commands", HELP)); h.addAction(a)

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        v = QVBoxLayout(root); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        # header bar with command line
        hb = QFrame(); hb.setObjectName("HeaderBar"); h = QHBoxLayout(hb); h.setContentsMargins(10, 6, 10, 6)
        brand = QLabel("UIE"); brand.setObjectName("Brand")
        self.cmd = QLineEdit(); self.cmd.setObjectName("CommandBar")
        self.cmd.setPlaceholderText("Command, then <GO>   (OPEN, RUN, FINDINGS, REPORT, ASK <question>, THEME, EXPORT, HELP)")
        comp = QCompleter(COMMANDS); comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive); self.cmd.setCompleter(comp)
        self.cmd.returnPressed.connect(self._command)
        go = QPushButton("GO"); go.setProperty("accent", True); go.clicked.connect(self._command)
        self.theme_combo = QComboBox()
        for t in self.themes.themes.values():
            self.theme_combo.addItem(t["name"], t["id"])
        self.theme_combo.currentIndexChanged.connect(lambda: self._set_theme(self.theme_combo.currentData()))
        self.clock = QLabel(); self.clock.setObjectName("Clock")
        h.addWidget(brand); h.addSpacing(10); h.addWidget(self.cmd, 1); h.addWidget(go)
        h.addSpacing(10); h.addWidget(self.theme_combo); h.addSpacing(10); h.addWidget(self.clock)
        v.addWidget(hb)
        # stat strip
        ss = QFrame(); ss.setObjectName("StatStrip"); s = QHBoxLayout(ss); s.setContentsMargins(10, 6, 10, 6); s.setSpacing(8)
        cap = QLabel("INTELLIGENCE\nCOMPRESSION"); cap.setObjectName("Dim"); s.addWidget(cap)
        self.tiles = {k: StatTile(k) for k in ("Records", "Original size", "Columns", "Patterns", "Anomalies", "Analysis time")}
        for t in self.tiles.values():
            s.addWidget(t, 1)
        v.addWidget(ss)
        # body
        split = QSplitter(Qt.Orientation.Horizontal); v.addWidget(split, 1)
        split.addWidget(self._build_left()); split.addWidget(self._build_tabs())
        split.setSizes([330, 1170]); split.setContentsMargins(8, 8, 8, 8)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setFixedWidth(200)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("READY. Open a dataset or drop a file on the window.")
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(1000); self._tick()

    def _build_left(self):
        p = Panel("Data source")
        self.open_btn = QPushButton("OPEN DATASET..."); self.open_btn.clicked.connect(lambda: self.open_dataset())
        self.path_lbl = QLabel("No dataset loaded"); self.path_lbl.setObjectName("Dim"); self.path_lbl.setWordWrap(True)
        self.table_combo = QComboBox(); self.table_combo.hide(); self.table_combo.currentIndexChanged.connect(self._on_table)
        self.time_combo = QComboBox(); self.time_combo.addItem("Time column: auto", None)
        self.group_combo = QComboBox(); self.group_combo.addItem("Group by: none", None)
        self.label_combo = QComboBox(); self.label_combo.addItem("Label: none", None)
        self.sem_lbl = QLabel("Semantics: none"); self.sem_lbl.setObjectName("Dim"); self.sem_lbl.setWordWrap(True)
        sem_row = QHBoxLayout(); sem_box = QWidget(); sem_box.setLayout(sem_row); sem_row.setContentsMargins(0, 0, 0, 0)
        lb = QPushButton("LOAD SEMANTICS..."); lb.clicked.connect(lambda: self.load_semantics())
        cb = QPushButton("CLEAR"); cb.clicked.connect(lambda: self.clear_semantics())
        sem_row.addWidget(lb, 1); sem_row.addWidget(cb)
        self.narr_combo = QComboBox(); self.narr_combo.addItem("Narration: template (no LLM)", "template")
        self.narr_combo.addItem("Narration: local LLM (Ollama)", "ollama")
        self.model_edit = QLineEdit(self.settings.value("model", "phi4-mini")); self.model_edit.setPlaceholderText("Ollama model")
        self.auto_chk = QCheckBox("Analyze automatically on open"); self.auto_chk.setChecked(True)
        self.run_btn = QPushButton("RUN ANALYSIS"); self.run_btn.setProperty("accent", True); self.run_btn.clicked.connect(self.run_analysis)
        for w in (self.open_btn, self.path_lbl, self.table_combo, self.time_combo, self.group_combo, self.label_combo, sem_box, self.sem_lbl,
                  self.narr_combo, self.model_edit, self.auto_chk, self.run_btn):
            p.body.addWidget(w)
        sc = Panel("Schema"); self.schema_tbl = table(["Column", "Type"]); sc.body.addWidget(self.schema_tbl)
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(8, 8, 4, 8); l.addWidget(p); l.addWidget(sc, 1)
        return w

    def _build_tabs(self):
        self.tabs = QTabWidget()
        # findings
        self.tab_find = QWidget(); fl = QVBoxLayout(self.tab_find)
        rb = QHBoxLayout()
        for text, st in (("DISMISS", "dismissed"), ("ACCEPT", "accepted"), ("RESTORE", "restore")):
            b = QPushButton(text); b.clicked.connect(lambda _=False, x=st: self._review(x)); rb.addWidget(b)
        self.show_reviewed = QCheckBox("Show reviewed findings"); self.show_reviewed.toggled.connect(lambda: self.sheet and self._fill_findings(keep_fp=self._selected_fp()))
        self.review_lbl = QLabel("Dismiss = false positive. Accept = known and fine. Reviewed findings stay hidden on later runs unless they get worse."); self.review_lbl.setObjectName("Dim")
        rb.addWidget(self.show_reviewed); rb.addWidget(self.review_lbl, 1)
        fl.addLayout(rb)
        self.find_tbl = table(["ID", "STATUS", "SEV", "TYPE", "FINDING"])
        self.find_tbl.setColumnWidth(0, 60); self.find_tbl.setColumnWidth(1, 80); self.find_tbl.setColumnWidth(2, 80); self.find_tbl.setColumnWidth(3, 130)
        self.find_tbl.itemSelectionChanged.connect(self._on_fact)
        self.detail = QTextBrowser(); self.chart = EvidenceChart()
        low = QSplitter(Qt.Orientation.Horizontal); low.addWidget(self.detail); low.addWidget(self.chart); low.setSizes([520, 620])
        vs = QSplitter(Qt.Orientation.Vertical); vs.addWidget(self.find_tbl); vs.addWidget(low); vs.setSizes([330, 420])
        fl.addWidget(vs)
        # report
        self.tab_report = QWidget(); rl = QVBoxLayout(self.tab_report)
        bar = QHBoxLayout()
        self.mode_combo = QComboBox()
        for label, key in (("Technical report", "technical"), ("Executive report", "executive"), ("Key findings only", "key")):
            self.mode_combo.addItem(label, key)
        self.mode_combo.currentIndexChanged.connect(self._render_report)
        bar.addWidget(self.mode_combo); bar.addStretch(1)
        for fmt in ("md", "html", "pdf", "json", "csv"):
            b = QPushButton(fmt.upper()); b.clicked.connect(lambda _=False, f=fmt: self.export(f)); bar.addWidget(b)
        self.report_view = QTextBrowser(); self.report_view.setOpenExternalLinks(True)
        rl.addLayout(bar); rl.addWidget(self.report_view)
        # ask
        self.tab_ask = QWidget(); al = QVBoxLayout(self.tab_ask)
        row = QHBoxLayout(); self.ask_in = QLineEdit()
        self.ask_in.setPlaceholderText("Ask in plain English (needs local LLM), or type SQL: SELECT ... FROM data. Shortcuts: head, schema, count")
        self.ask_in.returnPressed.connect(self.run_ask)
        ab = QPushButton("ASK"); ab.setProperty("accent", True); ab.clicked.connect(self.run_ask)
        row.addWidget(self.ask_in, 1); row.addWidget(ab)
        self.ask_sql = QLabel(""); self.ask_sql.setObjectName("Dim"); self.ask_sql.setWordWrap(True)
        self.ask_tbl = table([""], stretch_last=False)
        al.addLayout(row); al.addWidget(self.ask_sql); al.addWidget(self.ask_tbl, 1)
        # profile
        self.tab_prof = QWidget(); pl = QVBoxLayout(self.tab_prof)
        self.prof_tbl = table(["Column", "Type", "Null %", "Distinct", "Min", "Max", "Mean", "Std", "Median", "Top values"])
        pl.addWidget(self.prof_tbl)
        # compare (drift), reconcile, splits, expectations
        c = self._c()
        self.cmp_view = ReportView(self.status, c); self.rec_view = ReportView(self.status, c)
        self.spl_view = ReportView(self.status, c); self.exp_view = ReportView(self.status, c)
        self.tool_views = [self.cmp_view, self.rec_view, self.spl_view, self.exp_view]
        self.tab_cmp, self.tab_rec, self.tab_spl, self.tab_exp = self.cmp_view, self.rec_view, self.spl_view, self.exp_view
        cb = QPushButton("LOAD COMPARISON DATASET..."); cb.clicked.connect(self.run_compare)
        self.cmp_lbl = QLabel("Drift of another dataset (current) against the open one (reference): PSI, two-sample KS, chi-square with Cramer's V, null rates, schema changes."); self.cmp_lbl.setObjectName("Dim"); self.cmp_lbl.setWordWrap(True)
        self.cmp_view.controls.addWidget(cb); self.cmp_view.controls.addWidget(self.cmp_lbl)
        self.rec_key = QLineEdit(); self.rec_key.setPlaceholderText("Key column(s), comma separated  (required)")
        self.rec_cmp = QLineEdit(); self.rec_cmp.setPlaceholderText("Columns to compare, comma separated (blank = all shared numeric)")
        self.rec_tol = QDoubleSpinBox(); self.rec_tol.setDecimals(4); self.rec_tol.setRange(0, 1e9); self.rec_tol.setValue(0.01); self.rec_tol.setPrefix("tolerance ")
        self.rec_lbl = QLabel("Open dataset = left. Choose the right dataset, then RUN."); self.rec_lbl.setObjectName("Dim")
        rb1 = QPushButton("LOAD RIGHT DATASET..."); rb1.clicked.connect(lambda: self._pick_right("recon"))
        rb2 = QPushButton("RUN RECONCILE"); rb2.setProperty("accent", True); rb2.clicked.connect(self.run_reconcile)
        row = QHBoxLayout(); row.addWidget(rb1); row.addWidget(self.rec_tol); row.addWidget(rb2)
        for w in (self.rec_key, self.rec_cmp): self.rec_view.controls.addWidget(w)
        self.rec_view.controls.addLayout(row); self.rec_view.controls.addWidget(self.rec_lbl)
        self.spl_ent = QLineEdit(); self.spl_ent.setPlaceholderText("Entity column(s) to check for shared values (user_id, patient_id ...), comma separated")
        self.spl_ign = QLineEdit(); self.spl_ign.setPlaceholderText("Columns to ignore (ids, label is ignored automatically)")
        self.spl_col = QLineEdit(); self.spl_col.setPlaceholderText("Single-file mode: split column, train value, test value  (e.g.  split, train, test)")
        self.spl_lbl = QLabel("Open dataset = train. Load the test dataset, or use single-file mode."); self.spl_lbl.setObjectName("Dim")
        sb1 = QPushButton("LOAD TEST DATASET..."); sb1.clicked.connect(lambda: self._pick_right("split"))
        sb2 = QPushButton("RUN SPLIT CHECK"); sb2.setProperty("accent", True); sb2.clicked.connect(self.run_splits)
        row = QHBoxLayout(); row.addWidget(sb1); row.addWidget(sb2)
        for w in (self.spl_ent, self.spl_ign, self.spl_col): self.spl_view.controls.addWidget(w)
        self.spl_view.controls.addLayout(row); self.spl_view.controls.addWidget(self.spl_lbl)
        eb1 = QPushButton("EXPORT EXPECTATIONS FROM OPEN DATASET..."); eb1.clicked.connect(self.export_expectations)
        eb2 = QPushButton("VALIDATE A DATASET AGAINST AN EXPECTATIONS FILE..."); eb2.clicked.connect(self.validate_expectations)
        self.exp_lbl = QLabel("Expectations capture schema, null limits, value ranges, allowed categories and distributions (for PSI). Validate any later dataset against them."); self.exp_lbl.setObjectName("Dim"); self.exp_lbl.setWordWrap(True)
        row = QHBoxLayout(); row.addWidget(eb1); row.addWidget(eb2)
        self.exp_view.controls.addLayout(row); self.exp_view.controls.addWidget(self.exp_lbl)
        # model (regression + curve fit) and charts
        self.mod_view = ReportView(self.status, c)
        self.tool_views.append(self.mod_view)
        self.tab_mod = self.mod_view
        self.mod_target = QComboBox(); self.mod_feats = QListWidget(); self.mod_feats.setMaximumHeight(110)
        self.mod_x, self.mod_y = QComboBox(), QComboBox(); self.mod_at = QLineEdit(); self.mod_at.setPlaceholderText("Predict y at x = (comma separated, optional)")
        rg = QPushButton("RUN REGRESSION"); rg.setProperty("accent", True); rg.clicked.connect(self.run_model)
        cf = QPushButton("FIT CURVES"); cf.setProperty("accent", True); cf.clicked.connect(self.run_curvefit)
        r1 = QHBoxLayout(); r1.addWidget(QLabel("OLS:")); r1.addWidget(self.mod_target, 1); r1.addWidget(rg)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("Curve:")); r2.addWidget(self.mod_x, 1); r2.addWidget(self.mod_y, 1); r2.addWidget(self.mod_at, 1); r2.addWidget(cf)
        self.mod_view.controls.addLayout(r1); self.mod_view.controls.addWidget(self.mod_feats)
        self.mod_view.controls.addLayout(r2)
        hint = QLabel("Regression: pick a numeric target and tick the predictors (text columns become dummy variables). Exact least squares over all rows via SQL. "
                      "Results are associations, not causes."); hint.setObjectName("Dim"); hint.setWordWrap(True)
        self.mod_view.controls.addWidget(hint)
        self.plot_builder = PlotBuilder(lambda: self.engine, self._run, self.status, c)
        self.tool_views.append(self.plot_builder)
        self.tab_charts = self.plot_builder
        self.quant = QuantTab(self, c)
        self.tool_views.append(self.quant)
        self.tab_quant = self.quant
        for w, n in ((self.tab_find, "FINDINGS"), (self.tab_report, "REPORT"), (self.tab_ask, "ASK"),
                     (self.tab_prof, "PROFILE"), (self.tab_cmp, "COMPARE"), (self.tab_rec, "RECONCILE"),
                     (self.tab_spl, "SPLITS"), (self.tab_exp, "EXPECT"), (self.tab_mod, "MODEL"), (self.tab_charts, "CHARTS"), (self.tab_quant, "QUANT")):
            self.tabs.addTab(w, n)
        return self.tabs

    # ---------- helpers ----------
    def status(self, msg): self.statusBar().showMessage(msg)
    def _tick(self): self.clock.setText(datetime.now().strftime("%a %d %b %Y  %H:%M:%S"))
    def _c(self): return self.themes.current["colors"]

    def _run(self, fn, on_done, on_fail=None):
        w = Worker(fn, self)
        w.progress.connect(lambda m, p: (self.status(m.upper() + "..."), self.progress.setValue(p)))
        w.done.connect(on_done); w.failed.connect(on_fail or self._error)
        w.finished.connect(lambda w=w: self._workers.discard(w))
        self._workers.add(w); w.start()

    def _error(self, msg):
        self._busy(False); self.status("ERROR: " + msg[:160])
        QMessageBox.warning(self, "Error", msg)

    def _busy(self, b):
        for w in (self.open_btn, self.run_btn):
            w.setEnabled(not b)
        if not b:
            self.progress.setValue(0)

    def _set_theme(self, tid):
        self.settings.setValue("theme", tid); self.themes.apply(tid)

    def _reload_themes(self):
        self.themes.reload(); self.theme_combo.blockSignals(True); self.theme_combo.clear()
        for t in self.themes.themes.values():
            self.theme_combo.addItem(t["name"], t["id"])
        self.theme_combo.blockSignals(False); self._set_theme(self.themes.current.get("id", self.themes.default_id))

    def _on_theme(self, theme):
        i = self.theme_combo.findData(theme["id"])
        self.theme_combo.blockSignals(True); self.theme_combo.setCurrentIndex(max(i, 0)); self.theme_combo.blockSignals(False)
        self.chart.apply_theme(theme)
        c = theme["colors"]
        for v in self.tool_views: v.set_colors(c)
        self.report_view.document().setDefaultStyleSheet(
            f"h1,h2,h3 {{ color: {c['accent']}; }} code,pre {{ color: {c['info']}; }} th {{ color: {c['accent']}; }}")
        if self.sheet:
            self._fill_findings(keep_fp=self._selected_fp()); self._render_report()

    # ---------- open / analyze ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls: self.open_dataset(urls[0].toLocalFile())

    def open_dataset(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open dataset", self.settings.value("last_dir", ""), FILTER)
        if not path:
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        self._busy(True); self.status("LOADING...")
        self._run(lambda p: self.engine.open(path), self._after_open)

    def _after_open(self, src):
        self._busy(False); self.sheet = None
        self.path_lbl.setText(f"{src.name}\n{human_bytes(src.size_bytes)}")
        self.table_combo.blockSignals(True); self.table_combo.clear()
        for t in src.tables: self.table_combo.addItem(t)
        self.table_combo.setVisible(bool(src.tables)); self.table_combo.blockSignals(False)
        self._refresh_schema(); self._sync_from_sem(); self.status(f"LOADED {src.name}")
        if self.auto_chk.isChecked(): self.run_analysis()

    def _on_table(self):
        if not self.engine.src or not self.table_combo.currentText(): return
        self.engine.select_table(self.table_combo.currentText()); self._refresh_schema()

    def _refresh_schema(self):
        sch = self.engine.schema()
        self.schema_tbl.setRowCount(0)
        for n, t in sch:
            r = self.schema_tbl.rowCount(); self.schema_tbl.insertRow(r)
            self.schema_tbl.setItem(r, 0, cell(n)); self.schema_tbl.setItem(r, 1, cell(t))
        self.schema_tbl.resizeColumnToContents(0)
        self.plot_builder.set_columns(sch)
        self.quant.set_columns(sch, self.engine.temporal_columns(), self.engine.sem)
        keep = self.mod_target.currentData(); self.mod_target.clear(); self.mod_target.addItem("Target: choose", None)
        self.mod_x.clear(); self.mod_x.addItem("X: choose", None); self.mod_y.clear(); self.mod_y.addItem("Y: choose", None)
        checked = {self.mod_feats.item(i).text() for i in range(self.mod_feats.count()) if self.mod_feats.item(i).checkState() == Qt.CheckState.Checked}
        self.mod_feats.clear()
        for n, t in sch:
            self.mod_target.addItem(f"Target: {n}", n); self.mod_x.addItem(f"X: {n}", n); self.mod_y.addItem(f"Y: {n}", n)
            it = QListWidgetItem(n); it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if n in checked else Qt.CheckState.Unchecked); self.mod_feats.addItem(it)
        i = self.mod_target.findData(keep); self.mod_target.setCurrentIndex(max(i, 0))
        self.time_combo.clear(); self.time_combo.addItem("Time column: auto", None)
        for n in self.engine.temporal_columns(): self.time_combo.addItem(f"Time column: {n}", n)
        for combo, head in ((self.group_combo, "Group by"), (self.label_combo, "Label")):
            keep = combo.currentData(); combo.blockSignals(True); combo.clear(); combo.addItem(f"{head}: none", None)
            for n, _t in sch: combo.addItem(f"{head}: {n}", n)
            i = combo.findData(keep); combo.setCurrentIndex(max(i, 0)); combo.blockSignals(False)

    # ---------- semantics ----------
    def load_semantics(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Column semantics", self.settings.value("last_dir", ""), SEM_FILTER)
        if not path: return
        try:
            self.engine.set_semantics(path)
        except Exception as e:
            return self._error(f"Could not load semantics: {e}")
        self._sync_from_sem(); self.status(f"SEMANTICS LOADED: {Path(path).name}. Press RUN ANALYSIS to apply.")
        if self.engine.src and self.auto_chk.isChecked(): self.run_analysis()

    def clear_semantics(self):
        self.engine.set_semantics(None); self._sync_from_sem(); self.status("SEMANTICS CLEARED")

    def _sync_from_sem(self):
        sem = self.engine.sem
        active = bool(sem.columns or sem.dataset)
        self.sem_lbl.setText(f"Semantics: {Path(sem.source_path).name}" if sem.source_path else ("Semantics: in memory" if active else "Semantics: none (roles inferred from names)"))
        for combo, key in ((self.group_combo, "group_by"), (self.label_combo, "label")):
            v = sem.get(key)
            if v and combo.findData(v) >= 0: combo.setCurrentIndex(combo.findData(v))

    def run_analysis(self):
        if not self.engine.src:
            return self.status("NO DATASET. Use OPEN.")
        tc, mode, model = self.time_combo.currentData(), self.narr_combo.currentData(), self.model_edit.text().strip() or "phi4-mini"
        gb, lb = self.group_combo.currentData(), self.label_combo.currentData()
        self.settings.setValue("model", model); self._busy(True)
        self._run(lambda p: self.engine.analyze(time_col=tc, progress=p, group_by=gb, label=lb, command="UIE app"),
                  lambda s: self._after_analysis(s, mode, model))

    def _update_tiles(self):
        sheet, t = self.sheet, self.tiles
        t["Records"].set_value(human_count(sheet.rows)); t["Original size"].set_value(human_bytes(sheet.size_bytes))
        t["Columns"].set_value(str(sheet.cols)); t["Patterns"].set_value(str(sheet.patterns()))
        t["Anomalies"].set_value(str(sheet.anomalies())); t["Analysis time"].set_value(f"{sheet.seconds:.1f}s")

    def _after_analysis(self, sheet, mode, model):
        self._busy(False); self.sheet = sheet
        self._update_tiles(); self._fill_findings(); self._fill_profile(); self._render_report()
        self.tabs.setCurrentWidget(self.tab_find)
        extra = f"  ({len(sheet.triaged())} reviewed findings hidden)" if sheet.triaged() else ""
        self.status(f"ANALYSIS COMPLETE: {len(sheet.facts)} findings in {sheet.seconds:.1f}s{extra}")
        if sheet.warnings: self.status("NOTE: " + sheet.warnings[0][:150])
        if self.find_tbl.rowCount(): self.find_tbl.selectRow(0)
        if mode == "ollama":
            self.status("NARRATING WITH LOCAL LLM (numbers will be verified)...")
            def done(res):
                sheet.narrative, sheet.narrator = res; self._render_report(); self.status("NARRATION: " + res[1])
            self._run(lambda p: narrator.summarize(sheet, model), done)

    def _fill_findings(self, keep_fp=None):
        c = self._c(); sev_col = {"warning": c["warning"], "notable": c["accent"], "info": c["text_dim"]}
        show_all = self.show_reviewed.isChecked()
        self._row_facts = [f for f in self.sheet.facts if show_all or f.active]
        self.find_tbl.blockSignals(True); self.find_tbl.setRowCount(0)
        sel = None
        for f in self._row_facts:
            r = self.find_tbl.rowCount(); self.find_tbl.insertRow(r)
            dim = c["text_dim"] if not f.active else None
            st = {"new": "", "reopened": "REOPENED", "accepted": "ACCEPTED", "dismissed": "DISMISSED"}[f.status]
            self.find_tbl.setItem(r, 0, cell(f.id, color=dim or c["accent"]))
            self.find_tbl.setItem(r, 1, cell(st, color=c["warning"] if f.status == "reopened" else c["text_dim"]))
            self.find_tbl.setItem(r, 2, cell(f.severity.upper(), color=dim or sev_col[f.severity]))
            self.find_tbl.setItem(r, 3, cell(f.type.replace("quality_", "q:"), color=dim))
            self.find_tbl.setItem(r, 4, cell(f.narrate().replace("`", ""), color=dim))
            if keep_fp and f.fingerprint == keep_fp: sel = r
        self.find_tbl.blockSignals(False)
        if sel is not None: self.find_tbl.selectRow(sel)

    def _selected_row(self):
        rows = self.find_tbl.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _selected_fact(self):
        r = self._selected_row()
        return self._row_facts[r] if 0 <= r < len(self._row_facts) else None

    def _selected_fp(self):
        f = self._selected_fact()
        return f.fingerprint if f else None

    def _review(self, status, note=None, fact=None):
        f = fact or self._selected_fact()
        if not self.sheet or not f or not self.engine.triage:
            return self.status("SELECT A FINDING FIRST")
        if status == "restore":
            self.engine.restore(f)
        else:
            if note is None:
                note, ok = QInputDialog.getText(self, status.title(), "Reason (kept in the triage file for audit):")
                if not ok: return
            (self.engine.dismiss if status == "dismissed" else self.engine.accept)(f, note)
        self._update_tiles(); self._fill_findings(keep_fp=f.fingerprint if self.show_reviewed.isChecked() else None); self._render_report()
        self.status(f"{f.id} {status.upper()}  (saved to {self.engine.triage.path.name})")

    def _selected_row(self):
        rows = self.find_tbl.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _on_fact(self):
        f, c = self._selected_fact(), self._c()
        if not f or not self.sheet: return
        mt = "".join(f"<tr><td style='color:{c['text_dim']}'>{html.escape(str(k))}&nbsp;&nbsp;</td><td>{html.escape(str(v))}</td></tr>"
                     for k, v in f.metrics.items())
        st = f" &middot; {f.status.upper()}" + (f" ({html.escape(f.status_note)})" if f.status_note else "") if f.status != "new" else ""
        self.detail.setHtml(
            f"<h3 style='color:{c['accent']}'>{f.id} &middot; {f.type.upper()} &middot; {f.severity.upper()}{st}</h3>"
            f"<p>{html.escape(f.narrate())}</p><p style='color:{c['text_dim']}'>Confidence: {f.confidence} &middot; fingerprint {f.fingerprint}</p>"
            f"<p style='color:{c['accent']}'><b>EVIDENCE SQL</b></p><pre style='color:{c['info']}; white-space: pre-wrap'>{html.escape(f.evidence_sql)}</pre>"
            f"<p style='color:{c['accent']}'><b>METRICS</b></p><table>{mt}</table>")
        if not f.chart:
            return self.chart.show_message("No chart for this finding. Run its evidence SQL in the ASK tab.")
        self._chart_token += 1; tok = self._chart_token; spec = f.chart
        self.chart.show_message("Running evidence query...")
        def draw(res, tok=tok):
            if tok == self._chart_token: self.chart.show_data(spec, res[1])
        self._run(lambda p: self.engine.evidence(spec["sql"]), draw, lambda e: self.chart.show_message("Evidence query failed: " + e))

    def _fill_profile(self):
        t, rows = self.prof_tbl, self.sheet.profile
        t.setRowCount(0)
        for c in rows.columns:
            r = t.rowCount(); t.insertRow(r)
            top = ", ".join(f"{v} ({n:,})" for v, n in c.top[:3])
            vals = [c.name, c.dtype, pct(c.nulls / max(rows.rows, 1)), f"{c.distinct:,}",
                    fmt_num(c.min) if c.kind == "numeric" else (str(c.min) if c.min is not None else ""),
                    fmt_num(c.max) if c.kind == "numeric" else (str(c.max) if c.max is not None else ""),
                    fmt_num(c.mean) if c.mean is not None else "", fmt_num(c.std) if c.std is not None else "",
                    fmt_num(c.median) if c.median is not None else "", top]
            for i, v in enumerate(vals): t.setItem(r, i, cell(v, align_right=1 <= i <= 8 and i != 1))
        t.resizeColumnsToContents()

    def _render_report(self):
        if self.sheet: self.report_view.setMarkdown(render_markdown(self.sheet, self.mode_combo.currentData()))

    # ---------- ask / compare ----------
    def run_ask(self):
        q = self.ask_in.text().strip()
        if not q or not self.engine.src: return self.status("NO DATASET OR EMPTY QUESTION")
        use, model = self.narr_combo.currentData() == "ollama", self.model_edit.text().strip() or "phi4-mini"
        self._run(lambda p: qa.answer(self.engine, q, use, model), self._show_answer,
                  lambda e: (self.ask_sql.setText("ERROR: " + e), self.status("ASK FAILED")))

    def _show_answer(self, a):
        self.ask_sql.setText(f"[{a['via']}]  {a['sql']}" + ("   (first 1000 rows shown)" if a["truncated"] else ""))
        t = self.ask_tbl; t.clear(); t.setColumnCount(len(a["cols"])); t.setHorizontalHeaderLabels(a["cols"]); t.setRowCount(len(a["rows"]))
        for i, row in enumerate(a["rows"]):
            for j, v in enumerate(row):
                t.setItem(i, j, cell("" if v is None else (fmt_num(v) if isinstance(v, float) else v), isinstance(v, (int, float))))
        t.resizeColumnsToContents(); self.status(f"{len(a['rows'])} ROWS")

    def _need_dataset(self):
        if not self.engine.src:
            self.status("OPEN A DATASET FIRST"); return False
        return True

    def _pick(self, title):
        path, _ = QFileDialog.getOpenFileName(self, title, self.settings.value("last_dir", ""), FILTER)
        return path or None

    def run_compare(self):
        if not self._need_dataset(): return
        path = self._pick("Comparison (current) dataset")
        if not path: return
        src, lb = self.engine.src, self.label_combo.currentData()
        self._busy(True)
        self._run(lambda p: run_drift(src.path, path, lb, self.engine.cfg, self.engine.sem, table_left=src.table, progress=p, command="UIE app"),
                  lambda r: self._show_tool(r, self.cmp_view, self.tab_cmp))

    def _show_tool(self, report, view, tab):
        self._busy(False); view.show_report(report); self.tabs.setCurrentWidget(tab)
        self.status(f"{report.title}: {len(report.facts)} findings")

    def _pick_right(self, which):
        path = self._pick("Right dataset" if which == "recon" else "Test dataset")
        if not path: return
        if which == "recon": self.recon_right = path; self.rec_lbl.setText(f"Right dataset: {Path(path).name}")
        else: self.split_test = path; self.spl_lbl.setText(f"Test dataset: {Path(path).name}")

    @staticmethod
    def _csv(text): return [x.strip() for x in text.split(",") if x.strip()]

    def run_reconcile(self):
        if not self._need_dataset(): return
        if not self.recon_right: return self.status("LOAD THE RIGHT DATASET FIRST")
        key = self._csv(self.rec_key.text())
        if not key: return self.status("ENTER KEY COLUMN(S)")
        src, right, cmp_, tol = self.engine.src, self.recon_right, self._csv(self.rec_cmp.text()) or None, self.rec_tol.value()
        self._busy(True)
        self._run(lambda p: run_reconcile(src.path, right, key, cmp_, tol, None, self.engine.cfg, table_left=src.table, progress=p, command="UIE app"),
                  lambda r: self._show_tool(r, self.rec_view, self.tab_rec))

    def run_splits(self):
        if not self._need_dataset(): return
        sc = self._csv(self.spl_col.text()); lb = self.label_combo.currentData()
        if not self.split_test and len(sc) != 3: return self.status("LOAD A TEST DATASET, OR ENTER: split column, train value, test value")
        src, test, ent, ign = self.engine.src, (None if len(sc) == 3 and not self.split_test else self.split_test), self._csv(self.spl_ent.text()), self._csv(self.spl_ign.text())
        self._busy(True)
        self._run(lambda p: run_split_check(src.path, test, *(sc if test is None else (None, None, None)), entity=ent or None, ignore=ign, label=lb,
                                            cfg=self.engine.cfg, sem=self.engine.sem, table_left=src.table, progress=p, command="UIE app"),
                  lambda r: self._show_tool(r, self.spl_view, self.tab_spl))

    def run_model(self):
        if not self._need_dataset(): return
        tgt = self.mod_target.currentData()
        feats = [self.mod_feats.item(i).text() for i in range(self.mod_feats.count()) if self.mod_feats.item(i).checkState() == Qt.CheckState.Checked and self.mod_feats.item(i).text() != tgt]
        if not tgt or not feats: return self.status("CHOOSE A TARGET AND TICK AT LEAST ONE PREDICTOR")
        src = self.engine.src; self._busy(True)
        self._run(lambda p: run_regress(src.path, tgt, feats, self.engine.cfg, self.engine.sem, src.table, progress=p, command="UIE app"),
                  lambda r: self._show_tool(r, self.mod_view, self.tab_mod))

    def run_curvefit(self):
        if not self._need_dataset(): return
        x, y = self.mod_x.currentData(), self.mod_y.currentData()
        if not x or not y: return self.status("CHOOSE X AND Y FOR THE CURVE FIT")
        try: at = [float(v) for v in self._csv(self.mod_at.text())] or None
        except ValueError: return self.status("PREDICT-AT VALUES MUST BE NUMBERS")
        src = self.engine.src; self._busy(True)
        self._run(lambda p: fit_curves(src.path, x, y, self.engine.cfg, self.engine.sem, src.table, at, progress=p, command="UIE app"),
                  lambda r: self._show_tool(r, self.mod_view, self.tab_mod))

    def export_expectations(self):
        if not self._need_dataset(): return
        stem = Path(self.engine.src.name).stem
        path, _ = QFileDialog.getSaveFileName(self, "Export expectations", str(Path(self.settings.value("last_dir", "")) / f"{stem}.expectations.json"), "JSON (*.json);;YAML (*.yaml)")
        if not path: return
        self._busy(True)
        def done(exp):
            self._busy(False); expectations.save(exp, path)
            self.exp_lbl.setText(f"Wrote {path}: {len(exp['columns'])} columns, {exp['rows']['observed']:,} rows. Edit the limits if you like.")
            self.status(f"EXPECTATIONS EXPORTED {path}")
        self._run(lambda p: expectations.create(self.engine), done)

    def validate_expectations(self):
        data = self._pick("Dataset to validate")
        if not data: return
        exp_path, _ = QFileDialog.getOpenFileName(self, "Expectations file", self.settings.value("last_dir", ""), "Expectations (*.json *.yaml *.yml)")
        if not exp_path: return
        self._busy(True)
        self._run(lambda p: expectations.validate(data, expectations.load(exp_path), self.engine.sem, self.engine.cfg, progress=p, command="UIE app"),
                  lambda r: self._show_tool(r, self.exp_view, self.tab_exp))



    # ---------- export ----------
    def export(self, fmt):
        if not self.sheet: return self.status("NOTHING TO EXPORT. Run an analysis first.")
        fmt = fmt.lower(); stem = Path(self.sheet.dataset).stem
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}", str(Path(self.settings.value("last_dir", "")) / f"{stem}_report.{fmt}"))
        if not path: return
        try:
            write_export(fmt, path, render_markdown(self.sheet, self.mode_combo.currentData()), sheet_json(self.sheet), facts_csv(self.sheet), self.sheet.provenance)
            self.status(f"EXPORTED {path}" + ("  (+ .manifest.json)" if fmt == "csv" else ""))
        except Exception as e:
            self._error(str(e))

    # ---------- command line ----------
    def _command(self):
        text = self.cmd.text().strip(); self.cmd.clear()
        if not text: return
        head, _, rest = text.partition(" "); h, rest = head.upper(), rest.strip()
        tabs = {"FIND": self.tab_find, "FINDINGS": self.tab_find, "REPORT": self.tab_report, "PROFILE": self.tab_prof, "COMPARE": self.tab_cmp,
                "RECONCILE": self.tab_rec, "SPLITS": self.tab_spl, "EXPECT": self.tab_exp, "MODEL": self.tab_mod, "CHARTS": self.tab_charts, "QUANT": self.tab_quant}
        if h == "OPEN": self.open_dataset(rest.strip('"') or None)
        elif h in ("RUN", "GO", "ANALYZE"): self.run_analysis()
        elif h in tabs: self.tabs.setCurrentWidget(tabs[h])
        elif h == "ASK": self.tabs.setCurrentWidget(self.tab_ask); self.ask_in.setText(rest); self.run_ask()
        elif h == "SEMANTICS":
            if rest.lower() == "none": self.clear_semantics()
            else: self.load_semantics(rest.strip('"') or None)
        elif h in ("GROUP", "LABEL"):
            combo = self.group_combo if h == "GROUP" else self.label_combo
            i = -1 if rest.lower() in ("", "none") else next((k for k in range(combo.count()) if (combo.itemData(k) or "").lower() == rest.lower()), -2)
            if i == -2: self.status(f"NO COLUMN '{rest}'")
            else: combo.setCurrentIndex(max(i, 0)); self.status(f"{h} = {combo.currentData() or 'none'}. RUN to apply.")
        elif h in ("DISMISS", "ACCEPT", "RESTORE"):
            fid, _, note = rest.partition(" ")
            f = next((x for x in (self.sheet.facts if self.sheet else []) if x.id.upper() == fid.upper()), None)
            if f is None: self.status(f"NO FINDING '{fid}'")
            else: self._review({"DISMISS": "dismissed", "ACCEPT": "accepted", "RESTORE": "restore"}[h], note=note.strip(), fact=f)
        elif h == "THEME":
            m = [t for t in self.themes.themes.values() if rest.lower() in t["name"].lower() or rest == t["id"]]
            if m: self._set_theme(m[0]["id"])
            else: self.status("THEMES: " + ", ".join(t["name"] for t in self.themes.themes.values()))
        elif h == "EXPORT": self.export(rest or "md")
        elif h == "HELP": QMessageBox.information(self, "Commands", HELP)
        else: self.status(f"UNKNOWN COMMAND: {head}. Type HELP.")

