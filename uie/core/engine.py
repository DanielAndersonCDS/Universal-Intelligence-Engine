from __future__ import annotations
import time
from pathlib import Path
from . import provenance, triage as triage_mod
from .config import Config
from .facts import Fact, FactSheet, SEV_ORDER
from .detectors import REGISTRY, GROUPABLE, Context
from .ingest import DataSource
from .profile import compute_profile
from .report import template_summary
from .semantics import Semantics, build_clean_view, infer_roles, _lit
from .util import qident


def auto_time_col(profile):
    t = [c for c in profile.columns if c.kind == "temporal"]
    if not t:
        return None
    for c in t:
        if any(k in c.name.lower() for k in ("date", "time", "ts", "timestamp")):
            return c.name
    return t[0].name


class Engine:
    def __init__(self, cfg: Config | None = None):
        self.src: DataSource | None = None
        self.sheet: FactSheet | None = None
        self.cfg = cfg or Config()
        self.sem: Semantics = Semantics()
        self.sem_explicit = False
        self.sentinel_counts: dict = {}
        self.triage: triage_mod.TriageStore | None = None

    # ---- data ----
    def open(self, path, table=None) -> DataSource:
        src = DataSource().open(path, table)
        self.src, self.sheet = src, None
        self._rebuild_data()
        if not self.sem_explicit:                    # pick up <dataset>.uie.yaml next to the data, if any
            self.sem = Semantics.sidecar_for(path) or Semantics()
            self._rebuild_data()
        self.triage = triage_mod.TriageStore(triage_mod.TriageStore.default_path(path))
        return src

    def select_table(self, label):
        self.src.use_table(label)
        self.sheet = None
        self._rebuild_data()

    def set_semantics(self, sem):
        """Accepts a Semantics, a file path, or None."""
        if sem is None:
            self.sem, self.sem_explicit = Semantics(), False
        else:
            self.sem = sem if isinstance(sem, Semantics) else Semantics.from_file(sem)
            self.sem_explicit = True
        if self.src:
            self._rebuild_data()
        self.sheet = None

    def _rebuild_data(self):
        self.sentinel_counts = build_clean_view(self.src.con, "raw", "data", self.sem, qident)

    def schema(self):
        return [(r[0], str(r[1])) for r in self.src.con.execute("DESCRIBE data").fetchall()]

    def temporal_columns(self):
        return [n for n, t in self.schema() if t.upper() == "DATE" or t.upper().startswith("TIMESTAMP")]

    def evidence(self, sql: str):
        cur = self.src.con.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchall()

    # ---- triage ----
    def dismiss(self, fact, note=""):
        self.triage.set(fact, "dismissed", note)

    def accept(self, fact, note=""):
        self.triage.set(fact, "accepted", note)

    def restore(self, fact):
        self.triage.restore(fact)

    # ---- analysis ----
    def analyze(self, time_col=None, progress=lambda s, p: None, group_by=None, label=None, command=None) -> FactSheet:
        t0 = time.time()
        warnings = []
        if self.sem.clean_rules() or self.sentinel_counts:
            self._rebuild_data()
        con = self.src.con.cursor()
        progress("Profiling columns", 5)
        try:
            profile = compute_profile(con, "data", progress)
        except Exception as e:
            if not self.src.fallback_text():
                raise
            warnings.append(f"Type detection failed ({str(e)[:120]}); all columns loaded as text.")
            self._rebuild_data()
            con = self.src.con.cursor()
            profile = compute_profile(con, "data", progress)
        sem, cfg = self.sem, self.cfg
        warnings += sem.validate([c.name for c in profile.columns])
        if cfg.infer_roles:
            inferred = infer_roles(profile, sem, profile.rows)
            if inferred:
                warnings.append("Roles inferred from column names (declare them in a semantics file to confirm): "
                                + "; ".join(f"{r} = {', '.join(v)}" for r, v in inferred.items()))
        names = {c.name for c in profile.columns}
        tcol = time_col or sem.get("time") or auto_time_col(profile)
        group_by = group_by or sem.get("group_by")
        label = label or sem.get("label") or (sem.inferred.get("label") or [None])[0]
        if group_by and group_by not in names:
            warnings.append(f"Group column '{group_by}' not found; running without groups.")
            group_by = None
        if label and label not in names:
            warnings.append(f"Label column '{label}' not found; ML checks skipped.")
            label = None
        ctx = Context(con=con, rel="data", profile=profile, time_col=tcol, all_text=self.src.fallback_used, cfg=cfg, sem=sem,
                      label=label, sentinel_counts=self.sentinel_counts,
                      entity=sem.get("entity") or (sem.inferred.get("entity") or [None])[0])
        facts: list[Fact] = []
        n = len(REGISTRY)
        for i, (lab, fn) in enumerate(REGISTRY):
            progress(f"Detecting: {lab}", 30 + int(50 * i / n))
            try:
                facts += fn(ctx)
            except Exception as e:
                warnings.append(f"{lab} detector failed: {type(e).__name__}: {e}")
        if group_by:
            facts += self._run_groups(ctx, group_by, progress, warnings)
        facts.sort(key=lambda f: (SEV_ORDER[f.severity], -f.score))
        for i, f in enumerate(facts, 1):
            f.id = f"F{i:03d}"
        triage_mod.assign(facts)
        if self.triage:
            self.triage.apply(facts, cfg.reraise_factor)
        sheet = FactSheet(dataset=self.src.name, path=self.src.path, rows=profile.rows, cols=len(profile.columns),
                          size_bytes=self.src.size_bytes, seconds=time.time() - t0, time_col=ctx.time_col,
                          profile=profile, facts=facts, warnings=warnings, group_by=group_by, label=label,
                          semantics=sem.to_dict() if (sem.columns or sem.dataset or sem.inferred) else None)
        sheet.narrative = template_summary(sheet)
        params = {"time_col": ctx.time_col, "group_by": group_by, "label": label, "table": self.src.table}
        sheet.provenance = provenance.build(
            [provenance.input_record(self.src, "data", cfg.hash_data, progress)], params, cfg, sem if (sem.columns or sem.dataset or sem.inferred) else None,
            self.triage, facts, command)
        progress("Done", 100)
        self.sheet = sheet
        return sheet

    # ---- per-group analysis ----
    def _run_groups(self, ctx: Context, gcol: str, progress, warnings) -> list:
        cfg, con = self.cfg, ctx.con
        qg = qident(gcol)
        rows = con.execute(f"SELECT {qg}, count(*) n FROM data WHERE {qg} IS NOT NULL GROUP BY 1 HAVING count(*) >= {cfg.group_min_rows} "
                           f"ORDER BY 2 DESC, 1 LIMIT {cfg.group_max + 1}").fetchall()
        total_groups = con.execute(f"SELECT count(DISTINCT {qg}) FROM data").fetchone()[0]
        if not rows:
            warnings.append(f"No group in '{gcol}' has at least {cfg.group_min_rows} rows.")
            return []
        if total_groups > len(rows[:cfg.group_max]):
            warnings.append(f"'{gcol}' has {total_groups:,} values; analysed the largest {min(len(rows), cfg.group_max)} "
                            f"with at least {cfg.group_min_rows} rows.")
        out = []
        from .detectors import groups as groups_mod
        out += groups_mod.contrast(ctx, gcol, [r[0] for r in rows[:cfg.group_max]])
        for gi, (val, _n) in enumerate(rows[:cfg.group_max]):
            progress(f"Group {gcol} = {val}", 80 + int(18 * gi / max(1, min(len(rows), cfg.group_max))))
            subq = f"(SELECT * FROM data WHERE {qg} = {_lit(val)})"
            gcon = self.src.con.cursor()
            gcon.execute(f"CREATE OR REPLACE TEMP TABLE uie_grp AS SELECT * FROM data WHERE {qg} = {_lit(val)}")
            try:
                gprof = compute_profile(gcon, "uie_grp")
                gctx = Context(con=gcon, rel="uie_grp", profile=gprof, time_col=ctx.time_col, all_text=ctx.all_text, cfg=cfg,
                               sem=ctx.sem, label=ctx.label, sentinel_counts={}, derived=ctx.derived, entity=ctx.entity,
                               group={"column": gcol, "value": str(val)})
                gfacts = []
                for lab, fn in REGISTRY:
                    if lab not in GROUPABLE:
                        continue
                    try:
                        gfacts += fn(gctx)
                    except Exception as e:
                        warnings.append(f"{lab} detector failed in {gcol}={val}: {type(e).__name__}: {e}")
                gfacts = [f for f in gfacts if not f.is_quality]
                gfacts.sort(key=lambda f: (SEV_ORDER[f.severity], -f.score))
                for f in gfacts[:cfg.group_top_facts]:
                    f.group = gctx.group
                    f.evidence_sql = f.evidence_sql.replace("uie_grp", subq)
                    if f.chart and f.chart.get("sql"):
                        f.chart = {**f.chart, "sql": f.chart["sql"].replace("uie_grp", subq)}
                    f.title = f"[{gcol}={val}] {f.title}"
                    out.append(f)
            finally:
                gcon.execute("DROP TABLE IF EXISTS uie_grp")
        return out
