"""Headless interface:  python -m uie <command> ...   (or  python uie_cli.py ...)

Exit codes:  0 ok / gate passed   1 gate failed (findings at or above --fail-on)   2 bad usage or input   3 runtime error
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from . import __version__
from .core.config import Config
from .core.facts import SEV_ORDER

FAIL_CHOICES = ["warning", "notable", "info", "never"]


def _csv(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _emit(text: str, out):
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


def _progress(quiet):
    return (lambda s, p=0: None) if quiet else (lambda s, p=0: print(f"[{p:3d}%] {s}", file=sys.stderr))


def _cfg(a) -> Config:
    cfg = Config()
    cfg.hash_data = not getattr(a, "no_hash", False)
    cfg.infer_roles = not getattr(a, "no_infer", False)
    if getattr(a, "max_groups", None):
        cfg.group_max = a.max_groups
    for kv in getattr(a, "set", None) or []:
        k, _, v = kv.partition("=")
        if not hasattr(cfg, k):
            raise ValueError(f"unknown config key '{k}'")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true", "yes"))
    return cfg


def _sem(path):
    from .core.semantics import Semantics
    return Semantics.from_file(path) if path else None


def _gate_counts(facts):
    c = {"warning": 0, "notable": 0, "info": 0}
    for f in facts:
        if f.active:
            c[f.severity] += 1
    return c


def _failed(facts, fail_on):
    if fail_on == "never":
        return False
    lim = SEV_ORDER[fail_on]
    return any(f.active and SEV_ORDER[f.severity] <= lim for f in facts)


def _text_summary(title, facts, fail_on, extra=()):
    c = _gate_counts(facts)
    lines = [title] + list(extra)
    lines.append(f"active findings: {c['warning']} warning, {c['notable']} notable, {c['info']} info")
    for f in facts:
        if f.active:
            lines.append(f"  [{f.severity.upper():7}] {f.id} {f.fingerprint or ''} {f.narrate().replace('`', '')[:230]}")
    lines.append(f"gate (--fail-on {fail_on}): {'FAIL' if _failed(facts, fail_on) else 'PASS'}")
    return "\n".join(lines)


def _render(fmt, md_fn, json_obj_fn, csv_fn, text_fn, title):
    from .core.report import markdown_to_html
    if fmt == "json":
        return json_obj_fn()
    if fmt == "md":
        return md_fn()
    if fmt == "html":
        return markdown_to_html(md_fn(), title)
    if fmt == "csv":
        return csv_fn()
    return text_fn()


# ---------------- commands ----------------
def cmd_analyze(a):
    from .core.engine import Engine
    from .core.report import facts_csv, render_markdown
    from .core import triage as tri
    cfg = _cfg(a)
    e = Engine(cfg)
    if a.semantics:
        e.set_semantics(_sem(a.semantics))
    e.open(a.data, a.table)
    if a.triage == "none":
        e.triage = None
    elif a.triage:
        e.triage = tri.TriageStore(a.triage)
    sheet = e.analyze(time_col=a.time_col, group_by=a.group_by, label=a.label, progress=_progress(a.quiet),
                      command="uie " + " ".join(sys.argv[1:]))

    def js():
        d = sheet.to_dict()
        d["gate"] = {"fail_on": a.fail_on, "failed": _failed(sheet.facts, a.fail_on), "counts": _gate_counts(sheet.facts)}
        return json.dumps(d, indent=2, default=str)

    extra = [f"dataset: {sheet.dataset}  ({sheet.rows:,} rows, {sheet.cols} columns)", f"result digest: {sheet.provenance.get('result_digest')}"]
    extra += [f"warning: {w}" for w in sheet.warnings]
    _emit(_render(a.format, lambda: render_markdown(sheet, a.mode), js, lambda: facts_csv(sheet),
                  lambda: _text_summary("UIE analysis", sheet.facts, a.fail_on, extra), f"UIE: {sheet.dataset}"), a.output)
    return 1 if _failed(sheet.facts, a.fail_on) else 0


def _sub(report, a, title_fn=None, csv_table=None):
    from .core import subreport as sr
    from .core.report import markdown_to_html
    text = _render(a.format, lambda: sr.render_markdown(report), lambda: sr.to_json(report), lambda: sr.tables_csv(report, csv_table),
                   lambda: _text_summary(report.title, report.facts, a.fail_on, [f"{k}: {v}" for k, v in list(report.summary.items())[:12]]), report.title)
    _emit(text, a.output)
    return 1 if _failed(report.facts, a.fail_on) else 0


def cmd_reconcile(a):
    from .core.reconcile import reconcile
    r = reconcile(a.left, a.right, _csv(a.key), _csv(a.compare), a.tolerance, a.rel_tolerance, _cfg(a), progress=_progress(a.quiet),
                  command="uie " + " ".join(sys.argv[1:]))
    if a.exceptions_csv:
        from .core.subreport import tables_csv
        Path(a.exceptions_csv).write_text(tables_csv(r, "Exceptions"), encoding="utf-8", newline="")
        print(f"wrote {a.exceptions_csv}", file=sys.stderr)
    return _sub(r, a, csv_table="Exceptions")


def cmd_drift(a):
    from .core.drift import drift
    r = drift(a.reference, a.current, a.label, _cfg(a), _sem(a.semantics), ignore=_csv(a.ignore), progress=_progress(a.quiet),
              command="uie " + " ".join(sys.argv[1:]))
    return _sub(r, a, csv_table="Column drift")


def cmd_splits(a):
    from .core.splitcheck import split_check
    r = split_check(a.train, a.test, a.split_col, a.train_value, a.test_value, _csv(a.entity), _csv(a.ignore), a.label, a.time_col, _cfg(a), _sem(a.semantics),
                    progress=_progress(a.quiet), command="uie " + " ".join(sys.argv[1:]))
    return _sub(r, a, csv_table="Overlap")


def cmd_expect(a):
    from .core import expectations as ex
    from .core.engine import Engine
    cfg = _cfg(a)
    if a.action == "create":
        e = Engine(cfg)
        if a.semantics:
            e.set_semantics(_sem(a.semantics))
        e.open(a.data, a.table)
        exp = ex.create(e, cfg)
        ex.save(exp, a.output or "expectations.json")
        print(f"wrote {a.output or 'expectations.json'} ({len(exp['columns'])} columns, {exp['rows']['observed']:,} rows)", file=sys.stderr)
        return 0
    if not a.expectations:
        raise ValueError("validate needs an expectations file: uie expect validate DATA EXPECTATIONS")
    r = ex.validate(a.data, ex.load(a.expectations), _sem(a.semantics), cfg, a.table, progress=_progress(a.quiet), command="uie " + " ".join(sys.argv[1:]))
    return _sub(r, a)


def cmd_triage(a):
    from .core import triage as tri
    store = tri.TriageStore(a.store or tri.TriageStore.default_path(a.data))
    if a.action == "list":
        if not store.entries:
            print(f"no reviewed findings ({store.path})")
        for fp, e in sorted(store.entries.items(), key=lambda kv: kv[1].get("at", "")):
            print(f"{fp}  {e['status']:9}  {e.get('at', '')[:19]}  {e.get('by', '')}  {e.get('text', e.get('title', ''))[:110]}" + (f"  // {e['note']}" if e.get("note") else ""))
        return 0
    if not a.fingerprints:
        raise ValueError("give at least one fingerprint (from `analyze --format json` or csv, or `triage list`)")
    for fp in a.fingerprints:
        if a.action == "restore":
            n = store.remove(fp)
            print(f"{fp}: restored ({n} entry removed)")
        else:
            if fp not in store.entries and not any(k.startswith(fp) for k in store.entries):
                raise ValueError(f"fingerprint {fp} is not in the store; mark findings with `uie review DATA dismissed ID` or in the app first")
            key = store.set_fingerprint(fp, "dismissed" if a.action == "dismiss" else "accepted", a.note or "")
            print(f"{key}: {a.action}ed")
    return 0


def cmd_review(a):
    """analyze, then mark findings by id/fingerprint in one step (for scripts)."""
    from .core.engine import Engine
    cfg = _cfg(a)
    e = Engine(cfg)
    if a.semantics:
        e.set_semantics(_sem(a.semantics))
    e.open(a.data, a.table)
    sheet = e.analyze(time_col=a.time_col, group_by=a.group_by, label=a.label, progress=_progress(True))
    hit = 0
    for f in sheet.facts:
        if f.id in a.ids or any(f.fingerprint.startswith(x) for x in a.ids):
            (e.dismiss if a.status == "dismissed" else e.accept)(f, a.note or "")
            print(f"{f.id} {f.fingerprint}: {a.status}  {f.narrate()[:100]}")
            hit += 1
    if not hit:
        raise ValueError("no finding matched; run `uie analyze DATA --format csv` to see ids and fingerprints")
    print(f"triage store: {e.triage.path}", file=sys.stderr)
    return 0


# ---------------- parser ----------------
def build_parser():
    p = argparse.ArgumentParser(prog="uie", description="Universal Intelligence Engine, headless. Exit codes: 0 pass, 1 gate failed, 2 usage/input error, 3 runtime error.")
    p.add_argument("--version", action="version", version=f"uie {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, gate=True, fmt=True):
        sp.add_argument("--semantics", help="column semantics file (YAML/JSON)")
        sp.add_argument("-o", "--output", help="write to a file instead of stdout")
        sp.add_argument("--table", help="table to analyse inside a .duckdb/.sqlite file")
        sp.add_argument("--no-hash", action="store_true", help="skip SHA-256 of the input (faster on huge files)")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a Config threshold, e.g. --set outlier_t=5")
        sp.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")
        if fmt:
            sp.add_argument("--format", choices=["text", "json", "md", "html", "csv"], default="text")
        if gate:
            sp.add_argument("--fail-on", choices=FAIL_CHOICES, default="warning", help="exit 1 if any active finding is at or above this severity (default warning)")

    s = sub.add_parser("analyze", help="profile a dataset and report findings"); s.add_argument("data")
    common(s)
    s.add_argument("--time-col"); s.add_argument("--group-by", help="run detectors inside each value of this column"); s.add_argument("--label", help="ML label column")
    s.add_argument("--max-groups", type=int); s.add_argument("--no-infer", action="store_true", help="do not infer roles from column names")
    s.add_argument("--mode", choices=["technical", "executive", "key"], default="technical")
    s.add_argument("--triage", help="triage store path, or 'none' to ignore reviews")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("reconcile", help="row-level reconciliation of two datasets by key"); s.add_argument("left"); s.add_argument("right")
    common(s)
    s.add_argument("--key", required=True, help="comma-separated key columns"); s.add_argument("--compare", help="comma-separated numeric columns (default: all shared numeric)")
    s.add_argument("--tolerance", type=float, help="absolute tolerance (default 0.01)"); s.add_argument("--rel-tolerance", type=float)
    s.add_argument("--exceptions-csv", help="write every exception row to this CSV")
    s.set_defaults(fn=cmd_reconcile)

    s = sub.add_parser("drift", help="PSI / KS / chi-square drift between two datasets"); s.add_argument("reference"); s.add_argument("current")
    common(s)
    s.add_argument("--label"); s.add_argument("--ignore", help="comma-separated columns to skip")
    s.set_defaults(fn=cmd_drift)

    s = sub.add_parser("splits", help="train/test overlap: duplicate rows, shared entities, time overlap"); s.add_argument("train"); s.add_argument("test", nargs="?")
    common(s)
    s.add_argument("--split-col"); s.add_argument("--train-value"); s.add_argument("--test-value")
    s.add_argument("--entity", help="comma-separated entity columns (user_id, patient_id ...)"); s.add_argument("--ignore"); s.add_argument("--label"); s.add_argument("--time-col")
    s.set_defaults(fn=cmd_splits)

    s = sub.add_parser("expect", help="export an expectations file, or validate a dataset against one")
    s.add_argument("action", choices=["create", "validate"]); s.add_argument("data"); s.add_argument("expectations", nargs="?")
    common(s)
    s.set_defaults(fn=cmd_expect)

    s = sub.add_parser("triage", help="list / dismiss / accept / restore reviewed findings"); s.add_argument("action", choices=["list", "dismiss", "accept", "restore"])
    s.add_argument("data"); s.add_argument("fingerprints", nargs="*"); s.add_argument("--note"); s.add_argument("--store", help="triage store path")
    s.set_defaults(fn=cmd_triage)

    s = sub.add_parser("review", help="run an analysis and dismiss/accept findings by id or fingerprint"); s.add_argument("data"); s.add_argument("status", choices=["dismissed", "accepted"])
    s.add_argument("ids", nargs="+"); s.add_argument("--note"); common(s, gate=False, fmt=False)
    s.add_argument("--time-col"); s.add_argument("--group-by"); s.add_argument("--label"); s.add_argument("--no-infer", action="store_true")
    s.set_defaults(fn=cmd_review)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)          # argparse exits with 2 on usage errors
    try:
        return int(a.fn(a))
    except FileNotFoundError as e:
        print(f"uie: error: file not found: {e}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as e:
        print(f"uie: error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"uie: runtime error: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
