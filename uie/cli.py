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


def cmd_regress(a):
    from .core.regression import regress
    r = regress(a.data, a.target, _csv(a.features), _cfg(a), _sem(a.semantics), a.table, a.max_levels, progress=_progress(a.quiet),
                command="uie " + " ".join(sys.argv[1:]))
    return _sub(r, a, csv_table="Coefficients")


def cmd_fit(a):
    from .core.curvefit import fit_curves
    at = [float(v) for v in _csv(a.at)] if a.at else None
    r = fit_curves(a.data, a.x, a.y, _cfg(a), _sem(a.semantics), a.table, at, progress=_progress(a.quiet), command="uie " + " ".join(sys.argv[1:]))
    return _sub(r, a, csv_table="Models")


def _kv(text):
    out = {}
    for part in _csv(text) or []:
        k, _, v = part.partition("=")
        if not _:
            raise ValueError(f"expected key=value, got '{part}'")
        try:
            f = float(v)
            out[k] = int(f) if f == int(f) else f
        except ValueError:
            out[k] = v
    return out


def _series(a, value_is=None):
    from .core.quant.series import load_series
    return load_series(a.data, a.time, a.value, a.entity, a.entity_value, a.table, _sem(a.semantics), a.unit, value_is or ("returns" if a.returns else "price"), progress=_progress(a.quiet))


def _panel(a):
    from .core.quant.series import load_panel
    cols = _csv(a.columns)
    return load_panel(a.data, a.time, a.value, a.entity, cols, a.table, _sem(a.semantics), a.unit, "returns" if a.returns else "price", progress=_progress(a.quiet))


def _cmd(a):
    return "uie " + " ".join(sys.argv[1:])


def cmd_arima(a):
    from .core.quant.arima import arima
    order = tuple(int(x) for x in a.order.split(",")) if a.order else None
    if order and len(order) != 3:
        raise ValueError("--order must be p,d,q")
    return _sub(arima(_series(a), order, a.max_p, a.max_q, a.horizon, a.log, _cfg(a), _progress(a.quiet), _cmd(a)), a, csv_table="Forecast")


def cmd_garch(a):
    from .core.quant.garch import garch
    return _sub(garch(_series(a), horizon=a.horizon, cfg=_cfg(a), progress=_progress(a.quiet), command=_cmd(a)), a, csv_table="Parameters")


def cmd_coint(a):
    from .core.quant.cointegration import cointegration
    return _sub(cointegration(_panel(a), not a.no_log, cfg=_cfg(a), progress=_progress(a.quiet), command=_cmd(a)), a, csv_table="Pairs")


def cmd_var(a):
    from .core.quant.risk import risk
    conf = tuple(float(x) for x in _csv(a.confidence))
    methods = tuple(_csv(a.methods)) + (("garch",) if a.garch and "garch" not in (_csv(a.methods) or []) else ())
    return _sub(risk(_series(a), conf, methods, a.window, a.horizon, a.position, cfg=_cfg(a), progress=_progress(a.quiet), command=_cmd(a)), a, csv_table="VaR and ES")


def cmd_backtest(a):
    from .core.quant.backtest import backtest
    s = _series(a, "price")
    signal = None
    if a.signal_col:
        from .core.quant.series import load_series
        sg = load_series(a.data, a.time, a.signal_col, a.entity, a.entity_value, a.table, _sem(a.semantics), a.unit, "returns")
        m = dict(zip(sg.dates, sg.values))
        signal = [m.get(d, 0.0) for d in s.dates]
    strat = "signal" if a.signal_col else a.strategy
    r = backtest(s, strat, _kv(a.params) or None, None, a.long_short, a.cost_bps, a.slippage_bps, a.vol_target, a.rf, a.train, a.test, a.anchored, not a.no_walk, signal, a.objective, a.boot,
                 _cfg(a), _progress(a.quiet), _cmd(a))
    return _sub(r, a, csv_table="Performance")


def cmd_portfolio(a):
    from .core.quant.portfolio import portfolio
    return _sub(portfolio(_panel(a), tuple(_csv(a.methods)), a.rf, a.min_weight, a.max_weight, a.lookback, a.rebalance, a.cost_bps, a.stability_reps, _cfg(a), _progress(a.quiet), _cmd(a)), a, csv_table="Weights")


def cmd_price(a):
    from .core.quant import derivatives as D
    if a.model == "bond":
        r = D.price_bond(a.face, a.coupon, a.years, a.freq, a.ytm, a.bond_price)
    else:
        model = {"bs": "black-scholes", "binomial": "binomial", "mc": "monte-carlo"}[a.model]
        r = D.price_option(model, a.spot, a.strike, a.years, a.rate, a.div, a.vol, a.kind, a.american, a.style, a.barrier, a.steps, a.paths)
    return _sub(r, a, csv_table="Result")


def cmd_options(a):
    from .core.quant.derivatives import option_chain
    spot = a.spot
    try:
        spot = float(spot)
    except (TypeError, ValueError):
        pass
    r = option_chain(a.data, a.strike, a.type, a.price, a.bid, a.ask, a.expiry, a.years, spot, a.rate, a.div, a.valuation, a.table, _sem(a.semantics), _cfg(a), _progress(a.quiet), _cmd(a))
    return _sub(r, a, csv_table="Quotes")


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

    s = sub.add_parser("regress", help="OLS regression from exact SQL sufficient statistics (coefficients, CIs, VIF, residual checks)"); s.add_argument("data")
    common(s)
    s.add_argument("--target", required=True); s.add_argument("--features", required=True, help="comma-separated predictors (text columns become dummies)")
    s.add_argument("--max-levels", type=int, default=10)
    s.set_defaults(fn=cmd_regress, fail_on="never")

    s = sub.add_parser("fit", help="curve fitting: linear, quadratic, cubic, exponential, power, logarithmic"); s.add_argument("data")
    common(s)
    s.add_argument("--x", required=True); s.add_argument("--y", required=True); s.add_argument("--at", help="comma-separated x values to predict y at")
    s.set_defaults(fn=cmd_fit, fail_on="never")

    def seriesargs(sp):
        sp.add_argument("--time", help="time column (default: semantics or first date column)")
        sp.add_argument("--value", help="price (or return) column (default: semantics price role, or close/price)")
        sp.add_argument("--entity", help="entity column of a long table (ticker, symbol)"); sp.add_argument("--entity-value", help="which entity to analyse")
        sp.add_argument("--returns", action="store_true", help="the value column already holds simple returns (decimals)")
        sp.add_argument("--unit", choices=["day", "week", "month", "none"], default="day", help="bucket the series (last value per bucket)")

    s = sub.add_parser("arima", help="ARIMA(p,d,q) with AIC order selection, forecast and out-of-sample skill vs random walk"); s.add_argument("data")
    common(s); seriesargs(s)
    s.add_argument("--order", help="fix p,d,q instead of selecting"); s.add_argument("--max-p", type=int, default=2); s.add_argument("--max-q", type=int, default=2)
    s.add_argument("--horizon", type=int, default=20); s.add_argument("--log", action="store_true", help="model log prices")
    s.set_defaults(fn=cmd_arima, fail_on="never")

    s = sub.add_parser("garch", help="GARCH / GJR-GARCH (normal or t) volatility model, forecast and out-of-sample test"); s.add_argument("data")
    common(s); seriesargs(s); s.add_argument("--horizon", type=int, default=20)
    s.set_defaults(fn=cmd_garch, fail_on="never")

    s = sub.add_parser("coint", help="Engle-Granger cointegration scan of all pairs (long: --entity/--value, wide: --columns)"); s.add_argument("data")
    common(s); seriesargs(s); s.add_argument("--columns", help="comma-separated asset columns (wide table)"); s.add_argument("--no-log", action="store_true")
    s.set_defaults(fn=cmd_coint, fail_on="never")

    s = sub.add_parser("var", help="Value at Risk and Expected Shortfall with rolling backtests"); s.add_argument("data")
    common(s); seriesargs(s)
    s.add_argument("--confidence", default="0.95,0.99"); s.add_argument("--methods", default="historical,normal,t,cornish,ewma"); s.add_argument("--garch", action="store_true", help="add GARCH-filtered historical simulation (slower)")
    s.add_argument("--window", type=int, default=500); s.add_argument("--horizon", type=int, default=1); s.add_argument("--position", type=float, help="position value, to express VaR in currency")
    s.set_defaults(fn=cmd_var, fail_on="never")

    s = sub.add_parser("backtest", help="strategy backtest with drawdown analysis, walk-forward optimisation, deflated Sharpe and bootstrap"); s.add_argument("data")
    common(s); seriesargs(s)
    s.add_argument("--strategy", default="sma", choices=["buyhold", "sma", "momentum", "meanrev", "rsi", "breakout"]); s.add_argument("--params", help="e.g. fast=20,slow=100")
    s.add_argument("--signal-col", help="backtest the positions (-1..1) held in this column instead of a built-in rule")
    s.add_argument("--long-short", action="store_true"); s.add_argument("--cost-bps", type=float, default=5.0); s.add_argument("--slippage-bps", type=float, default=0.0)
    s.add_argument("--vol-target", type=float, help="annual volatility target, e.g. 0.10"); s.add_argument("--rf", type=float, default=0.0)
    s.add_argument("--train", type=int, default=750); s.add_argument("--test", type=int, default=125); s.add_argument("--anchored", action="store_true")
    s.add_argument("--no-walk", action="store_true", help="skip walk-forward"); s.add_argument("--objective", choices=["sharpe", "calmar"], default="sharpe"); s.add_argument("--boot", type=int, default=2000)
    s.set_defaults(fn=cmd_backtest, fail_on="never")

    s = sub.add_parser("portfolio", help="portfolio optimisation (min variance, max Sharpe, risk parity) with out-of-sample test"); s.add_argument("data")
    common(s); seriesargs(s); s.add_argument("--columns", help="comma-separated asset columns (wide table)")
    s.add_argument("--methods", default="equal,inv_vol,min_var,max_sharpe,risk_parity"); s.add_argument("--rf", type=float, default=0.0)
    s.add_argument("--min-weight", type=float, default=0.0); s.add_argument("--max-weight", type=float, default=1.0)
    s.add_argument("--lookback", type=int, default=252); s.add_argument("--rebalance", type=int, default=21); s.add_argument("--cost-bps", type=float, default=5.0); s.add_argument("--stability-reps", type=int, default=40)
    s.set_defaults(fn=cmd_portfolio, fail_on="never")

    s = sub.add_parser("price", help="price an option (Black-Scholes, binomial, Monte Carlo) or a bond"); s.add_argument("model", choices=["bs", "binomial", "mc", "bond"])
    s.add_argument("--spot", type=float); s.add_argument("--strike", type=float); s.add_argument("--years", type=float, default=1.0); s.add_argument("--rate", type=float, default=0.0)
    s.add_argument("--div", type=float, default=0.0); s.add_argument("--vol", type=float, default=0.2); s.add_argument("--kind", choices=["call", "put"], default="call")
    s.add_argument("--american", action="store_true"); s.add_argument("--style", choices=["european", "asian", "down_out", "up_out"], default="european"); s.add_argument("--barrier", type=float)
    s.add_argument("--steps", type=int, default=500); s.add_argument("--paths", type=int, default=200000)
    s.add_argument("--face", type=float, default=100.0); s.add_argument("--coupon", type=float, default=0.05); s.add_argument("--freq", type=int, default=2)
    s.add_argument("--ytm", type=float); s.add_argument("--bond-price", type=float)
    s.add_argument("--format", choices=["text", "json", "md", "html", "csv"], default="text"); s.add_argument("-o", "--output"); s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(fn=cmd_price, fail_on="never")

    s = sub.add_parser("options", help="option-chain diagnostics: implied vol, no-arbitrage checks, smile and term structure"); s.add_argument("data")
    common(s)
    s.add_argument("--strike", required=True); s.add_argument("--type", required=True, help="column holding call/put (or C/P)"); s.add_argument("--price", help="mid/last price column")
    s.add_argument("--bid"); s.add_argument("--ask"); s.add_argument("--expiry", help="expiry date column"); s.add_argument("--years", help="column with time to expiry in years")
    s.add_argument("--spot", required=True, help="underlying price (number) or column"); s.add_argument("--rate", type=float, default=0.0); s.add_argument("--div", type=float, default=0.0)
    s.add_argument("--valuation", help="valuation date YYYY-MM-DD (needed with --expiry)")
    s.set_defaults(fn=cmd_options)

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
    if getattr(a, "fn", None) is cmd_price and a.model != "bond" and (a.spot is None or a.strike is None):
        parser.error("price bs|binomial|mc needs --spot and --strike")
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
