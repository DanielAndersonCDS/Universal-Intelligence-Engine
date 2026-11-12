"""Run:  python -m unittest discover -s tests -v      (needs duckdb, numpy, pyyaml; no GUI libraries)"""
import csv, hashlib, io, json, os, sys, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datagen
from uie import cli
from uie.core import expectations, stats
from uie.core.drift import drift
from uie.core.engine import Engine
from uie.core.reconcile import reconcile
from uie.core.semantics import Semantics
from uie.core.splitcheck import split_check

TMP = Path(tempfile.mkdtemp(prefix="uie_test_"))


def write_csv(name, header, rows):
    p = TMP / name
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return str(p)


def analyze(path, sem=None, **kw):
    e = Engine()
    e.open(path)
    if sem:
        e.set_semantics(Semantics.from_dict(sem))
    return e, e.analyze(**kw)


def types(sheet):
    return [f.type for f in sheet.facts]


class Stats(unittest.TestCase):
    def test_chi2_sf(self):
        self.assertAlmostEqual(stats.chi2_sf(3.841, 1), 0.05, 3)
        self.assertAlmostEqual(stats.chi2_sf(15.507, 8), 0.05, 3)

    def test_adf_separates_random_walk_from_trend(self):
        rng = np.random.default_rng(1)
        rw = sum(stats.adf(np.cumsum(rng.normal(0, 1, 120))).reject5 for _ in range(100))
        ts = sum(stats.adf(0.1 * np.arange(120) + rng.normal(0, 1, 120)).reject5 for _ in range(100))
        self.assertLess(rw, 20)
        self.assertGreater(ts, 90)

    def test_psi_zero_for_identical(self):
        self.assertAlmostEqual(stats.psi([10, 20, 30], [10, 20, 30]), 0.0, 9)


class Foundation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        n = 6000
        a, b, c = rng.normal(100, 20, n).round(1), rng.normal(50, 10, n).round(1), rng.normal(70, 15, n).round(1)
        sq4 = np.where(rng.random(n) < 0.9, np.nan, -np.round(a * 1.1, 1))
        grp = rng.choice(["M", "F"], n)
        rows = [[grp[i], a[i], b[i], c[i], round(a[i] + b[i] + c[i], 1), "" if np.isnan(sq4[i]) else sq4[i]] for i in range(n)]
        cls.path = write_csv("lift.csv", ["Sex", "A", "B", "C", "Total", "Fail4"], rows)

    def test_sentinels_and_derived_and_suppression(self):
        sem = {"columns": {"Fail4": {"negative_is_missing": True, "note": "negative = failed"}}}
        _, raw = analyze(self.path)
        _, sh = analyze(self.path, sem)
        # without semantics the negative sentinel pollutes correlations; with it, the artefact disappears
        self.assertTrue(any(f.type == "correlation" and "Fail4" in f.columns for f in raw.facts))
        self.assertFalse(any(f.type in ("correlation", "near_duplicate") and "Fail4" in f.columns for f in sh.facts))
        self.assertIn("quality_sentinel", types(sh))
        d = [f for f in sh.facts if f.type == "quality_derived"]
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0].metrics["column"], "Total")
        self.assertEqual(sorted(d[0].metrics["sources"]), ["A", "B", "C"])
        for f in sh.facts:   # no tautological pairs involving Total survive
            if f.type == "correlation":
                self.assertNotIn("Total", f.columns)

    def test_declared_derived_from(self):
        sem = {"columns": {"Total": {"derived_from": ["A", "B", "C"]}}}
        e, sh = analyze(self.path, sem)
        self.assertEqual(e.sem.role("Total"), "derived")

    def test_group_analysis(self):
        _, sh = analyze(self.path, {"dataset": {"group_by": "Sex"}})
        self.assertEqual(sh.group_by, "Sex")
        grouped = [f for f in sh.facts if f.group]
        self.assertTrue(grouped)
        for f in grouped:
            self.assertIn(f.group["value"], ("M", "F"))
            self.assertIn("(SELECT * FROM data WHERE", f.evidence_sql)   # evidence SQL is reproducible against `data`
        e, _ = analyze(self.path, {"dataset": {"group_by": "Sex"}})
        for f in grouped[:3]:
            e.evidence(f.evidence_sql)                                   # and actually runs

    def test_unknown_semantics_column_warns(self):
        _, sh = analyze(self.path, {"columns": {"Nope": {"role": "id"}}})
        self.assertTrue(any("Nope" in w for w in sh.warnings))

    def test_bad_semantics_rejected(self):
        with self.assertRaises(ValueError):
            Semantics.from_dict({"columns": {"A": {"role": "wizard"}}})
        with self.assertRaises(ValueError):
            Semantics.from_dict({"columns": {"A": {"exclude": ["magic"]}}})

    def test_provenance_hash_and_determinism(self):
        _, s1 = analyze(self.path)
        _, s2 = analyze(self.path)
        want = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        self.assertEqual(s1.provenance["inputs"][0]["sha256"], want)
        self.assertEqual(s1.provenance["result_digest"], s2.provenance["result_digest"])
        self.assertIn("version", s1.provenance["tool"])
        self.assertIn("outlier_t", s1.provenance["config"])


class Triage(unittest.TestCase):
    def test_dismiss_persist_and_reopen(self):
        path = datagen.ml(str(TMP / "ml_triage.csv"))
        sem = {"dataset": {"label": "churn"}, "columns": {"user_id": {"role": "id"}}}
        e, sh = analyze(path, sem)
        leak = next(f for f in sh.facts if f.type == "target_leakage")
        before = sh.anomalies()
        e.dismiss(leak, "synthetic leak")
        e2, sh2 = analyze(path, sem)                                   # new engine, same store on disk
        again = next(f for f in sh2.facts if f.fingerprint == leak.fingerprint)
        self.assertEqual(again.status, "dismissed")
        self.assertEqual(sh2.anomalies(), before - 1)
        self.assertNotIn(again, sh2.active())
        # worsening re-raises: pretend it was dismissed when much smaller
        e2.triage.entries[leak.fingerprint]["magnitude"] = again.magnitude / 10
        e2.triage.save()
        _, sh3 = analyze(path, sem)
        self.assertEqual(next(f for f in sh3.facts if f.fingerprint == leak.fingerprint).status, "reopened")
        e2.restore(again)
        _, sh4 = analyze(path, sem)
        self.assertEqual(next(f for f in sh4.facts if f.fingerprint == leak.fingerprint).status, "new")


class Stationarity(unittest.TestCase):
    def test_random_walk_not_called_a_trend_but_real_trend_is(self):
        rng = np.random.default_rng(11)
        days = np.arange(400)
        n_rw_trends = 0
        # a daily series, one row per day, aggregated to weekly/monthly buckets by the engine
        def series(values, name):
            import datetime as dt
            rows = [[(dt.date(2023, 1, 1) + dt.timedelta(int(i))).isoformat(), round(float(v), 4)] for i, v in enumerate(values)]
            return write_csv(name, ["day", "v"], rows)
        for seed in range(10):
            r = np.random.default_rng(seed)
            _, sh = analyze(series(np.cumsum(r.normal(0, 1, 700)) + 200, f"rw{seed}.csv"))
            n_rw_trends += sum(f.type == "trend" for f in sh.facts)
        _, real = analyze(series(0.05 * np.arange(700) + rng.normal(0, 1, 700) + 10, "real.csv"))
        self.assertTrue(any(f.type == "trend" for f in real.facts))
        self.assertLessEqual(n_rw_trends, 3, "pure random walks should rarely be reported as trends")


class Finance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tx = datagen.transactions(str(TMP / "tx.csv"))
        cls.px = datagen.prices(str(TMP / "px.csv"))

    def test_transaction_patterns(self):
        sem = {"dataset": {"amount": "amount", "counterparty": "vendor", "time": "txn_date", "thresholds": [10000]}}
        _, sh = analyze(self.tx, sem)
        t = types(sh)
        for want in ("threshold_bunching", "near_duplicate_tx", "round_numbers", "time_gap", "variance_driver"):
            self.assertIn(want, t)
        b = next(f for f in sh.facts if f.type == "threshold_bunching")
        self.assertGreater(b.metrics["below"], 10 * max(b.metrics["above"], 1) / 2)
        self.assertEqual(b.severity, "warning")                       # declared threshold, strong ratio
        v = next(f for f in sh.facts if f.type == "variance_driver" and f.metrics["basis"] == "month over month")
        self.assertEqual(v.metrics["top"][0]["group"], "East")        # planted driver
        gap = next(f for f in sh.facts if f.type == "time_gap")
        self.assertGreaterEqual(gap.metrics["longest_days"], 35)

    def test_prices_returns_not_levels(self):
        sem = {"dataset": {"entity": "ticker", "price": "close", "time": "date"}}
        _, sh = analyze(self.px, sem)
        spur = [f for f in sh.facts if f.type == "spurious_correlation"]
        self.assertTrue(spur)
        for f in spur:
            self.assertGreaterEqual(abs(f.metrics["level_r"]), 0.8)
            self.assertLess(abs(f.metrics["return_r"]), 0.3)
        real = [f for f in sh.facts if f.type == "return_correlation"]
        self.assertTrue(any({"ticker=AAA", "ticker=BBB"} == {f.metrics["a"], f.metrics["b"]} for f in real))   # planted common factor
        shift = [f for f in sh.facts if f.type == "vol_shift"]
        self.assertTrue(shift and "CCC" in shift[0].metrics["label"])
        self.assertFalse(any(f.type in ("outliers", "trend") and not f.group for f in sh.facts if "close" in f.columns),
                         "prices must not be pooled across tickers")

    def test_reconcile(self):
        rng = np.random.default_rng(9)
        n = 1500
        ids, amt = np.arange(1, n + 1), np.round(rng.uniform(10, 5000, n), 2)
        left = write_csv("gl.csv", ["id", "amount"], list(zip(ids, amt)))
        miss = set(rng.choice(ids, 10, replace=False).tolist())
        mm = set(rng.choice(ids, 20, replace=False).tolist()) - miss
        rows = []
        for i, a in zip(ids, amt):
            if i in miss:
                continue
            a2 = a + (13.0 if i in mm else (0.004 if i % 7 == 0 else 0))     # tiny diffs stay within tolerance
            rows.append([f"{i}.0" if i % 2 else f" {i}", round(a2, 3)])        # key formats differ: 5 vs '5.0' vs ' 6'
        for k in range(5):
            rows.append([9000 + k, 100.0])
        right = write_csv("bank.csv", ["id", "amount"], rows)
        r = reconcile(left, right, ["id"], ["amount"])
        self.assertEqual((r.summary["missing_in_right"], r.summary["extra_in_right"], r.summary["mismatched"]), (10, 5, len(mm)))
        bridge = r.tables["Bridge"][1][0]
        self.assertAlmostEqual(bridge[-1], 0.0, 4)                             # the bridge explains the whole difference
        self.assertAlmostEqual(bridge[3], bridge[4] - bridge[5] + bridge[6] + bridge[7], 3)


class ML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ml = datagen.ml(str(TMP / "ml.csv"))

    def test_leakage_imbalance_features(self):
        _, sh = analyze(self.ml, {"dataset": {"label": "churn"}, "columns": {"user_id": {"role": "id"}}})
        t = types(sh)
        self.assertIn("target_leakage", t)
        self.assertIn("leakage_missingness", t)
        self.assertIn("class_imbalance", t)
        leak = next(f for f in sh.facts if f.type == "target_leakage")
        self.assertEqual(leak.metrics["feature"], "leak_score")
        self.assertFalse(any(f.type == "target_leakage" and f.metrics["feature"] in ("age", "spend") for f in sh.facts))
        self.assertTrue(any(f.type == "quality_constant" for f in sh.facts))

    def test_splits_and_drift_and_expectations(self):
        rng = np.random.default_rng(2)

        def mk(n, users, dup=None):
            rows = [[int(rng.integers(0, users)), round(float(rng.normal(50, 10)), 1), str(rng.choice(["a", "b", "c"])), int(rng.random() < 0.2)] for _ in range(n)]
            if dup:
                rows[:150] = dup[:150]
                rows[150:200] = [[r[0], r[1] + 0.001, r[2].upper(), r[3]] for r in dup[150:200]]
            return rows
        tr = mk(3000, 400)
        train = write_csv("train.csv", ["uid", "x", "cat", "y"], tr)
        test = write_csv("test.csv", ["uid", "x", "cat", "y"], mk(1000, 500, tr))
        r = split_check(train, test, entity="uid", label="y")
        exact = next(f for f in r.facts if f.type == "split_overlap" and f.metrics["kind"] == "exact")
        near = next(f for f in r.facts if f.type == "split_overlap" and f.metrics["kind"] == "near")
        self.assertGreaterEqual(exact.metrics["rows"], 150)
        self.assertEqual(near.metrics["rows"], 50)
        self.assertTrue(any(f.type == "split_entities" for f in r.facts))
        cur = write_csv("cur.csv", ["uid", "x", "cat", "y", "new"],
                        [[i % 400, round(float(rng.normal(58, 10)), 1), str(rng.choice(["a", "b", "c", "d"], p=[.2, .2, .2, .4])), 0, 1] for i in range(3000)])
        d = drift(train, cur)
        by = {f.metrics["column"]: f for f in d.facts if f.type == "drift"}
        self.assertEqual(by["x"].severity, "warning")
        self.assertEqual(by["cat"].metrics["kind"], "categorical")
        self.assertTrue(any(f.type == "drift_schema" for f in d.facts))
        e = Engine()
        e.open(train)
        exp = expectations.create(e)
        self.assertEqual(len(expectations.validate(train, exp).facts), 0)
        v = expectations.validate(cur, exp)
        self.assertTrue({"expect_schema", "expect_category", "expect_drift"} <= {f.type for f in v.facts})
        p = TMP / "exp.yaml"
        expectations.save(exp, p)
        self.assertEqual(expectations.load(p)["rows"]["observed"], 3000)


class CLI(unittest.TestCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main([str(a) for a in args])
        return code, buf.getvalue()

    def test_exit_codes_json_and_triage_gate(self):
        path = datagen.ml(str(TMP / "ml_cli.csv"))
        sem = TMP / "ml_cli.sem.yaml"
        sem.write_text("dataset:\n  label: churn\ncolumns:\n  user_id: {role: id}\n")
        code, out = self.run_cli("analyze", path, "--semantics", sem, "--format", "json", "-q")
        self.assertEqual(code, 1)                                      # warnings present
        d = json.loads(out)
        self.assertTrue(d["gate"]["failed"])
        self.assertIn("result_digest", d["provenance"])
        warn = [f for f in d["facts"] if f["severity"] == "warning"]
        self.assertTrue(warn and all(f["fingerprint"] for f in warn))
        self.assertEqual(self.run_cli("analyze", path, "--semantics", sem, "-q", "--fail-on", "never")[0], 0)
        for f in warn:                                                 # review every warning, then the gate passes
            self.assertEqual(self.run_cli("review", path, "accepted", f["fingerprint"], "--semantics", sem, "--note", "known", "-q")[0], 0)
        self.assertEqual(self.run_cli("analyze", path, "--semantics", sem, "-q")[0], 0)
        self.assertEqual(self.run_cli("analyze", path, "--semantics", sem, "-q", "--triage", "none")[0], 1)
        self.assertIn("accepted", self.run_cli("triage", "list", path)[1])

    def test_errors_and_other_commands(self):
        self.assertEqual(self.run_cli("analyze", TMP / "missing.csv", "-q")[0], 2)
        with self.assertRaises(SystemExit) as cm:
            self.run_cli("analyze", "--bogus")
        self.assertEqual(cm.exception.code, 2)
        left = write_csv("l.csv", ["id", "v"], [[1, 1.0], [2, 2.0]])
        right = write_csv("r.csv", ["id", "v"], [[1, 1.0], [3, 2.0]])
        code, out = self.run_cli("reconcile", left, right, "--key", "id", "--format", "json", "-q")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["summary"]["missing_in_right"], 1)
        self.assertEqual(self.run_cli("reconcile", left, left, "--key", "id", "-q")[0], 0)
        code, out = self.run_cli("expect", "create", left, "-o", TMP / "e.json", "-q")
        self.assertEqual(code, 0)
        self.assertEqual(self.run_cli("expect", "validate", left, TMP / "e.json", "-q")[0], 0)

    def test_html_and_csv_formats(self):
        path = datagen.ml(str(TMP / "ml_fmt.csv"))
        code, html_out = self.run_cli("analyze", path, "--format", "html", "-q", "--fail-on", "never")
        self.assertTrue(html_out.startswith("<!doctype html>") and "Provenance" in html_out)
        code, csv_out = self.run_cli("analyze", path, "--format", "csv", "-q", "--fail-on", "never")
        self.assertIn("fingerprint", csv_out.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
