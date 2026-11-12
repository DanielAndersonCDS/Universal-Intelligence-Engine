# UIE: Universal Intelligence Engine

Turns large datasets into a small, verifiable fact sheet, then into reports. No LLM ever reads your data.

    raw data -> DuckDB (profile, aggregate) -> detectors -> facts -> template narration -> report
                                                              \-> optional local LLM (sees only finished facts, output number-verified)

## Run
    pip install -r requirements.txt
    python make_sample_data.py          # optional: 400k-row demo file
    python main.py [dataset]

Supports CSV/TSV, Parquet, JSON/NDJSON, XLSX, DuckDB and SQLite files, and folders of Parquet/CSV. Drag a file onto the window.

## Layout
    main.py                 desktop app
    uie_cli.py / python -m uie   headless CLI
    themes/                 *.json themes + base.qss (see themes/README.md to add more)
    examples/               semantics files for finance, ML and the OpenPowerlifting data
    tests/                  python -m unittest discover -s tests
    uie/core/               ingest, semantics, profile, detectors/, narration, report, engine, provenance, triage,
                            reconcile, drift, splitcheck, expectations, stats, config
    uie/ui/                 theme_manager, main_window, tools, charts, widgets, worker, exporter

## App tabs
FINDINGS (with DISMISS / ACCEPT / RESTORE), REPORT (technical / executive / key), ASK (read-only SQL, or plain English via local Ollama; `data` is the cleaned view, `raw` the original),
PROFILE, COMPARE (drift), RECONCILE, SPLITS, EXPECT. Command bar: OPEN, RUN, FINDINGS, REPORT, ASK, PROFILE, COMPARE, RECONCILE, SPLITS, EXPECT, SEMANTICS, GROUP, LABEL,
DISMISS/ACCEPT/RESTORE <id>, THEME, EXPORT, HELP.

## What it finds
- General: data quality, outliers (median/MAD, log scale for skewed positive columns), correlations, trends and level shifts, segment concentration
- Stationarity-aware: trends are only reported if a unit-root test (ADF, constant + trend) says the series is trend-stationary, or the drift in the
  differences is significant. Pure random walks reported as trends: 16/40 before, 3/40 now.
- Derived columns: sums/differences and products/ratios are detected and verified in SQL on all rows; tautological correlations are suppressed.
- Finance: returns-based volatility, volatility regime shifts, fat tails, return correlation vs level correlation (flags spurious level correlation),
  Benford first digits, round-number excess, bunching just below thresholds, near-duplicate transactions, period-over-period variance bridges, time-series gaps
- ML: target leakage (rank AUC, balanced accuracy, missingness), class imbalance, near-constant columns, high-cardinality categoricals, group contrast
- Findings are associations, never causes. Anything statistical that could have a benign explanation says so in its wording.

## Column semantics (the biggest accuracy lever)
Put `<dataset>.uie.yaml` next to the data (or LOAD SEMANTICS in the app, `--semantics` in the CLI). See `examples/`.

    dataset:   time, group_by, entity, label, amount, counterparty, key, thresholds, dedup_window_days
    columns:   <name>: role (measure|id|derived|label|entity|time|amount|counterparty|price|category|ignore),
               negative_is_missing, sentinels: [..], note, unit, currency, exclude: [outliers|correlation|trends|segments|benford|rounding|returns|leakage|all],
               derived_from: [..]

`negative_is_missing` and `sentinels` are applied in the `data` view, so every finding and every evidence query sees cleaned values; `raw` is always the
original. If you declare nothing, roles (amount, counterparty, price, entity, label) are guessed from exact-ish column names and the report says so.

## Per-group analysis
Choose "Group by" (or `dataset.group_by`, `--group-by`). The detectors run inside each of the largest groups (default 12, at least 500 rows) with their
own medians, MADs and trends. Group findings are tagged `[column = value]`, and their evidence SQL filters `data`, so it runs as written. A group-contrast
finding says how much of each measure's variance the grouping explains.

## Reviewing findings
DISMISS (false positive) or ACCEPT (known and fine) a finding, with a reason. Decisions are stored next to the dataset as `<file>.uie-triage.json`
(or under `~/.uie/triage/` if that folder is read-only), keyed by a stable fingerprint (type, columns, group, key facts, not the numbers). Reviewed findings
are excluded from counts, reports and CI gates, and listed in a "Reviewed findings" section. A reviewed finding is re-raised as REOPENED if its magnitude grows
by more than 1.5x since the review.

## Reconcile, drift, train/test, expectations
- RECONCILE: two datasets by key. Keys are normalised (trim, case, `12` = `12.0`). Output: missing, extra, mismatched (absolute and relative tolerance), duplicate keys
  (amounts summed per key first), and a bridge: `total A - total B = missing - extra + mismatches + within-tolerance residue`, with a check column that must be ~0.
- COMPARE: PSI, two-sample KS, chi-square with Cramer's V, null-rate and schema changes. PSI bands: <0.10 stable, 0.10 to 0.25 moderate, >=0.25 significant.
  Numeric PSI/KS use a deterministic hash sample of up to 100,000 rows per side.
- SPLITS: exact and near-duplicate rows between train and test (near = numbers rounded to 2 decimals, text trimmed and lower-cased), shared entities, time overlap.
  Two files, or one file with `--split-col --train-value --test-value`.
- EXPECT: export schema, null limits, ranges, allowed categories and distributions from a dataset; validate later datasets (incl. PSI against the stored distribution).

## Provenance
Every export carries a manifest: tool version, UTC time, SHA-256 of each input file, parameters, thresholds, semantics digest, triage file, Python/DuckDB/NumPy versions and a
result digest over the findings (metrics rounded to 6 significant digits, because approximate aggregates can differ in the last digits). MD/HTML/PDF embed it, JSON includes it,
CSV gets `<file>.manifest.json`. The digest shows the result is unchanged; it is not a cryptographic signature.

## Headless CLI
    python -m uie analyze data.csv --semantics data.uie.yaml --format json -o report.json --fail-on warning
    python -m uie reconcile gl.csv bank.csv --key id --compare amount,fee --tolerance 0.01 --exceptions-csv exceptions.csv
    python -m uie drift train.csv current.csv --label churn
    python -m uie splits train.csv test.csv --entity user_id --label churn
    python -m uie expect create train.csv -o expectations.json ;  python -m uie expect validate new.csv expectations.json
    python -m uie review data.csv dismissed F007 --note "known"      # or by fingerprint
    python -m uie triage list data.csv

Formats: `text` (default), `json`, `md`, `html`, `csv`. Exit codes: **0** pass, **1** gate failed (an active finding at or above `--fail-on`, default `warning`),
**2** bad usage or input, **3** runtime error. `--fail-on never` always exits 0. `--set outlier_t=5` overrides any threshold in `uie/core/config.py`.

GitHub Actions example:

    - run: pip install duckdb numpy pyyaml
    - run: python -m uie analyze data/orders.csv --semantics data/orders.uie.yaml --format json -o uie-report.json -q
    - run: python -m uie expect validate data/orders.csv expectations.json -q
    - uses: actions/upload-artifact@v4
      with: {name: uie-report, path: uie-report.json}

The core and the CLI need only `duckdb`, `numpy` and `pyyaml`; PySide6 and pyqtgraph are only for the desktop app.

## Known limits
- Nothing here is validated for regulated decisions. It is a review aid; findings need a person.
- Derived-column detection finds exact sums/differences/products/ratios, not arbitrary formulas (declare those with `derived_from`).
- Leakage checks look at single features (and null patterns), not interactions or time-based leakage.
- ADF uses MacKinnon response-surface critical values; tests on very short series (<20 points) are skipped.
- The SQL guard on the ASK tab is a keyword filter that accepts only SELECT-style statements. It is not a security boundary; do not expose it to untrusted users.

## Add a detector
Create `uie/core/detectors/mydetector.py` with `detect(ctx) -> list[Fact]`, add a narration template in
`uie/core/narration.py`, and register it in `uie/core/detectors/__init__.py`.

## Package as .exe
    pip install pyinstaller
    pyinstaller --noconfirm --windowed --name UIE --add-data "themes;themes" main.py
