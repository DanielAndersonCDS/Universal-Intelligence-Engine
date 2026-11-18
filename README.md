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
    tests/                  python -m unittest discover -s tests   (GUI: QT_QPA_PLATFORM=offscreen python tests/gui_smoke.py data.csv outdir;
                            python tests/gui_quant_smoke.py prices_long.csv option_chain.csv)
    uie/core/               ingest, semantics, profile, detectors/, narration, report, engine, provenance, triage,
                            reconcile, drift, splitcheck, expectations, stats, config
    uie/ui/                 theme_manager, main_window, tools, plotbuilder, quanttab, charts, widgets, worker, exporter

## App tabs
FINDINGS (with DISMISS / ACCEPT / RESTORE), REPORT (technical / executive / key), ASK (read-only SQL, or plain English via local Ollama; `data` is the cleaned view, `raw` the original),
PROFILE, COMPARE (drift), RECONCILE, SPLITS, EXPECT, MODEL, CHARTS. Command bar: OPEN, RUN, FINDINGS, REPORT, ASK, PROFILE, COMPARE, RECONCILE, SPLITS, EXPECT, SEMANTICS, GROUP, LABEL,
DISMISS/ACCEPT/RESTORE <id>, MODEL, CHARTS, THEME, EXPORT, HELP.

## What it finds
- General: data quality, outliers (median/MAD, log scale for skewed positive columns), correlations, trends and level shifts, segment concentration
- Stationarity-aware: trends are only reported if a unit-root test (ADF, constant + trend) says the series is trend-stationary, or the drift in the
  differences is significant. Pure random walks reported as trends: 16/40 before, 3/40 now.
- Derived columns: sums/differences and products/ratios are detected and verified in SQL on all rows; tautological correlations are suppressed.
- Finance: returns-based volatility, volatility regime shifts, fat tails, return correlation vs level correlation (flags spurious level correlation),
  Benford first digits, round-number excess, bunching just below thresholds, near-duplicate transactions, period-over-period variance bridges, time-series gaps
- ML: target leakage (rank AUC, balanced accuracy, missingness), class imbalance, near-constant columns, high-cardinality categoricals, group contrast
- Statistics: distribution shape and best-fit family (normal, lognormal, exponential, gamma) with skew/kurtosis and Jarque-Bera; group differences (Welch t / ANOVA, confirmed
  by Kruskal-Wallis, Benjamini-Hochberg corrected, reported only at medium-or-larger effect size); associations Pearson misses (monotonic, U-shaped) and categorical
  association (Cramer's V); seasonality (autocorrelation of the detrended series); principal components and multicollinearity (VIF)
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

## Modelling and charts (on demand)
- **MODEL tab / `uie regress`**: OLS with coefficients, standard errors, t, p, 95% CI, standardized beta, VIF, R-squared, F-test and residual checks (normality, heteroscedasticity).
  Solved from exact sufficient statistics (centred X'X, X'y, y'y) computed in SQL over all complete rows, so it is exact for any row count and never loads the data. Text predictors become
  dummy variables (most common level is the reference). Numeric target only; logistic regression is not implemented yet.
- **MODEL tab / `uie fit`**: linear, quadratic, cubic, exponential, power and logarithmic fits ranked by AIC on the original scale, optional prediction at given x (interpolation vs
  extrapolation is labelled), plus the trapezoid area under the binned-mean curve and the steepest slope.
- **CHARTS tab**: histogram (optional fitted distribution, optional log X), scatter (optional fitted curves), line over time, bar, box plot, correlation heatmap, Q-Q plot and autocorrelation.
  Export PNG or SVG. Findings with charts (distribution, seasonality, PCA, group tests, nonlinear association, regression, curve fit) draw them in the finding detail pane.

## Quantitative finance (QUANT tab and `uie arima | garch | coint | var | backtest | portfolio | price | options`)
Every tool reads a price (or return) series from your dataset, states its assumptions, and reports findings with the same wording rules as the rest of UIE. Pick the time column, the price column and,
for a long table, the entity column and which entity (a long table with repeated timestamps is refused rather than silently mixed). Nothing here is investment advice.

- **ARIMA**: (p,d,q) by conditional least squares with stationarity/invertibility enforced, order by AIC, d from ADF tests. Forecast with 95% prediction intervals, residual checks (Ljung-Box,
  ARCH-LM, tails), and an out-of-sample test against a random-walk forecast (Diebold-Mariano). Prices usually do not beat the random walk, and the report says so when they do not.
- **GARCH**: GARCH(1,1) and GJR-GARCH(1,1) with normal or Student-t errors by maximum likelihood, chosen by BIC. Persistence, half-life, long-run vs current volatility, leverage and fat-tail evidence,
  residual diagnostics, and an out-of-sample volatility forecast test against EWMA and a constant variance (QLIKE loss).
- **Cointegration**: Engle-Granger on every pair of a panel (long table with an entity column, or ticked columns of a wide table), both orderings, exact MacKinnon p-values, Benjamini-Hochberg across
  pairs. Hedge ratio, spread half-life, current spread z-score, hedge-ratio stability across the two halves, a spurious-regression check, and a unit-root pre-test: a pair that includes a
  stationary-looking series is reported as a caveat, not a discovery.
- **VaR / expected shortfall**: historical, normal, Student-t (maximum likelihood), Cornish-Fisher, EWMA and optional GARCH-filtered simulation at your confidence levels, with rolling out-of-sample
  backtests: Kupiec coverage, Christoffersen independence, Basel traffic light, and realised loss versus expected shortfall. Worst-period stress table; optional position size for currency amounts.
- **Backtest**: moving-average crossover, momentum, z-score mean reversion, RSI, Donchian breakout, buy and hold, or your own position column. A position decided at the close of bar t earns bar t+1
  (verified no look-ahead for every rule). Costs and slippage on turnover, optional volatility targeting. Reports performance (Sharpe, Sortino, Calmar, probabilistic Sharpe), drawdown episodes
  (peak, trough, recovery, time under water, Ulcer index), cost sensitivity, **walk-forward** optimisation (rolling or anchored; in-sample vs out-of-sample Sharpe, walk-forward efficiency,
  parameter stability), a **deflated Sharpe ratio** that accounts for how many parameter sets were tried, and a block-bootstrap of the out-of-sample returns.
- **Portfolio**: minimum variance, maximum Sharpe, risk parity, inverse volatility and equal weight with weight bounds (long-only by default), Ledoit-Wolf shrinkage, efficient frontier, risk
  contributions and effective number of bets, weight stability under bootstrap, and an **out-of-sample rebalancing test** that says whether optimisation beats 1/N after costs (paired bootstrap).
- **Derivatives**: Black-Scholes-Merton with Greeks and implied volatility, CRR binomial tree (American exercise), Monte Carlo (European, Asian, barrier) with standard errors, bond analytics (price,
  yield, duration, convexity, DV01). `uie options` takes an option chain and reports implied volatility per quote, quotes outside no-arbitrage bounds, monotonicity, butterfly and put-call parity
  violations, the volatility skew, and the term structure. Quotes within a price tick of intrinsic value carry no volatility information and are not flagged.

Validated against reference implementations: ARIMA parameters, forecasts and prediction intervals against statsmodels; GARCH/GJR against `arch`; Engle-Granger and ADF statistics and p-values against statsmodels;
min-variance, tangency and long-only solutions against closed forms and SLSQP; Ledoit-Wolf against scikit-learn; Black-Scholes, Greeks (finite differences), binomial and Monte Carlo against each other.
Those libraries are only used by the tests, and only when installed.

    python -m uie garch prices.csv --entity ticker --entity-value AAPL
    python -m uie var prices.csv --entity ticker --entity-value AAPL --confidence 0.95,0.99 --position 1000000
    python -m uie backtest prices.csv --entity ticker --entity-value AAPL --strategy sma --cost-bps 5
    python -m uie coint prices.csv --entity ticker --value close
    python -m uie portfolio prices.csv --entity ticker --value close --max-weight 0.4
    python -m uie price bs --spot 100 --strike 105 --years 0.5 --rate 0.04 --vol 0.25 --kind put
    python -m uie options chain.csv --strike strike --type type --price mid --years yrs --spot 100 --rate 0.03

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
    python -m uie regress data.csv --target price --features sqft,rooms,city
    python -m uie fit data.csv --x dose --y response --at 2.5,10
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

## Deliberately not included
Symbolic math, ODE/PDE solvers, Laplace and Z transforms, general optimization (LP, integer, global), Simulink-style simulation, filter design and code generation: they do not analyse a dataset, and
SciPy, SymPy and CVXPY do them better. Also not included yet: seasonal ARIMA and EGARCH, Johansen/VECM, CVaR optimisation, Heston and local-volatility models, yield-curve bootstrapping, logistic regression.

## Known limits
- Quant: backtests ignore survivorship bias, taxes, capacity and regime change; ARIMA uses conditional least squares (close to, but not identical to, exact maximum likelihood); GARCH is order (1,1)
  only; cointegration is pairwise; portfolio expected returns are noisy historical means; implied volatilities use European Black-Scholes, so they are approximate for American options.
- The VaR backtest p-values are only as reliable as the number of exceptions: at 99% over 1,000 days you expect about 10, which gives limited power.
- Nothing here is validated for regulated decisions. It is a review aid; findings need a person.
- Derived-column detection finds exact sums/differences/products/ratios, not arbitrary formulas (declare those with `derived_from`).
- Seasonality needs at least 24 buckets with under 10% gaps, and reports the strongest cycle only. Group tests need 30+ rows per group.
- pyqtgraph 0.14's SVG exporter fails on some Qt builds; the app falls back to Qt's own SVG generator automatically.
- Charts are GUI-only (the CLI emits tables and JSON, not images).
- Leakage checks look at single features (and null patterns), not interactions or time-based leakage.
- ADF uses MacKinnon response-surface critical values; tests on very short series (<20 points) are skipped.
- The SQL guard on the ASK tab is a keyword filter that accepts only SELECT-style statements. It is not a security boundary; do not expose it to untrusted users.

## Add a detector
Create `uie/core/detectors/mydetector.py` with `detect(ctx) -> list[Fact]`, add a narration template in
`uie/core/narration.py`, and register it in `uie/core/detectors/__init__.py`.

## Package as .exe
    pip install pyinstaller
    pyinstaller --noconfirm --windowed --name UIE --add-data "themes;themes" main.py
