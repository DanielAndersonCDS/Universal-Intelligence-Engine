"""Template narration. Every sentence is built from fact metrics, so it cannot hallucinate.
Wording is deliberately associative: never 'caused by'."""
from __future__ import annotations
from .util import fmt_num, pct


def _extremes(m):
    parts = []
    if m.get("low") is not None:
        parts.append(f"as low as {fmt_num(m['low'])}")
    if m.get("high") is not None:
        parts.append(f"as high as {fmt_num(m['high'])}")
    return " and ".join(parts)


def _strength(r):
    a = abs(r)
    return "very strongly" if a >= 0.9 else "strongly" if a >= 0.7 else "moderately"


def _trend(m):
    s = (f"{m['label']} {m['direction']} {pct(m['pct'])} over {m['period']} "
         f"(from {fmt_num(m['start'])} to {fmt_num(m['end'])}; linear fit R-squared = {m['r2']:.2f}")
    if m.get("regime"):
        s += f"; {m['regime']}, ADF = {m['adf']:.2f}" + (f", drift t = {m['drift_t']:+.2f}" if m.get("drift_t") is not None else "")
    return s + ")."


def _a(word):
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _p(p):
    return "< 0.0001" if p < 1e-4 else f"= {p:.4f}"


def _score(measure, v):
    return f"{v:.3f}" if measure == "auc" else pct(v)


def _top_contrib(top):
    return ", ".join(f"{t['group']} ({fmt_num(t['delta'])}, {pct(t['share'], 0)} of the change)" for t in top)


def _amount_bits(d, sign=""):
    return ", ".join(f"`{k}` {sign}{fmt_num(v)}" for k, v in d.items())


_T = {
    "quality_empty": lambda m: "The dataset contains no rows.",
    "quality_nulls": lambda m: f"`{m['column']}` is missing {pct(m['frac'])} of its values ({m['nulls']:,} of {m['rows']:,} rows).",
    "quality_constant": lambda m: f"`{m['column']}` holds a single value ({m['value']}) in every row, so it carries no information.",
    "quality_id_like": lambda m: f"`{m['column']}` is unique (or nearly) per row, which suggests an identifier rather than a measurement.",
    "quality_duplicates": lambda m: f"{m['dupes']:,} rows ({pct(m['frac'], 2)}) are exact duplicates of another row.",
    "quality_text_numeric": lambda m: f"`{m['column']}` is stored as text, but {pct(m['frac'])} of its values parse as numbers.",
    "quality_sentinel": lambda m: f"{m['count']:,} values in `{m['column']}` were treated as missing per the semantics file: {m['rule']}.",
    "quality_derived": lambda m: (f"`{m['column']}` is derived: {m['expr']} holds in {pct(m['match_share'], 2)} of {m['n']:,} rows, "
                                  f"so correlations between them are tautological and are suppressed."),
    "outliers": lambda m: (f"{m['count']:,} values ({pct(m['share'], 2)}) in `{m['column']}` are statistical outliers "
                           f"(robust z-score above {m['threshold']:g}{', on a log scale' if m.get('log_scale') else ''}); they reach {_extremes(m)} against a typical value of {fmt_num(m['median'])}."),
    "correlation": lambda m: (f"`{m['a']}` and `{m['b']}` are strongly {'positively' if m['r'] > 0 else 'negatively'} associated "
                              f"(Pearson r = {m['r']:+.2f}, n = {m['n']:,}). This is an association, not evidence that one drives the other."),
    "near_duplicate": lambda m: f"`{m['a']}` and `{m['b']}` are almost perfectly correlated (r = {m['r']:.4f}); one may be derived from the other.",
    "trend": _trend,
    "trend_caveat": lambda m: (f"{m['label']} looks like it {m['direction']} {pct(m['pct'])} (linear R-squared = {m['r2']:.2f}), but the series is not "
                               f"distinguishable from a random walk (ADF = {m['adf']:.2f}, drift t = {m['drift_t']:+.2f}), so no trend is reported."),
    "changepoint": lambda m: (f"{m['label']} shifted around {m['when']}: the average {m['direction']} from "
                              f"{fmt_num(m['before'])} to {fmt_num(m['after'])} ({pct(m['pct'])} change)."),
    "concentration": lambda m: (f"`{m['group_value']}` (in `{m['group_column']}`) accounts for {pct(m['value_share'])} of `{m['column']}` "
                                f"but only {pct(m['row_share'])} of records, across {m['groups']} groups."),
    "group_contrast": lambda m: (f"`{m['group_column']}` explains {pct(m['eta2'])} of the variance in `{m['column']}` "
                                 f"(highest {m['hi_group']}: {fmt_num(m['hi_mean'])}; lowest {m['lo_group']}: {fmt_num(m['lo_mean'])}); "
                                 f"a pooled analysis may hide group-specific behavior."),
    # finance
    "volatility": lambda m: (f"{m['label']}: annualized volatility of log returns is {pct(m['ann_vol'])} over {m['n']:,} periods "
                             f"(worst {pct(m['worst'])}, best {pct(m['best'])}; excess kurtosis {m['kurtosis']:.1f}"
                             f"{', so returns are fat-tailed and normal-based thresholds understate extreme moves' if m['kurtosis'] >= 5 else ''})."),
    "vol_shift": lambda m: (f"{m['label']}: volatility {m['direction']} {m['ratio']:.1f}x in the most recent third of the period "
                            f"(from {pct(m['early_vol'])} to {pct(m['late_vol'])} annualized)."),
    "return_correlation": lambda m: (f"Returns of `{m['a']}` and `{m['b']}` are {_strength(m['r'])} {'positively' if m['r'] > 0 else 'negatively'} "
                                     f"associated (r = {m['r']:+.2f}, n = {m['n']:,}). This is an association, not evidence that one drives the other."),
    "spurious_correlation": lambda m: (f"`{m['a']}` and `{m['b']}` correlate at r = {m['level_r']:+.2f} in levels but only r = {m['return_r']:+.2f} "
                                       f"in returns (n = {m['n']:,}); the level correlation reflects shared trends rather than co-movement."),
    "benford": lambda m: (f"The first-digit distribution of `{m['column']}` deviates from Benford's expectation (MAD = {m['mad']:.4f}, {m['conformity']}; "
                          f"n = {m['n']:,}). Digit {m['worst_digit']} appears in {pct(m['observed'])} of values versus {pct(m['expected'])} expected. "
                          f"Price points, caps and rounding can cause this; it is a prompt for review, not evidence of error or fraud."),
    "round_numbers": lambda m: (f"{m['count']:,} values ({pct(m['share'])}) in `{m['column']}` are exact multiples of {m['modulus']:,}, "
                                f"versus about {pct(m['expected'], 2)} expected by chance."),
    "threshold_bunching": lambda m: (f"{m['below']:,} values in `{m['column']}` fall in the 5% just below {fmt_num(m['threshold'])} versus {m['above']:,} "
                                     f"just above it ({m['below_far']:,} in the band before that). This can reflect an approval limit, reporting "
                                     f"threshold or price point."),
    "near_duplicate_tx": lambda m: (f"{m['pairs']:,} transactions ({pct(m['share'], 2)}) repeat the same `{m['amount']}` for the same `{m['counterparty']}` "
                                    f"within {m['window']:g} days of an earlier one, totalling {fmt_num(m['amount_at_stake'])}. Legitimate recurring "
                                    f"payments can look like this."),
    "variance_driver": lambda m: (f"`{m['measure']}` {m['direction']} {pct(abs(m['pct']))} {m['basis']} ({m['prior_label']}: {fmt_num(m['prior'])} "
                                  f"to {m['current_label']}: {fmt_num(m['current'])}); in `{m['dimension']}`, the largest contributors were "
                                  f"{_top_contrib(m['top'])}."),
    "time_gap": lambda m: (f"`{m['column']}` has {m['gaps']:,} gap(s) longer than expected (typical spacing {m['typical_days']:g} days); the longest "
                           f"runs {m['longest_days']:g} days from {m['longest_start']} to {m['longest_end']}."),
    # AI
    "target_leakage": lambda m: (f"`{m['feature']}` almost determines `{m['label']}` ({m['method']}: {_score(m['measure'], m['purity'])} "
                                 f"against {_score(m['measure'], m['baseline'])} for chance). It is either a leakage candidate or derived from the label."),
    "leakage_missingness": lambda m: (f"Whether `{m['feature']}` is missing predicts `{m['label']}` (balanced accuracy {pct(m['purity'])} against "
                                      f"{pct(m['baseline'])} for chance); the field may only be filled in after the outcome is known."),
    "class_imbalance": lambda m: (f"`{m['label']}` is imbalanced: the rarest class ({m['minority']}) is {pct(m['minority_share'], 2)} of rows "
                                  f"(majority-to-minority ratio {m['ratio']:,.1f}:1 across {m['classes']} classes)."),
    "near_constant": lambda m: f"`{m['column']}` has the value {m['value']} in {pct(m['share'], 2)} of rows, so it carries almost no information.",
    "high_cardinality": lambda m: (f"`{m['column']}` has {m['distinct']:,} distinct values; {pct(m['rare_share'])} of rows belong to categories seen "
                                   f"fewer than {m['rare_count']} times. One-hot encoding will be sparse and target encoding risks leakage."),
    # statistics
    "distribution_fit": lambda m: (f"`{m['column']}` is not normally distributed (skewness {m['skew']:+.2f}, excess kurtosis {m['kurtosis']:.1f}); {_a(m['best'])} "
                                   f"fit describes it best (KS distance {m['best_ks']:.3f} versus {m['normal_ks']:.3f} for normal). Mean and standard-deviation "
                                   f"methods (z-scores, t-tests) are unreliable here; consider a log scale or rank-based tests."),
    "group_test": lambda m: (f"`{m['measure']}` differs across `{m['dimension']}` ({m['test']}: {m['stat_name']} = {m['stat']:.2f}, adjusted p {_p(m['p_adj'])}, "
                             f"{m['effect_name']} = {m['effect']:.2f}; Kruskal-Wallis p {_p(m['kruskal_p'])}). Highest {m['hi_group']}: {fmt_num(m['hi_mean'])}; "
                             f"lowest {m['lo_group']}: {fmt_num(m['lo_mean'])}. Groups may also differ in other ways, so this is an association."),
    "nonlinear_association": lambda m: (
        f"`{m['a']}` and `{m['b']}` move together monotonically but not in a straight line (Spearman rho = {m['spearman']:+.2f} versus Pearson r = {m['pearson']:+.2f})."
        if m["kind"] == "monotonic" else
        f"`{m['a']}` and `{m['b']}` have a curved relationship: a quadratic fit explains {pct(m['r2_quadratic'])} of the variance versus {pct(m['r2_linear'])} "
        f"for a straight line (Pearson r = {m['pearson']:+.2f} understates it)."),
    "categorical_association": lambda m: (f"`{m['a']}` and `{m['b']}` are associated (Cramer's V = {m['cramers_v']:.2f}, chi-square p {_p(m['p'])}, n = {m['n']:,})"
                                          f"{'; one may largely determine the other' if m['cramers_v'] >= 0.9 else ''}."),
    "seasonality": lambda m: (f"{m['label']} repeats on a cycle of about {m['period']} {m['unit']}s (autocorrelation {m['acf']:.2f} at that lag; the cycle explains "
                              f"{pct(m['strength'])} of the variation after removing {'the stochastic trend' if m['differenced'] else 'the linear trend'})."),
    "pca": lambda m: (f"The {m['columns']} numeric columns carry about {m['components90']} independent dimensions: {m['components90']} principal components explain 90% "
                      f"of the variance and the first explains {pct(m['pc1'])}. PC1 loads mainly on {', '.join('`' + n + '` (' + format(v, '+.2f') + ')' for n, v in m['loadings_pc1'])}."),
    "multicollinearity": lambda m: (f"{m['count']} column(s) are strongly collinear with the others (VIF above {m['threshold']:g}): "
                                    f"{', '.join('`' + n + '` (' + (format(v, '.0f') if v < 1e5 else 'infinite') + ')' for n, v in m['top'])}. "
                                    f"Regression coefficients on these columns will be unstable."),
    # regression / curve fitting (on-demand tools)
    "regress_fit": lambda m: (f"The model explains {pct(m['r2'])} of the variance in `{m['target']}` (adjusted R-squared {m['adj_r2']:.3f}, RMSE {fmt_num(m['rmse'])}, "
                              f"n = {m['n']:,}; F-test p {_p(m['f_p'])})."),
    "regress_predictor": lambda m: (f"Holding the other predictors fixed, a one-unit increase in `{m['term']}` is associated with a {fmt_num(m['coef'])} change in "
                                    f"`{m['target']}` (95% CI {fmt_num(m['ci_lo'])} to {fmt_num(m['ci_hi'])}; standardized beta {m['std_beta']:+.2f}; p {_p(m['p'])}). "
                                    f"This is an association, not evidence of cause."),
    "regress_warning": lambda m: m["message"],
    "curve_fit": lambda m: (f"`{m['y']}` is best described by {_a(m['model'])} function of `{m['x']}`: {m['equation']} (R-squared {m['r2']:.3f}, "
                            f"adjusted {m['adj_r2']:.3f}; AIC {m['delta_aic']:.1f} lower than a straight line)." if m["model"] != "linear" else
                            f"`{m['y']}` is well described by a straight line in `{m['x']}`: {m['equation']} (R-squared {m['r2']:.3f}); no simple curve fits meaningfully better."),
    # drift / split / reconcile
    "drift": lambda m: _drift(m),
    "drift_schema": lambda m: m["message"],
    "split_overlap": lambda m: (f"{m['rows']:,} {m['side']} rows ({pct(m['share'], 2)}) are {'exact ' if m['kind'] == 'exact' else 'near-'}duplicates of "
                                f"{m['ref_side']} rows (compared on {m['columns']} shared columns)."),
    "split_entities": lambda m: (f"{m['shared']:,} of {m['entities']:,} {m['side']} `{m['entity']}` values ({pct(m['share'])}) also appear in the "
                                 f"{m['ref_side']} data, covering {pct(m['row_share'])} of {m['side']} rows; entity-level leakage is likely."),
    "split_time": lambda m: m["message"],
    "recon_missing": lambda m: (f"{m['count']:,} keys exist in {m['left']} but not in {m['right']}" + (f" ({_amount_bits(m['amounts'])})" if m.get("amounts") else "") + "."),
    "recon_extra": lambda m: (f"{m['count']:,} keys exist in {m['right']} but not in {m['left']}" + (f" ({_amount_bits(m['amounts'])})" if m.get("amounts") else "") + "."),
    "recon_mismatch": lambda m: (f"{m['count']:,} matched keys differ beyond tolerance" + (f" (net difference {_amount_bits(m['amounts'])})" if m.get("amounts") else "") + "."),
    "recon_dupkeys": lambda m: f"{m['keys']:,} keys appear more than once in {m['side']} ({m['rows']:,} rows); amounts were summed per key before matching.",
    "recon_bridge": lambda m: (f"`{m['column']}`: {m['left']} total {fmt_num(m['total_left'])} vs {m['right']} total {fmt_num(m['total_right'])} "
                               f"(difference {fmt_num(m['diff'])}) = missing in {m['right']} {fmt_num(m['only_left'])} - extra in {m['right']} "
                               f"{fmt_num(m['only_right'])} + mismatches {fmt_num(m['mismatch'])} + within-tolerance residue {fmt_num(m['residue'])}."),
}


def _drift(m):
    if m["kind"] == "numeric":
        s = f"`{m['column']}` has shifted (PSI = {m['psi']:.3f}, KS D = {m['ks']:.3f}, mean shift {m['mean_shift']:+.2f} std"
    else:
        s = f"`{m['column']}` category mix changed (PSI = {m['psi']:.3f}, chi-square p {_p(m['p'])}, Cramer's V = {m['cramers_v']:.3f}"
    if abs(m.get("null_change", 0)) >= 0.05:
        s += f"; missing share changed by {m['null_change'] * 100:+.1f} points"
    return s + ")."


def narrate(f) -> str:
    fn = _T.get(f.type)
    try:
        if fn is None and (f.type.startswith("expect_") or f.type.startswith("q_")):
            text = f.metrics["message"]
        else:
            text = fn(f.metrics) if fn else f.title
    except Exception:
        text = f.title
    if f.group:
        text = f"[{f.group['column']} = {f.group['value']}] " + text
    return text
