"""Small, dependency-light statistics helpers (numpy only)."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

# ---- Augmented Dickey-Fuller -------------------------------------------------------------
# MacKinnon (2010) response-surface coefficients: crit = a + b/n + c/n^2
_ADF_CRIT = {
    "c": {1: (-3.43035, -6.5393, -16.786), 5: (-2.86154, -2.8903, -4.234), 10: (-2.56677, -1.5384, -2.809)},
    "ct": {1: (-3.95877, -9.0531, -28.428), 5: (-3.41049, -4.3904, -9.036), 10: (-3.12705, -2.5856, -3.925)},
}


@dataclass
class ADF:
    stat: float
    lags: int
    n: int
    crit5: float
    reject5: bool          # True => unit root rejected (series is stationary / trend-stationary)
    regression: str
    p: float = float("nan")   # MacKinnon approximate p-value


def adf(y, regression: str = "ct") -> ADF | None:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 15 or np.ptp(y) == 0:
        return None
    dy = np.diff(y)
    maxlag = max(0, min(int(math.ceil(12 * (n / 100) ** 0.25)), (n - 6) // 4))
    start = maxlag + 1                       # common sample for lag selection
    best = None
    for p in range(maxlag + 1):
        fit = _adf_fit(y, dy, p, start, regression)
        if fit is None:
            continue
        aic = fit[3]
        if best is None or aic < best[0]:
            best = (aic, p)
    if best is None:
        return None
    fit = _adf_fit(y, dy, best[1], best[1] + 1, regression)    # re-estimate on the longest usable sample
    if fit is None:
        return None
    stat, _, nobs, _ = fit
    if regression == "n":                       # residual-based tests (Engle-Granger): the caller applies cointegration critical values
        return ADF(float(stat), best[1], int(nobs), float("nan"), False, "n", float("nan"))
    a, b, c = _ADF_CRIT[regression][5]
    crit = a + b / nobs + c / nobs ** 2
    from .quant.mackinnon import mackinnon_p
    return ADF(float(stat), best[1], int(nobs), float(crit), bool(stat < crit), regression, float(mackinnon_p(float(stat), 1, regression)))


def _adf_fit(y, dy, p, start, regression):
    n = len(y)
    idx = np.arange(start, n)                # dy[t-1] = y[t] - y[t-1]
    if len(idx) < p + 8:
        return None
    cols = [y[idx - 1]]
    for i in range(1, p + 1):
        cols.append(dy[idx - 1 - i])
    if regression in ("c", "ct"):
        cols.append(np.ones(len(idx)))
    if regression == "ct":
        cols.append(idx.astype(float))
    X = np.column_stack(cols)
    t = dy[idx - 1]
    try:
        beta, *_ = np.linalg.lstsq(X, t, rcond=None)
        resid = t - X @ beta
        dof = len(t) - X.shape[1]
        if dof <= 0:
            return None
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = math.sqrt(max(cov[0, 0], 1e-300))
        rss = float(resid @ resid)
        aic = len(t) * math.log(max(rss / len(t), 1e-300)) + 2 * X.shape[1]
        return beta[0] / se, beta, len(t), aic
    except np.linalg.LinAlgError:
        return None


def nw_mean_t(d) -> float:
    """t-statistic of the mean of d with a Newey-West (Bartlett) long-run variance."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return 0.0
    x = d - d.mean()
    L = int(4 * (n / 100) ** (2 / 9))
    lrv = float(x @ x) / n
    for j in range(1, L + 1):
        lrv += 2 * (1 - j / (L + 1)) * float(x[j:] @ x[:-j]) / n
    if lrv <= 0:
        return 0.0
    return float(d.mean() / math.sqrt(lrv / n))


def robust_z(x) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    if not mad:
        mad = np.nanstd(x)
    return np.abs(x - med) / mad if mad else np.zeros_like(x)


# ---- distribution tests ------------------------------------------------------------------
def chi2_sf(x: float, df: int) -> float:
    """P(chi-square_df > x) via the regularised upper incomplete gamma function."""
    if x <= 0 or df <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1:
        s, term, n = 1.0 / a, 1.0 / a, 1
        while abs(term) > abs(s) * 1e-14 and n < 2000:
            term *= xx / (a + n)
            s += term
            n += 1
        p = s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        return max(0.0, min(1.0, 1 - p))
    tiny = 1e-300
    b = xx + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return max(0.0, min(1.0, math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h))


def ks_2samp(a, b):
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return None, None
    allv = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, allv, side="right") / n - np.searchsorted(b, allv, side="right") / m)))
    ne = n * m / (n + m)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    if lam < 1e-9:
        return d, 1.0
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, float(max(0.0, min(1.0, p)))


def psi(ref_counts, cur_counts, eps: float = 1e-4) -> float:
    r, c = np.asarray(ref_counts, float), np.asarray(cur_counts, float)
    if r.sum() <= 0 or c.sum() <= 0:
        return 0.0
    pr, pc = np.maximum(r / r.sum(), eps), np.maximum(c / c.sum(), eps)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def chi2_two_sample(ref_counts, cur_counts):
    """Chi-square test of homogeneity for two count vectors. Returns (chi2, df, p, cramers_v)."""
    r, c = np.asarray(ref_counts, float), np.asarray(cur_counts, float)
    keep = (r + c) > 0
    r, c = r[keep], c[keep]
    k = len(r)
    if k < 2 or r.sum() == 0 or c.sum() == 0:
        return 0.0, 0, 1.0, 0.0
    tot = r.sum() + c.sum()
    er, ec = (r + c) * r.sum() / tot, (r + c) * c.sum() / tot
    chi2 = float(np.sum((r - er) ** 2 / er) + np.sum((c - ec) ** 2 / ec))
    return chi2, k - 1, chi2_sf(chi2, k - 1), math.sqrt(chi2 / tot)


BENFORD = np.array([math.log10(1 + 1 / d) for d in range(1, 10)])


def benford_stats(counts):
    """counts: observed first-digit counts for digits 1..9. Returns dict with MAD, chi2, p and the worst digit."""
    obs = np.asarray(counts, float)
    n = obs.sum()
    if n <= 0:
        return None
    share = obs / n
    mad = float(np.mean(np.abs(share - BENFORD)))
    chi2 = float(np.sum((obs - n * BENFORD) ** 2 / (n * BENFORD)))
    w = int(np.argmax(np.abs(share - BENFORD)))
    return {"n": int(n), "mad": mad, "chi2": chi2, "p": chi2_sf(chi2, 8), "worst_digit": w + 1,
            "observed": float(share[w]), "expected": float(BENFORD[w])}


# ---- t / F distributions, group tests, multiple testing ------------------------------------
def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b) (continued fraction, Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (tiny if abs(d) < tiny else d)
        c = 1 + aa / c
        c = tiny if abs(c) < tiny else c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (tiny if abs(d) < tiny else d)
        c = 1 + aa / c
        c = tiny if abs(c) < tiny else c
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-14:
            break
    return h


def t_sf(t: float, df: float) -> float:
    """One-sided P(T > t)."""
    if df <= 0:
        return 0.5
    p = 0.5 * betainc(df / 2, 0.5, df / (df + t * t))
    return p if t > 0 else 1 - p


def t_two_sided(t: float, df: float) -> float:
    return min(1.0, 2 * t_sf(abs(t), df))


def f_sf(f: float, d1: float, d2: float) -> float:
    if f <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0
    return betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * f))


def welch_t(n1, m1, v1, n2, m2, v2):
    """Welch's t-test from summary statistics (sample variances). Returns t, df, two-sided p, Cohen's d."""
    se2 = v1 / n1 + v2 / n2
    if se2 <= 0 or n1 < 2 or n2 < 2:
        return 0.0, 0.0, 1.0, 0.0
    t = (m1 - m2) / math.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return t, df, t_two_sided(t, df), (m1 - m2) / sp if sp else 0.0


def anova_oneway(ns, means, variances):
    """One-way ANOVA from per-group n, mean and sample variance. Returns F, df1, df2, p, eta_squared."""
    ns, means, variances = np.asarray(ns, float), np.asarray(means, float), np.asarray(variances, float)
    k, N = len(ns), ns.sum()
    if k < 2 or N - k <= 0:
        return 0.0, 0, 0, 1.0, 0.0
    grand = float((ns * means).sum() / N)
    ssb = float((ns * (means - grand) ** 2).sum())
    ssw = float(((ns - 1) * variances).sum())
    if ssw <= 0:
        return float("inf"), k - 1, int(N - k), 0.0, 1.0
    f = (ssb / (k - 1)) / (ssw / (N - k))
    return f, k - 1, int(N - k), f_sf(f, k - 1, N - k), ssb / (ssb + ssw)


def avg_ranks(x):
    """Average ranks (1-based) with ties, vectorised."""
    x = np.asarray(x, float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    return ((cum - counts + 1) + cum)[inv] / 2.0


def kruskal(groups):
    """Kruskal-Wallis H with tie correction. Returns H, df, p, epsilon_squared."""
    groups = [np.asarray(g, float) for g in groups if len(g) > 0]
    k = len(groups)
    allv = np.concatenate(groups)
    N = len(allv)
    if k < 2 or N < 3:
        return 0.0, 0, 1.0, 0.0
    r = avg_ranks(allv)
    pos, h = 0, 0.0
    for g in groups:
        rg = r[pos:pos + len(g)]
        h += rg.sum() ** 2 / len(g)
        pos += len(g)
    h = 12 / (N * (N + 1)) * h - 3 * (N + 1)
    _, counts = np.unique(allv, return_counts=True)
    corr = 1 - (counts ** 3 - counts).sum() / (N ** 3 - N)
    if corr <= 0:
        return 0.0, k - 1, 1.0, 0.0
    h /= corr
    return float(h), k - 1, chi2_sf(h, k - 1), float(h / (N - 1))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return None
    rx, ry = avg_ranks(x[m]), avg_ranks(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def bh_adjust(pvals):
    """Benjamini-Hochberg adjusted p-values (false discovery rate), same order as input."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        adj[i] = prev
    return adj


def jarque_bera(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8 or x.std() == 0:
        return None
    z = (x - x.mean()) / x.std()
    s, k = float((z ** 3).mean()), float((z ** 4).mean()) - 3
    jb = n / 6 * (s * s + k * k / 4)
    return {"n": n, "skew": s, "kurtosis": k, "jb": jb, "p": chi2_sf(jb, 2)}


def acf(y, nlags):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    y = y - y.mean()
    den = float(y @ y)
    if n < 4 or den == 0:
        return np.zeros(nlags + 1)
    return np.array([1.0] + [float(y[k:] @ y[:-k]) / den for k in range(1, nlags + 1)])


# ---- distribution fitting ----------------------------------------------------------------
def _norm_cdf(x, mu, sd):
    return 0.5 * (1 + np.array([math.erf(v) for v in (np.asarray(x) - mu) / (sd * math.sqrt(2))]))


def fit_distributions(x, max_cdf_points: int = 4000):
    """Fit normal, lognormal, exponential and gamma (method of moments) and rank by the KS distance to the sample.
    The KS distance is a descriptive measure of fit; it is not a hypothesis test because parameters were estimated from the same data."""
    x = np.asarray(x, float)
    x = np.sort(x[np.isfinite(x)])
    n = len(x)
    if n < 50 or x.std() == 0:
        return []
    if n > max_cdf_points:
        x = x[np.linspace(0, n - 1, max_cdf_points).astype(int)]
        n = len(x)
    ecdf_hi, ecdf_lo = np.arange(1, n + 1) / n, np.arange(0, n) / n
    out = []

    def ks(F):
        return float(max(np.max(np.abs(ecdf_hi - F)), np.max(np.abs(ecdf_lo - F))))

    mu, sd = float(x.mean()), float(x.std())
    out.append({"dist": "normal", "params": {"mu": mu, "sigma": sd}, "ks": ks(_norm_cdf(x, mu, sd))})
    if x[0] > 0:
        lx = np.log(x)
        lm, ls = float(lx.mean()), float(lx.std())
        if ls > 0:
            out.append({"dist": "lognormal", "params": {"mu": lm, "sigma": ls}, "ks": ks(_norm_cdf(lx, lm, ls))})
    if x[0] >= 0 and mu > 0:
        out.append({"dist": "exponential", "params": {"rate": 1 / mu}, "ks": ks(1 - np.exp(-x / mu))})
        var = float(x.var())
        if var > 0:
            k, th = mu * mu / var, var / mu
            F = np.array([1 - chi2_sf(2 * v / th, 2 * k) for v in x])
            out.append({"dist": "gamma", "params": {"shape": k, "scale": th}, "ks": ks(F)})
    return sorted(out, key=lambda d: d["ks"])


def dist_pdf(dist: str, params: dict, xs):
    xs = np.asarray(xs, float)
    if dist == "normal":
        return np.exp(-0.5 * ((xs - params["mu"]) / params["sigma"]) ** 2) / (params["sigma"] * math.sqrt(2 * math.pi))
    if dist == "lognormal":
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.exp(-0.5 * ((np.log(xs) - params["mu"]) / params["sigma"]) ** 2) / (xs * params["sigma"] * math.sqrt(2 * math.pi))
        return np.where(xs > 0, y, 0.0)
    if dist == "exponential":
        return np.where(xs >= 0, params["rate"] * np.exp(-params["rate"] * xs), 0.0)
    if dist == "gamma":
        k, th = params["shape"], params["scale"]
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.exp((k - 1) * np.log(xs) - xs / th - math.lgamma(k) - k * math.log(th))
        return np.where(xs > 0, y, 0.0)
    return np.zeros_like(xs)


def qq_points(x, dist="normal", n_points=200):
    """Sample vs theoretical quantiles for a normal Q-Q plot (data is log-transformed first when dist='lognormal')."""
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x)]
    if dist == "lognormal":
        x = np.log(x[x > 0])
    n = len(x)
    if n < 10:
        return []
    idx = np.linspace(0, n - 1, min(n, n_points)).astype(int)
    p = (idx + 0.5) / n
    # inverse normal CDF by bisection on erf (no scipy)
    def ppf(q):
        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    return [(ppf(float(pi)), float(x[i])) for pi, i in zip(p, idx)]


# ---- least squares from sufficient statistics ----------------------------------------------
def ols_from_gram(xtx, xty, yty, n, means_x, mean_y, names):
    """OLS on *centred* data. xtx/xty/yty are centred cross-products; the intercept is recovered from the means.
    Returns None if the design is singular."""
    xtx = np.asarray(xtx, float)
    xty = np.asarray(xty, float)
    k = len(names)
    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inv)):
        return None
    beta = inv @ xty
    sse = max(float(yty - beta @ xty), 0.0)
    df = n - k - 1
    if df <= 0:
        return None
    s2 = sse / df
    se = np.sqrt(np.maximum(np.diag(inv) * s2, 0))
    tstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p = [t_two_sided(float(t), df) for t in tstat]
    tcrit = _t_crit(df)
    sst = float(yty)
    r2 = 1 - sse / sst if sst > 0 else 0.0
    f = ((sst - sse) / k) / s2 if s2 > 0 else float("inf")
    sd_y = math.sqrt(sst / (n - 1)) if n > 1 else 0.0
    diag = np.diag(xtx) / (n - 1)
    # VIF from the correlation matrix of the predictors
    d = np.sqrt(np.diag(xtx))
    corr = xtx / np.outer(d, d)
    try:
        vif = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        vif = np.full(k, np.inf)
    return {"intercept": float(mean_y - beta @ np.asarray(means_x)), "beta": beta, "se": se, "t": tstat, "p": np.array(p),
            "ci": np.column_stack([beta - tcrit * se, beta + tcrit * se]), "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / df,
            "f": f, "f_p": f_sf(f, k, df) if np.isfinite(f) else 0.0, "df": df, "n": n, "rmse": math.sqrt(s2), "sse": sse,
            "std_beta": [float(b * math.sqrt(dv) / sd_y) if sd_y else 0.0 for b, dv in zip(beta, diag)], "vif": vif}


def _t_crit(df, p=0.975):
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if 1 - t_sf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---- normal / t quantiles, residual diagnostics --------------------------------------------
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation refined by one Halley step; error < 1e-12)."""
    if not 0 < p < 1:
        return -math.inf if p <= 0 else math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    lo = 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - lo:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def t_cdf(x: float, df: float) -> float:
    return 1 - t_sf(x, df)


def t_ppf(p: float, df: float) -> float:
    if not 0 < p < 1:
        return -math.inf if p <= 0 else math.inf
    if p == 0.5:
        return 0.0
    lo, hi = -1e3, 1e3
    for _ in range(100):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def ljung_box(x, lags: int, dof: int = 0):
    """Ljung-Box Q statistic and p-value for autocorrelation up to `lags` (dof = fitted ARMA parameters)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    lags = max(1, min(lags, n // 2 - 1))
    ac = acf(x, lags)
    q = float(n * (n + 2) * np.sum(ac[1:] ** 2 / (n - np.arange(1, lags + 1))))
    df = max(lags - dof, 1)
    return q, chi2_sf(q, df), lags


def arch_lm(x, lags: int = 5):
    """Engle's ARCH-LM test: regress x^2 on its own lags; LM = n R^2 ~ chi2(lags)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    y = (x - x.mean()) ** 2
    n = len(y) - lags
    if n < lags + 10:
        return None
    X = np.column_stack([np.ones(n)] + [y[lags - k - 1:len(y) - k - 1] for k in range(lags)])
    t = y[lags:]
    beta = np.linalg.lstsq(X, t, rcond=None)[0]
    sst = float(((t - t.mean()) ** 2).sum())
    r2 = 1 - float(((t - X @ beta) ** 2).sum()) / sst if sst else 0.0
    lm = n * r2
    return lm, chi2_sf(lm, lags)
