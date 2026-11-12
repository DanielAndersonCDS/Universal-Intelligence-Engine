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
    a, b, c = _ADF_CRIT[regression][5]
    crit = a + b / nobs + c / nobs ** 2
    return ADF(float(stat), best[1], int(nobs), float(crit), bool(stat < crit), regression)


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
