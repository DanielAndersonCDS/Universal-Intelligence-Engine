"""MacKinnon (1994, 2010) response-surface tables for unit-root / Engle-Granger tests, N = number of variables (1 = ADF).
Values are the published coefficients (as distributed with statsmodels, BSD licence); used here for critical values and p-values.
"""
import math

SMALLP_C = [[2.1659, 1.4412, 0.038269], [2.92, 1.5012, 0.039796], [3.4699, 1.4856, 0.03164], [3.9673, 1.4777, 0.026315]]
LARGEP_C = [[1.7339, 0.93202, -0.12745, -0.010368], [2.1945, 0.64695, -0.29198, -0.042377], [2.5893, 0.45168, -0.36529, -0.050074], [3.0387, 0.45452, -0.33666, -0.041921]]
STAR_C = [-1.61, -2.62, -3.13, -3.47]
MIN_C = [-18.83, -18.86, -23.48, -28.07]
MAX_C = [2.74, 0.92, 0.55, 0.61]
CRIT_C = [[[-3.43035, -6.5393, -16.786, -79.433], [-2.86154, -2.8903, -4.234, -40.04], [-2.56677, -1.5384, -2.809, 0.0]], [[-3.89644, -10.9519, -33.527, 0.0], [-3.33613, -6.1101, -6.823, 0.0], [-3.04445, -4.2412, -2.72, 0.0]], [[-4.29374, -14.4354, -33.195, 47.433], [-3.74066, -8.5632, -10.852, 27.982], [-3.45218, -6.2143, -3.718, 0.0]], [[-4.64332, -18.1031, -37.972, 0.0], [-4.096, -11.2349, -11.175, 0.0], [-3.8102, -8.3931, -4.137, 0.0]]]
SMALLP_CT = [[3.2512, 1.6047, 0.049588], [3.6646, 1.5419, 0.036448], [4.0983, 1.5173, 0.029898], [4.5844, 1.5338, 0.028796]]
LARGEP_CT = [[2.5261, 0.61654, -0.37956, -0.060285], [2.85, 0.5272, -0.36622, -0.051695], [3.221, 0.5255, -0.32685, -0.041501], [3.652, 0.59758, -0.27483, -0.032081]]
STAR_CT = [-2.89, -3.19, -3.5, -3.65]
MIN_CT = [-16.18, -21.15, -25.37, -26.63]
MAX_CT = [0.7, 0.63, 0.71, 0.93]
CRIT_CT = [[[-3.95877, -9.0531, -28.428, -134.155], [-3.41049, -4.3904, -9.036, -45.374], [-3.12705, -2.5856, -3.925, -22.38]], [[-4.32762, -15.4387, -35.679, 0.0], [-3.78057, -9.5106, -12.074, 0.0], [-3.49631, -7.0815, -7.538, 21.892]], [[-4.66305, -18.7688, -49.793, 104.244], [-4.1189, -11.8922, -19.031, 77.332], [-3.83511, -9.0723, -8.504, 35.403]], [[-4.9694, -22.4694, -52.599, 51.314], [-4.42871, -14.5876, -18.228, 39.647], [-4.14633, -11.25, -9.873, 54.109]]]

_T = {"c": (SMALLP_C, LARGEP_C, STAR_C, MIN_C, MAX_C, CRIT_C), "ct": (SMALLP_CT, LARGEP_CT, STAR_CT, MIN_CT, MAX_CT, CRIT_CT)}


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mackinnon_p(stat: float, n_vars: int = 1, regression: str = "c") -> float:
    """Approximate p-value of an ADF (n_vars=1) or Engle-Granger (n_vars>=2) statistic. Regression 'c' or 'ct'."""
    small, large, star, mn, mx, _ = _T[regression]
    i = min(n_vars, 4) - 1
    if stat > mx[i]:
        return 1.0
    if stat < mn[i]:
        return 0.0
    coef = small[i] if stat <= star[i] else large[i]
    return _norm_cdf(sum(c * stat ** k for k, c in enumerate(coef)))


def mackinnon_crit(n_vars: int = 1, regression: str = "c", nobs: float = math.inf):
    """Critical values at 1%, 5%, 10% for a sample of `nobs` observations."""
    row = _T[regression][5][min(n_vars, 4) - 1]
    out = []
    for c in row:
        v = c[0] + (c[1] / nobs if nobs != math.inf else 0.0) + (c[2] / nobs ** 2 if nobs != math.inf else 0.0) + (c[3] / nobs ** 3 if nobs != math.inf else 0.0)
        out.append(v)
    return out
