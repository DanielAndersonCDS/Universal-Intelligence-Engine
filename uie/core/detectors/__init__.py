"""Detector registry. Add a module with detect(ctx) -> list[Fact] and list it here.
GROUPABLE detectors also run inside each group when a group column is chosen."""
from . import quality, derived, outliers, correlation, trends, segments
from .base import Context

# (label, fn). Optional modules are appended below as they are imported.
REGISTRY = [
    ("Data quality", quality.detect),
    ("Derived columns", derived.detect),
    ("Outliers", outliers.detect),
    ("Correlations", correlation.detect),
    ("Trends and shifts", trends.detect),
    ("Segment concentration", segments.detect),
]
GROUPABLE = {"Outliers", "Correlations", "Trends and shifts"}


def register(label, fn, groupable=False):
    REGISTRY.append((label, fn))
    if groupable:
        GROUPABLE.add(label)


for _mod, _label, _g in (("returns", "Returns and volatility", True), ("amounts", "Amount patterns", False),
                         ("txdupes", "Near-duplicate transactions", False), ("periods", "Period variance and gaps", True),
                         ("leakage", "Target leakage and label balance", False), ("features", "Feature health", False)):
    try:
        _m = __import__(f"{__name__}.{_mod}", fromlist=["detect"])
    except ModuleNotFoundError as e:           # module not present yet
        if e.name != f"{__name__}.{_mod}":
            raise
        continue
    register(_label, _m.detect, _g)
