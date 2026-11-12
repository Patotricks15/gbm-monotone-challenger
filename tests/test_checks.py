import numpy as np
import pandas as pd

from credit_core.models import fit_monotone_lgbm, lgbm_frame
from gbm_monotone.checks import partial_dependence, violates_monotone


def test_violates_monotone_detects_direction():
    up = np.array([0.1, 0.2, 0.3])
    assert not violates_monotone(up, +1)
    assert violates_monotone(up, -1)
    assert not violates_monotone(up[::-1], -1)


def test_constrained_lgbm_is_monotone_in_good_probability():
    rng = np.random.default_rng(0)
    n = 3000
    income = rng.normal(size=n)
    noise = rng.normal(size=n)
    p_good = 1 / (1 + np.exp(-(1.2 * income)))
    y = pd.Series((rng.uniform(size=n) < p_good).astype(int))
    X = pd.DataFrame({"annual_inc": income, "noise": noise})
    # Risk-direction constraint -1 for income means higher income gives higher good probability.
    model = fit_monotone_lgbm(X, y, {"annual_inc": -1}, {"n_estimators": 100, "min_child_samples": 20})
    grid = np.linspace(-2, 2, 25)
    curve = partial_dependence(lambda d: model.predict_proba(d)[:, 1], X, "annual_inc", grid)
    assert not violates_monotone(curve, +1)


def test_lgbm_frame_uses_fixed_categories():
    X = pd.DataFrame({"purpose": ["car", None, "other"], "x": [1.0, 2.0, 3.0]})
    out = lgbm_frame(X, {"purpose": ["car", "other", "debt"]})
    assert list(out["purpose"].cat.categories) == ["car", "other", "debt"]
