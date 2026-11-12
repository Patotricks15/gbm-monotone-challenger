"""Monotonicity check for a fitted model: partial dependence along each constrained feature."""

import numpy as np
import pandas as pd


def partial_dependence(predict_fn, X: pd.DataFrame, feature: str, grid: np.ndarray, sample: int = 500, seed: int = 0):
    """Average predicted probability when `feature` is set to each grid value, over a random sample."""
    rng = np.random.default_rng(seed)
    base = X.iloc[rng.choice(len(X), size=min(sample, len(X)), replace=False)].copy()
    values = []
    for v in grid:
        tmp = base.copy()
        tmp[feature] = v
        values.append(float(np.mean(predict_fn(tmp))))
    return np.array(values)


def violates_monotone(curve: np.ndarray, direction: int, tol: float = 1e-9) -> bool:
    """True if the curve moves against `direction` (+1 increasing, -1 decreasing) anywhere."""
    diffs = np.diff(curve)
    if direction > 0:
        return bool(np.any(diffs < -tol))
    return bool(np.any(diffs > tol))
