"""LendingClub out-of-time pipeline: data split, WOE scorecard baseline, monotone LightGBM challenger."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_core import RANDOM_STATE
from credit_core.datasets import (
    LC_CATEGORICAL,
    LC_MONOTONE,
    LC_NUMERIC,
    load_lendingclub,
    split_by_issue_date,
    to_frame_features,
)
from credit_core.models import lgbm_frame, make_encoder, fit_monotone_lgbm, tune_scorecard_C
from credit_core.scorecard import Scorecard

TRAIN_END = "2015-01-01"  # train: issued 2007 to 2014
TEST_START = "2015-01-01"  # test: issued 2015 to 2016
TEST_END = "2017-01-01"  # 2017 and 2018 loans are excluded: resolved outcomes are censored for them


@dataclass
class OotData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    issue_test: pd.Series
    categories: dict


def load_oot() -> OotData:
    df = load_lendingclub()
    train, test = split_by_issue_date(df, TRAIN_END, TEST_START)
    test = test[test["issue_date"] < TEST_END]
    X_tr, y_tr = to_frame_features(train)
    X_te, y_te = to_frame_features(test)
    categories = {c: sorted(X_tr[c].dropna().unique()) for c in LC_CATEGORICAL}
    return OotData(X_tr, y_tr, X_te, y_te, test["issue_date"].reset_index(drop=True), categories)


def fit_woe_scorecard(d: OotData, C: float | None = None, cv_rows: int = 200_000) -> tuple[Scorecard, float]:
    """Logistic scorecard on WOE. C is chosen by CV on a training subsample to keep runtime bounded."""
    if C is None:
        sub = d.X_train.sample(n=min(cv_rows, len(d.X_train)), random_state=RANDOM_STATE)
        C = tune_scorecard_C(LC_NUMERIC, LC_CATEGORICAL, sub, d.y_train.loc[sub.index])
    enc = make_encoder(LC_NUMERIC, LC_CATEGORICAL, n_bins=5, min_count=500, iv_min=0.02, iv_max=0.5)
    sc = Scorecard(enc, C=C).fit(d.X_train, d.y_train)
    return sc, C


def fit_challenger(d: OotData, params: dict | None = None):
    X_tr = lgbm_frame(d.X_train, d.categories)
    return fit_monotone_lgbm(X_tr, d.y_train, LC_MONOTONE, params)


def challenger_scores(model, d: OotData, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(lgbm_frame(X, d.categories))[:, 1]
