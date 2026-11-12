# gbm-monotone-challenger

LightGBM challenger with monotone constraints, compared with a WOE logistic scorecard on LendingClub. Validation is out-of-time.

## Results (out-of-time test: loans issued 2015 to 2016, n = 668,651)

| Model | AUC | Gini | KS | ECE |
|---|---|---|---|---|
| WOE scorecard | 0.690 | 0.380 | 0.270 | 0.048 |
| LightGBM monotone | 0.715 | 0.430 | 0.309 | 0.044 |

The Gini gain is +0.050. The challenger is higher in all eight test quarters. Both models lose Gini in the later 2016 quarters.

Monotone constraints hold on all seven constrained features (partial dependence check in the notebook).

Notebook: [notebooks/02_gbm_monotone_challenger.ipynb](notebooks/02_gbm_monotone_challenger.ipynb). Figures in [reports/figures/](reports/figures/).

## Method

- Split: train on loans issued before 2015 (453,809 rows). Test on 2015 to 2016 (668,651 rows). Loans issued 2017 and later are excluded because their resolved outcomes are censored.
- Target: 1 = Fully Paid, 0 = Charged Off or Default. Loans still in progress are dropped.
- Excluded: post-origination fields, and `grade`, `sub_grade`, `int_rate`, which are the lender's own risk price.
- Baseline: WOE logistic scorecard. `C` chosen by CV on a 200k subsample (picked 0.1).
- Challenger: LightGBM with the monotone constraints in `credit_core.datasets.LC_MONOTONE`, applied as risk direction.

## Limitations

- The resolved-only filter keeps shorter loans that finish sooner, so the sample is not a random sample of all originations.
- Test is a 2-year window, not a full cohort.

## Run

```bash
pip install -e ".[dev]"
make test
make notebook
```

Data is read through `credit_core.datasets.load_lendingclub` from `CREDIT_DATA_DIR`. See [data/README.md](data/README.md).
