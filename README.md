# ECN 372 Prediction Competition

This repository predicts individual-level healthcare expenditures from MEPS data and is organized as a full, reproducible modeling pipeline.

---

## 1) Project Goal

The goal is to predict yearly total healthcare expenditures (`TOTEXP`) for each person.

Important setup assumptions used in this repo:

- Data cleaning and exclusion rules have already been applied in `data/meps_clean.csv`.
- The target is nonnegative (health spending dollars).
- Evaluation metric is RMSLE, which emphasizes relative error and handles heavy right skew.
- The spending distribution has many zeros and a long right tail, so two-part modeling is used.

---

## 2) Repository Structure

### Core directories

- `data/`
  - `meps_clean.csv`: cleaned training data (required)
  - `meps_test.csv`: external/test set for final submission (required for final prediction step)
- `R/`
  - `utils.R`: shared utility functions (especially `rmsle`)
  - `01_eda.R`: exploratory analysis and figures
  - `02_features.R`: feature engineering + train/test split
  - `03_train_evaluate.R`: placeholder for extra model experimentation
  - `04_two_three_part.R`: two-part and three-part model training/evaluation
  - `05_ensemble.R`: blends available model predictions
  - `06_predict.R`: creates final prediction CSVs for the provided test file
- `outputs/`
  - `figures/`: EDA plot artifacts
  - `models/`: validation metrics and ensemble weights
  - `predictions/`: validation predictions and final test predictions
- `scripts/`
  - legacy scripts retained for reference from earlier workflow

### Convenience runner

- `makefile` provides one-command targets for each pipeline stage.

---

## 3) Metric Definition (Used Everywhere)

All scoring uses RMSLE:

```r
rmsle <- function(y_true, y_pred) {
  y_pred <- pmax(y_pred, 0)
  sqrt(mean((log1p(y_pred) - log1p(y_true))^2))
}
```

Why this metric is appropriate here:

- `TOTEXP` is nonnegative and highly skewed.
- Large absolute-dollar errors in high spenders are moderated on log scale.
- Predicting small values for zero/low spenders is heavily penalized if incorrect.

---

## 4) Step-by-Step Pipeline

Run all commands from repo root.

### Step A: EDA

Command:

- `make eda`

What it does:

- Loads `data/meps_clean.csv`
- Detects schema variants (`TOTEXP` vs `TOTEXPyy`, `YEAR` vs `DATAYEAR`, `AGE` vs `AGELAST`)
- Saves figures to `outputs/figures/`

Generated figures:

- Histogram of raw `TOTEXP` with log10 x-axis
- Histogram of `log1p(TOTEXP)`
- Zero-spender proportion by year
- Mean and median `TOTEXP` by 10-year age groups
- Boxplot by self-reported health status (if available)
- Correlation heatmap of top 20 numeric features vs `log1p(TOTEXP)`

---

### Step B: Feature Engineering

Command:

- `make features`

What it does (`R/02_features.R`):

- Reads `data/meps_clean.csv`
- Converts year field to factor with 2019 reference level
- Creates:
  - `AGE_SQ`
  - `CHRONIC_COUNT` from detected chronic binary indicators
  - `log1p_TOTEXP`
  - `ANY_SPEND`
  - `SPEND_TIER` (0 / 1 / 2 using 0 and 3000 thresholds)
- Performs 80/20 stratified split on `SPEND_TIER`

Outputs:

- `outputs/predictions/train_features.csv`
- `outputs/predictions/test_features.csv`
- `outputs/models/feature_objects.rds`

---

### Step C: Model Training and Validation

Command options:

- `make two_three_all` (runs all available model blocks)
- `make two_three_lgb` (runs LightGBM two-part only)
- `make two_three_xgb` (runs XGBoost-only blocks)

What `R/04_two_three_part.R` does:

- Two-part model:
  - Part 1: classify `ANY_SPEND`
  - Part 2: regress `log1p_TOTEXP` on positive spenders only
  - Final prediction: `p_spend * expm1(log_pred)`
- Three-part model:
  - Part 1: zero vs nonzero
  - Part 2: among nonzero, low vs high spender
  - Part 3: separate regressors for low and high tiers
  - Final soft-weighted prediction from all three parts

Outputs:

- `outputs/models/two_three_part_metrics.csv`
- `outputs/predictions/two_three_part_test_predictions.csv`

---

### Step D: Ensemble

Command:

- `make ensemble`

What it does (`R/05_ensemble.R`):

- Reads all available `*_test_predictions.csv` files in `outputs/predictions/`
- Scores each `pred_*` column with RMSLE
- Builds:
  - equal-weight mean ensemble
  - weighted-grid ensemble (top models)
- Selects and records best blend

Outputs:

- `outputs/models/ensemble_metrics.csv`
- `outputs/models/ensemble_weights.csv`
- `outputs/predictions/ensemble_test_predictions.csv`

---

### Step E: Final Prediction Files for Submission

Command:

- `make predict`

Prerequisite:

- `data/meps_test.csv` must exist.

What `R/06_predict.R` does:

- Reapplies feature engineering logic from training pipeline
- Trains/uses the best single model based on `two_three_part_metrics.csv`
- Builds ensemble prediction using `ensemble_weights.csv` when available
- Clips predictions to nonnegative

Final outputs:

- `outputs/predictions/predictions_best_model.csv`
- `outputs/predictions/predictions_ensemble.csv`

Both files contain:

- `id`
- `TOTEXP_pred`

---

## 5) Why We Chose the Final Model

We picked the final model based on the lowest validation RMSLE.

- `TOTEXP` has many zeros and a long right tail, so a two-part model fits the data better than one single regression.
- We selected the model (or blend) that performed best on the holdout set, not just on training data.
- We only use ensemble weights when they improve validation RMSLE.

`R/06_predict.R` uses the best model from `two_three_part_metrics.csv` and uses `ensemble_weights.csv` when it helps.

---

## 6) What We Learned from Comparing Models

- Two-part models were more reliable than single models because they handled zero spenders and positive spenders separately.
- Tree-based models (LightGBM/XGBoost) captured nonlinear patterns better than simpler baselines.
- Ensembles helped when strong models made different errors; when models were too similar, gains were small.
- Validation performance was a better guide than training performance for final selection.

---

## 7) Quick Command Reference

- `make help`
- `make eda`
- `make features`
- `make two_three_all`
- `make two_three_lgb`
- `make two_three_xgb`
- `make ensemble`
- `make predict`
- `make pipeline` (features -> full two/three-part run -> ensemble)

---

## 8) Suggested End-to-End Run Order

For a complete run from clean data to submission files:

1. `make features`
2. `make two_three_lgb` (or `make two_three_all`)
3. `make ensemble`
4. Place `data/meps_test.csv`
5. `make predict`

Then inspect:

- `outputs/models/two_three_part_metrics.csv`
- `outputs/models/ensemble_metrics.csv`
- `outputs/predictions/predictions_best_model.csv`
- `outputs/predictions/predictions_ensemble.csv`

