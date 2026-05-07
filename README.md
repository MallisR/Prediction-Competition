# ECN 372 Prediction Competition

Predict individual-level healthcare expenditures using MEPS data.

## Project Layout

- `data/`
  - `meps_clean.csv` (cleaned training data used by modeling scripts)
  - `meps_test.csv` (competition test set for final predictions)
- `R/`
  - `utils.R` - shared metric helpers (`rmsle`)
  - `01_eda.R` - EDA plots to `outputs/figures/`
  - `02_features.R` - feature engineering + train/test split
  - `03_train_evaluate.R` - reserved for additional CV harness work
  - `04_two_three_part.R` - two-part and three-part models
  - `05_ensemble.R` - blending/stacking over available model predictions
  - `06_predict.R` - final predictions for external test set
- `outputs/`
  - `figures/` - EDA figures
  - `models/` - model metrics and ensemble weights
  - `predictions/` - validation/test predictions
- `scripts/`
  - earlier preprocessing/CV utilities kept for reference

## Metric

All model evaluation uses RMSLE:

```r
rmsle <- function(y_true, y_pred) {
  y_pred <- pmax(y_pred, 0)
  sqrt(mean((log1p(y_pred) - log1p(y_true))^2))
}
```

## Quick Start

From repo root:

1. Build features
   - `make features`
2. Train models
   - Full: `make two_three_all`
   - LightGBM only: `make two_three_lgb`
   - XGBoost only: `make two_three_xgb`
3. Build ensembles
   - `make ensemble`
4. Generate final test predictions (requires `data/meps_test.csv`)
   - `make predict`

## Main Outputs

- Validation metrics:
  - `outputs/models/two_three_part_metrics.csv`
  - `outputs/models/ensemble_metrics.csv`
- Validation predictions:
  - `outputs/predictions/two_three_part_test_predictions.csv`
  - `outputs/predictions/ensemble_test_predictions.csv`
- Final competition predictions:
  - `outputs/predictions/predictions_best_model.csv`
  - `outputs/predictions/predictions_ensemble.csv`

## Notes

- `R/04_two_three_part.R` supports modes:
  - `all`
  - `xgboost_only`
  - `lightgbm_only`
- If a model package is unavailable, related outputs are set to `NA` and the run continues where possible.
