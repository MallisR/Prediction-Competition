# ECN 372 Prediction Competition

## Objective
Predict individual-level healthcare expenditures using MEPS data.

## Data
- MEPS Full-Year Consolidated Files (2019–2023)

## Target Variable
- TOTEXPyy (total healthcare expenditure)

## Restrictions
- Excludes all utilization and expenditure variables
- Excludes survey weights

## Approach
- Data cleaning: handle missing codes (-1, -7, -8)
- Feature selection from demographics, health, and insurance
- Log transformation of target (log(1 + y))
- Model: compare candidates with cross-validation; document the final choice in the write-up

## Evaluation Metric
- RMSLE (root mean squared log error)

## Reproducible Team Workflow
- Place the raw MEPS workbook at `raw_data/h251.xlsx` (team-standard path).
- From repo root, run:
  - `Rscript scripts/run_pipeline.R`

This runs the full pipeline in a fixed order:
1. `scripts/Filter data`
2. `scripts/explore_data.R`
3. `scripts/run_cv_tidymodels.R`

Expected outputs:
- `filtered_data.csv`
- `filtered_model_ready.csv`
- `figures/column_audit.tsv`
- `figures/eda_target.pdf`
- `outputs/cv_results.csv`

## Notes
- RMSLE scoring clips negative predictions to zero before computing `log1p` error.
- CV script uses 5-fold CV stratified by `SPEND_TIER`.
