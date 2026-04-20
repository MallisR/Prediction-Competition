
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
- Model: TBD

## Evaluation Metric
- RMSLE
## HOW TO DOWNLOAD DATA
## 