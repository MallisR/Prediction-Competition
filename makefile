.PHONY: help eda features two_three_all two_three_lgb two_three_xgb ensemble predict pipeline

help:
	@echo "Available targets:"
	@echo "  make eda             - Run EDA plots (R/01_eda.R)"
	@echo "  make features        - Build engineered train/test features (R/02_features.R)"
	@echo "  make two_three_all   - Run two-part + three-part models (R/04_two_three_part.R all)"
	@echo "  make two_three_lgb   - Run LightGBM two-part only"
	@echo "  make two_three_xgb   - Run XGBoost models only"
	@echo "  make ensemble        - Run stacking/weighted ensembles (R/05_ensemble.R)"
	@echo "  make predict         - Generate final test predictions (R/06_predict.R)"
	@echo "  make pipeline        - Run features -> models -> ensemble"

eda:
	Rscript R/01_eda.R

features:
	Rscript R/02_features.R

two_three_all:
	Rscript R/04_two_three_part.R all

two_three_lgb:
	Rscript R/04_two_three_part.R lightgbm_only

two_three_xgb:
	Rscript R/04_two_three_part.R xgboost_only

ensemble:
	Rscript R/05_ensemble.R

predict:
	Rscript R/06_predict.R

pipeline: features two_three_all ensemble
