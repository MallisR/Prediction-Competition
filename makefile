SHELL := /bin/bash

.PHONY: help doctor check_r setup_dirs install_core install_xgboost install_lightgbm install_all \
	check_train_data check_test_data \
	eda features two_three_all two_three_lgb two_three_xgb ensemble predict \
	pipeline_lgb pipeline_all full_lgb full_all

help:
	@echo "============================================================"
	@echo "MEPS Prediction Project - Reproducible Make Targets"
	@echo "============================================================"
	@echo ""
	@echo "1) FIRST-TIME SETUP"
	@echo "  make doctor        : verify R and Rscript are available"
	@echo "  make install_core  : install core R packages (dplyr/readr/ggplot2/etc.)"
	@echo "  make install_xgboost"
	@echo "  make install_lightgbm"
	@echo "  make install_all   : core + xgboost + lightgbm"
	@echo ""
	@echo "2) DATA CHECKS"
	@echo "  make check_train_data : ensure data/meps_clean.csv exists"
	@echo "  make check_test_data  : ensure data/meps_test.csv exists"
	@echo ""
	@echo "3) PIPELINE STAGES"
	@echo "  make eda           : run EDA plots (outputs/figures)"
	@echo "  make features      : run feature engineering split"
	@echo "  make two_three_all : run XGBoost + LightGBM sections"
	@echo "  make two_three_lgb : run LightGBM two-part only"
	@echo "  make two_three_xgb : run XGBoost sections only"
	@echo "  make ensemble      : run blending/stacking"
	@echo "  make predict       : create final submission CSVs"
	@echo ""
	@echo "4) ONE-COMMAND RUNS"
	@echo "  make pipeline_lgb  : features -> LightGBM two-part -> ensemble"
	@echo "  make pipeline_all  : features -> full two/three-part -> ensemble"
	@echo "  make full_lgb      : pipeline_lgb + final predict (requires meps_test.csv)"
	@echo "  make full_all      : pipeline_all + final predict (requires meps_test.csv)"
	@echo ""
	@echo "Recommended for grading (fast + best current score path):"
	@echo "  make doctor"
	@echo "  make install_all"
	@echo "  make pipeline_lgb"
	@echo "  make check_test_data"
	@echo "  make predict"
	@echo "============================================================"

doctor:
	@command -v R >/dev/null 2>&1 || { echo "ERROR: R is not installed or not on PATH."; exit 1; }
	@command -v Rscript >/dev/null 2>&1 || { echo "ERROR: Rscript is not installed or not on PATH."; exit 1; }
	@echo "R version:"
	@R --version | head -n 1
	@echo "Rscript path: $$(command -v Rscript)"
	@echo "Doctor check passed."

check_r: doctor

setup_dirs:
	@mkdir -p outputs/figures outputs/models outputs/predictions
	@echo "Ensured output directories exist."

install_core:
	@R -q -e 'install.packages(c("dplyr","readr","ggplot2","tidyr","scales","rsample","tibble"), repos="https://cloud.r-project.org")'

install_xgboost:
	@R -q -e 'install.packages("xgboost", repos="https://cloud.r-project.org")'

install_lightgbm:
	@R -q -e 'install.packages("lightgbm", repos="https://cloud.r-project.org")'

install_all: install_core install_xgboost install_lightgbm

check_train_data:
	@test -f data/meps_clean.csv || { echo "ERROR: data/meps_clean.csv is missing."; exit 1; }
	@echo "Found data/meps_clean.csv"

check_test_data:
	@test -f data/meps_test.csv || { echo "ERROR: data/meps_test.csv is missing."; exit 1; }
	@echo "Found data/meps_test.csv"

eda: check_train_data setup_dirs
	@echo "Running EDA..."
	Rscript R/01_eda.R

features: check_train_data setup_dirs
	@echo "Running feature engineering..."
	Rscript R/02_features.R

two_three_all: setup_dirs
	@echo "Running full two/three-part models (all)..."
	Rscript R/04_two_three_part.R all

two_three_lgb: setup_dirs
	@echo "Running LightGBM-only two-part model..."
	Rscript R/04_two_three_part.R lightgbm_only

two_three_xgb: setup_dirs
	@echo "Running XGBoost-only model blocks..."
	Rscript R/04_two_three_part.R xgboost_only

ensemble: setup_dirs
	@echo "Running ensemble..."
	Rscript R/05_ensemble.R

predict: check_test_data setup_dirs
	@echo "Generating final prediction CSVs..."
	Rscript R/06_predict.R
	@echo "Done. Check outputs/predictions/predictions_best_model.csv and predictions_ensemble.csv"

pipeline_lgb: features two_three_lgb ensemble
	@echo "LightGBM pipeline complete."

pipeline_all: features two_three_all ensemble
	@echo "Full modeling pipeline complete."

full_lgb: pipeline_lgb predict
	@echo "Full LightGBM run complete (including final test predictions)."

full_all: pipeline_all predict
	@echo "Full all-model run complete (including final test predictions)."
