## ECN 372 — full 5-fold CV RMSLE leaderboard (tidymodels + GAM)
## Run from repo root:
##   Rscript scripts/run_cv_tidymodels.R

options(repos = c(CRAN = "https://cloud.r-project.org"))

required_pkgs <- c(
  "tidymodels",
  "readr",
  "dplyr",
  "stringr",
  "purrr",
  "tibble",
  "mgcv",
  "bonsai"
)

for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

suppressPackageStartupMessages({
  library(tidymodels)
  library(readr)
  library(dplyr)
  library(stringr)
  library(purrr)
  library(tibble)
  library(mgcv)
  library(bonsai)
})

set.seed(372)

repo_root <- "."
data_path <- file.path(repo_root, "filtered_data.csv")
audit_path <- file.path(repo_root, "figures", "column_audit.tsv")
output_dir <- file.path(repo_root, "outputs")
output_path <- file.path(output_dir, "cv_results.csv")

if (!file.exists(data_path)) {
  stop("Missing filtered_data.csv at repo root.")
}
if (!file.exists(audit_path)) {
  stop("Missing figures/column_audit.tsv.")
}

df <- readr::read_csv(data_path, show_col_types = FALSE)
audit <- readr::read_tsv(audit_path, show_col_types = FALSE)

target_col <- audit %>%
  filter(role == "target") %>%
  pull(name) %>%
  unique()

if (length(target_col) != 1L) {
  stop("Expected exactly one target in column_audit.tsv.")
}
target_col <- target_col[[1]]

if (!target_col %in% names(df)) {
  stop("Target column from audit not found in filtered_data.csv: ", target_col)
}

allowed_predictors <- audit %>%
  filter(role == "allowed_predictor") %>%
  pull(name) %>%
  unique()

allowed_predictors <- intersect(allowed_predictors, names(df))
if (!length(allowed_predictors)) {
  stop("No allowed predictors found in filtered_data.csv.")
}

model_df <- df %>%
  transmute(
    TOTEXP23 = as.numeric(.data[[target_col]]),
    across(all_of(allowed_predictors), ~ .x)
  ) %>%
  mutate(
    TOTEXP23 = if_else(is.na(TOTEXP23), 0, TOTEXP23),
    TOTEXP23 = pmax(TOTEXP23, 0),
    log1p_TOTEXP = log1p(TOTEXP23),
    SPEND_TIER = case_when(
      TOTEXP23 == 0 ~ "zero",
      TOTEXP23 <= quantile(TOTEXP23[TOTEXP23 > 0], probs = 1 / 3, na.rm = TRUE) ~ "low",
      TOTEXP23 <= quantile(TOTEXP23[TOTEXP23 > 0], probs = 2 / 3, na.rm = TRUE) ~ "mid",
      TRUE ~ "high"
    ),
    SPEND_TIER = factor(SPEND_TIER, levels = c("zero", "low", "mid", "high"))
  )

folds <- rsample::vfold_cv(model_df, v = 5, strata = SPEND_TIER)

rmsle_fold_summary <- function(pred_df, truth_col, pred_col = ".pred", pred_scale = c("raw", "log")) {
  pred_scale <- match.arg(pred_scale)
  pred_df %>%
    mutate(
      truth_raw = if (truth_col == "log1p_TOTEXP") expm1(.data[[truth_col]]) else .data[[truth_col]],
      pred_raw = if (pred_scale == "log") expm1(.data[[pred_col]]) else .data[[pred_col]]
    ) %>%
    mutate(
      truth_raw = pmax(.data$truth_raw, 0),
      pred_raw = pmax(.data$pred_raw, 0),
      sq_log_err = (log1p(.data$pred_raw) - log1p(.data$truth_raw))^2
    ) %>%
    group_by(id) %>%
    summarise(fold_rmsle = sqrt(mean(.data$sq_log_err, na.rm = TRUE)), .groups = "drop") %>%
    summarise(
      cv_rmsle_mean = mean(.data$fold_rmsle, na.rm = TRUE),
      cv_rmsle_std = stats::sd(.data$fold_rmsle, na.rm = TRUE)
    )
}

rmsle_fold_summary_by_config <- function(pred_df, truth_col, pred_scale = c("raw", "log")) {
  pred_scale <- match.arg(pred_scale)
  tune_cols <- intersect(
    c("penalty", "mixture", "mtry", "min_n", "trees", "tree_depth", "learn_rate"),
    names(pred_df)
  )

  pred_df %>%
    mutate(
      truth_raw = if (truth_col == "log1p_TOTEXP") expm1(.data[[truth_col]]) else .data[[truth_col]],
      pred_raw = if (pred_scale == "log") expm1(.data$.pred) else .data$.pred
    ) %>%
    mutate(
      truth_raw = pmax(.data$truth_raw, 0),
      pred_raw = pmax(.data$pred_raw, 0),
      sq_log_err = (log1p(.data$pred_raw) - log1p(.data$truth_raw))^2
    ) %>%
    group_by(across(all_of(c("id", tune_cols)))) %>%
    summarise(fold_rmsle = sqrt(mean(.data$sq_log_err, na.rm = TRUE)), .groups = "drop") %>%
    group_by(across(all_of(tune_cols))) %>%
    summarise(
      cv_rmsle_mean = mean(.data$fold_rmsle, na.rm = TRUE),
      cv_rmsle_std = stats::sd(.data$fold_rmsle, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(.data$cv_rmsle_mean)
}

continuous_predictors <- model_df %>%
  select(all_of(allowed_predictors)) %>%
  select(where(is.numeric)) %>%
  names()

key_continuous_predictors <- setdiff(continuous_predictors, "AGE")
if (length(key_continuous_predictors)) {
  variances <- model_df %>%
    summarise(across(all_of(key_continuous_predictors), ~ stats::var(.x, na.rm = TRUE))) %>%
    unlist(use.names = TRUE)
  key_continuous_predictors <- names(sort(variances, decreasing = TRUE))[seq_len(min(3L, length(variances)))]
}

ns_terms <- unique(c(intersect("AGE", continuous_predictors), key_continuous_predictors))
poly_terms <- unique(c(intersect("AGE", continuous_predictors), key_continuous_predictors))
bmi_candidates <- names(model_df)[stringr::str_detect(names(model_df), regex("BMI", ignore_case = TRUE))]
bmi_smooth <- intersect(bmi_candidates, continuous_predictors)[1]

common_recipe <- function(outcome) {
  recipes::recipe(stats::as.formula(paste(outcome, "~ .")), data = model_df) %>%
    recipes::update_role(tidyselect::all_of("SPEND_TIER"), new_role = "stratification_only") %>%
    recipes::step_rm(recipes::has_role("stratification_only")) %>%
    recipes::step_zv(recipes::all_predictors()) %>%
    recipes::step_impute_median(recipes::all_numeric_predictors()) %>%
    recipes::step_impute_mode(recipes::all_nominal_predictors()) %>%
    recipes::step_novel(recipes::all_nominal_predictors()) %>%
    recipes::step_unknown(recipes::all_nominal_predictors()) %>%
    recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = TRUE)
}

natural_spline_recipe <- function(outcome) {
  rec <- common_recipe(outcome)
  if (length(ns_terms)) {
    rec <- rec %>% recipes::step_ns(tidyselect::all_of(ns_terms), deg_free = 4)
  }
  rec
}

poly_recipe <- function(outcome) {
  rec <- common_recipe(outcome)
  if (length(poly_terms)) {
    rec <- rec %>% recipes::step_poly(tidyselect::all_of(poly_terms), degree = 2)
  }
  rec
}

wf_ols <- workflow() %>%
  add_recipe(common_recipe("log1p_TOTEXP")) %>%
  add_model(linear_reg() %>% set_engine("lm"))

wf_ridge <- workflow() %>%
  add_recipe(common_recipe("log1p_TOTEXP")) %>%
  add_model(linear_reg(penalty = tune(), mixture = 0) %>% set_engine("glmnet"))

wf_lasso <- workflow() %>%
  add_recipe(common_recipe("log1p_TOTEXP")) %>%
  add_model(linear_reg(penalty = tune(), mixture = 1) %>% set_engine("glmnet"))

wf_elasticnet <- workflow() %>%
  add_recipe(common_recipe("log1p_TOTEXP")) %>%
  add_model(linear_reg(penalty = tune(), mixture = tune()) %>% set_engine("glmnet"))

wf_rf <- workflow() %>%
  add_recipe(common_recipe("log1p_TOTEXP")) %>%
  add_model(
    rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
      set_engine("ranger") %>%
      set_mode("regression")
  )

wf_xgb_log <- workflow() %>%
  add_recipe(common_recipe("log1p_TOTEXP")) %>%
  add_model(
    boost_tree(
      trees = tune(),
      tree_depth = tune(),
      learn_rate = tune(),
      min_n = tune()
    ) %>%
      set_engine("xgboost") %>%
      set_mode("regression")
  )

wf_xgb_tweedie <- workflow() %>%
  add_recipe(common_recipe("TOTEXP23")) %>%
  add_model(
    boost_tree(
      trees = tune(),
      tree_depth = tune(),
      learn_rate = tune(),
      min_n = tune()
    ) %>%
      set_engine("xgboost", objective = "reg:tweedie", tweedie_variance_power = 1.5) %>%
      set_mode("regression")
  )

wf_lgbm_log <- workflow() %>%
  add_recipe(common_recipe("log1p_TOTEXP")) %>%
  add_model(
    boost_tree(
      trees = tune(),
      tree_depth = tune(),
      learn_rate = tune(),
      min_n = tune()
    ) %>%
      set_engine("lightgbm", objective = "regression") %>%
      set_mode("regression")
  )

wf_lgbm_tweedie <- workflow() %>%
  add_recipe(common_recipe("TOTEXP23")) %>%
  add_model(
    boost_tree(
      trees = tune(),
      tree_depth = tune(),
      learn_rate = tune(),
      min_n = tune()
    ) %>%
      set_engine("lightgbm", objective = "tweedie", tweedie_variance_power = 1.5) %>%
      set_mode("regression")
  )

wf_ns <- workflow() %>%
  add_recipe(natural_spline_recipe("log1p_TOTEXP")) %>%
  add_model(linear_reg() %>% set_engine("lm"))

wf_poly <- workflow() %>%
  add_recipe(poly_recipe("log1p_TOTEXP")) %>%
  add_model(linear_reg() %>% set_engine("lm"))

control <- control_resamples(save_pred = TRUE, verbose = TRUE)
control_grid <- control_grid(save_pred = TRUE, verbose = TRUE)

rf_param <- parameters(wf_rf) %>%
  update(mtry = mtry(c(5L, min(100L, max(10L, length(allowed_predictors))))))

default_grid <- 12

fit_ols <- fit_resamples(wf_ols, resamples = folds, control = control)
fit_ns <- fit_resamples(wf_ns, resamples = folds, control = control)
fit_poly <- fit_resamples(wf_poly, resamples = folds, control = control)

tune_ridge <- tune_grid(wf_ridge, resamples = folds, grid = default_grid, control = control_grid)
tune_lasso <- tune_grid(wf_lasso, resamples = folds, grid = default_grid, control = control_grid)
tune_elasticnet <- tune_grid(wf_elasticnet, resamples = folds, grid = default_grid, control = control_grid)
tune_rf <- tune_grid(wf_rf, resamples = folds, grid = default_grid, param_info = rf_param, control = control_grid)
tune_xgb_log <- tune_grid(wf_xgb_log, resamples = folds, grid = default_grid, control = control_grid)
tune_xgb_tweedie <- tune_grid(wf_xgb_tweedie, resamples = folds, grid = default_grid, control = control_grid)
tune_lgbm_log <- tune_grid(wf_lgbm_log, resamples = folds, grid = default_grid, control = control_grid)
tune_lgbm_tweedie <- tune_grid(wf_lgbm_tweedie, resamples = folds, grid = default_grid, control = control_grid)

eval_gam_cv <- function(data, folds_obj) {
  numeric_predictors <- data %>% select(where(is.numeric)) %>% names()
  smooth_terms <- intersect("AGE", numeric_predictors)
  if (!is.na(bmi_smooth)) {
    smooth_terms <- c(smooth_terms, bmi_smooth)
  }
  smooth_terms <- unique(smooth_terms)

  factor_terms <- data %>%
    select(where(~ is.character(.x) || is.factor(.x))) %>%
    select(where(~ dplyr::n_distinct(.x, na.rm = TRUE) <= 20)) %>%
    names()
  factor_terms <- setdiff(factor_terms, "SPEND_TIER")

  formula_terms <- c(
    paste0("s(", smooth_terms, ")"),
    factor_terms
  )
  if (!length(formula_terms)) {
    stop("GAM formula has no predictors after filtering.")
  }
  gam_formula <- stats::as.formula(paste("log1p_TOTEXP ~", paste(formula_terms, collapse = " + ")))

  fold_scores <- purrr::map_dfr(seq_along(folds_obj$splits), function(i) {
    split <- folds_obj$splits[[i]]
    train_data <- rsample::analysis(split)
    test_data <- rsample::assessment(split)

    gam_fit <- mgcv::gam(gam_formula, data = train_data, family = gaussian())
    pred_log <- stats::predict(gam_fit, newdata = test_data, type = "response")

    score <- tibble(
      id = folds_obj$id[[i]],
      truth_raw = pmax(expm1(test_data$log1p_TOTEXP), 0),
      pred_raw = pmax(expm1(pred_log), 0)
    ) %>%
      mutate(fold_rmsle = sqrt(mean((log1p(.data$pred_raw) - log1p(.data$truth_raw))^2, na.rm = TRUE))) %>%
      summarise(id = first(.data$id), fold_rmsle = first(.data$fold_rmsle))

    score
  })

  fold_scores %>%
    summarise(
      cv_rmsle_mean = mean(.data$fold_rmsle, na.rm = TRUE),
      cv_rmsle_std = stats::sd(.data$fold_rmsle, na.rm = TRUE)
    ) %>%
    mutate(model_name = "gam_log1p")
}

result_rows <- list(
  collect_predictions(fit_ols) %>%
    rmsle_fold_summary(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    mutate(model_name = "ols_lm_log1p"),
  collect_predictions(fit_ns) %>%
    rmsle_fold_summary(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    mutate(model_name = "natural_spline_lm_log1p"),
  collect_predictions(fit_poly) %>%
    rmsle_fold_summary(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    mutate(model_name = "polynomial_lm_log1p"),
  collect_predictions(tune_ridge) %>%
    rmsle_fold_summary_by_config(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    slice(1) %>%
    mutate(model_name = "ridge_glmnet_log1p"),
  collect_predictions(tune_lasso) %>%
    rmsle_fold_summary_by_config(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    slice(1) %>%
    mutate(model_name = "lasso_glmnet_log1p"),
  collect_predictions(tune_elasticnet) %>%
    rmsle_fold_summary_by_config(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    slice(1) %>%
    mutate(model_name = "elasticnet_glmnet_log1p"),
  collect_predictions(tune_rf) %>%
    rmsle_fold_summary_by_config(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    slice(1) %>%
    mutate(model_name = "random_forest_ranger_log1p"),
  collect_predictions(tune_xgb_log) %>%
    rmsle_fold_summary_by_config(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    slice(1) %>%
    mutate(model_name = "xgboost_log1p"),
  collect_predictions(tune_xgb_tweedie) %>%
    rmsle_fold_summary_by_config(truth_col = "TOTEXP23", pred_scale = "raw") %>%
    slice(1) %>%
    mutate(model_name = "xgboost_tweedie_raw"),
  collect_predictions(tune_lgbm_log) %>%
    rmsle_fold_summary_by_config(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    slice(1) %>%
    mutate(model_name = "lightgbm_log1p"),
  collect_predictions(tune_lgbm_tweedie) %>%
    rmsle_fold_summary_by_config(truth_col = "TOTEXP23", pred_scale = "raw") %>%
    slice(1) %>%
    mutate(model_name = "lightgbm_tweedie_raw"),
  eval_gam_cv(model_df, folds)
)

results <- bind_rows(result_rows) %>%
  select(model_name, cv_rmsle_mean, cv_rmsle_std) %>%
  arrange(cv_rmsle_mean)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
readr::write_csv(results, output_path)

cat("\nCV Leaderboard (sorted by cv_rmsle_mean):\n")
print(results, n = nrow(results), width = Inf)
cat("\nSaved:", output_path, "\n")
