## ECN 372 — 5-fold CV with RMSLE leaderboard (tidymodels)
## Run from repo root:
##   Rscript scripts/run_cv_tidymodels.R

options(repos = c(CRAN = "https://cloud.r-project.org"))

required_pkgs <- c(
  "tidymodels",
  "readr",
  "dplyr",
  "stringr",
  "purrr",
  "tibble"
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

  scored <- pred_df %>%
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
    summarise(fold_rmsle = sqrt(mean(.data$sq_log_err, na.rm = TRUE)), .groups = "drop")

  tibble(
    cv_rmsle_mean = mean(scored$fold_rmsle, na.rm = TRUE),
    cv_rmsle_std = stats::sd(scored$fold_rmsle, na.rm = TRUE)
  )
}

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

log_glmnet_wf <- workflow() %>%
  add_recipe(common_recipe("log1p_TOTEXP")) %>%
  add_model(
    linear_reg(penalty = 0.001, mixture = 0.5) %>%
      set_engine("glmnet")
  )

log_rf_wf <- workflow() %>%
  add_recipe(common_recipe("log1p_TOTEXP")) %>%
  add_model(
    rand_forest(trees = 500, mtry = 50, min_n = 10) %>%
      set_engine("ranger") %>%
      set_mode("regression")
  )

tweedie_xgb_wf <- workflow() %>%
  add_recipe(common_recipe("TOTEXP23")) %>%
  add_model(
    boost_tree(
      trees = 800,
      tree_depth = 6,
      learn_rate = 0.05,
      min_n = 10,
      loss_reduction = 0
    ) %>%
      set_engine(
        "xgboost",
        objective = "reg:tweedie",
        tweedie_variance_power = 1.5
      ) %>%
      set_mode("regression")
  )

control <- control_resamples(save_pred = TRUE, verbose = TRUE)

fit_log_glmnet <- fit_resamples(log_glmnet_wf, resamples = folds, control = control)
fit_log_rf <- fit_resamples(log_rf_wf, resamples = folds, control = control)
fit_tweedie <- fit_resamples(tweedie_xgb_wf, resamples = folds, control = control)

results <- bind_rows(
  collect_predictions(fit_log_glmnet) %>%
    rmsle_fold_summary(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    mutate(model_name = "glmnet_log1p"),
  collect_predictions(fit_log_rf) %>%
    rmsle_fold_summary(truth_col = "log1p_TOTEXP", pred_scale = "log") %>%
    mutate(model_name = "ranger_log1p"),
  collect_predictions(fit_tweedie) %>%
    rmsle_fold_summary(truth_col = "TOTEXP23", pred_scale = "raw") %>%
    mutate(model_name = "xgboost_tweedie")
) %>%
  select(model_name, cv_rmsle_mean, cv_rmsle_std) %>%
  arrange(cv_rmsle_mean)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
readr::write_csv(results, output_path)

cat("\nCV Leaderboard (sorted by cv_rmsle_mean):\n")
print(results, n = nrow(results), width = Inf)
cat("\nSaved:", output_path, "\n")
