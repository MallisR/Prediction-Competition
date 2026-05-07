set.seed(42)
source("R/utils.R")

options(repos = c(CRAN = "https://cloud.r-project.org"))
required_pkgs <- c("dplyr", "readr", "xgboost")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs)) {
  stop(
    "Missing required packages: ",
    paste(missing_pkgs, collapse = ", "),
    ". Install them first, e.g. install.packages(c(\"",
    paste(missing_pkgs, collapse = "\", \""),
    "\"))"
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(xgboost)
})

train_path <- "outputs/predictions/train_features.csv"
test_path <- "outputs/predictions/test_features.csv"
if (!file.exists(train_path) || !file.exists(test_path)) {
  stop("Missing train/test feature files. Run R/02_features.R first.")
}

dir.create("outputs/models", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/predictions", recursive = TRUE, showWarnings = FALSE)

train_df <- readr::read_csv(train_path, show_col_types = FALSE)
test_df <- readr::read_csv(test_path, show_col_types = FALSE)

target_col <- if ("TOTEXP" %in% names(train_df)) {
  "TOTEXP"
} else {
  totexp_candidates <- grep("^TOTEXP[0-9]{2}$", names(train_df), value = TRUE)
  if (!length(totexp_candidates)) {
    stop("Expected target column TOTEXP or TOTEXPyy in train/test features.")
  }
  sort(totexp_candidates)[length(totexp_candidates)]
}

needed_cols <- c(target_col, "ANY_SPEND", "SPEND_TIER", "log1p_TOTEXP")
missing_needed <- setdiff(needed_cols, names(train_df))
if (length(missing_needed)) {
  stop("Missing required columns in feature files: ", paste(missing_needed, collapse = ", "))
}

if ("YEAR" %in% names(train_df)) {
  year_levels <- sort(unique(c("2019", as.character(train_df$YEAR), as.character(test_df$YEAR))))
  train_df$YEAR <- factor(as.character(train_df$YEAR), levels = year_levels)
  test_df$YEAR <- factor(as.character(test_df$YEAR), levels = year_levels)
}

train_df$SPEND_TIER <- factor(as.character(train_df$SPEND_TIER), levels = c("0", "1", "2"))
test_df$SPEND_TIER <- factor(as.character(test_df$SPEND_TIER), levels = c("0", "1", "2"))
train_df$ANY_SPEND <- as.integer(train_df$ANY_SPEND)
test_df$ANY_SPEND <- as.integer(test_df$ANY_SPEND)

feature_cols <- setdiff(names(train_df), c(target_col, "log1p_TOTEXP", "ANY_SPEND", "SPEND_TIER"))

make_xy <- function(df, cols) {
  design <- model.matrix(~ . - 1, data = df[, cols, drop = FALSE])
  design <- as.matrix(design)
  storage.mode(design) <- "numeric"
  design
}

x_train <- make_xy(train_df, feature_cols)
x_test <- make_xy(test_df, feature_cols)
y_test <- test_df[[target_col]]

# -----------------------------
# Two-part model (XGBoost)
# -----------------------------
part1_xgb <- xgboost::xgboost(
  data = x_train,
  label = train_df$ANY_SPEND,
  objective = "binary:logistic",
  nrounds = 300,
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

pos_idx <- which(train_df[[target_col]] > 0)
part2_xgb <- xgboost::xgboost(
  data = x_train[pos_idx, , drop = FALSE],
  label = train_df$log1p_TOTEXP[pos_idx],
  objective = "reg:squarederror",
  nrounds = 350,
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

p_spend_xgb <- as.numeric(predict(part1_xgb, x_test))
log_pred_xgb <- as.numeric(predict(part2_xgb, x_test))
y_pred_twopart_xgb <- p_spend_xgb * expm1(log_pred_xgb)
y_pred_twopart_xgb <- pmax(y_pred_twopart_xgb, 0)
rmsle_twopart_xgb <- rmsle(y_test, y_pred_twopart_xgb)

# -----------------------------
# Two-part model (LightGBM)
# -----------------------------
lightgbm_available <- requireNamespace("lightgbm", quietly = TRUE)
rmsle_twopart_lgb <- NA_real_
y_pred_twopart_lgb <- rep(NA_real_, nrow(test_df))

if (lightgbm_available) {
  lgb_train_cls <- lightgbm::lgb.Dataset(data = x_train, label = train_df$ANY_SPEND)
  part1_lgb <- lightgbm::lgb.train(
    params = list(
      objective = "binary",
      learning_rate = 0.05,
      num_leaves = 63,
      feature_fraction = 0.8,
      bagging_fraction = 0.8,
      bagging_freq = 1
    ),
    data = lgb_train_cls,
    nrounds = 350,
    verbose = -1
  )

  lgb_train_reg <- lightgbm::lgb.Dataset(
    data = x_train[pos_idx, , drop = FALSE],
    label = train_df$log1p_TOTEXP[pos_idx]
  )
  part2_lgb <- lightgbm::lgb.train(
    params = list(
      objective = "regression",
      learning_rate = 0.05,
      num_leaves = 63,
      feature_fraction = 0.8,
      bagging_fraction = 0.8,
      bagging_freq = 1
    ),
    data = lgb_train_reg,
    nrounds = 400,
    verbose = -1
  )

  p_spend_lgb <- as.numeric(predict(part1_lgb, x_test))
  log_pred_lgb <- as.numeric(predict(part2_lgb, x_test))
  y_pred_twopart_lgb <- p_spend_lgb * expm1(log_pred_lgb)
  y_pred_twopart_lgb <- pmax(y_pred_twopart_lgb, 0)
  rmsle_twopart_lgb <- rmsle(y_test, y_pred_twopart_lgb)
} else {
  message("Package `lightgbm` not installed; skipping two-part LightGBM fit.")
}

# -----------------------------
# Three-part model (XGBoost)
# -----------------------------
part1_three_xgb <- xgboost::xgboost(
  data = x_train,
  label = train_df$ANY_SPEND,
  objective = "binary:logistic",
  nrounds = 300,
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

nonzero_idx <- which(train_df$ANY_SPEND == 1L & !is.na(train_df$SPEND_TIER))
nonzero_binary <- as.integer(train_df$SPEND_TIER[nonzero_idx] == "2")

part2_three_xgb <- xgboost::xgboost(
  data = x_train[nonzero_idx, , drop = FALSE],
  label = nonzero_binary,
  objective = "binary:logistic",
  nrounds = 250,
  max_depth = 5,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

low_idx <- which(train_df$SPEND_TIER == "1")
high_idx <- which(train_df$SPEND_TIER == "2")

part3_low_xgb <- xgboost::xgboost(
  data = x_train[low_idx, , drop = FALSE],
  label = train_df$log1p_TOTEXP[low_idx],
  objective = "reg:squarederror",
  nrounds = 300,
  max_depth = 5,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

part3_high_xgb <- xgboost::xgboost(
  data = x_train[high_idx, , drop = FALSE],
  label = train_df$log1p_TOTEXP[high_idx],
  objective = "reg:squarederror",
  nrounds = 350,
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

p_any <- as.numeric(predict(part1_three_xgb, x_test))
p_high <- as.numeric(predict(part2_three_xgb, x_test))
pred_low <- expm1(as.numeric(predict(part3_low_xgb, x_test)))
pred_high <- expm1(as.numeric(predict(part3_high_xgb, x_test)))

y_pred_threepart_xgb <- p_any * (p_high * pred_high + (1 - p_high) * pred_low)
y_pred_threepart_xgb <- pmax(y_pred_threepart_xgb, 0)
rmsle_threepart_xgb <- rmsle(y_test, y_pred_threepart_xgb)

metrics <- dplyr::tibble(
  model = c("two_part_xgboost", "two_part_lightgbm", "three_part_xgboost"),
  rmsle = c(rmsle_twopart_xgb, rmsle_twopart_lgb, rmsle_threepart_xgb)
)

predictions <- dplyr::tibble(
  truth = y_test,
  pred_two_part_xgboost = y_pred_twopart_xgb,
  pred_two_part_lightgbm = y_pred_twopart_lgb,
  pred_three_part_xgboost = y_pred_threepart_xgb
)

readr::write_csv(metrics, "outputs/models/two_three_part_metrics.csv")
readr::write_csv(predictions, "outputs/predictions/two_three_part_test_predictions.csv")

cat("\nTwo-part / Three-part RMSLE:\n")
print(metrics, n = nrow(metrics), width = Inf)
cat("\nSaved: outputs/models/two_three_part_metrics.csv\n")
cat("Saved: outputs/predictions/two_three_part_test_predictions.csv\n")
