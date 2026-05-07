set.seed(42)
source("R/utils.R")

options(repos = c(CRAN = "https://cloud.r-project.org"))

required_pkgs <- c("dplyr", "readr")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs)) {
  stop("Missing required packages: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

train_path <- "data/meps_clean.csv"
test_path <- "data/meps_test.csv"
metrics_path <- "outputs/models/two_three_part_metrics.csv"
weights_path <- "outputs/models/ensemble_weights.csv"

if (!file.exists(train_path)) stop("Missing training data: ", train_path)
if (!file.exists(test_path)) stop("Missing test data: ", test_path)
if (!file.exists(metrics_path)) stop("Missing model metrics: ", metrics_path)

train_raw <- readr::read_csv(train_path, show_col_types = FALSE)
test_raw <- readr::read_csv(test_path, show_col_types = FALSE)
metrics <- readr::read_csv(metrics_path, show_col_types = FALSE)

target_col <- if ("TOTEXP" %in% names(train_raw)) {
  "TOTEXP"
} else {
  totexp_candidates <- grep("^TOTEXP[0-9]{2}$", names(train_raw), value = TRUE)
  if (!length(totexp_candidates)) stop("No TOTEXP/TOTEXPyy target in training data.")
  sort(totexp_candidates)[length(totexp_candidates)]
}

id_col <- if ("id" %in% names(test_raw)) {
  "id"
} else {
  id_candidates <- names(test_raw)[tolower(names(test_raw)) %in% c("id", "dupersid")]
  if (!length(id_candidates)) stop("No id column found in data/meps_test.csv.")
  id_candidates[[1]]
}

year_col <- if ("YEAR" %in% names(train_raw)) "YEAR" else if ("DATAYEAR" %in% names(train_raw)) "DATAYEAR" else NULL
age_col <- if ("AGE" %in% names(train_raw)) "AGE" else if ("AGELAST" %in% names(train_raw)) "AGELAST" else NULL
if (is.null(year_col) || is.null(age_col)) {
  stop("Training data must include YEAR/DATAYEAR and AGE/AGELAST.")
}

is_binary_01 <- function(x) {
  if (!is.numeric(x) && !is.integer(x)) return(FALSE)
  ux <- sort(unique(x[!is.na(x)]))
  length(ux) > 0 && all(ux %in% c(0, 1))
}

chronic_name_pattern <- "(CHRON|ARTH|ASTH|CANC|CHD|DIAB|EMPH|HIBP|STRK|ANGI|MIDX|OHRT|JTPAIN)"
binary_cols <- names(train_raw)[vapply(train_raw, is_binary_01, logical(1))]
chronic_binary_cols <- binary_cols[grepl(chronic_name_pattern, binary_cols, ignore.case = TRUE)]

engineer_features <- function(df, is_train) {
  out <- df %>%
    mutate(
      YEAR = {
        year_chr <- as.character(.data[[year_col]])
        year_levels <- sort(unique(c("2019", as.character(train_raw[[year_col]]), year_chr)))
        stats::relevel(factor(year_chr, levels = year_levels), ref = "2019")
      },
      AGE_SQ = .data[[age_col]]^2
    )

  chronic_use <- intersect(chronic_binary_cols, names(out))
  if (length(chronic_use)) {
    out <- out %>%
      mutate(CHRONIC_COUNT = rowSums(across(all_of(chronic_use)), na.rm = TRUE))
  } else {
    out <- out %>%
      mutate(CHRONIC_COUNT = 0L)
  }

  if (is_train) {
    out <- out %>%
      mutate(
        log1p_TOTEXP = log1p(.data[[target_col]]),
        ANY_SPEND = as.integer(.data[[target_col]] > 0),
        SPEND_TIER = dplyr::case_when(
          .data[[target_col]] == 0 ~ 0L,
          .data[[target_col]] > 0 & .data[[target_col]] <= 3000 ~ 1L,
          .data[[target_col]] > 3000 ~ 2L
        ),
        SPEND_TIER = factor(.data$SPEND_TIER, levels = c(0, 1, 2))
      )
  }

  out
}

train_df <- engineer_features(train_raw, is_train = TRUE)
test_df <- engineer_features(test_raw, is_train = FALSE)

feature_cols <- setdiff(names(train_df), c(target_col, "log1p_TOTEXP", "ANY_SPEND", "SPEND_TIER"))

make_xy_pair <- function(train_df, test_df, cols) {
  train_tag <- train_df[, cols, drop = FALSE]
  test_tag <- test_df[, cols, drop = FALSE]
  train_tag$.split_key <- "train"
  test_tag$.split_key <- "test"
  both <- dplyr::bind_rows(train_tag, test_tag)
  mm <- model.matrix(~ . - 1, data = both)
  is_train <- both$.split_key == "train"
  list(
    x_train = mm[is_train, , drop = FALSE],
    x_test = mm[!is_train, , drop = FALSE]
  )
}

xy <- make_xy_pair(train_df, test_df, feature_cols)
x_train <- as.matrix(xy$x_train)
x_test <- as.matrix(xy$x_test)
storage.mode(x_train) <- "numeric"
storage.mode(x_test) <- "numeric"

y_train <- train_df[[target_col]]
pos_idx <- which(y_train > 0)

train_two_part_xgboost <- function() {
  if (!requireNamespace("xgboost", quietly = TRUE)) return(NULL)
  part1 <- xgboost::xgboost(
    data = x_train, label = train_df$ANY_SPEND, objective = "binary:logistic",
    nrounds = 300, max_depth = 6, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8, verbose = 0
  )
  part2 <- xgboost::xgboost(
    data = x_train[pos_idx, , drop = FALSE], label = train_df$log1p_TOTEXP[pos_idx], objective = "reg:squarederror",
    nrounds = 350, max_depth = 6, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8, verbose = 0
  )
  p_spend <- as.numeric(predict(part1, x_test))
  log_pred <- as.numeric(predict(part2, x_test))
  pmax(p_spend * expm1(log_pred), 0)
}

train_two_part_lightgbm <- function() {
  if (!requireNamespace("lightgbm", quietly = TRUE)) return(NULL)
  part1 <- lightgbm::lgb.train(
    params = list(objective = "binary", learning_rate = 0.05, num_leaves = 63, feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 1),
    data = lightgbm::lgb.Dataset(data = x_train, label = train_df$ANY_SPEND),
    nrounds = 350, verbose = -1
  )
  part2 <- lightgbm::lgb.train(
    params = list(objective = "regression", learning_rate = 0.05, num_leaves = 63, feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 1),
    data = lightgbm::lgb.Dataset(data = x_train[pos_idx, , drop = FALSE], label = train_df$log1p_TOTEXP[pos_idx]),
    nrounds = 400, verbose = -1
  )
  p_spend <- as.numeric(predict(part1, x_test))
  log_pred <- as.numeric(predict(part2, x_test))
  pmax(p_spend * expm1(log_pred), 0)
}

train_three_part_xgboost <- function() {
  if (!requireNamespace("xgboost", quietly = TRUE)) return(NULL)
  part1 <- xgboost::xgboost(
    data = x_train, label = train_df$ANY_SPEND, objective = "binary:logistic",
    nrounds = 300, max_depth = 6, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8, verbose = 0
  )
  nonzero_idx <- which(train_df$ANY_SPEND == 1L & !is.na(train_df$SPEND_TIER))
  nonzero_binary <- as.integer(train_df$SPEND_TIER[nonzero_idx] == "2")
  part2 <- xgboost::xgboost(
    data = x_train[nonzero_idx, , drop = FALSE], label = nonzero_binary, objective = "binary:logistic",
    nrounds = 250, max_depth = 5, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8, verbose = 0
  )
  low_idx <- which(train_df$SPEND_TIER == "1")
  high_idx <- which(train_df$SPEND_TIER == "2")
  part3_low <- xgboost::xgboost(
    data = x_train[low_idx, , drop = FALSE], label = train_df$log1p_TOTEXP[low_idx], objective = "reg:squarederror",
    nrounds = 300, max_depth = 5, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8, verbose = 0
  )
  part3_high <- xgboost::xgboost(
    data = x_train[high_idx, , drop = FALSE], label = train_df$log1p_TOTEXP[high_idx], objective = "reg:squarederror",
    nrounds = 350, max_depth = 6, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8, verbose = 0
  )
  p_any <- as.numeric(predict(part1, x_test))
  p_high <- as.numeric(predict(part2, x_test))
  pred_low <- expm1(as.numeric(predict(part3_low, x_test)))
  pred_high <- expm1(as.numeric(predict(part3_high, x_test)))
  pmax(p_any * (p_high * pred_high + (1 - p_high) * pred_low), 0)
}

pred_cache <- list()
get_model_pred <- function(model_name) {
  if (!is.null(pred_cache[[model_name]])) return(pred_cache[[model_name]])
  pred <- switch(
    model_name,
    "two_part_xgboost" = train_two_part_xgboost(),
    "two_part_lightgbm" = train_two_part_lightgbm(),
    "three_part_xgboost" = train_three_part_xgboost(),
    NULL
  )
  pred_cache[[model_name]] <<- pred
  pred
}

best_row <- metrics %>%
  filter(!is.na(rmsle)) %>%
  arrange(rmsle) %>%
  slice(1)
if (!nrow(best_row)) stop("No valid best model found in metrics.")
best_model_name <- best_row$model[[1]]
best_pred <- get_model_pred(best_model_name)
if (is.null(best_pred)) stop("Could not generate predictions for best model: ", best_model_name)
best_pred <- pmax(best_pred, 0)

if (file.exists(weights_path)) {
  weights <- readr::read_csv(weights_path, show_col_types = FALSE)
  weights <- weights %>% filter(!is.na(weight), weight > 0)
} else {
  weights <- tibble::tibble(model = best_model_name, weight = 1)
}
if (!nrow(weights)) {
  weights <- tibble::tibble(model = best_model_name, weight = 1)
}

weights <- weights %>% mutate(model = sub("^pred_", "", model))

ensemble_pred <- rep(0, nrow(test_df))
total_w <- 0
for (i in seq_len(nrow(weights))) {
  mdl <- weights$model[[i]]
  w <- weights$weight[[i]]
  p <- get_model_pred(mdl)
  if (!is.null(p)) {
    ensemble_pred <- ensemble_pred + w * p
    total_w <- total_w + w
  }
}
if (total_w <= 0) {
  ensemble_pred <- best_pred
} else {
  ensemble_pred <- ensemble_pred / total_w
}
ensemble_pred <- pmax(ensemble_pred, 0)

dir.create("outputs/predictions", recursive = TRUE, showWarnings = FALSE)

best_out <- tibble::tibble(id = test_raw[[id_col]], TOTEXP_pred = best_pred)
ens_out <- tibble::tibble(id = test_raw[[id_col]], TOTEXP_pred = ensemble_pred)

readr::write_csv(best_out, "outputs/predictions/predictions_best_model.csv")
readr::write_csv(ens_out, "outputs/predictions/predictions_ensemble.csv")

cat("Best model used:", best_model_name, "\n")
cat("Saved: outputs/predictions/predictions_best_model.csv\n")
cat("Saved: outputs/predictions/predictions_ensemble.csv\n")
