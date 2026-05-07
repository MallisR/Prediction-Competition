set.seed(42)
source("R/utils.R")

options(repos = c(CRAN = "https://cloud.r-project.org"))
required_pkgs <- c("readr", "dplyr", "tibble")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs)) {
  stop(
    "Missing required packages: ",
    paste(missing_pkgs, collapse = ", "),
    ". Install them first."
  )
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
})

dir.create("outputs/models", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/predictions", recursive = TRUE, showWarnings = FALSE)

prediction_files <- list.files(
  "outputs/predictions",
  pattern = "_test_predictions\\.csv$",
  full.names = TRUE
)

if (!length(prediction_files)) {
  stop("No *_test_predictions.csv files found in outputs/predictions.")
}

read_prediction_file <- function(path) {
  dat <- readr::read_csv(path, show_col_types = FALSE)
  if (!"truth" %in% names(dat)) {
    return(NULL)
  }
  pred_cols <- grep("^pred_", names(dat), value = TRUE)
  if (!length(pred_cols)) {
    return(NULL)
  }
  list(path = path, truth = dat$truth, preds = dat[, pred_cols, drop = FALSE])
}

loaded <- lapply(prediction_files, read_prediction_file)
loaded <- loaded[!vapply(loaded, is.null, logical(1))]
if (!length(loaded)) {
  stop("No prediction files with columns `truth` and `pred_*` were found.")
}

base_truth <- loaded[[1]]$truth
ensemble_df <- tibble::tibble(truth = base_truth)

for (obj in loaded) {
  if (length(obj$truth) != length(base_truth) || any(obj$truth != base_truth, na.rm = TRUE)) {
    stop("Prediction files do not align on `truth`; ensure same held-out rows/order.")
  }
  for (nm in names(obj$preds)) {
    col_name <- nm
    if (col_name %in% names(ensemble_df)) {
      col_name <- paste0(col_name, "_", basename(obj$path))
    }
    ensemble_df[[col_name]] <- obj$preds[[nm]]
  }
}

pred_cols <- grep("^pred_", names(ensemble_df), value = TRUE)
valid_cols <- pred_cols[vapply(ensemble_df[pred_cols], function(x) !all(is.na(x)), logical(1))]

if (!length(valid_cols)) {
  stop("No usable prediction columns found (all NA).")
}

score_model <- function(y_true, y_pred) {
  ok <- !is.na(y_true) & !is.na(y_pred)
  if (!any(ok)) {
    return(NA_real_)
  }
  get("rmsle", mode = "function")(y_true[ok], y_pred[ok])
}

base_scores <- tibble::tibble(
  model = valid_cols,
  rmsle = vapply(valid_cols, function(col) score_model(ensemble_df$truth, ensemble_df[[col]]), numeric(1))
) %>%
  arrange(rmsle)

# Equal-weight mean ensemble across all available base predictions.
ensemble_df$pred_ensemble_mean <- rowMeans(ensemble_df[, valid_cols, drop = FALSE], na.rm = TRUE)
mean_score <- score_model(ensemble_df$truth, ensemble_df$pred_ensemble_mean)

# Weight search over best up to 3 base models.
top_k <- min(3L, nrow(base_scores))
top_models <- base_scores$model[seq_len(top_k)]

weight_grid <- as.data.frame(expand.grid(w1 = seq(0, 1, by = 0.05), w2 = seq(0, 1, by = 0.05), w3 = seq(0, 1, by = 0.05)))
weight_grid <- weight_grid[abs(rowSums(weight_grid) - 1) < 1e-8, , drop = FALSE]
if (top_k == 1L) {
  weight_grid <- data.frame(w1 = 1, w2 = 0, w3 = 0)
}
if (top_k == 2L) {
  weight_grid$w3 <- 0
  weight_grid <- weight_grid[abs(weight_grid$w1 + weight_grid$w2 - 1) < 1e-8, , drop = FALSE]
}

blend_pred <- function(row_weights) {
  w <- as.numeric(row_weights[1:top_k])
  preds <- as.matrix(ensemble_df[, top_models, drop = FALSE])
  as.numeric(preds %*% w)
}

grid_scores <- vapply(seq_len(nrow(weight_grid)), function(i) {
  pred <- blend_pred(weight_grid[i, ])
  score_model(ensemble_df$truth, pred)
}, numeric(1))

best_idx <- which.min(grid_scores)
best_weights <- as.numeric(weight_grid[best_idx, 1:top_k, drop = TRUE])
names(best_weights) <- top_models
ensemble_df$pred_ensemble_weighted <- blend_pred(weight_grid[best_idx, ])
weighted_score <- score_model(ensemble_df$truth, ensemble_df$pred_ensemble_weighted)

leaderboard <- bind_rows(
  base_scores,
  tibble::tibble(model = "ensemble_mean", rmsle = mean_score),
  tibble::tibble(model = "ensemble_weighted_grid", rmsle = weighted_score)
) %>%
  arrange(rmsle)

weight_table <- tibble::tibble(
  model = names(best_weights),
  weight = as.numeric(best_weights)
)

readr::write_csv(leaderboard, "outputs/models/ensemble_metrics.csv")
readr::write_csv(weight_table, "outputs/models/ensemble_weights.csv")
readr::write_csv(ensemble_df, "outputs/predictions/ensemble_test_predictions.csv")

cat("\nEnsemble leaderboard:\n")
print(leaderboard, n = nrow(leaderboard), width = Inf)
cat("\nBest weighted ensemble uses:\n")
print(weight_table, n = nrow(weight_table), width = Inf)
cat("\nSaved: outputs/models/ensemble_metrics.csv\n")
cat("Saved: outputs/models/ensemble_weights.csv\n")
cat("Saved: outputs/predictions/ensemble_test_predictions.csv\n")
