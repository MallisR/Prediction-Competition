set.seed(42)

options(repos = c(CRAN = "https://cloud.r-project.org"))
for (pkg in c("dplyr", "readr", "rsample")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(rsample)
})

source("R/utils.R")

input_path <- "data/meps_clean.csv"
if (!file.exists(input_path)) {
  stop("Missing cleaned input file at: ", input_path)
}

dir.create("outputs/predictions", recursive = TRUE, showWarnings = FALSE)

df <- readr::read_csv(input_path, show_col_types = FALSE)

year_col <- if ("YEAR" %in% names(df)) "YEAR" else if ("DATAYEAR" %in% names(df)) "DATAYEAR" else NULL
age_col <- if ("AGE" %in% names(df)) "AGE" else if ("AGELAST" %in% names(df)) "AGELAST" else NULL
if (is.null(year_col)) {
  stop("Expected `YEAR` (or `DATAYEAR`) column in data/meps_clean.csv.")
}
if (is.null(age_col)) {
  stop("Expected `AGE` (or `AGELAST`) column in data/meps_clean.csv.")
}

target_col <- if ("TOTEXP" %in% names(df)) {
  "TOTEXP"
} else {
  totexp_candidates <- grep("^TOTEXP[0-9]{2}$", names(df), value = TRUE)
  if (!length(totexp_candidates)) {
    stop("Expected target column `TOTEXP` or `TOTEXPyy` in data/meps_clean.csv.")
  }
  sort(totexp_candidates)[length(totexp_candidates)]
}

# Treat YEAR as categorical with 2019 reference level.
df <- df %>%
  mutate(
    YEAR = {
      year_chr <- as.character(.data[[year_col]])
      year_levels <- sort(unique(c("2019", year_chr)))
      stats::relevel(factor(year_chr, levels = year_levels), ref = "2019")
    }
  )

# Identify binary chronic-condition-like indicators (0/1) by values and names.
is_binary_01 <- function(x) {
  if (!is.numeric(x) && !is.integer(x)) {
    return(FALSE)
  }
  ux <- sort(unique(x[!is.na(x)]))
  length(ux) > 0 && all(ux %in% c(0, 1))
}

chronic_name_pattern <- "(CHRON|ARTH|ASTH|CANC|CHD|DIAB|EMPH|HIBP|STRK|ANGI|MIDX|OHRT|JTPAIN)"
binary_cols <- names(df)[vapply(df, is_binary_01, logical(1))]
chronic_binary_cols <- binary_cols[grepl(chronic_name_pattern, binary_cols, ignore.case = TRUE)]

if (length(chronic_binary_cols) > 0) {
  df <- df %>%
    mutate(CHRONIC_COUNT = rowSums(across(all_of(chronic_binary_cols)), na.rm = TRUE))
} else {
  df <- df %>%
    mutate(CHRONIC_COUNT = 0L)
}

df <- df %>%
  mutate(
    AGE_SQ = .data[[age_col]]^2,
    log1p_TOTEXP = log1p(.data[[target_col]]),
    ANY_SPEND = as.integer(.data[[target_col]] > 0),
    SPEND_TIER = dplyr::case_when(
      .data[[target_col]] == 0 ~ 0L,
      .data[[target_col]] > 0 & .data[[target_col]] <= 3000 ~ 1L,
      .data[[target_col]] > 3000 ~ 2L
    ),
    SPEND_TIER = factor(SPEND_TIER, levels = c(0, 1, 2))
  )

split_obj <- rsample::initial_split(df, prop = 0.80, strata = SPEND_TIER)
train_df <- rsample::training(split_obj)
test_df <- rsample::testing(split_obj)

readr::write_csv(train_df, "outputs/predictions/train_features.csv")
readr::write_csv(test_df, "outputs/predictions/test_features.csv")

saveRDS(
  list(
    split = split_obj,
    chronic_binary_cols = chronic_binary_cols
  ),
  "outputs/models/feature_objects.rds"
)

message("Feature engineering complete.")
message("Target column used: ", target_col)
message("YEAR source column: ", year_col)
message("AGE source column: ", age_col)
message("Rows total: ", nrow(df))
message("Train rows: ", nrow(train_df), " | Test rows: ", nrow(test_df))
message("Chronic binary columns used: ", length(chronic_binary_cols))
