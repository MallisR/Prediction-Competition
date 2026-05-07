set.seed(42)
source("R/utils.R")

options(repos = c(CRAN = "https://cloud.r-project.org"))
required_pkgs <- c("dplyr", "readr", "ggplot2", "tidyr", "scales")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs)) {
  stop("Missing required packages: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(tidyr)
  library(scales)
})

input_path <- "data/meps_clean.csv"
if (!file.exists(input_path)) {
  stop("Missing cleaned input file at: ", input_path)
}

dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

df <- readr::read_csv(input_path, show_col_types = FALSE)

target_col <- if ("TOTEXP" %in% names(df)) {
  "TOTEXP"
} else {
  totexp_candidates <- grep("^TOTEXP[0-9]{2}$", names(df), value = TRUE)
  if (!length(totexp_candidates)) {
    stop("Expected target column `TOTEXP` or `TOTEXPyy`.")
  }
  sort(totexp_candidates)[length(totexp_candidates)]
}

year_col <- if ("YEAR" %in% names(df)) "YEAR" else if ("DATAYEAR" %in% names(df)) "DATAYEAR" else NULL
age_col <- if ("AGE" %in% names(df)) "AGE" else if ("AGELAST" %in% names(df)) "AGELAST" else NULL
health_candidates <- c("RTHLTH", "RTHLTH31", "RTHLTH42", "RTHLTH53", "MNHLTH", "MNHLTH31", "MNHLTH42", "MNHLTH53")
health_col <- health_candidates[health_candidates %in% names(df)]
health_col <- if (length(health_col)) health_col[[1]] else NULL

eda_df <- df %>%
  mutate(
    TOTEXP_VAL = as.numeric(.data[[target_col]]),
    TOTEXP_VAL = pmax(TOTEXP_VAL, 0),
    log1p_TOTEXP = log1p(TOTEXP_VAL)
  )

# 1) Histogram of raw TOTEXP with log10 x-axis
p1 <- ggplot(eda_df, aes(x = TOTEXP_VAL + 1)) +
  geom_histogram(bins = 60, fill = "steelblue", color = "white") +
  scale_x_log10() +
  labs(
    title = "Histogram of Total Expenditures (log10 x-axis)",
    x = "TOTEXP + 1 (log10 scale)",
    y = "Count"
  ) +
  theme_minimal()
ggsave(
  filename = "outputs/figures/hist_totexp_log10x.png",
  plot = p1,
  width = 9,
  height = 6,
  dpi = 300
)

# 2) Histogram of log1p(TOTEXP)
p2 <- ggplot(eda_df, aes(x = log1p_TOTEXP)) +
  geom_histogram(bins = 60, fill = "darkseagreen4", color = "white") +
  labs(
    title = "Histogram of log1p(TOTEXP)",
    x = "log1p(TOTEXP)",
    y = "Count"
  ) +
  theme_minimal()
ggsave(
  filename = "outputs/figures/hist_log1p_totexp.png",
  plot = p2,
  width = 9,
  height = 6,
  dpi = 300
)

# 3) Bar chart: proportion of zero spenders by year
if (!is.null(year_col)) {
  zero_by_year <- eda_df %>%
    mutate(YEAR = as.factor(.data[[year_col]])) %>%
    group_by(YEAR) %>%
    summarise(prop_zero = mean(TOTEXP_VAL == 0, na.rm = TRUE), .groups = "drop")

  p3 <- ggplot(zero_by_year, aes(x = YEAR, y = prop_zero)) +
    geom_col(fill = "tomato3") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "Proportion of Zero Spenders by Year",
      x = "Year",
      y = "Proportion with TOTEXP = 0"
    ) +
    theme_minimal()

  ggsave(
    filename = "outputs/figures/zero_spenders_by_year.png",
    plot = p3,
    width = 9,
    height = 6,
    dpi = 300
  )
}

# 4) Line plot: mean and median TOTEXP by 10-year age group
if (!is.null(age_col)) {
  age_summary <- eda_df %>%
    mutate(
      AGE_NUM = as.numeric(.data[[age_col]]),
      age_group = cut(AGE_NUM, breaks = seq(0, 100, by = 10), include.lowest = TRUE, right = FALSE)
    ) %>%
    group_by(age_group) %>%
    summarise(
      mean_TOTEXP = mean(TOTEXP_VAL, na.rm = TRUE),
      median_TOTEXP = median(TOTEXP_VAL, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    tidyr::pivot_longer(
      cols = c(mean_TOTEXP, median_TOTEXP),
      names_to = "stat",
      values_to = "TOTEXP_value"
    )

  p4 <- ggplot(age_summary, aes(x = age_group, y = TOTEXP_value, group = stat, color = stat)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2) +
    labs(
      title = "Mean and Median TOTEXP by 10-Year Age Group",
      x = "Age Group",
      y = "TOTEXP",
      color = "Statistic"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  ggsave(
    filename = "outputs/figures/totexp_by_age_group.png",
    plot = p4,
    width = 10,
    height = 6,
    dpi = 300
  )
}

# 5) Boxplot: TOTEXP by self-reported health status (if present)
if (!is.null(health_col)) {
  health_df <- eda_df %>%
    mutate(health_raw = .data[[health_col]]) %>%
    mutate(
      health_status = case_when(
        as.character(health_raw) %in% c("1") ~ "Excellent",
        as.character(health_raw) %in% c("2") ~ "Very good",
        as.character(health_raw) %in% c("3") ~ "Good",
        as.character(health_raw) %in% c("4") ~ "Fair",
        as.character(health_raw) %in% c("5") ~ "Poor",
        TRUE ~ as.character(health_raw)
      ),
      health_status = factor(health_status, levels = c("Excellent", "Very good", "Good", "Fair", "Poor"))
    )

  p5 <- ggplot(health_df, aes(x = health_status, y = TOTEXP_VAL + 1)) +
    geom_boxplot(outlier.alpha = 0.15, fill = "orchid3") +
    scale_y_log10() +
    labs(
      title = paste0("TOTEXP by Self-Reported Health Status (", health_col, ")"),
      x = "Self-Reported Health Status",
      y = "TOTEXP + 1 (log10 scale)"
    ) +
    theme_minimal()

  ggsave(
    filename = "outputs/figures/totexp_by_health_status.png",
    plot = p5,
    width = 9,
    height = 6,
    dpi = 300
  )
}

# 6) Correlation plot of top 20 numeric features vs log1p(TOTEXP)
numeric_df <- eda_df %>%
  select(where(is.numeric))

cor_to_target <- sapply(names(numeric_df), function(nm) {
  if (nm == "log1p_TOTEXP") return(NA_real_)
  suppressWarnings(cor(numeric_df[[nm]], numeric_df$log1p_TOTEXP, use = "pairwise.complete.obs"))
})

cor_to_target <- cor_to_target[!is.na(cor_to_target)]
top_n <- min(20, length(cor_to_target))
top20_names <- names(sort(abs(cor_to_target), decreasing = TRUE))[seq_len(top_n)]

if (length(top20_names) >= 2) {
  corr_vars <- c("log1p_TOTEXP", top20_names)
  corr_mat <- suppressWarnings(cor(numeric_df[, corr_vars, drop = FALSE], use = "pairwise.complete.obs"))

  corr_long <- as.data.frame(as.table(corr_mat))
  names(corr_long) <- c("Var1", "Var2", "Correlation")

  p6 <- ggplot(corr_long, aes(x = Var1, y = Var2, fill = Correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "navy", mid = "white", high = "firebrick", midpoint = 0, limits = c(-1, 1)) +
    labs(
      title = "Correlation Matrix: Top 20 Numeric Features vs log1p(TOTEXP)",
      x = "",
      y = "",
      fill = "Corr"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      axis.text.y = element_text(size = 8)
    )

  ggsave(
    filename = "outputs/figures/correlation_top20_vs_log1p_totexp.png",
    plot = p6,
    width = 10,
    height = 9,
    dpi = 300
  )
}

message("EDA plots saved to outputs/figures/")
