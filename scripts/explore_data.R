## ECN 372 — EDA after filtering (run in R)
## From project root:  Rscript scripts/explore_data.R
## Requires filtered_data.csv from scripts/Filter data

options(repos = c(CRAN = "https://cloud.r-project.org"))

resolve_repo_root <- function() {
  for (root in c(".", "..")) {
    p <- file.path(root, "filtered_data.csv")
    if (file.exists(p)) {
      return(normalizePath(root, winslash = "/", mustWork = TRUE))
    }
  }
  stop(
    "Cannot find filtered_data.csv. Run from repo root or scripts/:\n",
    "  Rscript \"scripts/Filter data\"\n",
    "then run this script again."
  )
}

# Columns that must not be used as predictors (spend/charges/utilization).
# Target TOTEXP* is removed from this set after the call.
meps_forbidden_predictor_names <- function(nm) {
  spend_util <- nm[grepl(
    "^[A-Z]{2,12}(EXP|TCH|SLF|MCR|MCD|PRV|VA|TRI|OFD|STL|WCP|OSR|PTR|OTH)[0-9]{2}$",
    nm
  )]
  util_extra <- nm[grepl(
    "^(OBTOTV|OBVTCH|OBDRV|OPTOTV|OPTTCH|OPDRV|OPVTCH|ERTOT|ERTTCH|IPDIS|IPNGTD|DVTOT|RXTOT)",
    nm
  )]
  sort(unique(c(spend_util, util_extra)))
}

meps_to_na <- function(x) {
  if (is.numeric(x)) {
    out <- x
  } else {
    out <- suppressWarnings(as.numeric(x))
  }
  out[out %in% c(-1L, -7L, -8L, -9L)] <- NA_real_
  out
}

repo_root <- resolve_repo_root()
csv_path <- file.path(repo_root, "filtered_data.csv")
fig_dir <- file.path(repo_root, "figures")
dir.create(fig_dir, showWarnings = FALSE)

message("Reading: ", csv_path)
df <- read.csv(csv_path, check.names = FALSE)

totexp_cols <- grep("^TOTEXP", names(df), value = TRUE)
if (!length(totexp_cols)) {
  stop("No TOTEXP* column found in filtered_data.csv.")
}
target_col <- totexp_cols[[length(totexp_cols)]]

forbidden <- meps_forbidden_predictor_names(names(df))
forbidden <- setdiff(forbidden, target_col)
allowed_predictors <- setdiff(names(df), c(forbidden, target_col))

message(
  "Columns: ", length(names(df)),
  " | Forbidden predictors: ", length(forbidden),
  " | Allowed predictors: ", length(allowed_predictors),
  " | Target: ", target_col
)

audit_path <- file.path(fig_dir, "column_audit.tsv")
audit_lines <- c(
  "role\tname",
  paste0("target\t", target_col),
  vapply(forbidden, function(nm) paste0("forbidden_predictor\t", nm), character(1)),
  vapply(allowed_predictors, function(nm) paste0("allowed_predictor\t", nm), character(1))
)
writeLines(audit_lines, audit_path)
message("Wrote: ", audit_path)

y <- meps_to_na(df[[target_col]])
logy <- log1p(y)

pdf_path <- file.path(fig_dir, "eda_target.pdf")
pdf(pdf_path, width = 10, height = 8)

# ----- Page 1: scales chosen so bulk of the distribution is visible -----
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

# Raw dollars are extremely right-skewed; show where most mass sits.
p95 <- stats::quantile(y, 0.95, na.rm = TRUE)
p99 <- stats::quantile(y, 0.99, na.rm = TRUE)
y_bulk <- y[!is.na(y) & y <= p95]
hist(
  y_bulk,
  breaks = 50,
  main = "Total spend (bulk: up to 95th percentile)",
  sub = sprintf(
    "Excluded from panel: %s obs above $%s (95th pctl). 99th pctl = $%s.",
    format(sum(y > p95, na.rm = TRUE), big.mark = ","),
    format(round(p95), big.mark = ","),
    format(round(p99), big.mark = ",")
  ),
  xlab = sprintf("Dollars (x <= $%s)", format(round(p95), big.mark = ",")),
  col = "gray75",
  border = "white"
)

# Log10 dollars spreads the right tail for a readable histogram.
log10d <- log10(pmax(y, 1))
hist(
  log10d,
  breaks = 50,
  main = "log10(dollars), floor at $1",
  xlab = expression(log[10](max(y, 1))),
  col = "gray75",
  border = "white"
)

hist(
  logy,
  breaks = 60,
  main = expression(log(1 + y)) ,
  xlab = "log(1 + total expenditure)",
  col = "gray75",
  border = "white",
  freq = FALSE
)
lines(stats::density(logy, na.rm = TRUE, adjust = 1.2), col = "steelblue", lwd = 2)

plot(
  stats::ecdf(logy[!is.na(logy)]),
  main = "ECDF of log(1 + y)",
  xlab = "log(1 + y)",
  ylab = "F(x)",
  col.hor = "gray30",
  col.vert = "gray30"
)

mtext(
  paste("Target:", target_col, "(n =", sum(!is.na(y)), "non-missing)"),
  outer = TRUE,
  line = 0.2,
  cex = 1.05,
  font = 2
)

# ----- Page 2: relationships at readable scales -----
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 1, 0))

if ("AGELAST" %in% names(df)) {
  age <- meps_to_na(df[["AGELAST"]])
  ok <- !is.na(age) & !is.na(logy)
  if (sum(ok) > 50) {
    graphics::smoothScatter(
      age[ok],
      logy[ok],
      main = "log(1 + y) vs age (density)",
      xlab = "AGELAST (years)",
      ylab = "log(1 + y)",
      nrpoints = 400,
      colramp = colorRampPalette(c("white", "navy"))
    )
    lines(
      stats::lowess(age[ok], logy[ok], f = 0.35),
      col = "orangered",
      lwd = 2.5
    )
  }
} else {
  plot.new()
  title(main = "AGELAST not found — skipped age vs spend")
}

if ("SEX" %in% names(df)) {
  sx <- meps_to_na(df[["SEX"]])
  ok <- !is.na(sx) & !is.na(logy)
  if (sum(ok) > 50) {
    ylim <- stats::quantile(logy[ok], c(0.02, 0.98), na.rm = TRUE)
    boxplot(
      split(logy[ok], sx[ok]),
      main = "log(1 + y) by SEX (2–98% y-range)",
      xlab = "SEX code",
      ylab = "log(1 + y)",
      ylim = ylim,
      outline = FALSE,
      border = "gray25",
      col = "gray90"
    )
    stripchart(
      split(logy[ok], sx[ok]),
      vertical = TRUE,
      method = "jitter",
      jitter = 0.12,
      pch = 16,
      cex = 0.25,
      col = grDevices::adjustcolor("steelblue", alpha.f = 0.15),
      add = TRUE
    )
  }
} else {
  plot.new()
  title(main = "SEX not found — skipped")
}

mtext("Relationships (compressed outliers for readability)", outer = TRUE, line = -0.5, cex = 0.95)

dev.off()
message("Wrote: ", pdf_path)
