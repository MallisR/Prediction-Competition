## ECN 372 — reproducible end-to-end pipeline
## Run from repo root:
##   Rscript scripts/run_pipeline.R
##
## Team convention:
##   - Put the workbook at raw_data/h251.xlsx
##   - This script runs:
##       1) scripts/Filter data
##       2) scripts/explore_data.R
##       3) scripts/run_cv_tidymodels.R

resolve_repo_root <- function() {
  for (root in c(".", "..")) {
    if (file.exists(file.path(root, "scripts", "Filter data"))) {
      return(normalizePath(root, winslash = "/", mustWork = TRUE))
    }
  }
  stop("Cannot locate repo root (expected scripts/Filter data).")
}

repo_root <- resolve_repo_root()
scripts_dir <- file.path(repo_root, "scripts")

required_scripts <- c(
  file.path(scripts_dir, "Filter data"),
  file.path(scripts_dir, "explore_data.R"),
  file.path(scripts_dir, "run_cv_tidymodels.R")
)

missing_scripts <- required_scripts[!file.exists(required_scripts)]
if (length(missing_scripts)) {
  stop(
    "Missing required scripts:\n",
    paste0("  - ", missing_scripts, collapse = "\n")
  )
}

run_step <- function(path) {
  message("\n=== Running: ", path, " ===")
  sys.source(path, envir = new.env(parent = globalenv()))
  message("=== Done: ", path, " ===")
}

old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(repo_root)

run_step(file.path(scripts_dir, "Filter data"))
run_step(file.path(scripts_dir, "explore_data.R"))
run_step(file.path(scripts_dir, "run_cv_tidymodels.R"))

message("\nPipeline complete.")
message("Outputs:")
message("  - filtered_data.csv")
message("  - filtered_model_ready.csv")
message("  - figures/column_audit.tsv")
message("  - figures/eda_target.pdf")
message("  - outputs/cv_results.csv")
