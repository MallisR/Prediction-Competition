set.seed(42)

rmsle <- function(y_true, y_pred) {
  y_pred <- pmax(y_pred, 0)
  sqrt(mean((log1p(y_pred) - log1p(y_true))^2))
}
