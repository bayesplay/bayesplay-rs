# Reference Bayes factors for t-test models with (truncated) Cauchy priors.
#
# Regenerate with (from this directory):
#   Rscript bayesfactor_ttest.R
#
# For every design / t / prior interval / rscale in the grid this records:
#
# * `bayesfactor_logbf`: log BF10 from BayesFactor::ttest.tstat().
# * `reference_logbf`: an independent high-precision log BF10.
#
# BayesFactor is the standard for these tests, but its interval Bayes factors 
# are unreliable when the observed effect lies far outside the prior interval 
# and for very large t (it reports "t approximation invoked"). 
# `reference_logbf` is computed directly from the model definition, using only 
# positive integrands so there is no cancellation:
#
#   T = Z' / S,  Z' ~ N(delta * sqrt(n), 1),  S = sqrt(V / df),  V ~ chi^2_df
#   m1 = (1 / P(I)) int_I int_0^inf s phi(t s - delta sqrt(n)) h(s) cauchy(delta) ds ddelta
#   m0 = dt(t, df)

suppressMessages(library(BayesFactor))

log_bf_reference <- function(t, n, df, r, lo, hi) {
  log_mass <- log(pcauchy(hi, 0, r) - pcauchy(lo, 0, r))
  log_h <- function(s) {
    log(2) + (df / 2) * log(df / 2) - lgamma(df / 2) + (df - 1) * log(s) - df * s^2 / 2
  }
  log_f <- function(s, d) {
    log(s) + log_h(s) + dnorm(t * s - d * sqrt(n), log = TRUE) + dcauchy(d, 0, r, log = TRUE)
  }
  a <- max(lo, -1e3)
  b <- min(hi, 1e3)

  # Scale by the joint mode so deep tails do not underflow.
  start <- c(1, min(max(t / sqrt(n), a + 1e-9), b - 1e-9))
  mode <- optim(start, function(p) -log_f(p[1], p[2]),
    method = "L-BFGS-B", lower = c(1e-6, a), upper = c(10, b),
    control = list(factr = 1e2)
  )
  peak <- -mode$value
  s0 <- mode$par[1]

  inner <- function(s) {
    sapply(s, function(si) {
      # phi(t s - d sqrt(n)) has width 1/sqrt(n) in d and the Cauchy factor is
      # smooth, so the mass lies within 40 widths of the point of I nearest d*.
      d_star <- min(max(t * si / sqrt(n), a), b)
      w <- 40 / sqrt(n)
      cuts <- unique(c(max(a, d_star - w), if (d_star > a && d_star < b) d_star, min(b, d_star + w)))
      f <- function(d) exp(log_f(si, d) - peak)
      sum(sapply(seq_len(length(cuts) - 1), function(i) {
        integrate(f, cuts[i], cuts[i + 1], rel.tol = 1e-12, abs.tol = 0, subdivisions = 2000)$value
      }))
    })
  }
  w <- 1 / sqrt(df + t^2)
  cuts <- sort(unique(c(1e-9, max(1e-9, s0 - 60 * w), s0, s0 + 60 * w)))
  total <- sum(sapply(seq_len(length(cuts) - 1), function(i) {
    integrate(inner, cuts[i], cuts[i + 1], rel.tol = 1e-11, abs.tol = 0, subdivisions = 2000)$value
  }))
  peak + log(total) - log_mass - dt(t, df, log = TRUE)
}

log_bf_jzs <- function(t, n, df, r) {
  log_f <- function(g) {
    -0.5 * log1p(n * g * r^2) - (df + 1) / 2 * log1p(t^2 / ((1 + n * g * r^2) * df)) +
      (-0.5 * log(2 * pi) - 1.5 * log(g) - 1 / (2 * g))
  }
  opt <- optimize(function(lg) -log_f(exp(lg)), c(-30, 30))
  peak <- -opt$objective
  lg0 <- opt$minimum
  f <- function(lg) {
    v <- exp(log_f(exp(lg)) - peak + lg)
    v[!is.finite(v)] <- 0
    v
  }
  total <- integrate(f, lg0 - 40, lg0, rel.tol = 1e-13)$value +
    integrate(f, lg0, lg0 + 60, rel.tol = 1e-13)$value
  peak + log(total) + (df + 1) / 2 * log1p(t^2 / df)
}

designs <- list(c(20, 0), c(34, 0), c(100, 0), c(20, 25), c(50, 50))
t_values <- c(-20, -13, -8, -6, -3, 3, 6, 8, 13, 20, 30)
intervals <- list(c(-Inf, Inf), c(-Inf, 0), c(0, Inf), c(-1, 1), c(-0.5, 0.5), c(0.2, 0.8))
rscales <- c(0.707, 1)

rows <- list()
for (design in designs) {
  n1 <- design[1]
  n2 <- design[2]
  n <- if (n2 > 0) n1 * n2 / (n1 + n2) else n1
  df <- if (n2 > 0) n1 + n2 - 2 else n1 - 1
  for (t in t_values) for (interval in intervals) for (r in rscales) {
    null_interval <- if (all(is.infinite(interval))) NULL else interval
    t_approximation <- FALSE
    bf <- withCallingHandlers(
      ttest.tstat(t, n1, n2, nullInterval = null_interval, rscale = r),
      message = function(m) {
        t_approximation <<- TRUE
        invokeRestart("muffleMessage")
      }
    )
    reference <- log_bf_reference(t, n, df, r, interval[1], interval[2])
    if (all(is.infinite(interval))) {
      jzs <- log_bf_jzs(t, n, df, r)
      if (abs(jzs - reference) > 1e-6) {
        stop(sprintf("reference and JZS disagree at t=%g n1=%g n2=%g r=%g: %.9f vs %.9f", t, n1, n2, r, reference, jzs))
      }
    }
    rows[[length(rows) + 1]] <- data.frame(
      t = t, n1 = n1, n2 = n2, lower = interval[1], upper = interval[2], rscale = r,
      bayesfactor_logbf = bf$bf, bayesfactor_t_approximation = t_approximation,
      reference_logbf = reference
    )
  }
}
out <- do.call(rbind, rows)
write.csv(out, "bayesfactor_ttest.csv", row.names = FALSE, quote = FALSE)
cat(nrow(out), "rows written to bayesfactor_ttest.csv\n")
