//! R-compatible probability distributions and numerical integration.
//!
//! Functions follow R's names and argument order (`dnorm(x, mean, sd, log)`,
//! `pnorm(q, mean, sd, lower_tail, log_p)`, ...), and are written to stay
//! accurate in the tails and on the log scale.

pub mod distributions;
pub mod error;
pub mod integration;

pub use distributions::beta::{dbeta, lbeta, pbeta};
pub use distributions::binomial::{dbinom, pbinom};
pub use distributions::cauchy::{dcauchy, pcauchy};
pub use distributions::chi_squared::dchisq;
pub use distributions::exponential::{dexp, pexp};
pub use distributions::normal::{dnorm, pnorm};
pub use distributions::t::{dnt, dt, dt_scaled, pnt, pt, pt_scaled};
pub use distributions::uniform::{dunif, punif};
pub use error::{Distribution, Error, Result};
pub use integration::{IntegrationError, IntegratorResult, Quadrature, ToleranceMode, integrate};
