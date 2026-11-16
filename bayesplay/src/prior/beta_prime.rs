use rmath::{lbeta, pbeta};
use serde::{Deserialize, Serialize};

use super::{Normalize, PriorError, validate_positive_range};
use crate::common::{LogFunction, Range, Validate, truncated_normalization};

/// A symmetric beta-prime prior for a positive ratio.
///
/// `X ~ BetaPrime(alpha, alpha)`, or equivalently
/// `X / (1 + X) ~ Beta(alpha, alpha)`.
/// The distribution is reciprocal-symmetric and places half its mass on
/// either side of one.
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct BetaPrimePrior {
    pub alpha: f64,
    pub range: (Option<f64>, Option<f64>),
}

impl BetaPrimePrior {
    pub fn new(alpha: f64, range: (Option<f64>, Option<f64>)) -> Self {
        Self { alpha, range }
    }

    fn cdf(&self, q: f64, lower_tail: bool) -> Result<f64, PriorError> {
        let probability = if q <= 0.0 {
            if lower_tail { 0.0 } else { 1.0 }
        } else if q.is_infinite() {
            if lower_tail { 1.0 } else { 0.0 }
        } else {
            let (rho, beta_lower_tail) = if q <= 1.0 {
                (q / (1.0 + q), lower_tail)
            } else {
                let reciprocal = q.recip();
                (reciprocal / (1.0 + reciprocal), !lower_tail)
            };
            if rho < 1e-8 {
                let lower_probability = self.small_symmetric_beta_tail(rho)?;
                if beta_lower_tail {
                    lower_probability
                } else {
                    1.0 - lower_probability
                }
            } else {
                pbeta!(
                    q = rho,
                    shape1 = self.alpha,
                    shape2 = self.alpha,
                    lower_tail = beta_lower_tail
                )
                .map_err(PriorError::DistributionError)?
            }
        };
        Ok(probability)
    }

    /// Stable `I_x(alpha, alpha)` for very small `x`, where the generic beta
    /// CDF can round a representable tail probability to zero.
    fn small_symmetric_beta_tail(&self, x: f64) -> Result<f64, PriorError> {
        if x == 0.0 {
            return Ok(0.0);
        }
        let mut term = 1.0;
        let mut series = 1.0;
        for n in 1..=64 {
            let previous = (n - 1) as f64;
            term *= (self.alpha + previous) * (1.0 - self.alpha + previous)
                / ((self.alpha + 1.0 + previous) * n as f64)
                * x;
            series += term;
            if term.abs() <= series.abs() * f64::EPSILON {
                break;
            }
        }
        let log_leading = self.alpha * x.ln()
            - self.alpha.ln()
            - lbeta(self.alpha, self.alpha).map_err(PriorError::DistributionError)?;
        Ok(log_leading.exp() * series)
    }
}

impl Range for BetaPrimePrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        self.range
    }

    fn default_range(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
}

impl Validate<PriorError> for BetaPrimePrior {
    fn validate(&self) -> Result<(), PriorError> {
        let mut errors = Vec::new();
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            errors.push(PriorError::InvalidShapeParameter(self.alpha));
        }
        if let Err(error) = validate_positive_range(self.range) {
            errors.push(error);
        }
        PriorError::from_errors(errors)
    }
}

impl Normalize for BetaPrimePrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        self.validate()?;
        let (lower, upper) = self.range_or_default();
        let mass = truncated_normalization(lower, upper, |x, lower_tail| self.cdf(x, lower_tail))?;
        if !mass.is_finite() || mass <= 0.0 {
            return Err(PriorError::NormalizingError);
        }
        Ok(mass)
    }
}

impl LogFunction<PriorError> for BetaPrimePrior {
    fn ln_function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;
        let (lower, upper) = self.range_or_default();
        if !x.is_finite() || x <= 0.0 || !(lower..=upper).contains(&x) {
            return Ok(f64::NEG_INFINITY);
        }

        let log_x = x.ln();
        let log_one_plus_x = if x <= 1.0 {
            x.ln_1p()
        } else {
            log_x + (-log_x).exp().ln_1p()
        };
        let log_density = -lbeta(self.alpha, self.alpha).map_err(PriorError::DistributionError)?
            + (self.alpha - 1.0) * log_x
            - 2.0 * self.alpha * log_one_plus_x;
        Ok(log_density - self.normalize()?.ln())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::common::Family;
    use crate::common::Function;
    use crate::prior::{Prior, PriorFamily};

    #[test]
    fn known_densities_and_dispatch() {
        let prior = BetaPrimePrior::new(0.5, (None, None));
        assert_relative_eq!(
            prior.function(1.0).unwrap(),
            1.0 / (2.0 * std::f64::consts::PI),
            epsilon = 1e-14
        );
        assert_relative_eq!(
            prior.function(2.0).unwrap(),
            1.0 / (3.0 * 2.0_f64.sqrt() * std::f64::consts::PI),
            epsilon = 1e-14
        );
        let erased: Prior = prior.into();
        assert_eq!(erased.family(), PriorFamily::BetaPrime);
        assert_relative_eq!(
            erased.function(1.0).unwrap(),
            1.0 / (2.0 * std::f64::consts::PI),
            epsilon = 1e-14
        );
    }

    #[test]
    fn reciprocal_symmetry_and_support() {
        let prior = BetaPrimePrior::new(2.16, (None, None));
        for x in [0.4, 0.75, 1.5, 3.0] {
            assert_relative_eq!(
                prior.function(x).unwrap(),
                prior.function(1.0 / x).unwrap() / x.powi(2),
                epsilon = 1e-12
            );
        }
        for x in [-1.0, 0.0, f64::INFINITY, f64::NAN] {
            assert_eq!(prior.function(x).unwrap(), 0.0);
        }
    }

    #[test]
    fn representable_upper_tail_density_and_mass_do_not_underflow() {
        let prior = BetaPrimePrior::new(0.5, (Some(1e20), Some(4e20)));
        assert_relative_eq!(
            prior.normalize().unwrap(),
            3.183_098_861_837_907e-11,
            max_relative = 1e-12
        );
        let full = BetaPrimePrior::new(0.5, (None, None));
        assert_relative_eq!(
            full.function(1e200).unwrap(),
            3.183_098_861_837_907e-301,
            max_relative = 1e-12
        );
    }

    #[test]
    fn truncation_uses_exact_beta_mass() {
        let full = BetaPrimePrior::new(0.5, (None, None));
        let upper = BetaPrimePrior::new(0.5, (Some(1.0), None));
        let lower = BetaPrimePrior::new(0.5, (Some(0.0), Some(1.0)));
        assert_eq!(full.normalize().unwrap(), 1.0);
        assert_relative_eq!(upper.normalize().unwrap(), 0.5, epsilon = 1e-15);
        assert_relative_eq!(lower.normalize().unwrap(), 0.5, epsilon = 1e-15);
        assert_relative_eq!(
            upper.function(2.0).unwrap(),
            2.0 * full.function(2.0).unwrap(),
            epsilon = 1e-14
        );
        assert_eq!(upper.function(0.5).unwrap(), 0.0);
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(matches!(
            BetaPrimePrior::new(0.0, (None, None)).validate(),
            Err(PriorError::InvalidShapeParameter(_))
        ));
        assert!(matches!(
            BetaPrimePrior::new(0.5, (Some(-1.0), Some(1.0))).validate(),
            Err(PriorError::InvalidPositiveRangeBounds)
        ));
        assert!(matches!(
            BetaPrimePrior::new(0.5, (Some(2.0), Some(1.0))).validate(),
            Err(PriorError::InvalidRange)
        ));
    }
}
