use crate::common::truncated_normalization;
use rmath::{dbeta, pbeta};

use serde::Deserialize;
use serde::Serialize;

use crate::common::Range;
use crate::common::Validate;

use super::Normalize;
use super::PriorDensity;
use super::PriorError;

/// A Beta distribution prior for probability parameters.
///
/// The Beta distribution is defined on the interval [0, 1], making it ideal
/// for parameters that represent probabilities. It's the conjugate prior for
/// the binomial likelihood.
///
/// # Parameters
///
/// * `alpha` - First shape parameter (must be finite and positive)
/// * `beta` - Second shape parameter (must be finite and positive)
/// * `range` - Optional bounds (must be within [0, 1])
///
/// # Common Parameterizations
///
/// | alpha | beta | Interpretation |
/// |-------|------|----------------|
/// | 1 | 1 | Uniform (uninformative) |
/// | 0.5 | 0.5 | Jeffreys' prior |
/// | 2 | 2 | Symmetric, mode at 0.5 |
/// | 1 | 2 | Favors lower values |
/// | 2 | 1 | Favors higher values |
///
/// # Mathematical Form
///
/// The Beta PDF is:
///
/// f(x; α, β) = x^(α-1) × (1-x)^(β-1) / B(α, β)
///
/// where B(α, β) is the beta function.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Uniform prior (Beta(1,1))
/// let uniform = BetaPrior::new(1.0, 1.0, (None, None));
///
/// // Jeffreys' prior (Beta(0.5, 0.5))
/// let jeffreys = BetaPrior::new(0.5, 0.5, (None, None));
///
/// // Prior favoring values around 0.5
/// let centered = BetaPrior::new(5.0, 5.0, (None, None));
///
/// // Combine with a binomial likelihood (conjugate pair)
/// let likelihood: Likelihood = BinomialLikelihood::new(7.0, 10.0).into();
/// let model: Model = likelihood * Prior::from(uniform);
/// ```
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct BetaPrior {
    /// First shape parameter (α).
    pub alpha: f64,
    /// Second shape parameter (β).
    pub beta: f64,
    /// Optional truncation bounds (must be within [0, 1]).
    pub range: (Option<f64>, Option<f64>),
}

impl BetaPrior {
    /// Creates a new Beta distribution prior.
    ///
    /// # Arguments
    ///
    /// * `alpha` - First shape parameter (must be finite and positive)
    /// * `beta` - Second shape parameter (must be finite and positive)
    /// * `range` - Optional bounds (must be within [0, 1])
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// // Uniform prior
    /// let uniform = BetaPrior::new(1.0, 1.0, (None, None));
    ///
    /// // Informative prior centered around 0.7
    /// let informative = BetaPrior::new(7.0, 3.0, (None, None));
    /// ```
    pub fn new(alpha: f64, beta: f64, range: (Option<f64>, Option<f64>)) -> Self {
        BetaPrior { alpha, beta, range }
    }
}

/// The natural range of a Beta prior is [0, 1].
impl Range for BetaPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        self.range
    }

    fn default_range(&self) -> (f64, f64) {
        (0.0, 1.0)
    }
}

/// Validates the Beta prior parameters.
///
/// Ensures:
/// 1. Both shape parameters are finite and positive
/// 2. The range bounds are within [0, 1]
/// 3. The lower bound is strictly below the upper bound
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Valid parameters
/// let valid = BetaPrior::new(1.0, 1.0, (None, None));
/// assert!(valid.validate().is_ok());
///
/// // Invalid: negative shape parameter
/// let invalid = BetaPrior::new(-1.0, 1.0, (None, None));
/// assert!(matches!(invalid.validate(), Err(PriorError::InvalidShapeParameter(_))));
///
/// // Invalid: range outside [0, 1]
/// let invalid = BetaPrior::new(1.0, 1.0, (Some(-0.5), None));
/// assert!(matches!(invalid.validate(), Err(PriorError::InvalidRangeBounds)));
/// ```
impl Validate<PriorError> for BetaPrior {
    fn validate(&self) -> Result<(), PriorError> {
        let mut errors: Vec<PriorError> = Vec::new();
        let (lower, upper) = self.range_or_default();

        if !lower.is_finite() || !upper.is_finite() || lower < 0.0 || upper > 1.0 {
            errors.push(PriorError::InvalidRangeBounds);
        }

        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            errors.push(PriorError::InvalidShapeParameter(self.alpha));
        }

        if !self.beta.is_finite() || self.beta <= 0.0 {
            errors.push(PriorError::InvalidShapeParameter(self.beta));
        }

        if lower >= upper {
            errors.push(PriorError::InvalidRange);
        }

        PriorError::from_errors(errors)
    }
}

/// Evaluates the Beta PDF at a given value.
///
/// Returns an error if x is outside [0, 1].
///
/// Within [0, 1], values outside the truncation interval have zero density.
/// Inside the interval, the density is divided by its prior probability mass.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Beta(1,1) is uniform on [0,1], so density = 1 everywhere
/// let uniform = BetaPrior::new(1.0, 1.0, (None, None));
/// assert!((uniform.function(0.5).unwrap() - 1.0).abs() < 1e-10);
///
/// // Beta(2,2) has mode at 0.5
/// let beta22 = BetaPrior::new(2.0, 2.0, (None, None));
/// let at_mode = beta22.function(0.5).unwrap();
/// let off_mode = beta22.function(0.3).unwrap();
/// assert!(at_mode > off_mode);
/// ```
impl PriorDensity for BetaPrior {
    fn ln_density(&self, x: f64, ln_normalizer: f64) -> Result<f64, PriorError> {
        if !(0.0..=1.0).contains(&x) {
            Err(PriorError::InvalidValue(x, 0.0, 1.0))?;
        }

        let (lower, upper) = self.range_or_default();
        if !(lower..=upper).contains(&x) {
            return Ok(f64::NEG_INFINITY);
        }
        let log_density = dbeta!(x = x, shape1 = self.alpha, shape2 = self.beta, log = true)
            .map_err(PriorError::DistributionError)?;
        Ok(log_density - ln_normalizer)
    }
}

/// Probability mass of the truncation interval under the untruncated Beta prior.
impl Normalize for BetaPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        self.validate()?;
        let (lower, upper) = self.range_or_default();
        let mass = truncated_normalization(lower, upper, |q, lower_tail| {
            pbeta!(
                q = q,
                shape1 = self.alpha,
                shape2 = self.beta,
                lower_tail = lower_tail
            )
            .map_err(PriorError::DistributionError)
        })?;
        if !mass.is_finite() || mass <= 0.0 {
            return Err(PriorError::NormalizingError);
        }
        Ok(mass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Function;
    use crate::prior::Prior;
    use approx::assert_relative_eq;

    #[test]
    fn truncated_beta_density_and_tail_mass() {
        let prior = BetaPrior::new(2.0, 2.0, (Some(0.25), Some(0.75)));
        assert_relative_eq!(prior.normalize().unwrap(), 11.0 / 16.0, epsilon = 1e-14);
        assert_relative_eq!(prior.function(0.5).unwrap(), 24.0 / 11.0, epsilon = 1e-14);
        assert_eq!(prior.function(0.2).unwrap(), 0.0);
        assert_eq!(prior.function(0.8).unwrap(), 0.0);
        let tail = BetaPrior::new(1.0, 5.0, (Some(0.999), None));
        assert_relative_eq!(
            tail.normalize().unwrap(),
            (1.0_f64 - 0.999).powi(5),
            max_relative = 1e-12
        );
        let lower = BetaPrior::new(2.0, 1.0, (None, Some(0.5)));
        assert_relative_eq!(lower.normalize().unwrap(), 0.25, epsilon = 1e-14);
    }

    #[test]
    fn truncated_beta_binomial_model_matches_polynomial_integral() {
        use crate::prelude::*;
        let likelihood: Likelihood = BinomialLikelihood::new(1.0, 1.0).into();
        let prior: Prior = BetaPrior::new(2.0, 1.0, (Some(0.5), None)).into();
        let model = likelihood * prior;
        assert_relative_eq!(model.integral().unwrap().value, 7.0 / 9.0, epsilon = 1e-12);
        let posterior = model.posterior().unwrap();
        assert_relative_eq!(
            posterior.integrate(None, Some(0.75)).unwrap(),
            19.0 / 56.0,
            epsilon = 1e-12
        );
        assert_eq!(posterior.function(0.25).unwrap(), 0.0);
    }

    #[test]
    fn invalid_beta_parameters_return_errors_without_panicking() {
        for shape in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(
                BetaPrior::new(shape, 1.0, (None, None))
                    .normalize()
                    .is_err()
            );
            assert!(
                BetaPrior::new(1.0, shape, (None, None))
                    .normalize()
                    .is_err()
            );
        }
        for bounds in [
            (Some(0.8), Some(0.2)),
            (Some(0.5), Some(0.5)),
            (Some(f64::NAN), None),
            (None, Some(f64::INFINITY)),
        ] {
            assert!(BetaPrior::new(1.0, 1.0, bounds).normalize().is_err());
        }
    }

    #[test]
    fn test_beta_prior_creation() {
        let prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).into();
        if let Prior::Beta(p) = prior {
            assert_eq!(p.alpha, 1.0);
            assert_eq!(p.beta, 1.0);
            assert_eq!(p.range, (None, None));
        } else {
            panic!("Expected Beta prior");
        }
    }

    #[test]
    fn test_beta_uniform_density() {
        // Beta(1, 1) is uniform on [0,1], so density = 1.0 everywhere
        let beta = BetaPrior::new(1.0, 1.0, (None, None));
        assert_relative_eq!(beta.function(0.5).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(beta.function(0.1).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(beta.function(0.9).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_beta_symmetric_mode() {
        // Beta(2, 2) has mode at 0.5
        let beta = BetaPrior::new(2.0, 2.0, (None, None));
        let at_mode = beta.function(0.5).unwrap();
        let off_mode = beta.function(0.3).unwrap();
        assert!(at_mode > off_mode);
    }

    #[test]
    fn test_beta_density_known_value() {
        // Beta(2,5) at x=0.3: dbeta(0.3, 2, 5) ≈ 2.1609
        // R: dbeta(0.3, 2, 5) = 2.1609
        let beta = BetaPrior::new(2.0, 5.0, (None, None));
        let density = beta.function(0.3).unwrap();
        assert_relative_eq!(density, 2.1609, epsilon = 1e-3);
    }

    #[test]
    fn test_beta_validation_valid() {
        let beta = BetaPrior::new(1.0, 1.0, (None, None));
        assert!(beta.validate().is_ok());
    }

    #[test]
    fn test_beta_validation_negative_alpha() {
        let beta = BetaPrior::new(-1.0, 1.0, (None, None));
        assert!(matches!(
            beta.validate(),
            Err(PriorError::InvalidShapeParameter(_))
        ));
    }

    #[test]
    fn test_beta_validation_negative_beta() {
        let beta = BetaPrior::new(1.0, -1.0, (None, None));
        assert!(matches!(
            beta.validate(),
            Err(PriorError::InvalidShapeParameter(_))
        ));
    }

    #[test]
    fn test_beta_validation_invalid_range_bounds() {
        let beta = BetaPrior::new(1.0, 1.0, (Some(-0.5), None));
        assert!(matches!(
            beta.validate(),
            Err(PriorError::InvalidRangeBounds)
        ));
    }

    #[test]
    fn test_beta_value_outside_0_1() {
        // function() should return InvalidValue for x outside [0, 1]
        let beta = BetaPrior::new(1.0, 1.0, (None, None));
        assert!(beta.function(1.5).is_err());
        assert!(beta.function(-0.1).is_err());
    }

    #[test]
    fn test_beta_normalization() {
        let beta = BetaPrior::new(2.0, 5.0, (None, None));
        assert_eq!(beta.normalize().unwrap(), 1.0);
    }

    #[test]
    fn test_beta_default_range() {
        let beta = BetaPrior::new(1.0, 1.0, (None, None));
        assert_eq!(beta.default_range(), (0.0, 1.0));
    }
}
