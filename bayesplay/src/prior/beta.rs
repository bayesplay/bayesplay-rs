use rmath::dbeta;

use serde::Deserialize;
use serde::Serialize;

use crate::common::Function;
use crate::common::Range;
use crate::common::Validate;

use super::Normalize;
use super::PriorError;

/// A Beta distribution prior for probability parameters.
///
/// The Beta distribution is defined on the interval [0, 1], making it ideal
/// for parameters that represent probabilities. It's the conjugate prior for
/// the binomial likelihood.
///
/// # Parameters
///
/// * `alpha` - First shape parameter (must be non-negative)
/// * `beta` - Second shape parameter (must be non-negative)
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
    /// * `alpha` - First shape parameter (must be non-negative)
    /// * `beta` - Second shape parameter (must be non-negative)
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
/// 1. Both shape parameters are non-negative
/// 2. The range bounds are within [0, 1]
/// 3. The range bounds are not equal
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

        if lower < 0.0 || upper > 1.0 {
            errors.push(PriorError::InvalidRangeBounds);
        }

        if self.alpha.is_sign_negative() {
            errors.push(PriorError::InvalidShapeParameter(self.alpha));
        }

        if self.beta.is_sign_negative() {
            errors.push(PriorError::InvalidShapeParameter(self.beta));
        }

        if lower == upper {
            errors.push(PriorError::InvalidRange);
        }

        PriorError::from_errors(errors)
    }
}

/// Evaluates the Beta PDF at a given value.
///
/// Returns an error if x is outside [0, 1].
///
/// # Note
///
/// Truncated Beta priors are not currently supported.
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
impl Function<f64, f64, PriorError> for BetaPrior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;

        // Truncated betas are not currently supported
        if !self.has_default_range() {
            return Err(PriorError::InvalidRange);
        };

        let shape1 = self.alpha;
        let shape2 = self.beta;
        let k = 1.0;

        if !(0.0..=1.0).contains(&x) {
            Err(PriorError::InvalidValue(x, 0.0, 1.0))?;
        }

        Ok(dbeta!(x = x, shape1 = shape1, shape2 = shape2)
            .map_err(PriorError::DistributionError)?
            * k)
    }
}

/// Beta priors over the full [0, 1] range are already normalized.
///
/// # Note
///
/// Truncated Beta priors are not currently supported and will panic.
impl Normalize for BetaPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        if !self.has_default_range() {
            todo!("Truncated beta priors are not allowed!")
        }
        Ok(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::Prior;
    use approx::assert_relative_eq;

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
