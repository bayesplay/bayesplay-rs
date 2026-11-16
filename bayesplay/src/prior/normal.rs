use serde::{Deserialize, Serialize};

use super::Normalize;
use super::PriorDensity;
use super::PriorError;
use super::{is_valid_parameter, validate_range};
use crate::common::Range;
use crate::common::Validate;
use crate::common::truncated_ln_normalization;
///
/// A normal distribution prior with customizable parameters.
///
/// # Parameters
///
/// * `mean` - The mean (μ) of the normal distribution
/// * `sd` - The standard deviation (σ) of the normal distribution
/// * `range` - Optional bounds for the distribution as (lower, upper)
///
/// # Examples
///
/// ```
/// use bayesplay::prelude::*;
///
/// // Create a standard normal distribution (mean=0, sd=1)
/// let standard_normal = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
///
/// // Create a normal distribution with custom mean and standard deviation
/// let custom_normal = NormalPrior::new(2.5, 0.5, (None, None)).unwrap();
///
/// // Create a truncated normal distribution (bounded between 0 and 5)
/// let truncated_normal = NormalPrior::new(2.5, 1.0, (Some(0.0), Some(5.0))).unwrap();
/// ```

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct NormalPrior {
    pub mean: f64,
    pub sd: f64,
    pub range: (Option<f64>, Option<f64>),
}

/// Implementation of the NormalPrior struct.
impl NormalPrior {
    /// Creates a new normal distribution prior.
    ///
    /// # Arguments
    ///
    /// * `mean` - The mean (μ) of the distribution
    /// * `sd` - The standard deviation (σ) of the distribution
    /// * `range` - Optional bounds as (lower_bound, upper_bound)
    ///
    /// # Returns
    ///
    /// A `Prior` enum variant containing the normal distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// // Unbounded normal distribution
    /// let normal = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
    ///
    /// // Normal distribution with lower bound only
    /// let lower_bounded = NormalPrior::new(0.0, 1.0, (Some(-2.0), None)).unwrap();
    ///
    /// // Normal distribution with upper bound only
    /// let upper_bounded = NormalPrior::new(0.0, 1.0, (None, Some(2.0))).unwrap();
    ///
    /// // Normal distribution with both bounds
    /// let bounded = NormalPrior::new(0.0, 1.0, (Some(-2.0), Some(2.0))).unwrap();
    /// ```
    pub fn new(mean: f64, sd: f64, range: (Option<f64>, Option<f64>)) -> Result<Self, PriorError> {
        let value = NormalPrior { mean, sd, range };
        value.validate()?;
        Ok(value)
    }
}

/// Implementation of validation for the NormalPrior.
///
/// Validates that:
/// 1. The standard deviation is positive
/// 2. The range bounds are not equal (except for default infinite bounds)
impl Validate<PriorError> for NormalPrior {
    /// Validates the parameters of the normal distribution.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the parameters are valid
    /// * `Err(PriorError)` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// // Valid parameters
    /// let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
    /// assert!(prior.validate().is_ok());
    ///
    ///
    /// // Parameters changed after construction are caught by `validate`
    /// let mut prior = prior;
    /// prior.sd = -1.0;
    /// assert!(matches!(
    ///     prior.validate(),
    ///     Err(PriorError::InvalidStandardDeviation(sd)) if sd == -1.0
    /// ));
    ///
    /// // Invalid: Equal bounds
    /// prior.sd = 1.0;
    /// prior.range = (Some(2.0), Some(2.0));
    /// assert!(matches!(prior.validate(), Err(PriorError::InvalidRange)));
    /// ```
    fn validate(&self) -> Result<(), PriorError> {
        let mut errors = Vec::new();

        if !is_valid_parameter(self.mean, false) {
            errors.push(PriorError::InvalidLocation(self.mean));
        }

        if !is_valid_parameter(self.sd, true) {
            errors.push(PriorError::InvalidStandardDeviation(self.sd));
        }

        if let Err(e) = validate_range(self.range_or_default()) {
            errors.push(e);
        }

        PriorError::from_errors(errors)
    }
}

/// Implementation of the Range trait for NormalPrior.
impl Range for NormalPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        self.range
    }
    fn default_range(&self) -> (f64, f64) {
        (-f64::INFINITY, f64::INFINITY)
    }
}

/// Implementation of normalization for the NormalPrior.
impl Normalize for NormalPrior {
    /// Calculates the normalization constant for the normal distribution.
    ///
    /// For unbounded normal distributions, this returns 1.0.
    /// For truncated distributions, calculates the area under the curve within the bounds.
    ///
    /// # Returns
    ///
    /// * `Ok(normalization_constant)` - The normalization constant
    /// * `Err(PriorError)` - If normalization fails (e.g., the area is zero)
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    /// use std::f64::consts::PI;
    ///
    /// // Standard normal distribution (unbounded)
    /// if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (None, None)).unwrap().into() {
    ///     assert_eq!(prior.normalize().unwrap(), 1.0);
    /// }
    ///
    /// // Truncated normal to include only values between -1.0 and 1.0
    /// if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).unwrap().into() {
    ///     let norm = prior.normalize().unwrap();
    ///     // The normalization should equal the probability mass between -1 and 1
    ///     // for a standard normal, which is about 0.6827
    ///     assert!((norm - 0.6827).abs() < 0.01);
    /// }
    /// ```
    fn normalize(&self) -> Result<f64, PriorError> {
        Ok(self.ln_normalize()?.exp())
    }

    fn ln_normalize(&self) -> Result<f64, PriorError> {
        let (lower, upper) = self.range_or_default();
        let ln_mass = truncated_ln_normalization(lower, upper, |x, lower_tail| {
            rmath::pnorm(x, self.mean, self.sd, lower_tail, true)
                .map_err(PriorError::DistributionError)
        })?;
        if ln_mass.is_nan() || ln_mass == f64::NEG_INFINITY {
            return Err(PriorError::NormalizingError);
        }
        Ok(ln_mass)
    }
}

/// Implementation of the Function trait for NormalPrior to calculate probability density.
/// Calculates the probability density function (PDF) of the normal distribution at point x.
///
/// For truncated distributions, returns 0.0 if x is outside the bounds.
///
/// # Arguments
///
/// * `x` - The point at which to evaluate the PDF
///
/// # Returns
///
/// * `Ok(density)` - The probability density at point x
/// * `Err(PriorError)` - If validation fails or distribution calculations error
///
/// # Examples
///
/// ```
/// use bayesplay::prelude::*;
/// use std::f64::consts::PI;
///
/// // Standard normal PDF at x=0 should be 1/sqrt(2π)
/// if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (None, None)).unwrap().into() {
///     let expected = 1.0 / (2.0 * PI).sqrt();
///     let actual = prior.function(0.0).unwrap();
///     assert!((actual - expected).abs() < 1e-10);
/// }
///
/// // Evaluate at the mean of a custom normal distribution
/// if let Prior::Normal(prior) = NormalPrior::new(2.0, 0.5, (None, None)).unwrap().into() {
///     let expected = 1.0 / (2.0 * PI * 0.5 * 0.5).sqrt();
///     let actual = prior.function(2.0).unwrap();
///     assert!((actual - expected).abs() < 1e-10);
/// }
///
/// // Truncated normal: value outside bounds should return 0
/// if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).unwrap().into() {
///     assert_eq!(prior.function(2.0).unwrap(), 0.0);
/// }
///
/// // Truncated normal: value inside bounds needs normalization
/// if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).unwrap().into() {
///     let density = prior.function(0.0).unwrap();
///     let unnormalized = 1.0 / (2.0 * PI).sqrt();
///     let normalization = prior.normalize().unwrap();
///     assert!((density - unnormalized/normalization).abs() < 1e-10);
/// }
/// ```
impl PriorDensity for NormalPrior {
    fn ln_density(&self, x: f64, ln_normalizer: f64) -> Result<f64, PriorError> {
        let log_density =
            rmath::dnorm(x, self.mean, self.sd, true).map_err(PriorError::DistributionError)?;
        let (lower, upper) = self.range_or_default();
        if !(lower..=upper).contains(&x) {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(log_density - ln_normalizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Function;
    use crate::prior::Prior;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_normal_prior_creation() {
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap().into();
        if let Prior::Normal(p) = prior {
            assert_eq!(p.mean, 0.0);
            assert_eq!(p.sd, 1.0);
            assert_eq!(p.range, (None, None));
        } else {
            panic!("Expected Normal prior");
        }
    }

    #[test]
    fn test_normal_prior_validation() {
        // Test valid prior
        {
            let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
            assert!(prior.validate().is_ok());
        }

        // Test invalid standard deviation
        {
            let prior = NormalPrior {
                mean: 0.0,
                sd: -1.0,
                range: (None, None),
            };
            let result = prior.validate();
            assert!(result.is_err());
            match result {
                Err(PriorError::InvalidStandardDeviation(sd)) => assert_eq!(sd, -1.0),
                _ => panic!("Expected InvalidStandardDeviation error"),
            }
        }

        // Test invalid range
        {
            let prior = NormalPrior {
                mean: 0.0,
                sd: 1.0,
                range: (Some(2.0), Some(2.0)),
            };
            let result = prior.validate();
            assert!(result.is_err());
            match result {
                Err(PriorError::InvalidRange) => {}
                _ => panic!("Expected InvalidRange error"),
            }
        }

        // Test multiple validation errors
        {
            let prior = NormalPrior {
                mean: 0.0,
                sd: -1.0,
                range: (Some(2.0), Some(2.0)),
            };
            let result = prior.validate();
            assert!(result.is_err());
            match result {
                Err(PriorError::MultipleErrors(errors)) => {
                    assert_eq!(errors.len(), 2);
                    assert!(matches!(
                        errors[0],
                        PriorError::InvalidStandardDeviation(-1.0)
                    ));
                    assert!(matches!(errors[1], PriorError::InvalidRange));
                }
                _ => panic!("Expected MultipleErrors"),
            }
        }
    }

    #[test]
    fn test_normalization() {
        // Test unbounded normal distribution
        {
            let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
            assert_eq!(prior.normalize().unwrap(), 1.0);
        }

        // Test lower bounded normal distribution
        {
            let prior = NormalPrior::new(0.0, 1.0, (Some(-1.0), None)).unwrap();
            let norm = prior.normalize().unwrap();
            let expected = rmath::pnorm(-1.0, 0.0, 1.0, false, false).unwrap();
            assert_relative_eq!(norm, expected, epsilon = 1e-10);
        }

        // Test upper bounded normal distribution
        {
            let prior = NormalPrior::new(0.0, 1.0, (None, Some(1.0))).unwrap();
            let norm = prior.normalize().unwrap();
            let expected = rmath::pnorm(1.0, 0.0, 1.0, true, false).unwrap();
            assert_relative_eq!(norm, expected, epsilon = 1e-10);
        }

        // Test bounded normal distribution
        {
            let prior = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).unwrap();
            let norm = prior.normalize().unwrap();
            // Expected: approximately 0.6827
            let expected = 0.6827;
            assert_relative_eq!(norm, expected, epsilon = 1e-3);
        }

        // Test normalization error case (impossible range)
        {
            let mut prior = NormalPrior {
                mean: 0.0,
                sd: 1.0,
                range: (Some(5.0), Some(1.0)),
            };
            // Set the range to 0 so that normalization fails
            prior.range = (Some(5.0), Some(5.0));
            assert!(prior.normalize().is_err());
        }
    }

    #[test]
    fn test_pdf_calculation() {
        // Test standard normal PDF at various points
        {
            let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
            let expected_at_mean = 1.0 / (2.0 * PI).sqrt();
            assert_relative_eq!(
                prior.function(0.0).unwrap(),
                expected_at_mean,
                epsilon = 1e-10
            );

            // Test at x = 1 (one standard deviation away from mean)
            let expected_at_1sd = expected_at_mean * (-0.5_f64).exp();
            assert_relative_eq!(
                prior.function(1.0).unwrap(),
                expected_at_1sd,
                epsilon = 1e-10
            );

            // Test at x = -1 (should be symmetric)
            assert_relative_eq!(
                prior.function(-1.0).unwrap(),
                expected_at_1sd,
                epsilon = 1e-10
            );
        }

        // Test custom normal distribution
        {
            let prior = NormalPrior::new(2.0, 0.5, (None, None)).unwrap();
            let expected_at_mean = 1.0 / (2.0 * PI * 0.5 * 0.5).sqrt();
            assert_relative_eq!(
                prior.function(2.0).unwrap(),
                expected_at_mean,
                epsilon = 1e-10
            );
        }

        // Test truncated normal distribution - value outside bounds
        {
            let prior = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).unwrap();
            assert_eq!(prior.function(2.0).unwrap(), 0.0);
            assert_eq!(prior.function(-2.0).unwrap(), 0.0);
        }

        // Test truncated normal distribution - value inside bounds
        {
            let prior = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).unwrap();
            let unnormalized = 1.0 / (2.0 * PI).sqrt();
            let normalization = prior.normalize().unwrap();
            assert_relative_eq!(
                prior.function(0.0).unwrap(),
                unnormalized / normalization,
                epsilon = 1e-10
            );
        }

        // Test with invalid parameters
        {
            let prior = NormalPrior {
                mean: 0.0,
                sd: -1.0,
                range: (None, None),
            };
            assert!(prior.function(0.0).is_err());
        }
    }

    #[test]
    fn test_range_behavior() {
        {
            let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
            assert_eq!(prior.range(), (None, None));
            assert_eq!(prior.default_range(), (-f64::INFINITY, f64::INFINITY));
            assert!(prior.has_default_range());
        }

        {
            let prior = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).unwrap();
            assert_eq!(prior.range(), (Some(-1.0), Some(1.0)));
            assert!(!prior.has_default_range());
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test with extremely large values
        {
            let prior = NormalPrior::new(0.0, 1.0, (None, Some(1e10))).unwrap();
            // Should be close to zero because the area under the curve beyond 10^10 is negligible
            assert_relative_eq!(prior.normalize().unwrap(), 1.0, epsilon = 1e-5);
        }

        // Test with extremely small standard deviation
        {
            let prior = NormalPrior::new(0.0, 1e-10, (None, None)).unwrap();
            // The PDF at the mean should be very large
            let value = prior.function(0.0).unwrap();
            assert!(value > 1e9);
        }
    }
}
