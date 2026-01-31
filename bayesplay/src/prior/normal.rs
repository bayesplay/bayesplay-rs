use rmath::dnorm;

use rmath::pnorm;

use serde::{Deserialize, Serialize};

use super::Normalize;
use super::PriorError;
use crate::common::truncated_normalization;
use crate::common::Function;
use crate::common::Range;
use crate::common::Validate;
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
/// let standard_normal = NormalPrior::new(0.0, 1.0, (None, None));
///
/// // Create a normal distribution with custom mean and standard deviation
/// let custom_normal = NormalPrior::new(2.5, 0.5, (None, None));
///
/// // Create a truncated normal distribution (bounded between 0 and 5)
/// let truncated_normal = NormalPrior::new(2.5, 1.0, (Some(0.0), Some(5.0)));
/// ```

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct NormalPrior {
    pub mean: f64,
    pub sd: f64,
    range: (Option<f64>, Option<f64>),
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
    /// let normal = NormalPrior::new(0.0, 1.0, (None, None));
    ///
    /// // Normal distribution with lower bound only
    /// let lower_bounded = NormalPrior::new(0.0, 1.0, (Some(-2.0), None));
    ///
    /// // Normal distribution with upper bound only
    /// let upper_bounded = NormalPrior::new(0.0, 1.0, (None, Some(2.0)));
    ///
    /// // Normal distribution with both bounds
    /// let bounded = NormalPrior::new(0.0, 1.0, (Some(-2.0), Some(2.0)));
    /// ```
    pub fn new(mean: f64, sd: f64, range: (Option<f64>, Option<f64>)) -> Self {
        NormalPrior { mean, sd, range }
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
    /// let prior = NormalPrior::new(0.0, 1.0, (None, None));
    /// assert!(prior.validate().is_ok());
    ///
    ///
    /// // Invalid: Negative standard deviation
    /// let prior = NormalPrior::new(0.0, -1.0, (None, None));
    /// let result =prior.validate();
    /// assert!(result.is_err());
    /// if let Err(PriorError::InvalidStandardDeviation(sd)) = result {
    ///     assert_eq!(sd, -1.0);
    ///   }
    ///  
    ///
    /// // Invalid: Equal bounds
    /// let prior = NormalPrior::new(0.0, 1.0, (Some(2.0), Some(2.0)));
    /// let result = prior.validate();
    /// assert!(result.is_err());
    /// assert!(matches!(result, Err(PriorError::InvalidRange)));
    ///
    /// ```
    fn validate(&self) -> Result<(), PriorError> {
        let mut errors = Vec::new();

        if self.sd <= 0.0 {
            errors.push(PriorError::InvalidStandardDeviation(self.sd));
        }

        let (lower, upper) = self.range_or_default();
        if lower == upper {
            errors.push(PriorError::InvalidRange);
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
    /// if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (None, None)).into() {
    ///     assert_eq!(prior.normalize().unwrap(), 1.0);
    /// }
    ///
    /// // Truncated normal to include only values between -1.0 and 1.0
    /// if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).into() {
    ///     let norm = prior.normalize().unwrap();
    ///     // The normalization should equal the probability mass between -1 and 1
    ///     // for a standard normal, which is about 0.6827
    ///     assert!((norm - 0.6827).abs() < 0.01);
    /// }
    /// ```
    fn normalize(&self) -> Result<f64, PriorError> {
        let (lower, upper) = self.range_or_default();
        let res = truncated_normalization(lower, upper, |x, lower_tail| {
            pnorm!(
                q = x,
                mean = self.mean,
                sd = self.sd,
                lower_tail = lower_tail
            )
            .map_err(PriorError::DistributionError)
        })?;

        if res == 0.0 {
            Err(PriorError::NormalizingError)?;
        }
        Ok(res)
    }
}

/// Implementation of the Function trait for NormalPrior to calculate probability density.
impl Function<f64, f64, PriorError> for NormalPrior {
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
    /// if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (None, None)).into() {
    ///     let expected = 1.0 / (2.0 * PI).sqrt();
    ///     let actual = prior.function(0.0).unwrap();
    ///     assert!((actual - expected).abs() < 1e-10);
    /// }
    ///
    /// // Evaluate at the mean of a custom normal distribution
    /// if let Prior::Normal(prior) = NormalPrior::new(2.0, 0.5, (None, None)).into() {
    ///     let expected = 1.0 / (2.0 * PI * 0.5 * 0.5).sqrt();
    ///     let actual = prior.function(2.0).unwrap();
    ///     assert!((actual - expected).abs() < 1e-10);
    /// }
    ///
    /// // Truncated normal: value outside bounds should return 0
    /// if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).into() {
    ///     assert_eq!(prior.function(2.0).unwrap(), 0.0);
    /// }
    ///
    /// // Truncated normal: value inside bounds needs normalization
    /// if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).into() {
    ///     let density = prior.function(0.0).unwrap();
    ///     let unnormalized = 1.0 / (2.0 * PI).sqrt();
    ///     let normalization = prior.normalize().unwrap();
    ///     assert!((density - unnormalized/normalization).abs() < 1e-10);
    /// }
    /// ```
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;
        let mean = self.mean;
        let sd = self.sd;
        if self.has_default_range() {
            return dnorm!(x = mean, mean = x, sd = sd).map_err(PriorError::DistributionError);
        }

        let (lower, upper) = self.range_or_default();

        if !(lower..=upper).contains(&x) {
            return Ok(0.0);
        }
        let k = self.normalize()?;
        let density = dnorm!(x = x, mean = mean, sd = sd).map_err(PriorError::DistributionError)?;
        Ok(density / k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::Prior;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_normal_prior_creation() {
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
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
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (None, None)).into() {
            assert!(prior.validate().is_ok());
        }

        // Test invalid standard deviation
        if let Prior::Normal(prior) = NormalPrior::new(0.0, -1.0, (None, None)).into() {
            let result = prior.validate();
            assert!(result.is_err());
            match result {
                Err(PriorError::InvalidStandardDeviation(sd)) => assert_eq!(sd, -1.0),
                _ => panic!("Expected InvalidStandardDeviation error"),
            }
        }

        // Test invalid range
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(2.0), Some(2.0))).into() {
            let result = prior.validate();
            assert!(result.is_err());
            match result {
                Err(PriorError::InvalidRange) => {}
                _ => panic!("Expected InvalidRange error"),
            }
        }

        // Test multiple validation errors
        if let Prior::Normal(prior) = NormalPrior::new(0.0, -1.0, (Some(2.0), Some(2.0))).into() {
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
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (None, None)).into() {
            assert_eq!(prior.normalize().unwrap(), 1.0);
        }

        // Test lower bounded normal distribution
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), None)).into() {
            let norm = prior.normalize().unwrap();
            let expected = pnorm!(q = -1.0, mean = 0.0, sd = 1.0, lower_tail = false).unwrap();
            assert_relative_eq!(norm, expected, epsilon = 1e-10);
        }

        // Test upper bounded normal distribution
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (None, Some(1.0))).into() {
            let norm = prior.normalize().unwrap();
            let expected = pnorm!(q = 1.0, mean = 0.0, sd = 1.0, lower_tail = true).unwrap();
            assert_relative_eq!(norm, expected, epsilon = 1e-10);
        }

        // Test bounded normal distribution
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).into() {
            let norm = prior.normalize().unwrap();
            // Expected: approximately 0.6827
            let expected = 0.6827;
            assert_relative_eq!(norm, expected, epsilon = 1e-3);
        }

        // Test normalization error case (impossible range)
        if let Prior::Normal(mut prior) = NormalPrior::new(0.0, 1.0, (Some(5.0), Some(1.0))).into()
        {
            // Set the range to 0 so that normalization fails
            prior.range = (Some(5.0), Some(5.0));
            assert!(prior.normalize().is_err());
        }
    }

    #[test]
    fn test_pdf_calculation() {
        // Test standard normal PDF at various points
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (None, None)).into() {
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
        if let Prior::Normal(prior) = NormalPrior::new(2.0, 0.5, (None, None)).into() {
            let expected_at_mean = 1.0 / (2.0 * PI * 0.5 * 0.5).sqrt();
            assert_relative_eq!(
                prior.function(2.0).unwrap(),
                expected_at_mean,
                epsilon = 1e-10
            );
        }

        // Test truncated normal distribution - value outside bounds
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).into() {
            assert_eq!(prior.function(2.0).unwrap(), 0.0);
            assert_eq!(prior.function(-2.0).unwrap(), 0.0);
        }

        // Test truncated normal distribution - value inside bounds
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).into() {
            let unnormalized = 1.0 / (2.0 * PI).sqrt();
            let normalization = prior.normalize().unwrap();
            assert_relative_eq!(
                prior.function(0.0).unwrap(),
                unnormalized / normalization,
                epsilon = 1e-10
            );
        }

        // Test with invalid parameters
        if let Prior::Normal(prior) = NormalPrior::new(0.0, -1.0, (None, None)).into() {
            assert!(prior.function(0.0).is_err());
        }
    }

    #[test]
    fn test_range_behavior() {
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (None, None)).into() {
            assert_eq!(prior.range(), (None, None));
            assert_eq!(prior.default_range(), (-f64::INFINITY, f64::INFINITY));
            assert!(prior.has_default_range());
        }

        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).into() {
            assert_eq!(prior.range(), (Some(-1.0), Some(1.0)));
            assert!(!prior.has_default_range());
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test with extremely large values
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1.0, (None, Some(1e10))).into() {
            // Should be close to zero because the area under the curve beyond 10^10 is negligible
            assert_relative_eq!(prior.normalize().unwrap(), 1.0, epsilon = 1e-5);
        }

        // Test with extremely small standard deviation
        if let Prior::Normal(prior) = NormalPrior::new(0.0, 1e-10, (None, None)).into() {
            // The PDF at the mean should be very large
            let value = prior.function(0.0).unwrap();
            assert!(value > 1e9);
        }
    }
}
