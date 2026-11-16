use serde::{Deserialize, Serialize};

use super::Normalize;
use super::PriorDensity;
use super::PriorError;
use super::{is_valid_parameter, validate_range};
use crate::common::Range;
use crate::common::Validate;
use crate::common::truncated_ln_normalization;

/// A Student's t distribution prior with customizable parameters.
///
/// The Student's t prior is similar to the normal distribution but has heavier
/// tails. The weight of the tails is controlled by the degrees of freedom (df)
/// parameter. As df → ∞, the t-distribution approaches the normal distribution.
/// With df = 1, it equals the Cauchy distribution.
///
/// # Parameters
///
/// * `mean` - The location parameter (mean/mode of the distribution)
/// * `sd` - The scale parameter (must be positive)
/// * `df` - The degrees of freedom (must be > 1)
/// * `range` - Optional bounds for truncation as (lower, upper)
///
/// # Mathematical Form
///
/// The scaled t-distribution PDF is proportional to:
///
/// (1 + ((x - mean) / sd)² / df)^(-(df+1)/2) / sd
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Standard t-prior with 3 degrees of freedom
/// let t_prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).unwrap();
///
/// // t-prior with heavier tails (lower df)
/// let heavy_tails = StudentTPrior::new(0.0, 1.0, 2.0, (None, None)).unwrap();
///
/// // Truncated t-prior (positive values only)
/// let positive_t = StudentTPrior::new(0.0, 1.0, 3.0, (Some(0.0), None)).unwrap();
/// ```
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct StudentTPrior {
    /// The location parameter (mean/mode).
    pub mean: f64,
    /// The scale parameter.
    pub sd: f64,
    /// The degrees of freedom.
    pub df: f64,
    /// Optional truncation bounds as (lower, upper).
    pub range: (Option<f64>, Option<f64>),
}

impl StudentTPrior {
    /// Creates a new Student's t distribution prior.
    ///
    /// # Arguments
    ///
    /// * `mean` - The location parameter
    /// * `sd` - The scale parameter (must be positive)
    /// * `df` - The degrees of freedom (must be > 1)
    /// * `range` - Optional bounds as (lower_bound, upper_bound)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// // Unbounded t-prior
    /// let t_prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).unwrap();
    ///
    /// // Half-t prior (positive values)
    /// let half_t = StudentTPrior::new(0.0, 1.0, 3.0, (Some(0.0), None)).unwrap();
    /// ```
    pub fn new(
        mean: f64,
        sd: f64,
        df: f64,
        range: (Option<f64>, Option<f64>),
    ) -> Result<Self, PriorError> {
        let value = StudentTPrior {
            mean,
            sd,
            df,
            range,
        };
        value.validate()?;
        Ok(value)
    }
}

/// Validates the Student's t prior parameters.
///
/// Ensures:
/// 1. The scale (sd) is positive
/// 2. The degrees of freedom is > 1
/// 3. The range bounds are not equal
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Valid parameters
/// let valid = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).unwrap();
/// assert!(valid.validate().is_ok());
///
/// // Invalid: sd <= 0
/// assert!(matches!(StudentTPrior::new(0.0, -1.0, 3.0, (None, None)), Err(PriorError::InvalidStandardDeviation(_))));
///
/// // Invalid: df <= 1
/// assert!(matches!(StudentTPrior::new(0.0, 1.0, 0.5, (None, None)), Err(PriorError::InvalidDegreesOfFreedom(_))));
///
/// // Multiple errors
/// assert!(matches!(StudentTPrior::new(0.0, -1.0, 0.5, (None, None)), Err(PriorError::MultipleErrors(_))));
/// ```
impl Validate<PriorError> for StudentTPrior {
    fn validate(&self) -> Result<(), PriorError> {
        let mut errors: Vec<PriorError> = Vec::new();

        if !is_valid_parameter(self.mean, false) {
            errors.push(PriorError::InvalidLocation(self.mean));
        }

        if !is_valid_parameter(self.sd, true) {
            errors.push(PriorError::InvalidStandardDeviation(self.sd));
        }

        if self.df.is_nan() || self.df <= 1.0 {
            errors.push(PriorError::InvalidDegreesOfFreedom(self.df));
        }

        if let Err(e) = validate_range(self.range_or_default()) {
            errors.push(e);
        }

        PriorError::from_errors(errors)
    }
}

/// Evaluates the Student's t PDF at a given value.
///
/// For truncated distributions, returns 0 outside the bounds and applies
/// proper normalization within the bounds.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let t_prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).unwrap();
///
/// // Evaluate at the mean
/// let density_at_mean = t_prior.function(0.0).unwrap();
///
/// // Truncated t: value outside bounds returns 0
/// let truncated = StudentTPrior::new(0.0, 1.0, 3.0, (Some(-1.0), Some(1.0))).unwrap();
/// assert_eq!(truncated.function(5.0).unwrap(), 0.0);
/// ```
impl PriorDensity for StudentTPrior {
    fn ln_density(&self, x: f64, ln_normalizer: f64) -> Result<f64, PriorError> {
        let log_density = rmath::dt_scaled(x, self.mean, self.sd, self.df, true)
            .map_err(PriorError::DistributionError)?;
        let (lower, upper) = self.range_or_default();
        if !(lower..=upper).contains(&x) {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(log_density - ln_normalizer)
    }
}

/// Computes the normalization constant for the Student's t prior.
///
/// For unbounded distributions, returns 1.0. For truncated distributions,
/// calculates the probability mass within the truncation bounds.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Unbounded t has normalization constant of 1
/// let unbounded = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).unwrap();
/// assert_eq!(unbounded.normalize().unwrap(), 1.0);
///
/// // Half-t (positive values) has normalization constant of 0.5
/// let half_t = StudentTPrior::new(0.0, 1.0, 3.0, (Some(0.0), None)).unwrap();
/// assert!((half_t.normalize().unwrap() - 0.5).abs() < 0.001);
/// ```
impl Normalize for StudentTPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        Ok(self.ln_normalize()?.exp())
    }

    fn ln_normalize(&self) -> Result<f64, PriorError> {
        let (lower, upper) = self.range_or_default();
        let ln_mass = truncated_ln_normalization(lower, upper, |x, lower_tail| {
            rmath::pt_scaled(x, self.mean, self.sd, self.df, lower_tail, true)
                .map_err(PriorError::DistributionError)
        })?;
        if ln_mass.is_nan() || ln_mass == f64::NEG_INFINITY {
            return Err(PriorError::NormalizingError);
        }
        Ok(ln_mass)
    }
}

/// Implementation of the Range trait for StudentTPrior.
impl Range for StudentTPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        self.range
    }

    fn default_range(&self) -> (f64, f64) {
        (-f64::INFINITY, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Function;
    use crate::prior::Prior;
    use approx::assert_relative_eq;

    // ---- Tests from R's test-priors.R ----

    #[test]
    fn test_half_t_density_at_zero() {
        // R: prior("student_t", 0, 1, 10, c(0, Inf)) at 0 == dt(0, 10) * 2
        // For a scaled t with mean=0, sd=1, dt_scaled(0, 0, 1, 10) = dt(0, 10)
        // The half-t at 0 should be twice the untruncated value
        let full_t = StudentTPrior::new(0.0, 1.0, 10.0, (None, None)).unwrap();
        let half_t = StudentTPrior::new(0.0, 1.0, 10.0, (Some(0.0), None)).unwrap();

        let full_density = full_t.function(0.0).unwrap();
        let half_density = half_t.function(0.0).unwrap();

        // Half-t at 0 should equal full-t at 0 times 2
        assert_relative_eq!(half_density, full_density * 2.0, epsilon = 5e-7);
    }

    #[test]
    fn test_half_t_density_outside_range() {
        // R: prior("student_t", 0, 1, 10, c(0, Inf)) at -1 == 0
        let half_t = StudentTPrior::new(0.0, 1.0, 10.0, (Some(0.0), None)).unwrap();
        let density = half_t.function(-1.0).unwrap();
        assert_eq!(density, 0.0);
    }

    // ---- Creation and validation tests ----

    #[test]
    fn test_student_t_prior_creation() {
        let prior: Prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None))
            .unwrap()
            .into();
        if let Prior::StudentT(p) = prior {
            assert_eq!(p.mean, 0.0);
            assert_eq!(p.sd, 1.0);
            assert_eq!(p.df, 3.0);
            assert_eq!(p.range, (None, None));
        } else {
            panic!("Expected StudentT prior");
        }
    }

    #[test]
    fn test_student_t_validation_valid() {
        let t_prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).unwrap();
        assert!(t_prior.validate().is_ok());
    }

    #[test]
    fn test_student_t_validation_negative_sd() {
        let t_prior = StudentTPrior {
            mean: 0.0,
            sd: -1.0,
            df: 3.0,
            range: (None, None),
        };
        assert!(matches!(
            t_prior.validate(),
            Err(PriorError::InvalidStandardDeviation(_))
        ));
    }

    #[test]
    fn test_student_t_validation_zero_sd() {
        let t_prior = StudentTPrior {
            mean: 0.0,
            sd: 0.0,
            df: 3.0,
            range: (None, None),
        };
        assert!(matches!(
            t_prior.validate(),
            Err(PriorError::InvalidStandardDeviation(_))
        ));
    }

    #[test]
    fn test_student_t_validation_invalid_df() {
        let t_prior = StudentTPrior {
            mean: 0.0,
            sd: 1.0,
            df: 0.5,
            range: (None, None),
        };
        assert!(matches!(
            t_prior.validate(),
            Err(PriorError::InvalidDegreesOfFreedom(_))
        ));
    }

    #[test]
    fn test_student_t_validation_df_equals_one() {
        // df must be > 1, so df=1 should fail
        let t_prior = StudentTPrior {
            mean: 0.0,
            sd: 1.0,
            df: 1.0,
            range: (None, None),
        };
        assert!(matches!(
            t_prior.validate(),
            Err(PriorError::InvalidDegreesOfFreedom(_))
        ));
    }

    #[test]
    fn test_student_t_validation_multiple_errors() {
        let t_prior = StudentTPrior {
            mean: 0.0,
            sd: -1.0,
            df: 0.5,
            range: (None, None),
        };
        match t_prior.validate() {
            Err(PriorError::MultipleErrors(errors)) => {
                assert_eq!(errors.len(), 2);
            }
            _ => panic!("Expected MultipleErrors"),
        }
    }

    #[test]
    fn test_student_t_validation_equal_range() {
        let t_prior = StudentTPrior {
            mean: 0.0,
            sd: 1.0,
            df: 3.0,
            range: (Some(0.0), Some(0.0)),
        };
        let result = t_prior.validate();
        assert!(result.is_err());
    }

    // ---- Normalization tests ----

    #[test]
    fn test_student_t_normalization_unbounded() {
        let t_prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).unwrap();
        assert_eq!(t_prior.normalize().unwrap(), 1.0);
    }

    #[test]
    fn test_student_t_normalization_half() {
        let half_t = StudentTPrior::new(0.0, 1.0, 3.0, (Some(0.0), None)).unwrap();
        assert_relative_eq!(half_t.normalize().unwrap(), 0.5, epsilon = 1e-10);
    }

    // ---- Range tests ----

    #[test]
    fn test_student_t_range_behavior() {
        let t_prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).unwrap();
        assert_eq!(t_prior.range(), (None, None));
        assert!(t_prior.has_default_range());

        let truncated = StudentTPrior::new(0.0, 1.0, 3.0, (Some(-1.0), Some(1.0))).unwrap();
        assert!(!truncated.has_default_range());
    }

    #[test]
    fn test_truncated_student_t_outside_bounds() {
        let truncated = StudentTPrior::new(0.0, 1.0, 3.0, (Some(-1.0), Some(1.0))).unwrap();
        assert_eq!(truncated.function(5.0).unwrap(), 0.0);
        assert_eq!(truncated.function(-5.0).unwrap(), 0.0);
    }
}
