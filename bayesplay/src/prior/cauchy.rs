use rmath::dcauchy;
use rmath::pcauchy;

use serde::{Deserialize, Serialize};

use super::Normalize;
use super::PriorError;
use crate::common::truncated_normalization;
use crate::common::Function;
use crate::common::Range;
use crate::common::Validate;

/// A Cauchy distribution prior with customizable parameters.
///
/// The Cauchy distribution has heavier tails than the normal distribution,
/// making it appropriate for priors where extreme values should not be
/// ruled out too strongly. A common choice is the "medium" Cauchy prior
/// with location=0 and scale=0.707 (≈1/√2).
///
/// # Parameters
///
/// * `location` - The location parameter (mode/median of the distribution)
/// * `scale` - The scale parameter (half-width at half-maximum, must be positive)
/// * `range` - Optional bounds for truncation as (lower, upper)
///
/// # Mathematical Form
///
/// The Cauchy PDF is:
///
/// f(x) = 1 / (π × scale × (1 + ((x - location) / scale)²))
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Standard Cauchy prior (location=0, scale=1)
/// let standard = CauchyPrior::new(0.0, 1.0, (None, None));
///
/// // "Medium" Cauchy prior commonly used for effect sizes
/// let medium = CauchyPrior::new(0.0, 0.707, (None, None));
///
/// // Half-Cauchy (positive values only) - useful for scale parameters
/// let half_cauchy = CauchyPrior::new(0.0, 1.0, (Some(0.0), None));
///
/// // Bounded Cauchy
/// let bounded = CauchyPrior::new(0.0, 0.707, (Some(-2.0), Some(2.0)));
/// ```
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct CauchyPrior {
    /// The location parameter (mode/median).
    pub location: f64,
    /// The scale parameter (half-width at half-maximum).
    pub scale: f64,
    /// Optional truncation bounds as (lower, upper).
    pub range: (Option<f64>, Option<f64>),
}

impl CauchyPrior {
    /// Creates a new Cauchy distribution prior.
    ///
    /// # Arguments
    ///
    /// * `location` - The location parameter (mode/median)
    /// * `scale` - The scale parameter (must be positive)
    /// * `range` - Optional bounds as (lower_bound, upper_bound)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// // Unbounded Cauchy
    /// let cauchy = CauchyPrior::new(0.0, 0.707, (None, None));
    ///
    /// // Half-Cauchy (positive values only)
    /// let half_cauchy = CauchyPrior::new(0.0, 1.0, (Some(0.0), None));
    /// ```
    pub fn new(location: f64, scale: f64, range: (Option<f64>, Option<f64>)) -> Self {
        CauchyPrior {
            location,
            scale,
            range,
        }
    }
}

/// Validates the parameters of the Cauchy prior.
///
/// Ensures:
/// 1. The scale parameter is positive
/// 2. The range bounds are not equal (except for default infinite bounds)
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Valid parameters
/// let valid = CauchyPrior::new(0.0, 0.707, (None, None));
/// assert!(valid.validate().is_ok());
///
/// // Invalid: negative scale
/// let invalid = CauchyPrior::new(0.0, -1.0, (None, None));
/// assert!(matches!(invalid.validate(), Err(PriorError::InvalidScale(_))));
///
/// // Invalid: equal bounds
/// let invalid = CauchyPrior::new(0.0, 1.0, (Some(0.0), Some(0.0)));
/// assert!(matches!(invalid.validate(), Err(PriorError::InvalidRange)));
/// ```
impl Validate<PriorError> for CauchyPrior {
    fn validate(&self) -> Result<(), PriorError> {
        if self.scale <= 0.0 {
            Err(PriorError::InvalidScale(self.scale))?;
        }

        if !self.has_default_range() & (self.range.0 == self.range.1) {
            Err(PriorError::InvalidRange)?;
        }
        Ok(())
    }
}

/// Implementation of the Range trait for CauchyPrior.
impl Range for CauchyPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        self.range
    }

    fn default_range(&self) -> (f64, f64) {
        (-f64::INFINITY, f64::INFINITY)
    }
}

/// Implementation of normalization for the CauchyPrior.
///
/// For unbounded Cauchy distributions, returns 1.0. For truncated distributions,
/// calculates the probability mass within the truncation bounds.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Unbounded Cauchy has normalization constant of 1
/// let unbounded = CauchyPrior::new(0.0, 1.0, (None, None));
/// assert_eq!(unbounded.normalize().unwrap(), 1.0);
///
/// // Half-Cauchy (positive values) has normalization constant of 0.5
/// let half_cauchy = CauchyPrior::new(0.0, 1.0, (Some(0.0), None));
/// assert!((half_cauchy.normalize().unwrap() - 0.5).abs() < 0.001);
/// ```
impl Normalize for CauchyPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        let (lower, upper) = self.range_or_default();
        let res = truncated_normalization(lower, upper, |x, lower_tail| {
            pcauchy!(
                q = x,
                location = self.location,
                scale = self.scale,
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

/// Implementation of the Function trait for evaluating the Cauchy PDF.
///
/// For truncated distributions, returns 0 outside the bounds and applies
/// proper normalization within the bounds.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
/// use std::f64::consts::PI;
///
/// // Standard Cauchy PDF at x=0 should be 1/π
/// let cauchy = CauchyPrior::new(0.0, 1.0, (None, None));
/// let density = cauchy.function(0.0).unwrap();
/// assert!((density - 1.0/PI).abs() < 1e-10);
///
/// // Truncated Cauchy: value outside bounds returns 0
/// let truncated = CauchyPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0)));
/// assert_eq!(truncated.function(5.0).unwrap(), 0.0);
/// ```
impl Function<f64, f64, PriorError> for CauchyPrior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;
        let location = self.location;
        let scale = self.scale;

        if self.has_default_range() {
            return dcauchy!(x = x, location = location, scale = scale)
                .map_err(PriorError::DistributionError);
        }

        let (lower, upper) = self.range_or_default();

        if !(lower..=upper).contains(&x) {
            return Ok(0.0);
        }
        let k = self.normalize()?;
        Ok(dcauchy!(x = x, location = location, scale = scale)
            .map_err(PriorError::DistributionError)?
            / k)
    }
}
