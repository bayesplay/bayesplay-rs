use rmath::dunif;
use serde::{Deserialize, Serialize};

use super::Normalize;
use crate::common::Function;
use crate::common::Range;
use crate::common::Validate;

use super::PriorError;

/// A uniform distribution prior over a specified interval.
///
/// The uniform prior assigns equal probability density to all values within
/// the interval [min, max]. It's often used as an uninformative or "flat" prior
/// when there's no reason to prefer any value over another within a range.
///
/// # Parameters
///
/// * `min` - The lower bound of the interval
/// * `max` - The upper bound of the interval (must be greater than min)
///
/// # Mathematical Form
///
/// The uniform PDF is:
///
/// f(x) = 1 / (max - min)  for min ≤ x ≤ max
/// f(x) = 0                otherwise
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Uniform prior from 0 to 1 (useful for probability parameters)
/// let unit_uniform = UniformPrior::new(0.0, 1.0);
///
/// // Uniform prior from -1 to 1
/// let symmetric = UniformPrior::new(-1.0, 1.0);
///
/// // The density is constant within the interval
/// let density = symmetric.function(0.0).unwrap();
/// assert!((density - 0.5).abs() < 1e-10); // 1/(1-(-1)) = 0.5
///
/// // Zero outside the interval
/// assert_eq!(symmetric.function(2.0).unwrap(), 0.0);
/// ```
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct UniformPrior {
    /// The lower bound of the interval.
    pub min: f64,
    /// The upper bound of the interval.
    pub max: f64,
}

impl UniformPrior {
    /// Creates a new uniform prior over the interval [min, max].
    ///
    /// # Arguments
    ///
    /// * `min` - The lower bound
    /// * `max` - The upper bound (must be greater than min)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// let uniform = UniformPrior::new(0.0, 1.0);
    /// assert_eq!(uniform.min, 0.0);
    /// assert_eq!(uniform.max, 1.0);
    /// ```
    pub fn new(min: f64, max: f64) -> Self {
        UniformPrior { min, max }
    }
}

/// The range of a uniform prior is [min, max].
impl Range for UniformPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        (Some(self.min), Some(self.max))
    }

    fn default_range(&self) -> (f64, f64) {
        (-f64::INFINITY, f64::INFINITY)
    }
}

/// Evaluates the uniform PDF at a given value.
///
/// Returns 1/(max-min) for values within [min, max], 0 otherwise.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let uniform = UniformPrior::new(0.0, 2.0);
///
/// // Inside the interval: density = 1/(2-0) = 0.5
/// assert!((uniform.function(1.0).unwrap() - 0.5).abs() < 1e-10);
///
/// // Outside the interval: density = 0
/// assert_eq!(uniform.function(3.0).unwrap(), 0.0);
/// ```
impl Function<f64, f64, PriorError> for UniformPrior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;
        let min = self.min;
        let max = self.max;
        Ok(dunif!(x = x, min = min, max = max))
    }
}

/// Validates the uniform prior parameters.
///
/// Ensures that min < max.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Valid: min < max
/// let valid = UniformPrior::new(0.0, 1.0);
/// assert!(valid.validate().is_ok());
///
/// // Invalid: min == max
/// let invalid = UniformPrior::new(1.0, 1.0);
/// assert!(matches!(invalid.validate(), Err(PriorError::InvalidRange)));
///
/// // Invalid: min > max
/// let invalid = UniformPrior::new(2.0, 1.0);
/// assert!(matches!(invalid.validate(), Err(PriorError::InvalidRange)));
/// ```
impl Validate<PriorError> for UniformPrior {
    fn validate(&self) -> Result<(), PriorError> {
        if self.min == self.max {
            Err(PriorError::InvalidRange)?;
        }

        if self.min > self.max {
            Err(PriorError::InvalidRange)?;
        }

        Ok(())
    }
}

/// Uniform priors are already normalized by construction.
impl Normalize for UniformPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        Ok(1.0)
    }
}
