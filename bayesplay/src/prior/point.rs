use super::Normalize;
use super::PriorError;
use crate::common::Function;
use crate::common::Range;
use crate::common::Validate;
use serde::{Deserialize, Serialize};

/// A point prior (Dirac delta) that places all probability mass at a single value.
///
/// Point priors are used to represent null hypotheses in Bayesian hypothesis
/// testing. For example, a point prior at 0 represents the hypothesis that
/// an effect is exactly zero.
///
/// # Parameters
///
/// * `point` - The value at which all probability mass is concentrated
///
/// # Mathematical Form
///
/// The point prior is a Dirac delta function δ(x - point), which is:
/// - 1 when x = point
/// - 0 otherwise
///
/// Note: This is a simplified representation for computational purposes.
/// The actual Dirac delta is not a function but a distribution.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Null hypothesis: effect size is exactly 0
/// let null_prior = PointPrior::new(0.0);
///
/// // Combine with a likelihood
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
/// let null_model: Model = likelihood * Prior::from(null_prior);
///
/// // The marginal likelihood equals the likelihood evaluated at the point
/// let marginal = null_model.integral().unwrap();
/// ```
///
/// # Computing Bayes Factors
///
/// Point priors are commonly used to compute Bayes factors comparing
/// a point null hypothesis to an alternative hypothesis:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
///
/// // Null: effect is exactly 0
/// let h0: Model = likelihood * Prior::from(PointPrior::new(0.0));
///
/// // Alternative: effect follows a normal distribution
/// let h1: Model = likelihood * Prior::from(NormalPrior::new(0.0, 1.0, (None, None)));
///
/// // Bayes factor (BF10 = evidence for H1 over H0)
/// let bf10 = h1.integral().unwrap() / h0.integral().unwrap();
/// ```
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct PointPrior {
    /// The value at which all probability mass is concentrated.
    pub point: f64,
}

impl PointPrior {
    /// Creates a new point prior at the specified value.
    ///
    /// # Arguments
    ///
    /// * `point` - The value where all probability mass is placed
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// let null = PointPrior::new(0.0);
    /// assert_eq!(null.point, 0.0);
    ///
    /// // Point prior at a non-zero value
    /// let specific = PointPrior::new(0.5);
    /// assert_eq!(specific.point, 0.5);
    /// ```
    pub fn new(point: f64) -> Self {
        PointPrior { point }
    }
}

/// Validates the point prior parameters.
///
/// The point value must not be NaN.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let valid = PointPrior::new(0.0);
/// assert!(valid.validate().is_ok());
///
/// let invalid = PointPrior { point: f64::NAN };
/// assert!(invalid.validate().is_err());
/// ```
impl Validate<PriorError> for PointPrior {
    fn validate(&self) -> Result<(), PriorError> {
        if self.point.is_nan() {
            return Err(PriorError::InvalidValue(
                self.point,
                f64::NEG_INFINITY,
                f64::INFINITY,
            ));
        }
        Ok(())
    }
}

/// Evaluates the point prior at a given value.
///
/// Returns 1.0 if x equals the point, 0.0 otherwise.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let prior = PointPrior::new(0.0);
///
/// assert_eq!(prior.function(0.0).unwrap(), 1.0);
/// assert_eq!(prior.function(0.5).unwrap(), 0.0);
/// assert_eq!(prior.function(-1.0).unwrap(), 0.0);
/// ```
impl Function<f64, f64, PriorError> for PointPrior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;
        if x == self.point {
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    }
}

/// Point priors always have a normalization constant of 1.
impl Normalize for PointPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        Ok(1.0)
    }
}

/// The range of a point prior is just the single point.
impl Range for PointPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        (Some(self.point), Some(self.point))
    }
    fn default_range(&self) -> (f64, f64) {
        (self.point, self.point)
    }
}
