use rmath::dt;
use serde::{Deserialize, Serialize};

use super::{LikelihoodError, Observation};
use crate::common::Function;
use crate::common::Validate;

/// A noncentral t likelihood for t-statistic data.
///
/// This likelihood is used when the observed data is a t-statistic, and we want
/// to make inferences about the noncentrality parameter (which relates to the
/// true effect size).
///
/// # Parameters
///
/// * `t` - The observed t-statistic
/// * `df` - The degrees of freedom (must be ≥ 1)
///
/// # Mathematical Form
///
/// Given observed t and degrees of freedom df, the likelihood of a noncentrality
/// parameter x is given by the noncentral t-distribution:
///
/// L(x) = dt(t; df, ncp=x)
///
/// where dt is the noncentral t-distribution density.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // t-statistic of 2.5 with 29 degrees of freedom
/// let likelihood = NoncentralTLikelihood::new(2.5, 29.0);
///
/// // Evaluate the likelihood at ncp = 0 (null hypothesis)
/// let null_value = likelihood.function(0.0).unwrap();
///
/// // Evaluate at ncp = 2.0
/// let alt_value = likelihood.function(2.0).unwrap();
/// ```
#[derive(Default, Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct NoncentralTLikelihood {
    /// The observed t-statistic.
    pub t: f64,
    /// The degrees of freedom.
    pub df: f64,
}

impl NoncentralTLikelihood {
    /// Creates a new `NoncentralTLikelihood`.
    ///
    /// # Arguments
    ///
    /// * `t` - The observed t-statistic
    /// * `df` - The degrees of freedom
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood = NoncentralTLikelihood::new(2.5, 29.0);
    /// assert_eq!(likelihood.t, 2.5);
    /// assert_eq!(likelihood.df, 29.0);
    /// ```
    pub fn new(t: f64, df: f64) -> Self {
        NoncentralTLikelihood { t, df }
    }
}

/// # Examples
///
/// ```
/// use bayesplay::prelude::*;
///
/// let likelihood = NoncentralTLikelihood { t: 2.0, df: 10.0 };
///
/// // Compute the likelihood function value for a given x
/// let result = likelihood.function(1.5);
/// assert!(result.is_ok());
/// assert!((result.unwrap() - 0.31460591844887925150).abs() < 1e-10);
///
/// ```
impl Function<f64, f64, LikelihoodError> for NoncentralTLikelihood {
    fn function(&self, x: f64) -> Result<f64, LikelihoodError> {
        let df = self.df;
        let t = self.t;
        dt!(x = t, df = df, ncp = x).map_err(LikelihoodError::DistributionError)
    }
}

/// Implements observation management for `NoncentralTLikelihood`.
///
/// The observation is the t-statistic.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let mut likelihood = NoncentralTLikelihood::new(2.5, 29.0);
///
/// assert_eq!(likelihood.get_observation(), Some(2.5));
///
/// likelihood.update_observation(3.0);
/// assert_eq!(likelihood.get_observation(), Some(3.0));
/// ```
impl Observation for NoncentralTLikelihood {
    fn update_observation(&mut self, observation: f64) {
        self.t = observation;
    }

    fn get_observation(&self) -> Option<f64> {
        Some(self.t)
    }
}

/// Validates the `NoncentralTLikelihood` parameters.
///
/// The degrees of freedom must be at least 1.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Valid parameters
/// let valid = NoncentralTLikelihood::new(2.5, 29.0);
/// assert!(valid.validate().is_ok());
///
/// // Invalid: df < 1
/// let invalid = NoncentralTLikelihood::new(2.5, 0.5);
/// assert!(matches!(invalid.validate(), Err(LikelihoodError::InvalidDF(_))));
/// ```
impl Validate<LikelihoodError> for NoncentralTLikelihood {
    fn validate(&self) -> Result<(), LikelihoodError> {
        if self.df < 1.0 {
            return Err(LikelihoodError::InvalidDF(self.df));
        }
        Ok(())
    }
}
