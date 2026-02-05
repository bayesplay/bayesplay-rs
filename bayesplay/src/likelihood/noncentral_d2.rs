use rmath::dt;
use serde::{Deserialize, Serialize};

use super::{LikelihoodError, Observation};
use crate::common::Function;
use crate::common::Validate;

/// A noncentral d likelihood for two-sample effect sizes (Cohen's d).
///
/// This likelihood is used when the observed data is a standardized effect size
/// from a two-sample (independent groups) design. The effect size represents
/// the standardized difference between two group means.
///
/// # Parameters
///
/// * `d` - The observed effect size (Cohen's d for two samples)
/// * `n1` - The sample size of the first group (must be ≥ 1)
/// * `n2` - The sample size of the second group (must be ≥ 1)
///
/// # Relationship to t-statistic
///
/// For a two-sample design:
/// - Pooled n: n_pooled = (n1 × n2) / (n1 + n2)
/// - t = d × √n_pooled
/// - df = n1 + n2 - 2
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Effect size d = 0.5 with group sizes of 25 and 30
/// let likelihood = NoncentralD2Likelihood::new(0.5, 25.0, 30.0);
///
/// // Validate parameters
/// assert!(likelihood.validate().is_ok());
///
/// // Combine with a prior
/// let prior: Prior = CauchyPrior::new(0.0, 0.707, (None, None)).into();
/// let model: Model = Likelihood::from(likelihood) * prior;
/// ```
#[derive(Default, Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct NoncentralD2Likelihood {
    /// The observed effect size (Cohen's d).
    pub d: f64,
    /// The sample size of the first group.
    pub n1: f64,
    /// The sample size of the second group.
    pub n2: f64,
}

impl NoncentralD2Likelihood {
    /// Creates a new `Likelihood` instance with the given parameters.
    ///
    /// # Parameters
    /// - `d`: The effect size.
    /// - `n1`: The sample size for the first group.
    /// - `n2`: The sample size for the second group.
    ///
    /// # Returns
    /// A `Likelihood` enum variant containing the `NoncentralD2Likelihood`.
    ///
    /// # Examples
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood = NoncentralD2Likelihood::new(0.5, 10.0, 15.0);
    /// assert_eq!(likelihood.d, 0.5);
    /// assert_eq!(likelihood.n1, 10.0);
    /// assert_eq!(likelihood.n2, 15.0);
    /// ```
    pub fn new(d: f64, n1: f64, n2: f64) -> Self {
        NoncentralD2Likelihood { d, n1, n2 }
    }
}

// TODO: Implement the function to get the t-value for approximations
// pub fn get_tvalue(&self) -> (f64, f64, f64) {
//     let n1 = self.n1;
//     let n2 = self.n2;
//     let d = self.d;
//
//     let n = (n1 * n2) / (n1 + n2);
//     let t = d * n.sqrt();
//     let df = n1 + n2 - 2.0;
//     (t, df, n)
// }
//
// pub fn into_t(&self) -> (Likelihood, f64) {
//     let (t, df, n) = self.get_tvalue();
//     (NoncentralTLikelihood::new(t, df), n)
// }

impl Function<f64, f64, LikelihoodError> for NoncentralD2Likelihood {
    /// Computes the likelihood function for the given input `x`.
    ///
    /// # Parameters
    /// - `x`: The input value for the likelihood function.
    ///
    /// # Returns
    /// A `Result` containing the computed likelihood value or a `LikelihoodError`.
    ///
    /// # Examples
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood = NoncentralD2Likelihood { d: 0.5, n1: 10.0, n2: 15.0 };
    /// let result = likelihood.function(1.0);
    /// assert!(result.is_ok());
    /// assert!((result.unwrap() - 0.19076591569612219579).abs() < 1e-10);
    /// ```
    fn function(&self, x: f64) -> Result<f64, LikelihoodError> {
        let d = self.d;
        let n1 = self.n1;
        let n2 = self.n2;

        let ncp = x * ((n1 * n2) / (n1 + n2)).sqrt();
        let df = n1 + n2 - 2.0;
        let x = d / (1. / n1 + 1. / n2).sqrt();
        dt!(x = x, df = df, ncp = ncp).map_err(LikelihoodError::DistributionError)
    }
}

/// Validates the `NoncentralD2Likelihood` parameters.
///
/// Both sample sizes must be at least 1.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Valid parameters
/// let valid = NoncentralD2Likelihood::new(0.5, 25.0, 30.0);
/// assert!(valid.validate().is_ok());
///
/// // Invalid: n1 < 1
/// let invalid = NoncentralD2Likelihood::new(0.5, 0.5, 30.0);
/// assert!(matches!(invalid.validate(), Err(LikelihoodError::InvalidN1(_))));
///
/// // Invalid: both n1 and n2 < 1
/// let both_invalid = NoncentralD2Likelihood::new(0.5, 0.5, 0.5);
/// assert!(matches!(both_invalid.validate(), Err(LikelihoodError::MultipleErrors(_))));
/// ```
impl Validate<LikelihoodError> for NoncentralD2Likelihood {
    fn validate(&self) -> Result<(), LikelihoodError> {
        let mut errors = Vec::new();

        if self.n1 < 1.0 {
            errors.push(LikelihoodError::InvalidN1(self.n1));
        }

        if self.n2 < 1.0 {
            errors.push(LikelihoodError::InvalidN2(self.n2));
        }

        LikelihoodError::from_errors(errors)
    }
}

/// Implements observation management for `NoncentralD2Likelihood`.
///
/// The observation is the effect size d.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let mut likelihood = NoncentralD2Likelihood::new(0.5, 25.0, 30.0);
///
/// assert_eq!(likelihood.get_observation(), Some(0.5));
///
/// likelihood.update_observation(0.8);
/// assert_eq!(likelihood.get_observation(), Some(0.8));
/// ```
impl Observation for NoncentralD2Likelihood {
    fn update_observation(&mut self, observation: f64) {
        self.d = observation;
    }

    fn get_observation(&self) -> Option<f64> {
        Some(self.d)
    }
}
