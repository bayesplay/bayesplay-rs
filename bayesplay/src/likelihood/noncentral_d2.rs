use serde::{Deserialize, Serialize};

use super::{LikelihoodError, Observation};
use crate::common::LogFunction;
use crate::common::Validate;
use crate::prelude::NoncentralTLikelihood;

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
/// let likelihood = NoncentralD2Likelihood::new(0.5, 25.0, 30.0).unwrap();
///
/// // Validate parameters
/// assert!(likelihood.validate().is_ok());
///
/// // Combine with a prior
/// let prior: Prior = CauchyPrior::new(0.0, 0.707, (None, None)).unwrap().into();
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
    /// let likelihood = NoncentralD2Likelihood::new(0.5, 10.0, 15.0).unwrap();
    /// assert_eq!(likelihood.d, 0.5);
    /// assert_eq!(likelihood.n1, 10.0);
    /// assert_eq!(likelihood.n2, 15.0);
    /// ```
    pub fn new(d: f64, n1: f64, n2: f64) -> Result<Self, LikelihoodError> {
        let value = NoncentralD2Likelihood { d, n1, n2 };
        value.validate()?;
        Ok(value)
    }

    /// The effective sample size `n1 * n2 / (n1 + n2)` relating d to t
    /// (`t = d * sqrt(effective_n)`).
    pub fn effective_n(&self) -> f64 {
        self.n1 * self.n2 / (self.n1 + self.n2)
    }

    /// The equivalent noncentral t likelihood (t = d√n_eff, df = n1 + n2 − 2),
    /// whose parameter is the noncentrality δ√n_eff rather than δ.
    pub fn to_noncentral_t(&self) -> NoncentralTLikelihood {
        NoncentralTLikelihood {
            t: self.d * self.effective_n().sqrt(),
            df: self.n1 + self.n2 - 2.0,
        }
    }
}

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
impl LogFunction<LikelihoodError> for NoncentralD2Likelihood {
    fn ln_function(&self, x: f64) -> Result<f64, LikelihoodError> {
        let sqrt_n = self.effective_n().sqrt();
        let df = self.n1 + self.n2 - 2.0;
        rmath::dnt(self.d * sqrt_n, df, x * sqrt_n, true)
            .map_err(LikelihoodError::DistributionError)
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
/// let valid = NoncentralD2Likelihood::new(0.5, 25.0, 30.0).unwrap();
/// assert!(valid.validate().is_ok());
///
/// // Invalid: n1 < 1
/// assert!(matches!(NoncentralD2Likelihood::new(0.5, 0.5, 30.0), Err(LikelihoodError::InvalidN1(_))));
///
/// // Invalid: both n1 and n2 < 1
/// assert!(matches!(NoncentralD2Likelihood::new(0.5, 0.5, 0.5), Err(LikelihoodError::MultipleErrors(_))));
/// ```
impl Validate<LikelihoodError> for NoncentralD2Likelihood {
    fn validate(&self) -> Result<(), LikelihoodError> {
        let mut errors = Vec::new();

        if !(self.n1 >= 1.0 && self.n1.is_finite()) {
            errors.push(LikelihoodError::InvalidN1(self.n1));
        }

        if !(self.n2 >= 1.0 && self.n2.is_finite()) {
            errors.push(LikelihoodError::InvalidN2(self.n2));
        }

        let df = self.n1 + self.n2 - 2.0;
        if errors.is_empty() && df < 1.0 {
            errors.push(LikelihoodError::InvalidDF(df));
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
/// let mut likelihood = NoncentralD2Likelihood::new(0.5, 25.0, 30.0).unwrap();
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
