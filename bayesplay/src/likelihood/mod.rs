//! Likelihood distributions for Bayesian inference.
//!
//! This module provides various likelihood functions that represent the probability
//! of observing data given parameter values. Likelihoods are combined with priors
//! to compute posterior distributions.
//!
//! # Available Likelihoods
//!
//! | Type | Use Case | Parameters |
//! |------|----------|------------|
//! | [`NormalLikelihood`] | Continuous data with known SE | mean, se |
//! | [`BinomialLikelihood`] | Count data | successes, trials |
//! | [`StudentTLikelihood`] | t-distributed data | mean, sd, df |
//! | [`NoncentralDLikelihood`] | One-sample effect size | d, n |
//! | [`NoncentralD2Likelihood`] | Two-sample effect size | d, n1, n2 |
//! | [`NoncentralTLikelihood`] | Noncentral t data | t, df |
//! | [`VarianceRatioLikelihood`] | Variance ratio against a reference | u, df |
//!
//! # Examples
//!
//! ```rust
//! use bayesplay::prelude::*;
//!
//! // Create different types of likelihoods
//! let normal = NormalLikelihood::new(0.5, 0.2);
//! let binomial = BinomialLikelihood::new(7.0, 10.0);
//! let student_t = StudentTLikelihood::new(2.5, 1.0, 29.0);
//!
//! // Convert to the Likelihood enum for use with models
//! let likelihood: Likelihood = normal.into();
//! ```

use thiserror::Error;

pub(crate) mod binomial;
pub(crate) mod noncentral_d;
pub(crate) mod noncentral_d2;
pub(crate) mod noncentral_t;
pub(crate) mod normal;
pub(crate) mod student_t;
pub(crate) mod variance_ratio;

use crate::common::Family;
use crate::common::Function;
use crate::common::LogFunction;
use crate::common::Validate;
use crate::common::impl_from_variants;

pub use binomial::BinomialLikelihood;
pub use noncentral_d::NoncentralDLikelihood;
pub use noncentral_d2::NoncentralD2Likelihood;
pub use noncentral_t::NoncentralTLikelihood;
pub use normal::NormalLikelihood;
pub use student_t::StudentTLikelihood;
pub use variance_ratio::VarianceRatioLikelihood;

use serde::{Deserialize, Serialize};

/// Errors that can occur when working with likelihood functions.
///
/// This enum covers all validation errors for likelihood parameters across
/// all likelihood types. When multiple validation errors occur, they are
/// collected into a [`LikelihoodError::MultipleErrors`] variant.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Invalid standard error
/// let invalid = NormalLikelihood { mean: 0.0, se: -1.0 };
/// match invalid.validate() {
///     Err(LikelihoodError::InvalidSE(se)) => assert_eq!(se, -1.0),
///     _ => panic!("Expected InvalidSE error"),
/// }
///
/// // Multiple errors
/// let invalid = BinomialLikelihood { successes: 15.0, trials: 0.5 };
/// match invalid.validate() {
///     Err(LikelihoodError::MultipleErrors(errors)) => {
///         assert!(errors.len() >= 2);
///     }
///     _ => panic!("Expected multiple errors"),
/// }
/// ```
#[derive(Error, Debug, Serialize, Deserialize)]
pub enum LikelihoodError {
    #[error("SD value of {0} is invalid. SD must be positive")]
    InvalidSD(f64),
    #[error("SE value of {0} is invalid. SE must be positive")]
    InvalidSE(f64),
    #[error("N value of {0} is invalid. N must be at least 2")]
    InvalidN(f64),
    #[error("N1 value of {0} is invalid. N1 must be at least 1")]
    InvalidN1(f64),
    #[error("N2 value of {0} is invalid. N2 must be at least 1")]
    InvalidN2(f64),
    #[error("Trials value of {0} is invalid. Trials must be a positive integer")]
    InvalidTrials(f64),
    #[error(
        "Successes value of {0} is invalid. Successes must be a whole number between 0 and Trials"
    )]
    InvalidSuccess(f64),
    #[error(
        "DF value of {0} is invalid. DF must be positive (at least 1 for noncentral t/d likelihoods)"
    )]
    InvalidDF(f64),
    #[error("U value of {0} is invalid. U must be positive")]
    InvalidU(f64),
    #[error("Invalid probability value of {0}. Probability must be between 0 and 1")]
    InvalidProbability(f64),
    #[error("Error with distribution: {0}")]
    DistributionError(&'static str),
    #[error("Multiple validation errors: {}", .0.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "))]
    MultipleErrors(Vec<LikelihoodError>),
}

/// Result type alias for likelihood operations
pub type LikelihoodResult<T> = Result<T, LikelihoodError>;

impl LikelihoodError {
    /// Consolidate multiple errors into a single error result.
    /// Returns Ok(()) if the vector is empty, a single error if only one,
    /// or MultipleErrors if more than one.
    pub fn from_errors(errors: Vec<LikelihoodError>) -> Result<(), LikelihoodError> {
        match errors.len() {
            0 => Ok(()),
            1 => Err(errors.into_iter().next().unwrap()),
            _ => Err(LikelihoodError::MultipleErrors(errors)),
        }
    }
}

/// Trait for managing observed data in likelihood functions.
///
/// This trait allows retrieving and updating the observed value in a likelihood.
/// It is used internally for computing predictive distributions, where the
/// observation is varied across a range of values.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let mut likelihood = NormalLikelihood::new(0.5, 0.2);
///
/// // Get the current observation (the mean)
/// assert_eq!(likelihood.get_observation(), Some(0.5));
///
/// // Update to a new observed value
/// likelihood.update_observation(1.0);
/// assert_eq!(likelihood.get_observation(), Some(1.0));
/// ```
pub trait Observation {
    /// Updates the observed data value.
    ///
    /// The meaning of "observation" depends on the likelihood type:
    /// - `NormalLikelihood`: the mean
    /// - `BinomialLikelihood`: the number of successes
    /// - `NoncentralDLikelihood`/`NoncentralD2Likelihood`: the effect size d
    /// - `StudentTLikelihood`: the mean
    /// - `NoncentralTLikelihood`: the t statistic
    /// - `VarianceRatioLikelihood`: the observed variance ratio
    fn update_observation(&mut self, observation: f64);

    /// Returns the current observed data value.
    fn get_observation(&self) -> Option<f64>;
}

/// A likelihood function for Bayesian inference.
///
/// This enum wraps all available likelihood types, allowing them to be used
/// polymorphically with prior distributions to form models.
///
/// # Creating Likelihoods
///
/// Likelihoods are typically created using the specific type's constructor,
/// then converted to the enum using `.into()`:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Create specific likelihood types
/// let normal = NormalLikelihood::new(0.5, 0.2);
/// let binomial = BinomialLikelihood::new(7.0, 10.0);
///
/// // Convert to Likelihood enum
/// let likelihood: Likelihood = normal.into();
/// ```
///
/// # Combining with Priors
///
/// Likelihoods are combined with priors using multiplication to create models:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
///
/// // Create a model
/// let model: Model = likelihood * prior;
/// ```
///
/// # Evaluating Likelihoods
///
/// Use the [`Function`] trait to evaluate the likelihood at parameter values:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
///
/// // Evaluate the likelihood at different parameter values
/// let value = likelihood.function(0.5).unwrap();
/// ```
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub enum Likelihood {
    /// Normal (Gaussian) likelihood for continuous data with known standard error.
    Normal(NormalLikelihood),
    /// Binomial likelihood for count data (successes out of trials).
    Binomial(BinomialLikelihood),
    /// Student's t likelihood for t-distributed data.
    StudentT(StudentTLikelihood),
    /// Noncentral d likelihood for one-sample effect sizes.
    NoncentralD(NoncentralDLikelihood),
    /// Noncentral d likelihood for two-sample effect sizes.
    NoncentralD2(NoncentralD2Likelihood),
    /// Noncentral t likelihood.
    NoncentralT(NoncentralTLikelihood),
    /// Likelihood for a population variance ratio.
    VarianceRatio(VarianceRatioLikelihood),
}

/// Evaluates `$body` with `$inner` bound to the likelihood inside a
/// [`Likelihood`], whichever variant it is. Adding a variant makes every use
/// of this a compile error until it is listed here.
macro_rules! on_likelihood {
    ($likelihood:expr, $inner:ident => $body:expr) => {
        match $likelihood {
            Likelihood::Normal($inner) => $body,
            Likelihood::Binomial($inner) => $body,
            Likelihood::StudentT($inner) => $body,
            Likelihood::NoncentralD($inner) => $body,
            Likelihood::NoncentralD2($inner) => $body,
            Likelihood::NoncentralT($inner) => $body,
            Likelihood::VarianceRatio($inner) => $body,
        }
    };
}

impl_from_variants!(Likelihood {
    Normal(NormalLikelihood),
    Binomial(BinomialLikelihood),
    StudentT(StudentTLikelihood),
    NoncentralD(NoncentralDLikelihood),
    NoncentralD2(NoncentralD2Likelihood),
    NoncentralT(NoncentralTLikelihood),
    VarianceRatio(VarianceRatioLikelihood),
});

impl LogFunction<LikelihoodError> for Likelihood {
    fn ln_function(&self, x: f64) -> Result<f64, LikelihoodError> {
        on_likelihood!(self, l => l.ln_function(x))
    }
}

impl Validate<LikelihoodError> for Likelihood {
    fn validate(&self) -> Result<(), LikelihoodError> {
        on_likelihood!(self, l => l.validate())
    }
}

impl Observation for Likelihood {
    fn update_observation(&mut self, observation: f64) {
        on_likelihood!(self, l => l.update_observation(observation))
    }

    fn get_observation(&self) -> Option<f64> {
        on_likelihood!(self, l => l.get_observation())
    }
}

impl Function<&[f64], Vec<Option<f64>>, LikelihoodError> for Likelihood {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, LikelihoodError> {
        Ok(x.iter().map(|&x| self.function(x).ok()).collect())
    }
}

impl Likelihood {
    /// Where the likelihood (as a function of the parameter) peaks, and the
    /// width of that peak, as `(centre, width)`. Used to place integration
    /// windows so a narrow peak cannot be missed.
    pub(crate) fn peak_hint(&self) -> (f64, f64) {
        match self {
            Likelihood::Normal(l) => (l.mean, l.se),
            Likelihood::StudentT(l) => (l.mean, l.sd),
            Likelihood::NoncentralT(l) => (l.t, 1.0),
            Likelihood::NoncentralD(l) => (l.d, l.effective_n().recip().sqrt()),
            Likelihood::NoncentralD2(l) => (l.d, l.effective_n().recip().sqrt()),
            Likelihood::Binomial(l) => {
                let p = l.successes / l.trials;
                (p, ((p * (1.0 - p)).max(l.trials.recip()) / l.trials).sqrt())
            }
            Likelihood::VarianceRatio(l) => (l.u, l.u * (2.0 / l.df).sqrt()),
        }
    }
}

#[derive(PartialEq, Debug, Serialize, Clone, Copy)]
pub enum LikelihoodFamily {
    Normal,
    NoncentralD,
    StudentT,
    NoncentralD2,
    NoncentralT,
    Binomial,
    VarianceRatio,
}

impl Family<LikelihoodFamily> for Likelihood {
    fn family(&self) -> LikelihoodFamily {
        match self {
            Likelihood::Normal(_) => LikelihoodFamily::Normal,
            Likelihood::Binomial(_) => LikelihoodFamily::Binomial,
            Likelihood::StudentT(_) => LikelihoodFamily::StudentT,
            Likelihood::NoncentralD(_) => LikelihoodFamily::NoncentralD,
            Likelihood::NoncentralD2(_) => LikelihoodFamily::NoncentralD2,
            Likelihood::NoncentralT(_) => LikelihoodFamily::NoncentralT,
            Likelihood::VarianceRatio(_) => LikelihoodFamily::VarianceRatio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_likelihood_families_dispatch() {
        let cases = [(
            Likelihood::from(VarianceRatioLikelihood::new(1.2, 9.0)),
            LikelihoodFamily::VarianceRatio,
        )];

        for (likelihood, expected) in cases {
            assert_eq!(likelihood.family(), expected);
            assert!(likelihood.validate().is_ok());
        }
    }
}
