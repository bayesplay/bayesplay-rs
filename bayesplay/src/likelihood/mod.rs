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

use enum_dispatch::enum_dispatch;
use thiserror::Error;

pub(crate) mod binomial;
pub(crate) mod noncentral_d;
pub(crate) mod noncentral_d2;
pub(crate) mod noncentral_t;
pub(crate) mod normal;
pub(crate) mod student_t;

use crate::common::Family;
use crate::common::Function;
use crate::common::Validate;

pub use binomial::BinomialLikelihood;
pub use noncentral_d::NoncentralDLikelihood;
pub use noncentral_d2::NoncentralD2Likelihood;
pub use noncentral_t::NoncentralTLikelihood;
pub use normal::NormalLikelihood;
pub use student_t::StudentTLikelihood;

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
    #[error("N value of {0} is invalid. N must be positive")]
    InvalidN(f64),
    #[error("N1 value of {0} is invalid. N1 must be positive")]
    InvalidN1(f64),
    #[error("N2 value of {0} is invalid. N2 must be positive")]
    InvalidN2(f64),
    #[error("Trials value of {0} is invalid. Trails must be a positive integer")]
    InvalidTrials(f64),
    #[error("Success value of {0} is invalid. Success must be positive integer and less Trials")]
    InvalidSuccess(f64),
    #[error("DF value of {0} is invalid. DF must be positive")]
    InvalidDF(f64),
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
#[enum_dispatch]
pub trait Observation {
    /// Updates the observed data value.
    ///
    /// The meaning of "observation" depends on the likelihood type:
    /// - `NormalLikelihood`: the mean
    /// - `BinomialLikelihood`: the number of successes
    /// - `NoncentralDLikelihood`/`NoncentralD2Likelihood`: the effect size d
    /// - `StudentTLikelihood`: the mean
    /// - `NoncentralTLikelihood`: the t statistic
    fn update_observation(&mut self, observation: f64);

    /// Returns the current observed data value.
    fn get_observation(&self) -> Option<f64>;
}

/// Concrete trait for likelihood function evaluation (used by enum_dispatch)
#[enum_dispatch]
pub trait LikelihoodFn {
    fn likelihood_function(&self, x: f64) -> Result<f64, LikelihoodError>;
}

/// Concrete trait for likelihood validation (used by enum_dispatch)
#[enum_dispatch]
pub trait LikelihoodValidate {
    fn likelihood_validate(&self) -> Result<(), LikelihoodError>;
}

// Implement the concrete traits for each likelihood type by delegating to the generic traits
impl LikelihoodFn for NormalLikelihood {
    fn likelihood_function(&self, x: f64) -> Result<f64, LikelihoodError> {
        Function::function(self, x)
    }
}

impl LikelihoodFn for BinomialLikelihood {
    fn likelihood_function(&self, x: f64) -> Result<f64, LikelihoodError> {
        Function::function(self, x)
    }
}

impl LikelihoodFn for StudentTLikelihood {
    fn likelihood_function(&self, x: f64) -> Result<f64, LikelihoodError> {
        Function::function(self, x)
    }
}

impl LikelihoodFn for NoncentralDLikelihood {
    fn likelihood_function(&self, x: f64) -> Result<f64, LikelihoodError> {
        Function::function(self, x)
    }
}

impl LikelihoodFn for NoncentralD2Likelihood {
    fn likelihood_function(&self, x: f64) -> Result<f64, LikelihoodError> {
        Function::function(self, x)
    }
}

impl LikelihoodFn for NoncentralTLikelihood {
    fn likelihood_function(&self, x: f64) -> Result<f64, LikelihoodError> {
        Function::function(self, x)
    }
}

impl LikelihoodValidate for NormalLikelihood {
    fn likelihood_validate(&self) -> Result<(), LikelihoodError> {
        Validate::validate(self)
    }
}

impl LikelihoodValidate for BinomialLikelihood {
    fn likelihood_validate(&self) -> Result<(), LikelihoodError> {
        Validate::validate(self)
    }
}

impl LikelihoodValidate for StudentTLikelihood {
    fn likelihood_validate(&self) -> Result<(), LikelihoodError> {
        Validate::validate(self)
    }
}

impl LikelihoodValidate for NoncentralDLikelihood {
    fn likelihood_validate(&self) -> Result<(), LikelihoodError> {
        Validate::validate(self)
    }
}

impl LikelihoodValidate for NoncentralD2Likelihood {
    fn likelihood_validate(&self) -> Result<(), LikelihoodError> {
        Validate::validate(self)
    }
}

impl LikelihoodValidate for NoncentralTLikelihood {
    fn likelihood_validate(&self) -> Result<(), LikelihoodError> {
        Validate::validate(self)
    }
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
#[enum_dispatch(Observation, LikelihoodFn, LikelihoodValidate)]
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
}

// Note: From implementations are generated automatically by enum_dispatch

// Implement the generic Function trait for Likelihood using enum_dispatch
impl Function<f64, f64, LikelihoodError> for Likelihood {
    fn function(&self, x: f64) -> Result<f64, LikelihoodError> {
        self.likelihood_function(x)
    }
}

impl Function<&[f64], Vec<Option<f64>>, LikelihoodError> for Likelihood {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, LikelihoodError> {
        Ok(x.iter()
            .map(|x| self.likelihood_function(*x).ok())
            .collect::<Vec<Option<_>>>())
    }
}

// Implement the generic Validate trait for Likelihood using enum_dispatch
impl Validate<LikelihoodError> for Likelihood {
    fn validate(&self) -> Result<(), LikelihoodError> {
        self.likelihood_validate()
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
             
         } 
    }
}
