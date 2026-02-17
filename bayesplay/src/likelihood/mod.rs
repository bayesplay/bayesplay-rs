use enum_dispatch::enum_dispatch;
use thiserror::Error;

pub(crate) mod binomial;
pub(crate) mod noncentral_d;
pub(crate) mod noncentral_d2;
pub(crate) mod noncentral_t;
pub(crate) mod normal;
pub(crate) mod student_t;

use crate::common::Function;
use crate::common::Validate;

pub use binomial::BinomialLikelihood;
pub use noncentral_d::NoncentralDLikelihood;
pub use noncentral_d2::NoncentralD2Likelihood;
pub use noncentral_t::NoncentralTLikelihood;
pub use normal::NormalLikelihood;
pub use student_t::StudentTLikelihood;

use serde::{Deserialize, Serialize};

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

/// Trait for likelihood observation management (used by enum_dispatch)
#[enum_dispatch]
pub trait Observation {
    fn update_observation(&mut self, observation: f64);
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

#[enum_dispatch(Observation, LikelihoodFn, LikelihoodValidate)]
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub enum Likelihood {
    Normal(NormalLikelihood),
    Binomial(BinomialLikelihood),
    StudentT(StudentTLikelihood),
    NoncentralD(NoncentralDLikelihood),
    NoncentralD2(NoncentralD2Likelihood),
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
