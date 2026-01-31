// TODO: Still to write doc tests here
//
use rmath::dt;
use serde::{Deserialize, Serialize};

use super::{LikelihoodError, Observation};
use crate::common::Function;
use crate::common::Validate;

#[derive(Default, Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct NoncentralTLikelihood {
    pub t: f64,
    pub df: f64,
}

impl NoncentralTLikelihood {
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

impl Observation for NoncentralTLikelihood {
    fn update_observation(&mut self, observation: f64) {
        self.t = observation;
    }

    fn get_observation(&self) -> Option<f64> {
        Some(self.t)
    }
}

impl Validate<LikelihoodError> for NoncentralTLikelihood {
    fn validate(&self) -> Result<(), LikelihoodError> {
        if self.df < 1.0 {
            return Err(LikelihoodError::InvalidDF(self.df));
        }
        Ok(())
    }
}
