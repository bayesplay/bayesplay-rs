// TODO: Still to write doc tests here

use rmath::dt;
use serde::{Deserialize, Serialize};

use super::{LikelihoodError, Observation};
use crate::common::Function;
use crate::common::Validate;

// use crate::likelihood::NoncentralTLikelihood;

#[derive(Default, Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct NoncentralD2Likelihood {
    pub d: f64,
    pub n1: f64,
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

impl Observation for NoncentralD2Likelihood {
    fn update_observation(&mut self, observation: f64) {
        self.d = observation;
    }

    fn get_observation(&self) -> Option<f64> {
        Some(self.d)
    }
}
