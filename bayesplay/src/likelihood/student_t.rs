use rmath::dt;

use serde::{Deserialize, Serialize};

use super::{LikelihoodError, Observation};
use crate::common::Function;
use crate::common::Validate;

#[derive(Default, Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct StudentTLikelihood {
    pub mean: f64,
    pub sd: f64,
    pub df: f64,
}

impl StudentTLikelihood {
    /// Creates a new `StudentTLikelihood` wrapped in a `Likelihood::StudentT`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let mean = 0.0;
    /// let sd = 1.0;
    /// let df = 10.0;
    /// let likelihood = StudentTLikelihood::new(mean, sd, df);
    ///
    /// assert_eq!(likelihood.mean, mean);
    /// assert_eq!(likelihood.sd, sd);
    /// assert_eq!(likelihood.df, df);
    /// ```
    pub fn new(mean: f64, sd: f64, df: f64) -> Self {
        StudentTLikelihood { mean, sd, df }
    }
}

impl Validate<LikelihoodError> for StudentTLikelihood {
    /// Validates the `StudentTLikelihood` parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let valid_likelihood = StudentTLikelihood { mean: 0.0, sd: 1.0, df: 10.0 };
    /// assert!(valid_likelihood.validate().is_ok());
    ///
    /// let invalid_sd = StudentTLikelihood { mean: 0.0, sd: -1.0, df: 10.0 };
    /// assert!(matches!(invalid_sd.validate(), Err(LikelihoodError::InvalidSD(_))));
    ///
    /// let invalid_df = StudentTLikelihood { mean: 0.0, sd: 1.0, df: -5.0 };
    /// assert!(matches!(invalid_df.validate(), Err(LikelihoodError::InvalidDF(_))));
    ///
    /// let both_invalid = StudentTLikelihood { mean: 0.0, sd: -1.0, df: -5.0 };
    /// assert!(matches!(both_invalid.validate(), Err(LikelihoodError::MultipleErrors(_))));
    /// ```
    fn validate(&self) -> Result<(), LikelihoodError> {
        let mut errors = Vec::new();

        if self.sd <= 0.0 {
            errors.push(LikelihoodError::InvalidSD(self.sd));
        }

        if self.df <= 0.0 {
            errors.push(LikelihoodError::InvalidDF(self.df));
        }

        LikelihoodError::from_errors(errors)
    }
}

impl Observation for StudentTLikelihood {
    /// Updates and retrieves observations for the `StudentTLikelihood`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let mut likelihood = StudentTLikelihood { mean: 0.0, sd: 10.0, df: 10.0 };
    ///
    /// // Update the observation
    /// likelihood.update_observation(5.0);
    /// assert_eq!(likelihood.get_observation(), Some(5.0));
    ///
    /// // Check if the observation is a whole number
    /// if let Some(observation) = likelihood.get_observation() {
    ///     assert!(observation.fract() == 0.0); // True, because 5.0 is a whole number
    /// }
    /// ```
    fn update_observation(&mut self, observation: f64) {
        self.mean = observation;
    }

    fn get_observation(&self) -> Option<f64> {
        Some(self.mean)
    }
}

impl Function<f64, f64, LikelihoodError> for StudentTLikelihood {
    /// Evaluates the StudentT likelihood function for a given value `x`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood = StudentTLikelihood { mean: 0.0, sd: 1.0, df: 10.0 };
    ///
    /// // Valid input
    /// let result = likelihood.function(0.5);
    /// assert!(result.is_ok());
    /// assert!((result.unwrap() - 0.33969513635207776447).abs() < 1e-10);
    /// ```
    fn function(&self, x: f64) -> Result<f64, LikelihoodError> {
        let mean = self.mean;
        let df = self.df;
        let sd = self.sd;

        dt_scaled(mean, x, sd, df)
    }
}

fn dt_scaled(x: f64, mean: f64, sd: f64, df: f64) -> Result<f64, LikelihoodError> {
    Ok(dt!(x = (x - mean) / sd, df = df).map_err(LikelihoodError::DistributionError)? / sd)
}
