use rmath::dnorm;
use serde::{Deserialize, Serialize};

/// A normal likelihood function with specified mean and standard error.
///
/// # Parameters
///
/// * `mean` - The observed mean or point estimate
/// * `se` - The standard error of the estimate
///
/// # Examples
///
/// ```
/// use bayesplay::prelude::*;
///
/// // Create a normal likelihood with mean 0.0 and standard error 1.0
/// let likelihood = NormalLikelihood::new(0.0, 1.0);
///
/// // Access the parameters directly
/// assert_eq!(likelihood.mean, 0.0);
/// assert_eq!(likelihood.se, 1.0);
/// ```
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
#[allow(dead_code)]
pub struct NormalLikelihood {
    pub mean: f64,
    pub se: f64,
}

use crate::common::Function;
use crate::common::Validate;
use crate::likelihood::LikelihoodError;
use crate::likelihood::Observation;

/// Constructor methods for creating normal likelihood functions.
///
/// # Examples
///
/// ```
/// use bayesplay::prelude::*;
///
/// // Create a normal likelihood with mean 0.5 and standard error 0.2
/// let likelihood = NormalLikelihood::new(0.5, 0.2);
///
/// // Access the parameters directly
/// assert_eq!(likelihood.mean, 0.5);
/// assert_eq!(likelihood.se, 0.2);
///
/// // Create a likelihood for different experimental results
/// let new_result = NormalLikelihood::new(1.2, 0.3);
/// ```
impl NormalLikelihood {
    pub fn new(mean: f64, se: f64) -> Self {
        NormalLikelihood { mean, se }
    }
}

/// Implements the `Observation` trait for `NormalLikelihood`.
///
/// This trait allows retrieving and updating the observed mean value.
///
/// # Examples
///
/// ```
/// use bayesplay::prelude::*;
///
/// // Create a normal likelihood with mean 0.0 and standard error 1.0
/// let mut likelihood = NormalLikelihood::new(0.0, 1.0);
///
/// // The observation is the mean value
/// assert_eq!(likelihood.get_observation(), Some(0.0));
///
/// // Update the observation to a new value
/// likelihood.update_observation(2.5);
///
/// // Verify the observation was updated
/// assert_eq!(likelihood.get_observation(), Some(2.5));
/// assert_eq!(likelihood.mean, 2.5);
///
/// // The standard error remains unchanged
/// assert_eq!(likelihood.se, 1.0);
/// ```
impl Observation for NormalLikelihood {
    fn get_observation(&self) -> Option<f64> {
        Some(self.mean)
    }

    fn update_observation(&mut self, observation: f64) {
        self.mean = observation;
    }
}

/// Implements the Function trait for calculating the likelihood value.
///
/// This calculates the likelihood of parameter value x given the observed data,
/// which follows a normal distribution with mean and standard error.
///
/// # Examples
///
/// ```
/// use bayesplay::prelude::*;
/// use std::f64::consts::PI;
///
/// // Create a normal likelihood with mean 0.0 and standard error 1.0
/// let likelihood = NormalLikelihood::new(0.0, 1.0);
///
/// // When x equals the mean, the value should be at maximum (1/√(2πσ²))
/// let max_value = likelihood.function(0.0).unwrap();
/// let expected_max = 1.0 / (2.0 * PI).sqrt();
/// assert!((max_value - expected_max).abs() < 1e-10);
///
/// // At one standard error away from mean (x = 1.0), the likelihood decreases
/// let value_at_1sd = likelihood.function(1.0).unwrap();
/// assert!(value_at_1sd < max_value);
///
/// // The function is symmetric around the mean
/// let value_at_minus_1sd = likelihood.function(-1.0).unwrap();
/// assert!((value_at_1sd - value_at_minus_1sd).abs() < 1e-10);
///
/// // With invalid standard error, the function should return an error
/// let invalid_likelihood = NormalLikelihood { mean: 0.0, se: -1.0 };
/// assert!(invalid_likelihood.function(0.0).is_err());
/// ```
impl Function<f64, f64, LikelihoodError> for NormalLikelihood {
    fn function(&self, x: f64) -> Result<f64, LikelihoodError> {
        self.validate()?;
        let mean = self.mean;
        let se = self.se;

        dnorm!(x = mean, mean = x, sd = se).map_err(LikelihoodError::DistributionError)
    }
}

/// Validates the `NormalLikelihood` structure parameters.
///
/// Ensures the standard error (se) is positive.
///
/// # Examples
///
/// ```
/// use bayesplay::prelude::*;
///
/// // Create a normal likelihood with valid parameters
/// let valid_likelihood = NormalLikelihood::new(0.0, 1.0);
///
/// // Validation should pass
/// assert!(valid_likelihood.validate().is_ok());
///
/// // Function should also work since it calls validate()
/// assert!(valid_likelihood.function(0.0).is_ok());
///
/// // Create a normal likelihood with invalid standard error
/// let invalid_likelihood = NormalLikelihood { mean: 0.0, se: 0.0 };
///
/// // Validation should fail with InvalidSE error
/// let validation_result = invalid_likelihood.validate();
/// assert!(validation_result.is_err());
/// assert!(matches!(validation_result, Err(LikelihoodError::InvalidSE(se)) if se == 0.0));
///
/// // Function should also fail with the same error
/// let function_result = invalid_likelihood.function(0.0);
/// assert!(function_result.is_err());
/// assert!(matches!(function_result, Err(LikelihoodError::InvalidSE(se)) if se == 0.0));
///
/// // Negative standard error is also invalid
/// let negative_se_likelihood = NormalLikelihood { mean: 0.0, se: -1.5 };
/// assert!(matches!(
///     negative_se_likelihood.validate(),
///     Err(LikelihoodError::InvalidSE(se)) if se == -1.5
/// ));
/// ```
impl Validate<LikelihoodError> for NormalLikelihood {
    fn validate(&self) -> Result<(), LikelihoodError> {
        if self.se <= 0.0 {
            Err(LikelihoodError::InvalidSE(self.se))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    macro_rules! approx_eq {
        ($a:expr, $b:expr) => {
            assert_eq!($a as f32, $b as f32)
        };
    }

    #[test]
    fn test_normal_likelihood() {
        let p = NormalLikelihood { mean: 0.0, se: 1.0 };
        let x = 0.0;
        let y = p.function(x).unwrap();
        approx_eq!(y, 0.3989423);
    }
}
