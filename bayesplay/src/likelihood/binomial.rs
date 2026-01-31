use rmath::dbinom;

use serde::{Deserialize, Serialize};

use crate::common::Function;
use crate::common::Validate;
use crate::likelihood::LikelihoodError;
use crate::likelihood::Observation;

#[derive(Default, Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct BinomialLikelihood {
    pub successes: f64,
    pub trials: f64,
}

impl BinomialLikelihood {
    /// Creates a new `BinomialLikelihood`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let successes = 5.0;
    /// let trials = 10.0;
    /// let likelihood = BinomialLikelihood::new(successes, trials);
    ///
    /// let likelihood: Likelihood = likelihood.into();
    /// match likelihood {
    ///     Likelihood::Binomial(binomial) => {
    ///         assert_eq!(binomial.successes, successes);
    ///         assert_eq!(binomial.trials, trials);
    ///     }
    ///     _ => panic!("Expected a Binomial likelihood"),
    /// }
    /// ```
    pub fn new(successes: f64, trials: f64) -> Self {
        BinomialLikelihood { successes, trials }
    }
}

impl Function<f64, f64, LikelihoodError> for BinomialLikelihood {
    /// Evaluates the binomial likelihood function for a given probability `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood = BinomialLikelihood { successes: 3.0, trials: 10.0 };
    ///
    /// // Valid probability
    /// let result = likelihood.function(0.5);
    /// assert!(result.is_ok());
    /// assert!((result.unwrap() - 0.11718750000000013878).abs() < 1e-10);
    ///
    /// ```
    fn function(&self, p: f64) -> Result<f64, LikelihoodError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(LikelihoodError::InvalidProbability(p));
        }
        let successes = self.successes;
        let trials = self.trials;

        dbinom!(prob = p, size = trials, x = successes).map_err(LikelihoodError::DistributionError)
    }
}

// Tests for the Validate implementation for BinomialLikelihood
///
/// ```
/// use bayesplay::prelude::*;
///
/// // Case 1: Valid parameters
/// let valid = BinomialLikelihood {
///     successes: 5.0,
///     trials: 10.0,
/// };
/// assert!(valid.validate().is_ok());
///
/// // Case 2: Invalid trials (trials < 1)
/// let invalid_trials = BinomialLikelihood {
///     successes: 0.0,
///     trials: 0.0,
/// };
/// match invalid_trials.validate() {
///     Err(LikelihoodError::InvalidTrials(value)) => assert_eq!(value, 0.0),
///     other => panic!("Expected InvalidTrials error, got {:?}", other),
/// }
///
/// // Case 3: Invalid successes (successes > trials)
/// let invalid_successes = BinomialLikelihood {
///     successes: 11.0,
///     trials: 10.0,
/// };
/// match invalid_successes.validate() {
///     Err(LikelihoodError::InvalidSuccess(value)) => assert_eq!(value, 11.0),
///     other => panic!("Expected InvalidSuccess error, got {:?}", other),
/// }
///
/// // Case 4: Trials is not an whole number
/// let invalid_trials = BinomialLikelihood {
///     successes: 0.0,
///     trials: 1.1,
/// };
///
/// match invalid_trials.validate() {
///     Err(LikelihoodError::InvalidTrials(value)) => assert_eq!(value, 1.1),
///     other => panic!("Expected InvalidTrials error, got {:?}", other),
/// }
///
/// // Case 5: Successes is not a whole number
/// let invalid_successes = BinomialLikelihood {
///     successes: 11.2,
///     trials: 12.0,
/// };
/// match invalid_successes.validate() {
///     Err(LikelihoodError::InvalidSuccess(value)) => assert_eq!(value, 11.2),
///     other => panic!("Expected InvalidSuccess error, got {:?}", other),
/// }
///
/// // Case 6: Both invalid (trials < 1 and successes > trials)
/// let both_invalid = BinomialLikelihood {
///     successes: 5.0,
///     trials: 0.0,
/// };
/// match both_invalid.validate() {
///     Err(LikelihoodError::MultipleErrors(errors)) => {
///         assert_eq!(errors.len(), 2);
///         assert!(matches!(errors[0], LikelihoodError::InvalidTrials(0.0)));
///         assert!(matches!(errors[1], LikelihoodError::InvalidSuccess(5.0)));
///     },
///     other => panic!("Expected MultipleErrors, got {:?}", other),
/// }
/// ```
impl Validate<LikelihoodError> for BinomialLikelihood {
    fn validate(&self) -> Result<(), LikelihoodError> {
        let mut errors = Vec::new();

        let trials_less_than_one = self.trials < 1.0;
        let trials_is_not_whole_number = self.trials.fract() != 0.0;
        if trials_less_than_one || trials_is_not_whole_number {
            errors.push(LikelihoodError::InvalidTrials(self.trials));
        }

        let successes_greater_than_trials = self.successes > self.trials;
        let successes_is_not_whole_number = self.successes.fract() != 0.0;
        if successes_greater_than_trials || successes_is_not_whole_number {
            errors.push(LikelihoodError::InvalidSuccess(self.successes));
        }

        LikelihoodError::from_errors(errors)
    }
}

/// Tests for the Observation implementation for BinomialLikelihood
///
/// ```
/// use bayesplay::prelude::*;
///
/// // Create a new BinomialLikelihood
/// let mut likelihood = BinomialLikelihood {
///     successes: 0.0,
///     trials: 10.0,
/// };
///
/// // Test initial state
/// assert_eq!(likelihood.get_observation(), Some(0.0));
///
/// // Update observation with a new value
/// likelihood.update_observation(5.0);
///
/// // Verify the observation was updated correctly
/// assert_eq!(likelihood.get_observation(), Some(5.0));
///
/// // Update with another value
/// likelihood.update_observation(8.0);
///
/// // Verify the new observation
/// assert_eq!(likelihood.get_observation(), Some(8.0));
///
/// ```
impl Observation for BinomialLikelihood {
    fn update_observation(&mut self, observation: f64) {
        self.successes = observation
    }

    fn get_observation(&self) -> Option<f64> {
        Some(self.successes)
    }
}
