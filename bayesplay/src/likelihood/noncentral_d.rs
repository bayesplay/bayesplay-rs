use rmath::dt;
use serde::{Deserialize, Serialize};

/// A noncentral d likelihood for one-sample effect sizes (Cohen's d).
///
/// This likelihood is used when the observed data is a standardized effect size
/// from a one-sample design. The effect size d is related to the t-statistic
/// and sample size, and follows a noncentral t-distribution.
///
/// # Parameters
///
/// * `d` - The observed effect size (Cohen's d)
/// * `n` - The sample size (must be ≥ 1)
///
/// # Relationship to t-statistic
///
/// For a one-sample design:
/// - t = d × √n
/// - df = n - 1
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Effect size d = 0.5 from a sample of 30
/// let likelihood = NoncentralDLikelihood::new(0.5, 30.0);
///
/// // Combine with a Cauchy prior on effect size
/// let prior: Prior = CauchyPrior::new(0.0, 0.707, (None, None)).into();
/// let model: Model = Likelihood::from(likelihood) * prior;
///
/// // Compute the marginal likelihood
/// let marginal = model.integral().unwrap();
/// ```
#[derive(Default, Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct NoncentralDLikelihood {
    /// The observed effect size (Cohen's d).
    pub d: f64,
    /// The sample size.
    pub n: f64,
}

use super::{LikelihoodError, Observation};
use crate::common::Function;
use crate::common::Validate;
use crate::prelude::NoncentralTLikelihood;

// use super::NoncentralTLikelihood;

impl NoncentralDLikelihood {
    /// Creates a new `NoncentralDLikelihood` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood = NoncentralDLikelihood::new(2.5, 10.0);
    /// let likelihood: Likelihood = likelihood.into();
    /// match likelihood {
    ///     Likelihood::NoncentralD(noncentral_d) => {
    ///         assert_eq!(noncentral_d.d, 2.5);
    ///         assert_eq!(noncentral_d.n, 10.0);
    ///     },
    ///     _ => panic!("Expected NoncentralD variant"),
    /// }
    /// ```
    pub fn new(d: f64, n: f64) -> Self {
        NoncentralDLikelihood { d, n }
    }

    pub fn get_tvalue(&self) -> (f64, f64, f64) {
        let n = self.n;
        let d = self.d;

        let t = d * n.sqrt();
        let df = n - 1.0;

        (t, df, n)
    }

    pub fn into_t(&self) -> (NoncentralTLikelihood, Option<f64>) {
        let (t, df, n) = self.get_tvalue();
        (NoncentralTLikelihood::new(t, df), Some(n))
    }
}
//
// pub fn new_checked(d: f64, n: f64) -> Result<Likelihood, LikelihoodError> {
//     let likelihood = NoncentralDLikelihood { d, n };
//     likelihood.validate()?;
//     Ok(Likelihood::NoncentralD(likelihood))
// }

impl Function<f64, f64, LikelihoodError> for NoncentralDLikelihood {
    /// Evaluates the noncentral D likelihood function.
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood = NoncentralDLikelihood { d: 2.5, n: 10.0 };
    /// let result = likelihood.function(1.2);
    /// assert!(result.is_ok());
    /// assert!((result.unwrap() - 0.01991719281946398995).abs() < 1e-10);
    ///
    /// ```
    fn function(&self, x: f64) -> Result<f64, LikelihoodError> {
        let d = self.d;
        let n = self.n;
        let df = n - 1.0;
        let ncp = x * n.sqrt();
        dt!(x = d * n.sqrt(), df = df, ncp = ncp).map_err(LikelihoodError::DistributionError)
    }
}

impl Observation for NoncentralDLikelihood {
    /// Updates and retrieves observations for `NoncentralDLikelihood`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let mut likelihood = NoncentralDLikelihood { d: 0.0, n: 10.0 };
    ///
    /// likelihood.update_observation(2.5);
    /// assert_eq!(likelihood.get_observation(), Some(2.5));
    /// ```
    fn update_observation(&mut self, observation: f64) {
        self.d = observation;
    }

    fn get_observation(&self) -> Option<f64> {
        Some(self.d)
    }
}

impl Validate<LikelihoodError> for NoncentralDLikelihood {
    /// Validates the `NoncentralDLikelihood` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let valid_likelihood = NoncentralDLikelihood { d: 2.5, n: 10.0 };
    /// assert!(valid_likelihood.validate().is_ok());
    ///
    /// let invalid_likelihood = NoncentralDLikelihood { d: 2.5, n: 0.5 };
    /// assert!(matches!(
    ///     invalid_likelihood.validate(),
    ///     Err(LikelihoodError::InvalidN(0.5))
    /// ));
    /// ```
    fn validate(&self) -> Result<(), LikelihoodError> {
        if self.n < 1.0 {
            return Err(LikelihoodError::InvalidN(self.n));
        }
        Ok(())
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
    fn test_noncentral_d_likelihood() {
        let d = 0.5;
        let n = 10.0;
        let likelihood = NoncentralDLikelihood::new(d, n);
        let x = 1.0;
        let result = likelihood.function(x).unwrap();
        approx_eq!(result, 0.128_440_99);
    }
}
