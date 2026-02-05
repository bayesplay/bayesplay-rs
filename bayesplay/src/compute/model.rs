//! Bayesian model computation.
//!
//! This module provides types for combining priors and likelihoods into models,
//! and computing posteriors, marginal likelihoods, and predictive distributions.

use itertools::izip;
use rmath::integrate;

use std::ops::Mul;
use thiserror::Error;

use serde::{Deserialize, Serialize};

use crate::common::Function;
use crate::common::Integrate;
use crate::common::Range;
use crate::common::Validate;

use crate::likelihood::*;
use crate::prior::*;

/// A Bayesian model combining a prior distribution with a likelihood function.
///
/// A model is the fundamental building block for Bayesian inference. It combines:
/// - A **prior** representing beliefs about parameter values before seeing data
/// - A **likelihood** representing the probability of the observed data given parameters
///
/// # Creating Models
///
/// Models are created by multiplying a prior and a likelihood (order doesn't matter):
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
///
/// // Both of these create the same model
/// let model1: Model = likelihood * prior;
/// let model2: Model = prior * likelihood;
/// ```
///
/// # Computing Posteriors
///
/// Use [`Model::posterior()`] to compute the posterior distribution:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
/// let model: Model = likelihood * prior;
///
/// let posterior = model.posterior().unwrap();
///
/// // Evaluate the posterior at a point
/// let density = posterior.function(0.4).unwrap();
/// ```
///
/// # Computing Bayes Factors
///
/// The marginal likelihood (integral of the model) can be used to compute
/// Bayes factors for hypothesis testing:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
///
/// // H1: effect follows a normal distribution
/// let h1_prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
/// let h1_model: Model = likelihood * h1_prior;
///
/// // H0: effect is exactly zero
/// let h0_prior: Prior = PointPrior::new(0.0).into();
/// let h0_model: Model = likelihood * h0_prior;
///
/// // Bayes factor (BF10)
/// let bf10 = h1_model.integral().unwrap() / h0_model.integral().unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct Model {
    /// The prior distribution over parameter values.
    pub prior: Prior,
    /// The likelihood function for the observed data.
    pub likelihood: Likelihood,
    /// The integration range for the model.
    pub range: (f64, f64),
}

/// Errors that can occur during model integration.
///
/// Integration can fail due to issues with the prior, likelihood, or the
/// numerical integration algorithm itself.
#[derive(Error, Debug)]
pub enum IntegralError {
    /// An error occurred with the prior distribution.
    #[error("Error with prior: {0}")]
    Prior(#[from] PriorError),
    /// An error occurred with the likelihood function.
    #[error("Error with likelihood: {0}")]
    Likelihood(#[from] LikelihoodError),
    /// The numerical integration algorithm failed.
    #[error("Error with integration: {0}")]
    Integration(rmath::integration::IntegrationError),
}

impl Model {
    /// Returns the observed data value from the likelihood.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
    /// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
    /// let model: Model = likelihood * prior;
    ///
    /// assert_eq!(model.get_observation(), Some(0.5));
    /// ```
    pub fn get_observation(&self) -> Option<f64> {
        self.likelihood.get_observation()
    }

    /// Computes the posterior distribution.
    ///
    /// The posterior is computed by normalizing the product of the prior and
    /// likelihood by the marginal likelihood (the integral of the model).
    ///
    /// # Returns
    ///
    /// A [`Posterior`] object that can be evaluated at any parameter value.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be integrated (e.g., invalid
    /// parameters or numerical integration failure).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
    /// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
    /// let model: Model = likelihood * prior;
    ///
    /// let posterior = model.posterior().unwrap();
    ///
    /// // The posterior integrates to 1
    /// let area = posterior.integral().unwrap();
    /// assert!((area - 1.0).abs() < 0.001);
    /// ```
    pub fn posterior(&self) -> Result<Posterior, IntegralError> {
        let constant = self.integral()?;

        let model = *self;

        Ok(Posterior { model, constant })
    }

    /// Creates a predictive distribution from the model.
    ///
    /// The predictive distribution gives the probability of future observations
    /// by integrating over the prior distribution of parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
    /// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
    /// let model: Model = likelihood * prior;
    ///
    /// let predictive = model.predictive();
    ///
    /// // Evaluate the predictive probability of a future observation
    /// let prob = predictive.function(0.3).unwrap();
    /// ```
    pub fn predictive(&self) -> Predictive {
        Predictive(*self)
    }
}

impl Range for Model {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        let prior = self.prior;
        prior.range()
    }
    fn default_range(&self) -> (f64, f64) {
        let prior = self.prior;
        prior.default_range()
    }
}

impl Function<f64, f64, anyhow::Error> for Model {
    fn function(&self, x: f64) -> Result<f64, anyhow::Error> {
        let prior = self.prior;
        let likelihood = self.likelihood;
        let res = prior.function(x)? * likelihood.function(x)?;
        Ok(res)
    }
}

impl Integrate<IntegralError, anyhow::Error> for Model {
    /// Integrate a model object
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// # fn main() {
    /// let likelihood = NormalLikelihood::new(0.2, 4.0);
    /// let prior = NormalPrior::new(0.0, 1.0, (Some(-f64::INFINITY), Some(f64::INFINITY)));
    ///
    /// let model = Model {prior: prior.into(), likelihood: likelihood.into(), range: (-f64::INFINITY, f64::INFINITY)};
    /// let auc = model.integral().unwrap();
    ///
    /// assert_eq!(auc as f32, 0.09664394965841581 as f32);
    /// # }
    /// ```
    fn integral(&self) -> Result<f64, IntegralError> {
        let prior = self.prior;
        let likelihood = self.likelihood;

        let (lower, upper) = self.range_or_default();
        prior.validate()?;
        likelihood.validate()?;

        match prior {
            Prior::Point(point) => Ok(likelihood.function(point.point)?),
            _ => {
                let model = *self;
                let f = move |x| model.function(x).unwrap();
                let h = integrate!(f = f, lower = lower, upper = upper);

                match h {
                    Ok(v) => Ok(v.value),
                    Err(e) => Err(IntegralError::Integration(e)),
                }
            }
        }
    }
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        let (_, _) = (lb, ub);
        self.integral()
    }
}

/// A posterior distribution computed from a Bayesian model.
///
/// The posterior represents the updated beliefs about parameter values after
/// observing data. It is proportional to the product of the prior and likelihood,
/// normalized to integrate to 1.
///
/// # Computing Posteriors
///
/// Posteriors are obtained from a [`Model`] using the [`Model::posterior()`] method:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
/// let model: Model = likelihood * prior;
///
/// let posterior = model.posterior().unwrap();
/// ```
///
/// # Evaluating the Posterior
///
/// Use the [`Function`] trait to evaluate the posterior density at parameter values:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
/// let model: Model = likelihood * prior;
/// let posterior = model.posterior().unwrap();
///
/// // Evaluate at a single point
/// let density = posterior.function(0.4).unwrap();
///
/// // Evaluate at multiple points
/// let points = vec![0.0, 0.25, 0.5, 0.75, 1.0];
/// let densities = posterior.function(points.as_slice()).unwrap();
/// ```
///
/// # Integration
///
/// The posterior supports integration to compute probability intervals:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
/// let model: Model = likelihood * prior;
/// let posterior = model.posterior().unwrap();
///
/// // Probability that parameter is between 0 and 1
/// let prob = posterior.integrate(Some(0.0), Some(1.0)).unwrap();
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Posterior {
    model: Model,
    constant: f64,
}

/// A predictive distribution from a Bayesian model.
///
/// The predictive distribution gives the probability of future observations
/// by averaging the likelihood over all possible parameter values, weighted
/// by the prior.
///
/// This is useful for:
/// - Predicting future observations
/// - Model checking (comparing predictions to actual data)
/// - Computing marginal likelihoods for model comparison
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
/// let model: Model = likelihood * prior;
///
/// let predictive = model.predictive();
///
/// // What's the probability of observing a value of 0.3?
/// let prob = predictive.function(0.3).unwrap();
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Predictive(Model);

impl Function<f64, f64, anyhow::Error> for Posterior {
    fn function(&self, x: f64) -> Result<f64, anyhow::Error> {
        let prior = self.model.prior;
        let likelihood = self.model.likelihood;
        let k = self.constant;
        let res = (likelihood.function(x)? * prior.function(x)?) / k;
        Ok(res)
    }
}

impl Function<&[f64], Vec<Option<f64>>, anyhow::Error> for Posterior {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, anyhow::Error> {
        let prior = self.model.prior;
        let prior = prior.function(x).unwrap();
        let likelihood = self.model.likelihood;
        let likelihood = likelihood.function(x).unwrap();
        let norm = (self.model.likelihood * self.model.prior).integral()?;
        let res = izip!(prior, likelihood)
            .map(|(p, l)| match (p, l, norm) {
                (Some(p), Some(l), norm) => Some((p * l) / norm),
                _ => None,
            })
            .collect();
        Ok(res)
    }
}

impl Function<f64, f64, anyhow::Error> for Predictive {
    fn function(&self, x: f64) -> Result<f64, anyhow::Error> {
        let mut likelihood = self.0.likelihood;
        likelihood.update_observation(x);
        let model = self.0.prior * likelihood;
        model
            .integral()
            .map_err(|_| anyhow::Error::msg("Error with integration"))
    }
}

impl Function<&[f64], Vec<Option<f64>>, anyhow::Error> for Predictive {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, anyhow::Error> {
        let mut likelihood = self.0.likelihood;
        let res: Vec<Option<f64>> = x
            .iter()
            .map(|x| {
                likelihood.update_observation(*x);
                let model = self.0.prior * likelihood;
                model.integral().ok()
            })
            .collect();
        Ok(res)
    }
}

impl Mul<Prior> for Likelihood {
    type Output = Model;
    /// Create a model from a prior and a likelihood
    /// ```
    /// # use bayesplay::prelude::*;
    /// # fn main() {
    /// //let likelihood = likelihood!(family = "normal", mean = 0.2, se = 4.0);
    /// //let prior = prior!(family = "normal", mean = 0.0, sd = 1.0);
    /// //let model = likelihood * prior;
    /// # }
    /// ```
    fn mul(self, rhs: Prior) -> Self::Output {
        let prior = rhs;
        let likelihood = self;
        let range = prior.range_or_default();
        Model {
            prior,
            likelihood,
            range,
        }
    }
}

impl Mul<Likelihood> for Prior {
    type Output = Model;
    /// Create a model from a prior and a likelihood
    /// ```
    /// # use bayesplay::prelude::*;
    /// # fn main() {
    /// //let likelihood = likelihood!(family = "normal", mean = 0.2, se = 4.0);
    /// //let prior = prior!(family = "normal", mean = 0.0, sd = 1.0);
    /// //let model = prior * likelihood;
    /// # }
    /// ```
    fn mul(self, rhs: Likelihood) -> Self::Output {
        let likelihood = rhs;
        let prior = self;
        let range = prior.range_or_default();
        Model {
            prior,
            likelihood,
            range,
        }
    }
}

impl Range for Posterior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        let prior = self.model.prior;
        prior.range()
    }
    fn default_range(&self) -> (f64, f64) {
        let prior = self.model.prior;
        prior.default_range()
    }
}

impl Integrate<IntegralError, anyhow::Error> for Posterior {
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        let posterior = *self;
        let lb = lb.unwrap_or(posterior.range_or_default().0);
        let ub = ub.unwrap_or(posterior.range_or_default().1);

        let f = move |x| posterior.function(x).unwrap();
        let h = integrate!(f = f, lower = lb, upper = ub);
        match h {
            Ok(v) => Ok(v.value),
            Err(e) => Err(IntegralError::Integration(e)),
        }
    }
    fn integral(&self) -> Result<f64, IntegralError> {
        let posterior = *self;
        let (lb, ub) = posterior.model.prior.range_or_default();
        let f = move |x| posterior.function(x).unwrap();
        let h = integrate!(f = f, lower = lb, upper = ub);
        match h {
            Ok(v) => Ok(v.value),
            Err(e) => Err(IntegralError::Integration(e)),
        }
    }
}
