//! Bayesian model computation.
//!
//! This module provides types for combining priors and likelihoods into models,
//! and computing posteriors, marginal likelihoods, and predictive distributions.

use itertools::izip;
use rmath::integrate;
use rmath::pcauchy;
use rmath::pexp;
use rmath::pt;

use std::ops::Mul;
use thiserror::Error;

use serde::{Deserialize, Serialize};

use crate::common::Auc;
use crate::common::Family;
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
    #[error("Have to approximate")]
    Approximation,
    /// A distribution function call failed (e.g. invalid parameters).
    #[error("Distribution error: {0}")]
    Distribution(String),
    /// `estimate_marginal` requires a prior with explicit finite bounds.
    #[error("Prior must have finite bounds for estimate_marginal")]
    UnboundedPrior,
    /// `estimate_marginal` requires a sample size, but the `ApproximateModel` has `n = None`.
    #[error("ApproximateModel has no sample size (n); estimate_marginal requires n")]
    MissingN,
}

/// The result of [`estimate_marginal`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarginalResult {
    /// The marginal likelihood (integrating out the effect size parameter).
    pub marginal: f64,
    /// The Bayes factor relative to the point-null (effect = 0).
    pub bf: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct ApproximateModel {
    pub approximation: bool,
    supported_prior: bool,
    is_large: bool,
    n: Option<f64>,
    t: f64,
    df: f64,
}

impl Default for ApproximateModel {
    fn default() -> Self {
        ApproximateModel {
            approximation: false,
            supported_prior: false,
            is_large: false,
            n: None,
            t: 0.0,
            df: 0.0,
        }
    }
}

impl ApproximateModel {
    /// Returns true if approximation is needed
    pub fn needs_approximation(&self) -> bool {
        self.approximation
    }

    /// Returns true if the prior is supported for approximation
    pub fn has_supported_prior(&self) -> bool {
        self.supported_prior
    }

    /// Returns true if the test statistic is large (|t| > 5)
    pub fn is_large_statistic(&self) -> bool {
        self.is_large
    }

    /// Creates a new builder for constructing an ApproximateModel
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bayesplay::compute::model::ApproximateModel;
    ///
    /// let model = ApproximateModel::builder()
    ///     .approximation(true)
    ///     .supported_prior(true)
    ///     .t(10.0)
    ///     .df(20.0)
    ///     .build();
    /// ```
    pub fn builder() -> ApproximateModel {
        ApproximateModel::default()
    }

    /// Sets whether approximation is needed
    pub fn approximation(mut self, approximation: bool) -> Self {
        self.approximation = approximation;
        self
    }

    /// Sets whether the prior is supported for approximation
    pub fn supported_prior(mut self, supported_prior: bool) -> Self {
        self.supported_prior = supported_prior;
        self
    }

    /// Sets whether the test statistic is large (|t| > 5)
    pub fn is_large(mut self, is_large: bool) -> Self {
        self.is_large = is_large;
        self
    }

    /// Sets the sample size (optional)
    pub fn n(mut self, n: Option<f64>) -> Self {
        self.n = n;
        self
    }

    /// Sets the test statistic value
    pub fn t(mut self, t: f64) -> Self {
        self.t = t;
        self
    }

    /// Sets the degrees of freedom
    pub fn df(mut self, df: f64) -> Self {
        self.df = df;
        self
    }

    /// Builds the ApproximateModel
    pub fn build(self) -> ApproximateModel {
        ApproximateModel {
            approximation: self.approximation,
            supported_prior: self.supported_prior,
            is_large: self.is_large,
            n: self.n,
            t: self.t,
            df: self.df,
        }
    }
}

impl From<ApproximateModel> for bool {
    fn from(model: ApproximateModel) -> bool {
        model.approximation
    }
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
        let auc = self.integral()?;
        let constant = auc.value;

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

    pub fn is_approximation(&self) -> ApproximateModel {
        const SUPPORTED_PRIORS: [PriorFamily; 1] = [PriorFamily::Cauchy];
        const SUPPORTED_LIKELIHOODS: [LikelihoodFamily; 3] = [
            LikelihoodFamily::NoncentralD,
            LikelihoodFamily::NoncentralD2,
            LikelihoodFamily::NoncentralT,
        ];

        let prior_family = self.prior.family();
        let likelihood_family = self.likelihood.family();

        let supported = SUPPORTED_LIKELIHOODS.contains(&likelihood_family)
            && SUPPORTED_PRIORS.contains(&prior_family);

        if !supported {
            return ApproximateModel::builder().supported_prior(false);
        }

        let (t_likelihood, n, df) = match self.likelihood {
            Likelihood::NoncentralD(likelihood) => {
                let (t, n) = likelihood.into_t();
                (t, n, t.df)
            }
            Likelihood::NoncentralD2(likelihood) => {
                let (t, n) = likelihood.into_t();
                (t, n, t.df)
            }
            Likelihood::NoncentralT(likelihood) => (likelihood, None, likelihood.df),
            _ => unreachable!(),
        };

        let t_likelihood: Likelihood = t_likelihood.into();
        let observation = t_likelihood.get_observation();
        let t = observation.unwrap_or(0.0);
        let abs_t = t.abs();

        let prior_limits = self.prior.range_or_default();
        let in_range = observation >= Some(prior_limits.0) && observation <= Some(prior_limits.1);

        let more_than_15 = abs_t > 15.0;
        let more_than_5 = abs_t > 5.0;

        let needs_approximation = more_than_15 || (more_than_5 && !in_range);

        ApproximateModel::builder()
            .approximation(needs_approximation)
            .supported_prior(supported)
            .is_large(more_than_15 || more_than_5)
            .n(n)
            .t(t)
            .df(df)
            .build()
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
    /// assert_eq!(auc.value as f32, 0.09664394965841581 as f32);
    /// # }
    /// ```
    fn integral(&self) -> Result<Auc, IntegralError> {
        let prior = self.prior;
        let likelihood = self.likelihood;

        let (lower, upper) = self.range_or_default();
        prior.validate()?;
        likelihood.validate()?;

        if self.is_approximation().needs_approximation() {
            let result = estimate_marginal(&self.is_approximation(), &self.prior);
            let marginal_result = result.ok();
            let marginal = marginal_result.map(|r| r.marginal).unwrap_or(0.0);
            return Ok(Auc::new(marginal, likelihood, prior, marginal_result));
        }

        match prior {
            Prior::Point(point) => {
                let value = likelihood.function(point.point)?;
                let auc = Auc::new(value, likelihood, prior, None);
                Ok(auc)
            }
            _ => {
                let model = *self;
                let f = move |x| model.function(x).unwrap();
                let h = integrate!(f = f, lower = lower, upper = upper);

                match h {
                    Ok(v) => {
                        let auc = Auc::new(v.value, likelihood, prior, None);
                        Ok(auc)
                    }
                    Err(e) => Err(IntegralError::Integration(e)),
                }
            }
        }
    }
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        let (_, _) = (lb, ub);
        self.integral().map(|auc| auc.value)
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
/// // NoncentralD(-2.24, 34): |t| ≈ 13 > 5, observation outside [0, Inf) -> approximation path
/// let likelihood: Likelihood = NoncentralDLikelihood::new(-2.24, 34.0).into();
/// let prior = CauchyPrior::new(0.0, 0.707, (Some(0.0), Some(f64::INFINITY)));
/// let approx = (likelihood * Prior::from(prior)).is_approximation();
/// assert!(approx.needs_approximation());
/// let prior_enum: Prior = prior.into();
/// let result = estimate_marginal(&approx, &prior_enum).unwrap();
/// assert!(result.bf < 1.0); // weak evidence against the positive-effect hypothesis
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
            .map(|auc| auc.value)
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
                model.integral().ok().map(|auc| auc.value)
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
    fn integral(&self) -> Result<Auc, IntegralError> {
        let posterior = *self;
        let (lb, ub) = posterior.model.prior.range_or_default();
        let f = move |x| posterior.function(x).unwrap();
        let h = integrate!(f = f, lower = lb, upper = ub);
        match h {
            Ok(v) => Ok(Auc::new(
                v.value,
                self.model.likelihood,
                self.model.prior,
                None,
            )),
            Err(e) => Err(IntegralError::Integration(e)),
        }
    }
}

/// Computes the marginal likelihood and Bayes factor for a truncated Cauchy prior
/// using a two-part correction strategy.
///
/// This mirrors the R function `estimate_marginal(n, t, df, prior)`. The approach
/// decomposes the log-BF into:
///
/// 1. **Interval correction** (`log_bf_interval`): compares how much probability
///    mass the posterior (approximated as a scaled t) and the prior (Cauchy at
///    `rscale = prior.scale * √n`) each assign to `[lower, upper]`, working
///    entirely in log space via `pexp` for numerical stability.
///
/// 2. **Uncorrected BF** (`log_bf_uncorrected`): the BF for the *unbounded* model
///    `cauchy(location, rscale, (-∞, ∞))` vs the point null, computed by
///    numerical integration.
///
/// The final BF combines both terms: `log_bf = log_bf_interval + log_bf_uncorrected`.
///
/// # Arguments
///
/// * `approx` – an [`ApproximateModel`] supplying `t`, `df`, and `n`
///              (constructed via [`Model::is_approximation`])
/// * `prior`  – a `CauchyPrior` with explicit finite bounds `(Some(lower), Some(upper))`
///
/// # Errors
///
/// Returns [`IntegralError::MissingN`] if `approx.n` is `None`.
/// Returns [`IntegralError::UnboundedPrior`] if `prior.range` is not `(Some(_), Some(_))`.
/// Returns [`IntegralError::Distribution`] if any CDF call fails.
/// Returns [`IntegralError::Integration`] if numerical integration of the unbounded
/// model fails.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NoncentralDLikelihood::new(0.5, 20.0).into();
/// let prior = CauchyPrior::new(0.0, 0.707, (Some(0.0), Some(f64::INFINITY)));
/// let approx = (likelihood * Prior::from(prior)).is_approximation();
/// let prior_enum: Prior = prior.into();
/// let result = estimate_marginal(&approx, &prior_enum).unwrap();
/// assert!(result.bf > 1.0);
/// ```
pub fn estimate_marginal(
    approx: &ApproximateModel,
    prior: &Prior,
) -> Result<MarginalResult, IntegralError> {
    let prior = match prior {
        Prior::Cauchy(cauchy) => cauchy,
        _ => return Err(IntegralError::UnboundedPrior),
    };

    let n = approx.n.ok_or(IntegralError::MissingN)?;
    let t = approx.t;
    let df = approx.df;
    // Require explicit finite-ish bounds (at least one bound must be Some).
    // Both must be Some for the interval correction to be well-defined.
    let (lower, upper) = match prior.range {
        (Some(lo), Some(hi)) => (lo, hi),
        (Some(lo), None) => (lo, f64::INFINITY),
        (None, Some(hi)) => (f64::NEG_INFINITY, hi),
        _ => return Err(IntegralError::UnboundedPrior),
    };

    // --- 1. Posterior interval (approximate posterior is a scaled t) ---
    // var_delta = 1/n,  mean_delta = t / sqrt(n)
    let sqrt_var_delta = (1.0_f64 / n).sqrt(); // = 1/sqrt(n)
    let mean_delta = t / n.sqrt();

    // Guard infinite bounds: CDF(+∞) = 1 → log = 0.0; CDF(-∞) = 0 → log = -∞.
    // We bypass the CDF calls for infinite bounds to avoid any floating-point
    // imprecision in the underlying distribution implementations.
    let log_post_lower = if lower == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        pt!(
            q = (lower - mean_delta) / sqrt_var_delta,
            df = df,
            lower_tail = true,
            log_p = true
        )
        .map_err(|e| IntegralError::Distribution(e.to_string()))?
    };

    let log_post_upper = if upper == f64::INFINITY {
        0.0 // log(1) = 0
    } else {
        pt!(
            q = (upper - mean_delta) / sqrt_var_delta,
            df = df,
            lower_tail = true,
            log_p = true
        )
        .map_err(|e| IntegralError::Distribution(e.to_string()))?
    };

    // log(CDF_upper - CDF_lower) computed stably:
    // log_p[[2]] + pexp(diff(log_p), 1, lower.tail=TRUE, log.p=TRUE)
    // = log_post_upper + log(1 - exp(log_post_lower - log_post_upper))
    let post_interval = log_post_upper
        + pexp!(
            q = log_post_upper - log_post_lower,
            rate = 1.0,
            lower_tail = true,
            log_p = true
        )
        .map_err(|e| IntegralError::Distribution(e.to_string()))?;

    // --- 2. Prior interval (Cauchy at scaled rscale) ---
    let rscale = prior.scale * n.sqrt();

    let log_prior_lower = if lower == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        pcauchy!(
            q = lower,
            location = prior.location,
            scale = rscale,
            lower_tail = true,
            log_p = true
        )
        .map_err(|e| IntegralError::Distribution(e.to_string()))?
    };

    let log_prior_upper = if upper == f64::INFINITY {
        0.0 // log(1) = 0
    } else {
        pcauchy!(
            q = upper,
            location = prior.location,
            scale = rscale,
            lower_tail = true,
            log_p = true
        )
        .map_err(|e| IntegralError::Distribution(e.to_string()))?
    };

    let prior_interval = log_prior_upper
        + pexp!(
            q = log_prior_upper - log_prior_lower,
            rate = 1.0,
            lower_tail = true,
            log_p = true
        )
        .map_err(|e| IntegralError::Distribution(e.to_string()))?;

    // --- 3. Interval log-BF ---
    let log_bf_interval = post_interval - prior_interval;

    // --- 4. Uncorrected BF via numerical integration (unbounded model) ---
    // The integrand dnt(t; df, ncp=x) * dcauchy(x; 0, rscale) has its mass concentrated
    // near x = t (the ncp that maximises the likelihood). When |t| is large, this peak
    // lies far from x = 0, which is where gkquad's infinite-interval transformation
    // (x -> x/(1+|x|)) clusters its initial quadrature points. Without guidance, the
    // adaptive integrator can declare convergence before it finds the peak.
    //
    // The fix is to pass `t` as a split point, forcing gkquad to place a subdivision
    // boundary exactly at the peak location so the algorithm concentrates subdivisions
    // in the right region. This matches the accuracy achieved by R's QUADPACK
    // (integrate(..., subdivisions=1000, abs.tol=1e-14)).
    let unbounded_prior: Prior = CauchyPrior::new(prior.location, rscale, (None, None)).into();
    let new_likelihood: Likelihood = NoncentralTLikelihood::new(t, df).into();
    let unbounded_model = new_likelihood * unbounded_prior;

    let f = move |x| unbounded_model.function(x).unwrap();
    let auc_h1 = integrate!(f = f, points = vec![t])
        .map_err(IntegralError::Integration)?
        .value;
    let auc_h0 = new_likelihood
        .function(0.0)
        .map_err(|e| IntegralError::Distribution(e.to_string()))?;

    let log_bf_uncorrected = (auc_h1 / auc_h0).ln();

    // --- 5. Combine ---
    let log_bf = log_bf_interval + log_bf_uncorrected;
    let bf = log_bf.exp();

    Ok(MarginalResult {
        marginal: bf * auc_h0,
        bf,
    })
}
