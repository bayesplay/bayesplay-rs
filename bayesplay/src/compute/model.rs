//! Bayesian model computation.
//!
//! This module provides types for combining priors and likelihoods into models,
//! and computing posteriors, marginal likelihoods, and predictive distributions.

use std::ops::Mul;

use rmath::integration::{
    Integrate as Quadrature, IntegrationError, ToleranceMode, integratewrapper,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::common::Auc;
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
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct Model {
    /// The prior distribution over parameter values.
    pub prior: Prior,
    /// The likelihood function for the observed data.
    pub likelihood: Likelihood,
}

/// Errors that can occur when evaluating or integrating a model.
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
    Integration(#[from] IntegrationError),
    /// The requested integration bounds are NaN or reversed.
    #[error("Invalid integration bounds ({0}, {1})")]
    InvalidBounds(f64, f64),
    /// The model has no (finite, positive) mass, so it cannot be normalized.
    #[error("Posterior requires a finite, positive normalizing constant, got {0}")]
    DegeneratePosterior(f64),
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
    /// # Errors
    ///
    /// Returns an error if the model cannot be integrated (e.g., invalid
    /// parameters or numerical integration failure), or if it has no mass.
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
        let constant = self.integrate(None, None)?;
        if !constant.is_finite() || constant <= 0.0 {
            return Err(IntegralError::DegeneratePosterior(constant));
        }
        Ok(Posterior {
            model: *self,
            constant,
        })
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
        self.prior.range()
    }
    fn default_range(&self) -> (f64, f64) {
        self.prior.default_range()
    }
}

impl Function<f64, f64, IntegralError> for Model {
    fn function(&self, x: f64) -> Result<f64, IntegralError> {
        Ok(self.prior.function(x)? * self.likelihood.function(x)?)
    }
}

impl Integrate<IntegralError, IntegralError> for Model {
    /// Computes the marginal likelihood: the integral of prior × likelihood
    /// over the prior's support.
    ///
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// let likelihood: Likelihood = NormalLikelihood::new(0.2, 4.0).into();
    /// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
    /// let auc = (likelihood * prior).integral().unwrap();
    ///
    /// assert_eq!(auc.value as f32, 0.09664394965841581 as f32);
    /// ```
    fn integral(&self) -> Result<Auc, IntegralError> {
        Ok(Auc::new(self.integrate(None, None)?))
    }

    /// Integrates prior × likelihood over `[lb, ub]` intersected with the
    /// prior's support. Missing bounds default to the support.
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        self.prior.validate()?;
        self.likelihood.validate()?;
        let (lb, ub) = requested_bounds(lb, ub, self.range_or_default())?;
        if let Prior::Point(point) = self.prior {
            return if (lb..=ub).contains(&point.point) {
                Ok(self.likelihood.function(point.point)?)
            } else {
                Ok(0.0)
            };
        }
        let likelihood_peak = self.likelihood.peak_hint();
        // A prior at least as wide as the likelihood is smooth across the
        // likelihood's window, so it only needs a window of its own when it is
        // the narrower of the two.
        let prior_peak = self
            .prior
            .peak_hint()
            .filter(|&(_, width)| width < likelihood_peak.1);
        let hints = [Some(likelihood_peak), prior_peak];
        integrate_kernel(|x| self.function(x), lb, ub, hints.into_iter().flatten())
    }
}

/// Resolves optional user bounds against a support: NaN or reversed user
/// bounds are errors, and the result is clamped to the support (so it may be
/// empty, i.e. `lower > upper`, when the request misses the support).
pub(crate) fn requested_bounds(
    lb: Option<f64>,
    ub: Option<f64>,
    (lower, upper): (f64, f64),
) -> Result<(f64, f64), IntegralError> {
    let (l, u) = (lb.unwrap_or(f64::NEG_INFINITY), ub.unwrap_or(f64::INFINITY));
    if l.is_nan() || u.is_nan() || l > u {
        return Err(IntegralError::InvalidBounds(l, u));
    }
    Ok((l.max(lower), u.min(upper)))
}

/// Number of peak widths either side of a peak that are integrated as a finite
/// window. Nothing is truncated (the tails are integrated separately); this
/// only decides where the quadrature concentrates its effort.
const PEAK_WINDOW_WIDTHS: f64 = 8.0;

/// Integrates a density kernel over `[lower, upper]`, either of which may be
/// infinite.
///
/// Adaptive quadrature over an infinite interval maps it onto a finite one,
/// which squeezes a narrow peak far from the origin into a sliver it can miss
/// entirely (e.g. a Normal(8, 0.01) likelihood under a Normal(0, 10) prior
/// came out at half its true value). Each `(centre, width)` hint therefore
/// becomes a finite window of ±[`PEAK_WINDOW_WIDTHS`] widths, split at the
/// centre; the window and the remaining tails are integrated separately.
///
/// The first evaluation error raised by `f` is returned instead of being
/// silently turned into zero mass.
pub(crate) fn integrate_kernel<E>(
    f: impl Fn(f64) -> Result<f64, E>,
    lower: f64,
    upper: f64,
    hints: impl IntoIterator<Item = (f64, f64)>,
) -> Result<f64, IntegralError>
where
    IntegralError: From<E>,
{
    if lower >= upper {
        return Ok(0.0);
    }
    let mut cuts: Vec<f64> = hints
        .into_iter()
        .filter(|(centre, width)| centre.is_finite() && width.is_finite() && *width > 0.0)
        .flat_map(|(centre, width)| {
            let half = PEAK_WINDOW_WIDTHS * width;
            // Clamp so a peak outside the support still gets a window at the
            // nearest edge, where the truncated kernel is largest.
            let centre = centre.clamp(lower, upper);
            [centre - half, centre, centre + half]
        })
        .map(|x| x.clamp(lower, upper))
        .collect();
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    // The core spans the peak windows; without hints it is the whole range.
    let (core_lower, core_upper) = match (cuts.first(), cuts.last()) {
        (Some(&first), Some(&last)) if first < last => (first, last),
        _ => (lower, upper),
    };
    cuts.retain(|x| *x > core_lower && *x < core_upper);
    let core_pieces = std::iter::once(core_lower)
        .chain(cuts)
        .chain(std::iter::once(core_upper))
        .collect::<Vec<_>>();

    let mut error = None;
    let mut integrate_piece = |a: f64, b: f64, abs_tol: f64, tolerance: ToleranceMode| {
        let result = integratewrapper(Quadrature {
            f: Some(|x: f64| match f(x) {
                Ok(value) => value,
                Err(e) => {
                    error.get_or_insert(e);
                    f64::NAN
                }
            }),
            lower: Some(a),
            upper: Some(b),
            abs_tol,
            tolerance,
            ..Default::default()
        });
        match error.take() {
            Some(e) => Err(IntegralError::from(e)),
            None => Ok(result?.value),
        }
    };

    let defaults = Quadrature::<fn(f64) -> f64>::default();
    let mut core = 0.0;
    for piece in core_pieces.windows(2) {
        core += integrate_piece(piece[0], piece[1], defaults.abs_tol, ToleranceMode::Both)?;
    }

    // The tails hold little mass, so they only need to be accurate relative to
    // the core. Demanding relative accuracy on their own tiny values is slow
    // (densities there can be expensive) and can fail to converge at all.
    let (tail_tol, tail_mode) = if core > 0.0 {
        (defaults.rel_tol * core, ToleranceMode::Either)
    } else {
        (defaults.abs_tol, ToleranceMode::Both)
    };
    let mut tails = 0.0;
    for (a, b) in [(lower, core_lower), (core_upper, upper)] {
        if a < b {
            tails += integrate_piece(a, b, tail_tol, tail_mode)?;
        }
    }
    Ok(core + tails)
}

/// A posterior distribution computed from a Bayesian model.
///
/// The posterior represents the updated beliefs about parameter values after
/// observing data. It is proportional to the product of the prior and likelihood,
/// normalized to integrate to 1.
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

impl Function<f64, f64, IntegralError> for Posterior {
    fn function(&self, x: f64) -> Result<f64, IntegralError> {
        Ok(self.model.function(x)? / self.constant)
    }
}

impl Function<&[f64], Vec<Option<f64>>, IntegralError> for Posterior {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, IntegralError> {
        Ok(x.iter().map(|&x| self.function(x).ok()).collect())
    }
}

impl Function<f64, f64, IntegralError> for Predictive {
    fn function(&self, x: f64) -> Result<f64, IntegralError> {
        let mut likelihood = self.0.likelihood;
        likelihood.update_observation(x);
        // Effect-size likelihoods are t densities in t = d * sqrt(n); the
        // Jacobian makes the predictive a density over d.
        let jacobian = match likelihood {
            Likelihood::NoncentralD(l) => l.effective_n().sqrt(),
            Likelihood::NoncentralD2(l) => l.effective_n().sqrt(),
            _ => 1.0,
        };
        Ok((self.0.prior * likelihood).integrate(None, None)? * jacobian)
    }
}

impl Function<&[f64], Vec<Option<f64>>, IntegralError> for Predictive {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, IntegralError> {
        Ok(x.iter().map(|&x| self.function(x).ok()).collect())
    }
}

impl Mul<Prior> for Likelihood {
    type Output = Model;
    /// Creates a model from a likelihood and a prior.
    fn mul(self, prior: Prior) -> Model {
        Model {
            prior,
            likelihood: self,
        }
    }
}

impl Mul<Likelihood> for Prior {
    type Output = Model;
    /// Creates a model from a prior and a likelihood.
    fn mul(self, likelihood: Likelihood) -> Model {
        likelihood * self
    }
}

impl Range for Posterior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        self.model.range()
    }
    fn default_range(&self) -> (f64, f64) {
        self.model.default_range()
    }
}

impl Integrate<IntegralError, IntegralError> for Posterior {
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        Ok(self.model.integrate(lb, ub)? / self.constant)
    }
    fn integral(&self) -> Result<Auc, IntegralError> {
        Ok(Auc::new(self.integrate(None, None)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn integrate_fn(
        f: impl Fn(f64) -> Result<f64, IntegralError>,
        lower: f64,
        upper: f64,
    ) -> Result<f64, IntegralError> {
        integrate_kernel(f, lower, upper, [])
    }

    #[test]
    fn predictive_effect_size_jacobians_preserve_kernels() {
        let prior: Prior = PointPrior::new(0.0).into();
        for (likelihood, jacobian) in [
            (Likelihood::from(NoncentralDLikelihood::new(0.2, 25.0)), 5.0),
            (
                Likelihood::from(NoncentralD2Likelihood::new(0.2, 20.0, 30.0)),
                12.0_f64.sqrt(),
            ),
        ] {
            let model = likelihood * prior;
            let kernel = likelihood.function(0.0).unwrap();
            assert_relative_eq!(model.integral().unwrap().value, kernel);
            assert_relative_eq!(model.predictive().function(0.2).unwrap(), kernel * jacobian);
            assert_relative_eq!(
                model.predictive().function(&[0.2][..]).unwrap()[0].unwrap(),
                kernel * jacobian
            );
            let predictive = model.predictive();
            let area = integrate_fn(|x| predictive.function(x), -10.0, 10.0).unwrap();
            assert_relative_eq!(area, 1.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn large_noncentral_t_statistics_integrate_exactly() {
        // Formerly rejected with "requires reparameterization".
        let likelihood: Likelihood = NoncentralTLikelihood::new(20.0, 20.0).into();
        let model = likelihood * Prior::from(CauchyPrior::new(0.0, 0.7, (None, None)));
        let value = model.integral().unwrap().value;
        assert!(value.is_finite() && value > 0.0);
        assert_relative_eq!(
            model.predictive().function(20.0).unwrap(),
            value,
            epsilon = 1e-12
        );
    }

    #[test]
    fn density_errors_are_not_zero_mass() {
        let error = integrate_kernel(
            |_| Err(IntegralError::DegeneratePosterior(0.0)),
            0.0,
            1.0,
            [],
        )
        .unwrap_err();
        assert!(matches!(error, IntegralError::DegeneratePosterior(_)));
    }

    #[test]
    fn point_posterior_and_bounded_model_integrals() {
        let likelihood: Likelihood = NormalLikelihood::new(0.0, 1.0).into();
        let model = likelihood * Prior::from(PointPrior::new(0.5));
        let posterior = model.posterior().unwrap();
        assert_eq!(posterior.integral().unwrap().value, 1.0);
        assert_eq!(posterior.integrate(Some(0.5), Some(0.5)).unwrap(), 1.0);
        assert_eq!(posterior.integrate(Some(0.6), None).unwrap(), 0.0);
        assert_eq!(model.integrate(Some(-1.0), Some(0.0)).unwrap(), 0.0);
        let model = likelihood * Prior::from(NormalPrior::new(0.0, 1.0, (None, None)));
        assert_relative_eq!(
            model.integrate(None, Some(0.0)).unwrap(),
            model.integral().unwrap().value / 2.0,
            epsilon = 1e-10
        );
        assert!(matches!(
            model.integrate(Some(1.0), Some(-1.0)),
            Err(IntegralError::InvalidBounds(..))
        ));
        assert!(matches!(
            model.integrate(Some(f64::NAN), None),
            Err(IntegralError::InvalidBounds(..))
        ));
    }

    #[test]
    fn posterior_of_massless_model_is_an_error() {
        let model =
            Likelihood::from(NormalLikelihood::new(1e100, 1.0)) * Prior::from(PointPrior::new(0.0));
        assert!(matches!(
            model.posterior(),
            Err(IntegralError::DegeneratePosterior(_))
        ));
    }

    #[test]
    fn sharp_likelihoods_match_normal_normal_closed_form() {
        // N(x; m, se) * N(x; 0, sd) integrates to N(m; 0, sqrt(se^2 + sd^2)).
        for mean in [-35.0, -8.0, 0.0, 0.5, 2.0, 3.0, 8.0, 15.0, 25.0, 35.0] {
            for (se, sd) in [
                (1.0, 1.0),
                (0.2, 10.0),
                (0.05, 1.0),
                (1.0, 0.05),
                (0.01, 10.0),
                (0.001, 1.0),
                (1e-4, 100.0),
            ] {
                let likelihood: Likelihood = NormalLikelihood::new(mean, se).into();
                let prior: Prior = NormalPrior::new(0.0, sd, (None, None)).into();
                let want = rmath::dnorm!(x = mean, mean = 0.0, sd = f64::hypot(se, sd)).unwrap();
                if want < 1e-280 {
                    continue; // below the representable range of the marginal
                }
                let got = (likelihood * prior).integral().unwrap().value;
                assert_relative_eq!(got, want, max_relative = 1e-7);
            }
        }
    }

    #[test]
    fn sharp_likelihood_posterior_is_centred() {
        let model = Likelihood::from(NormalLikelihood::new(8.0, 0.01))
            * Prior::from(NormalPrior::new(0.0, 10.0, (None, None)));
        let posterior = model.posterior().unwrap();
        assert_relative_eq!(
            posterior.integrate(None, Some(8.0)).unwrap(),
            0.5,
            epsilon = 1e-3
        );
        assert_relative_eq!(posterior.integral().unwrap().value, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn sharp_likelihoods_under_heavy_tailed_priors() {
        // With se = 1e-3 the likelihood is ~ a delta at 8, so the marginal is
        // the prior density at 8 (to O(se^2)).
        let likelihood: Likelihood = NormalLikelihood::new(8.0, 1e-3).into();
        for prior in [
            Prior::from(CauchyPrior::new(0.0, 1.0, (None, None))),
            StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).into(),
            NormalPrior::new(0.0, 10.0, (Some(0.0), None)).into(),
            UniformPrior::new(0.0, 20.0).into(),
        ] {
            let got = (likelihood * prior).integral().unwrap().value;
            assert_relative_eq!(got, prior.function(8.0).unwrap(), max_relative = 1e-5);
        }
    }

    #[test]
    fn model_round_trips_through_json() {
        let model = Likelihood::from(NormalLikelihood::new(0.5, 0.2))
            * Prior::from(NormalPrior::new(0.0, 1.0, (None, None)));
        let json = serde_json::to_string(&model).unwrap();
        let back: Model = serde_json::from_str(&json).unwrap();
        assert_eq!(back, model);
    }
}
