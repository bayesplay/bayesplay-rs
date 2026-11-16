//! Bayesian model computation.
//!
//! This module provides types for combining priors and likelihoods into models,
//! and computing posteriors, marginal likelihoods, and predictive distributions.

use std::ops::Mul;

use rmath::{IntegrationError, Quadrature, ToleranceMode};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::common::Auc;
use crate::common::Function;
use crate::common::Integrate;
use crate::common::LogFunction;
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
        let ln_constant = self.ln_integrate(None, None)?;
        if !ln_constant.is_finite() {
            return Err(IntegralError::DegeneratePosterior(ln_constant.exp()));
        }
        Ok(Posterior {
            model: *self,
            ln_constant,
        })
    }

    /// The natural logarithm of [`Integrate::integrate`]: the log of the
    /// integral of prior × likelihood over `[lb, ub]` (intersected with the
    /// prior's support), computed without underflow.
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// // The marginal is ~1e-174, far below what `integrate` can represent
    /// // relative to its tails, but its logarithm is exact.
    /// let likelihood: Likelihood = NormalLikelihood::new(40.0, 1.0).into();
    /// let model = likelihood * Prior::from(NormalPrior::new(0.0, 1.0, (None, None)));
    /// let expected = -400.0 - 0.5 * (4.0 * std::f64::consts::PI).ln();
    /// assert!((model.ln_integrate(None, None).unwrap() - expected).abs() < 1e-8);
    /// ```
    pub fn ln_integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        self.prior.validate()?;
        self.likelihood.validate()?;
        let (lb, ub) = requested_bounds(lb, ub, self.range_or_default())?;
        if let Prior::Point(point) = self.prior {
            return if (lb..=ub).contains(&point.point) {
                Ok(self.likelihood.ln_function(point.point)?)
            } else {
                Ok(f64::NEG_INFINITY)
            };
        }
        let likelihood_peak = self.likelihood.peak_hint();
        let prior_peak = self.prior.peak_hint();
        // A prior at least as wide as the likelihood is smooth across the
        // likelihood's window, so it only needs a window of its own when it is
        // the narrower of the two. Its centre still guides the mode search.
        let windows = [
            Some(likelihood_peak),
            prior_peak.filter(|&(_, width)| width < likelihood_peak.1),
        ];
        let landmarks = prior_peak.map(|(centre, _)| centre);
        let ln_prior = self.prior.ln_kernel()?;
        let likelihood = self.likelihood;
        ln_integrate_kernel(
            |x| -> Result<f64, IntegralError> { Ok(ln_prior(x)? + likelihood.ln_function(x)?) },
            lb,
            ub,
            windows.into_iter().flatten(),
            landmarks,
        )
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

impl LogFunction<IntegralError> for Model {
    fn ln_function(&self, x: f64) -> Result<f64, IntegralError> {
        Ok(self.prior.ln_function(x)? + self.likelihood.ln_function(x)?)
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
        Ok(Auc::from_ln(self.ln_integrate(None, None)?))
    }

    /// Integrates prior × likelihood over `[lb, ub]` intersected with the
    /// prior's support. Missing bounds default to the support. See
    /// [`Model::ln_integrate`] for the logarithm, which cannot underflow.
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        Ok(self.ln_integrate(lb, ub)?.exp())
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

/// Maximum iterations of golden-section search used to locate the kernel's
/// mode; each shrinks the bracket by a factor of 0.618.
const MODE_SEARCH_ITERATIONS: usize = 100;

/// Cut points closer together than this many (narrowest) peak widths are
/// merged.
const MERGE_CUTS_WIDTHS: f64 = 0.5;

/// The mode search stops once its bracket is this many (narrowest) peak
/// widths wide.
const MODE_TOLERANCE_WIDTHS: f64 = 0.05;

/// Returns the natural log of the integral of `exp(ln_f)` over
/// `[lower, upper]`, either of which may be infinite.
///
/// **Scaling.** The kernel is integrated as `exp(ln_f(x) - M)`, where `M` is
/// (approximately) its maximum, and `M` is added back to the log of the result.
/// The integrand therefore peaks near 1 however small the kernel is, so
/// marginals far below `f64::MIN_POSITIVE` keep full relative precision.
///
/// **Placement.** Adaptive quadrature over an infinite interval maps it onto a
/// finite one, which squeezes a narrow peak far from the origin into a sliver
/// it can miss entirely (e.g. a Normal(8, 0.01) likelihood under a
/// Normal(0, 10) prior came out at half its true value). Each `(centre, width)`
/// in `windows` therefore becomes a finite window of ±[`PEAK_WINDOW_WIDTHS`]
/// widths. The kernel's mode is located by golden-section search between the
/// window centres and `landmarks` (other places mass may concentrate, such as a
/// wide prior's centre) and gets a window too, since the product of two peaks
/// can concentrate between them. The windows are integrated first, then the
/// tails to a tolerance relative to the windows' mass.
///
/// The first evaluation error raised by `ln_f` is returned instead of being
/// silently turned into zero mass.
pub(crate) fn ln_integrate_kernel<E>(
    ln_f: impl Fn(f64) -> Result<f64, E>,
    lower: f64,
    upper: f64,
    windows: impl IntoIterator<Item = (f64, f64)>,
    landmarks: impl IntoIterator<Item = f64>,
) -> Result<f64, IntegralError>
where
    IntegralError: From<E>,
{
    if lower >= upper {
        return Ok(f64::NEG_INFINITY);
    }
    let mut windows: Vec<(f64, f64)> = windows
        .into_iter()
        .filter(|(centre, width)| centre.is_finite() && width.is_finite() && *width > 0.0)
        .map(|(centre, width)| (centre.clamp(lower, upper), width))
        .collect();

    // --- Find the scale M (the kernel's maximum) and the mode's window ---
    let eval = |x: f64| -> Result<f64, IntegralError> {
        let value = ln_f(x)?;
        Ok(if value.is_nan() {
            f64::NEG_INFINITY
        } else {
            value
        })
    };
    let mut points: Vec<f64> = windows
        .iter()
        .map(|&(centre, _)| centre)
        .chain(landmarks.into_iter().filter(|x| x.is_finite()))
        .map(|x| x.clamp(lower, upper))
        .filter(|x| x.is_finite())
        .collect();
    if points.is_empty() {
        let fallback = match (lower.is_finite(), upper.is_finite()) {
            (true, true) => 0.5 * (lower + upper),
            _ => 0.0_f64.clamp(lower, upper),
        };
        points.push(fallback);
    }
    let mut best = (points[0], eval(points[0])?);
    for &x in &points[1..] {
        let value = eval(x)?;
        if value > best.1 {
            best = (x, value);
        }
    }
    let (a, b) = points
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    let min_width = windows
        .iter()
        .map(|&(_, width)| width)
        .fold(f64::INFINITY, f64::min);
    if a < b {
        // Locating the mode to a small fraction of the narrowest peak is
        // plenty: it shifts the log scale by < 1e-3 and the window negligibly.
        let tolerance = if min_width.is_finite() {
            MODE_TOLERANCE_WIDTHS * min_width
        } else {
            0.0
        };
        let mode = golden_section_max(&eval, a, b, tolerance)?;
        if mode.1 > best.1 {
            best = mode;
        }
    }
    let (mode, peak) = best;
    if let Some(width) = curvature_width(&eval, mode, min_width, lower, upper)?
        && !windows
            .iter()
            .any(|&(centre, w)| (centre - mode).abs() < w.min(width))
    {
        // Only a mode away from every existing window needs one of its own.
        windows.push((mode, width));
    }
    let scale = if peak.is_finite() { peak } else { 0.0 };

    // --- Split into windows (core) and tails ---
    let mut cuts: Vec<f64> = windows
        .iter()
        .flat_map(|&(centre, width)| {
            let half = PEAK_WINDOW_WIDTHS * width;
            [centre - half, centre, centre + half]
        })
        .map(|x| x.clamp(lower, upper))
        .collect();
    cuts.sort_by(f64::total_cmp);
    // Overlapping windows leave near-duplicate cuts; each sliver between them
    // would cost a full quadrature rule, so merge cuts closer than this.
    let min_gap = MERGE_CUTS_WIDTHS
        * windows
            .iter()
            .map(|&(_, w)| w)
            .fold(f64::INFINITY, f64::min);
    cuts.dedup_by(|later, kept| *later - *kept < min_gap);
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
        let result = Quadrature {
            lower: a,
            upper: b,
            abs_tol,
            tolerance,
            ..Default::default()
        }
        .integrate(|x: f64| match ln_f(x) {
            Ok(value) => (value - scale).exp(),
            Err(e) => {
                error.get_or_insert(e);
                f64::NAN
            }
        });
        match error.take() {
            Some(e) => Err(IntegralError::from(e)),
            None => Ok(result?.value),
        }
    };

    let defaults = Quadrature::default();
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
    Ok(scale + (core + tails).ln())
}

/// Maximizes `f` over `[a, b]` by golden-section search until the bracket is
/// narrower than `tolerance`, returning `(argmax, max)`. Exact for unimodal `f`
/// (e.g. a log-concave kernel); for other kernels it finds a local maximum,
/// which is all the scaling needs.
fn golden_section_max(
    f: &impl Fn(f64) -> Result<f64, IntegralError>,
    mut a: f64,
    mut b: f64,
    tolerance: f64,
) -> Result<(f64, f64), IntegralError> {
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let mut c = b - INV_PHI * (b - a);
    let mut d = a + INV_PHI * (b - a);
    let (mut fc, mut fd) = (f(c)?, f(d)?);
    for _ in 0..MODE_SEARCH_ITERATIONS {
        if (b - a).abs() <= tolerance.max(f64::EPSILON * (a.abs() + b.abs())) {
            break;
        }
        if fc >= fd {
            b = d;
            d = c;
            fd = fc;
            c = b - INV_PHI * (b - a);
            fc = f(c)?;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = f(d)?;
        }
    }
    Ok(if fc >= fd { (c, fc) } else { (d, fd) })
}

/// Estimates the width of the kernel's peak at `mode` from the curvature of
/// its log, `1 / sqrt(-(ln f)'')`. Returns `None` where that is not defined
/// (e.g. the mode is on a boundary of `[lower, upper]` or the log is not
/// locally concave).
fn curvature_width(
    f: &impl Fn(f64) -> Result<f64, IntegralError>,
    mode: f64,
    scale_hint: f64,
    lower: f64,
    upper: f64,
) -> Result<Option<f64>, IntegralError> {
    let h = if scale_hint.is_finite() {
        1e-3 * scale_hint
    } else {
        1e-3 * mode.abs().max(1.0)
    };
    if !(mode - h > lower && mode + h < upper) {
        return Ok(None);
    }
    let (left, centre, right) = (f(mode - h)?, f(mode)?, f(mode + h)?);
    let curvature = (left - 2.0 * centre + right) / (h * h);
    Ok((curvature.is_finite() && curvature < 0.0).then(|| (-curvature).sqrt().recip()))
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
    /// Log of the model's integral (the marginal likelihood).
    ln_constant: f64,
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

impl LogFunction<IntegralError> for Posterior {
    fn ln_function(&self, x: f64) -> Result<f64, IntegralError> {
        Ok(self.model.ln_function(x)? - self.ln_constant)
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
        Ok(((self.0.prior * likelihood).ln_integrate(None, None)? + jacobian.ln()).exp())
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
        Ok((self.model.ln_integrate(lb, ub)? - self.ln_constant).exp())
    }
    fn integral(&self) -> Result<Auc, IntegralError> {
        Ok(Auc::from_ln(
            self.model.ln_integrate(None, None)? - self.ln_constant,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Integrates a plain (non-log) function with the model's kernel.
    fn integrate_fn(
        f: impl Fn(f64) -> Result<f64, IntegralError>,
        lower: f64,
        upper: f64,
    ) -> Result<f64, IntegralError> {
        Ok(ln_integrate_kernel(
            |x| Ok::<_, IntegralError>(f(x)?.ln()),
            lower,
            upper,
            [],
            None,
        )?
        .exp())
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
        let error = ln_integrate_kernel(
            |_| Err(IntegralError::DegeneratePosterior(0.0)),
            0.0,
            1.0,
            [],
            None,
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
        // A variance ratio is positive, so a prior on negative values has no mass.
        let model = Likelihood::from(VarianceRatioLikelihood::new(1.5, 9.0))
            * Prior::from(UniformPrior::new(-2.0, -1.0));
        assert!(matches!(
            model.posterior(),
            Err(IntegralError::DegeneratePosterior(_))
        ));
        assert_eq!(model.ln_integrate(None, None).unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn point_prior_marginals_keep_their_log_when_they_underflow() {
        let model =
            Likelihood::from(NormalLikelihood::new(1e100, 1.0)) * Prior::from(PointPrior::new(0.0));
        let auc = model.integral().unwrap();
        assert_eq!(auc.value, 0.0);
        let expected = -0.5 * 1e200 - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert_relative_eq!(auc.log_value, expected, max_relative = 1e-15);
        // The posterior is still a valid point mass.
        let posterior = model.posterior().unwrap();
        assert_eq!(posterior.integrate(Some(-1.0), Some(1.0)).unwrap(), 1.0);
    }

    #[test]
    fn marginals_match_normal_normal_closed_form_in_log_space() {
        // N(x; m, se) * N(x; 0, sd) integrates to N(m; 0, sqrt(se^2 + sd^2)).
        // Compared on the log scale, so marginals far below f64's range count.
        for mean in [
            -35.0, -8.0, 0.0, 0.5, 2.0, 3.0, 8.0, 15.0, 25.0, 35.0, 60.0, 1e3,
        ] {
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
                let want = rmath::dnorm(mean, 0.0, f64::hypot(se, sd), true).unwrap();
                let got = (likelihood * prior).integral().unwrap().log_value;
                assert!(
                    (got - want).abs() < 1e-7 * want.abs().max(1.0),
                    "mean={mean} se={se} sd={sd}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn bayes_factors_with_underflowing_marginals() {
        // N(m, 1) likelihood: H1 = N(0, 1) prior, H0 = point at 0.
        // log BF10 = ln N(m; 0, sqrt 2) - ln N(m; 0, 1) = m^2 / 4 - ln(2) / 2.
        for mean in [5.0, 40.0, 60.0, 100.0] {
            let likelihood: Likelihood = NormalLikelihood::new(mean, 1.0).into();
            let h1 = (likelihood * Prior::from(NormalPrior::new(0.0, 1.0, (None, None))))
                .integral()
                .unwrap();
            let h0 = (likelihood * Prior::from(PointPrior::new(0.0)))
                .integral()
                .unwrap();
            let want = mean * mean / 4.0 - std::f64::consts::LN_2 / 2.0;
            assert_relative_eq!(h1.log_value - h0.log_value, want, max_relative = 1e-10);
            if want < f64::MAX.ln() {
                // The f64 ratio is exact as long as the Bayes factor itself fits.
                assert_relative_eq!((h1 / h0).ln(), want, max_relative = 1e-10);
            }
        }
    }

    #[test]
    fn posteriors_of_extreme_models() {
        // N(60, 1) likelihood x N(0, 1) prior: posterior N(30, 1/sqrt 2), while
        // the marginal is ~1e-391.
        let model = Likelihood::from(NormalLikelihood::new(60.0, 1.0))
            * Prior::from(NormalPrior::new(0.0, 1.0, (None, None)));
        assert_eq!(model.integral().unwrap().value, 0.0);
        let posterior = model.posterior().unwrap();
        assert_relative_eq!(posterior.integral().unwrap().value, 1.0, epsilon = 1e-9);
        assert_relative_eq!(
            posterior.integrate(None, Some(30.0)).unwrap(),
            0.5,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            posterior.function(30.0).unwrap(),
            1.0 / std::f64::consts::PI.sqrt(),
            max_relative = 1e-9
        );
    }

    #[test]
    fn priors_truncated_far_into_a_tail() {
        // N(0, 1) truncated to [40, inf): mass Q(40) = exp(-804.608...), which
        // underflows. R: pnorm(40, lower.tail = FALSE, log.p = TRUE).
        let ln_mass = -804.608_442_013_753_8;
        let prior = NormalPrior::new(0.0, 1.0, (Some(40.0), None));
        assert_relative_eq!(prior.ln_normalize().unwrap(), ln_mass, max_relative = 1e-14);
        let ln_phi_41 = -0.5 * 41.0 * 41.0 - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert_relative_eq!(
            prior.ln_function(41.0).unwrap(),
            ln_phi_41 - ln_mass,
            max_relative = 1e-12
        );
        let prior: Prior = prior.into();
        assert_relative_eq!(prior.integral().unwrap().value, 1.0, epsilon = 1e-8);

        // A likelihood centred in that tail sees an ordinary marginal.
        let model = Likelihood::from(NormalLikelihood::new(40.5, 0.1)) * prior;
        let value = model.integral().unwrap().value;
        assert!(value.is_finite() && value > 0.0);

        // Cauchy and Student-t far tails go through the same log path.
        for prior in [
            Prior::from(CauchyPrior::new(0.0, 1.0, (Some(1e10), Some(2e10)))),
            StudentTPrior::new(0.0, 1.0, 3.0, (Some(1e6), None)).into(),
        ] {
            assert_relative_eq!(prior.integral().unwrap().value, 1.0, epsilon = 1e-7);
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
