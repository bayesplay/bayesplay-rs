//! Prior distributions for Bayesian inference.
//!
//! This module provides various prior distributions that represent beliefs about
//! parameter values before observing data. Priors are combined with likelihoods
//! to compute posterior distributions.
//!
//! # Available Priors
//!
//! | Type | Use Case | Parameters |
//! |------|----------|------------|
//! | [`NormalPrior`] | Symmetric, unbounded parameters | mean, sd |
//! | [`CauchyPrior`] | Heavy-tailed priors | location, scale |
//! | [`StudentTPrior`] | Adjustable tail weight | mean, sd, df |
//! | [`UniformPrior`] | Equal probability in interval | min, max |
//! | [`BetaPrior`] | Probabilities (0-1) | alpha, beta |
//! | [`BetaPrimePrior`] | Positive ratios | alpha |
//! | [`PointPrior`] | Point null hypothesis | point |
//!
//! # Truncation
//!
//! Most continuous priors support truncation to restrict the parameter space.
//! This is useful when prior knowledge constrains parameters to certain regions.
//!
//! ```rust
//! use bayesplay::prelude::*;
//!
//! // Normal prior truncated to positive values
//! let positive_normal = NormalPrior::new(0.0, 1.0, (Some(0.0), None));
//!
//! // Cauchy prior truncated between -2 and 2
//! let bounded_cauchy = CauchyPrior::new(0.0, 0.707, (Some(-2.0), Some(2.0)));
//!
//! // Student's t prior with lower bound only
//! let lower_bounded_t = StudentTPrior::new(0.0, 1.0, 3.0, (Some(-1.0), None));
//! ```
//!
//! # Examples
//!
//! ```rust
//! use bayesplay::prelude::*;
//!
//! // Create different types of priors
//! let normal = NormalPrior::new(0.0, 1.0, (None, None));
//! let cauchy = CauchyPrior::new(0.0, 0.707, (None, None));
//! let point = PointPrior::new(0.0);
//!
//! // Convert to the Prior enum for use with models
//! let prior: Prior = normal.into();
//!
//! // Combine with a likelihood
//! let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
//! let model: Model = likelihood * prior;
//! ```

use thiserror::Error;

pub mod beta;
pub mod beta_prime;
pub mod cauchy;
pub mod normal;
pub mod point;
pub mod student_t;
pub mod uniform;

use crate::common::Auc;
use crate::common::Family;
use crate::common::Function;
use crate::common::Integrate;
use crate::common::LogFunction;
use crate::common::Range;
use crate::common::Validate;
use crate::common::impl_from_variants;
use crate::compute::model::{IntegralError, ln_integrate_kernel, requested_bounds};

pub use beta::BetaPrior;
pub use beta_prime::BetaPrimePrior;
pub use cauchy::CauchyPrior;
pub use normal::NormalPrior;
pub use point::PointPrior;
pub use student_t::StudentTPrior;
pub use uniform::UniformPrior;

use serde::{Deserialize, Serialize};

/// Errors that can occur when working with prior distributions.
///
/// This enum covers all validation errors for prior parameters across
/// all prior types. When multiple validation errors occur, they are
/// collected into a [`PriorError::MultipleErrors`] variant.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Invalid standard deviation
/// let invalid = NormalPrior::new(0.0, -1.0, (None, None));
/// match invalid.validate() {
///     Err(PriorError::InvalidStandardDeviation(sd)) => assert_eq!(sd, -1.0),
///     _ => panic!("Expected InvalidStandardDeviation error"),
/// }
///
/// // Invalid range (equal bounds)
/// let invalid = NormalPrior::new(0.0, 1.0, (Some(0.0), Some(0.0)));
/// assert!(matches!(invalid.validate(), Err(PriorError::InvalidRange)));
/// ```
#[derive(Error, Debug, Serialize, Deserialize, Clone)]
pub enum PriorError {
    #[error("Invalid standard deviation ({0}). Must be positive.")]
    InvalidStandardDeviation(f64),
    #[error("Invalid scale ({0}). Must be positive.")]
    InvalidScale(f64),
    #[error("Invalid rate ({0}). Must be positive.")]
    InvalidRate(f64),
    #[error("Invalid location ({0}). Must be finite.")]
    InvalidLocation(f64),
    #[error("Invalid df ({0}). Must be greater than 1.")]
    InvalidDegreesOfFreedom(f64),
    #[error("Invalid range. The lower bound must be below the upper bound.")]
    InvalidRange,
    #[error("Invalid range bounds. Must be between 0 and 1")]
    InvalidRangeBounds,
    #[error("Invalid range bounds. Positive-scale priors require bounds at or above 0.")]
    InvalidPositiveRangeBounds,
    #[error("Invalid shape parameter {0}. Must be positive.")]
    InvalidShapeParameter(f64),
    #[error("Invalid value {0}. Must be between {1} and {2}.")]
    InvalidValue(f64, f64, f64),
    #[error("Error normalizing prior.")]
    NormalizingError,
    #[error("Error with distribution: {0}")]
    DistributionError(&'static str),
    #[error("Multiple validation errors: {}", .0.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "))]
    MultipleErrors(Vec<PriorError>),
}

/// Result type alias for prior operations
pub type PriorResult<T> = Result<T, PriorError>;

impl PriorError {
    /// Consolidate multiple errors into a single error result.
    /// Returns Ok(()) if the vector is empty, a single error if only one,
    /// or MultipleErrors if more than one.
    pub fn from_errors(errors: Vec<PriorError>) -> Result<(), PriorError> {
        match errors.len() {
            0 => Ok(()),
            1 => Err(errors.into_iter().next().unwrap()),
            _ => Err(PriorError::MultipleErrors(errors)),
        }
    }
}

/// Checks that a prior's effective support is a non-empty interval: no NaN
/// bounds, and lower strictly below upper.
pub(crate) fn validate_range(range: (f64, f64)) -> Result<(), PriorError> {
    let (lower, upper) = range;
    if lower.is_nan() || upper.is_nan() || lower >= upper {
        return Err(PriorError::InvalidRange);
    }
    Ok(())
}

/// Checks that a location or scale-like parameter is finite and (if `positive`) > 0.
pub(crate) fn is_valid_parameter(value: f64, positive: bool) -> bool {
    value.is_finite() && (!positive || value > 0.0)
}

pub(crate) fn validate_positive_range(range: (Option<f64>, Option<f64>)) -> Result<(), PriorError> {
    let lower = range.0.unwrap_or(0.0);
    let upper = range.1.unwrap_or(f64::INFINITY);
    if lower.is_nan() || upper.is_nan() || lower < 0.0 || upper < 0.0 {
        return Err(PriorError::InvalidPositiveRangeBounds);
    }
    if lower >= upper {
        return Err(PriorError::InvalidRange);
    }
    Ok(())
}

/// Enum representing the family (type) of a prior distribution.
///
/// This is used for runtime type identification of priors.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
/// assert_eq!(prior.family(), PriorFamily::Normal);
///
/// let prior: Prior = PointPrior::new(0.0).into();
/// assert_eq!(prior.family(), PriorFamily::Point);
/// assert!(prior.is_point());
/// ```
#[derive(PartialEq, Debug, Serialize, Clone, Copy)]
pub enum PriorFamily {
    /// Cauchy distribution prior.
    Cauchy,
    /// Normal (Gaussian) distribution prior.
    Normal,
    /// Point mass prior (Dirac delta).
    Point,
    /// Uniform distribution prior.
    Uniform,
    /// Student's t distribution prior.
    StudentT,
    /// Beta distribution prior.
    Beta,
    /// Symmetric beta-prime distribution prior.
    BetaPrime,
}

/// Trait for computing normalization constants of prior distributions.
///
/// For truncated distributions, the normalization constant is the probability
/// mass within the truncation bounds. This ensures the truncated distribution
/// integrates to 1.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Unbounded normal: normalization constant is 1
/// let unbounded = NormalPrior::new(0.0, 1.0, (None, None));
/// assert_eq!(unbounded.normalize().unwrap(), 1.0);
///
/// // Truncated normal: normalization constant is the probability mass in bounds
/// let truncated = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0)));
/// let norm = truncated.normalize().unwrap();
/// assert!((norm - 0.6827).abs() < 0.01); // ~68% of mass is within ±1 SD
/// ```
pub trait Normalize {
    /// Computes the normalization constant for the distribution.
    ///
    /// Returns 1.0 for untruncated distributions. For truncated distributions,
    /// returns the probability mass within the truncation bounds.
    fn normalize(&self) -> Result<f64, PriorError>;

    /// The natural logarithm of [`Normalize::normalize`]. Priors that can be
    /// truncated far into a tail override this to avoid underflow.
    fn ln_normalize(&self) -> Result<f64, PriorError> {
        Ok(self.normalize()?.ln())
    }
}

/// A prior distribution for Bayesian inference.
///
/// This enum wraps all available prior types, allowing them to be used
/// polymorphically with likelihood functions to form models.
///
/// # Creating Priors
///
/// Priors are typically created using the specific type's constructor,
/// then converted to the enum using `.into()`:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Create specific prior types
/// let normal = NormalPrior::new(0.0, 1.0, (None, None));
/// let cauchy = CauchyPrior::new(0.0, 0.707, (None, None));
/// let point = PointPrior::new(0.0);
///
/// // Convert to Prior enum
/// let prior: Prior = normal.into();
/// ```
///
/// # Combining with Likelihoods
///
/// Priors are combined with likelihoods using multiplication to create models:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
///
/// // Create a model (order doesn't matter)
/// let model: Model = prior * likelihood;
/// // or: let model: Model = likelihood * prior;
/// ```
///
/// # Evaluating Priors
///
/// Use the [`Function`] trait to evaluate the prior density at parameter values:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
///
/// // Evaluate the prior at different parameter values
/// let density = prior.function(0.0).unwrap();
/// assert!((density - 0.3989).abs() < 0.001);
/// ```
///
/// # Type Identification
///
/// Use the [`Family`] trait to identify the type of a prior at runtime:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let prior: Prior = PointPrior::new(0.0).into();
/// assert!(prior.is_point());
/// assert_eq!(prior.family(), PriorFamily::Point);
/// ```
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum Prior {
    /// Normal (Gaussian) prior distribution.
    Normal(NormalPrior),
    /// Point mass prior (Dirac delta) for null hypotheses.
    Point(PointPrior),
    /// Cauchy prior distribution (heavy tails).
    Cauchy(CauchyPrior),
    /// Uniform prior distribution over an interval.
    Uniform(UniformPrior),
    /// Student's t prior distribution.
    StudentT(StudentTPrior),
    /// Beta prior distribution for probabilities.
    Beta(BetaPrior),
    /// Symmetric beta-prime prior for positive ratios.
    BetaPrime(BetaPrimePrior),
}

/// A prior's log density given precomputed constants.
///
/// Each prior implements this; [`LogFunction`] follows from it, validating
/// and normalizing on every call. Integration instead uses
/// [`Prior::ln_kernel`], which does both once.
pub(crate) trait PriorDensity {
    /// The log density at `x`, assuming the parameters are valid and
    /// `ln_normalizer` is [`Normalize::ln_normalize`].
    fn ln_density(&self, x: f64, ln_normalizer: f64) -> Result<f64, PriorError>;
}

impl<T: PriorDensity + Validate<PriorError> + Normalize> LogFunction<PriorError> for T {
    fn ln_function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;
        self.ln_density(x, self.ln_normalize()?)
    }
}

/// Evaluates `$body` with `$inner` bound to the distribution inside a
/// [`Prior`], whichever variant it is. Adding a variant makes every use of this
/// a compile error until it is listed here.
macro_rules! on_prior {
    ($prior:expr, $inner:ident => $body:expr) => {
        match $prior {
            Prior::Normal($inner) => $body,
            Prior::Point($inner) => $body,
            Prior::Cauchy($inner) => $body,
            Prior::Uniform($inner) => $body,
            Prior::StudentT($inner) => $body,
            Prior::Beta($inner) => $body,
            Prior::BetaPrime($inner) => $body,
        }
    };
}

impl_from_variants!(Prior {
    Normal(NormalPrior),
    Point(PointPrior),
    Cauchy(CauchyPrior),
    Uniform(UniformPrior),
    StudentT(StudentTPrior),
    Beta(BetaPrior),
    BetaPrime(BetaPrimePrior),
});

impl LogFunction<PriorError> for Prior {
    fn ln_function(&self, x: f64) -> Result<f64, PriorError> {
        on_prior!(self, p => p.ln_function(x))
    }
}

impl Function<&[f64], Vec<Option<f64>>, PriorError> for Prior {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, PriorError> {
        Ok(x.iter().map(|&x| self.function(x).ok()).collect())
    }
}

impl Validate<PriorError> for Prior {
    fn validate(&self) -> Result<(), PriorError> {
        on_prior!(self, p => p.validate())
    }
}

impl Range for Prior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        on_prior!(self, p => p.range())
    }
    fn default_range(&self) -> (f64, f64) {
        on_prior!(self, p => p.default_range())
    }
}

impl Normalize for Prior {
    fn normalize(&self) -> Result<f64, PriorError> {
        on_prior!(self, p => p.normalize())
    }
    fn ln_normalize(&self) -> Result<f64, PriorError> {
        on_prior!(self, p => p.ln_normalize())
    }
}

impl Family<PriorFamily> for Prior {
    /// The prior's family.
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
    /// assert_eq!(prior.family(), PriorFamily::Normal);
    /// ```
    fn family(&self) -> PriorFamily {
        match self {
            Prior::Normal(_) => PriorFamily::Normal,
            Prior::Point(_) => PriorFamily::Point,
            Prior::Cauchy(_) => PriorFamily::Cauchy,
            Prior::Uniform(_) => PriorFamily::Uniform,
            Prior::StudentT(_) => PriorFamily::StudentT,
            Prior::Beta(_) => PriorFamily::Beta,
            Prior::BetaPrime(_) => PriorFamily::BetaPrime,
        }
    }
}

impl Prior {
    /// Returns true if this is a point prior (a point mass), which is
    /// integrated by evaluation rather than quadrature.
    ///
    /// ```rust
    /// use bayesplay::prelude::*;
    ///
    /// assert!(Prior::from(PointPrior::new(0.0)).is_point());
    /// assert!(!Prior::from(NormalPrior::new(0.0, 1.0, (None, None))).is_point());
    /// ```
    pub fn is_point(&self) -> bool {
        matches!(self, Prior::Point(_))
    }

    /// The location and scale of the prior's peak, as `(centre, width)`, for
    /// families whose peak can be narrow relative to their support. Used to
    /// place integration windows.
    pub(crate) fn peak_hint(&self) -> Option<(f64, f64)> {
        match self {
            Prior::Normal(p) => Some((p.mean, p.sd)),
            Prior::Cauchy(p) => Some((p.location, p.scale)),
            Prior::StudentT(p) => Some((p.mean, p.sd)),
            Prior::Point(_) | Prior::Uniform(_) | Prior::Beta(_) | Prior::BetaPrime(_) => None,
        }
    }

    /// The log density as a closure, with validation and the normalizing
    /// constant done once up front rather than on every evaluation.
    pub(crate) fn ln_kernel(
        &self,
    ) -> Result<impl Fn(f64) -> Result<f64, PriorError> + '_, PriorError> {
        self.validate()?;
        let ln_normalizer = self.ln_normalize()?;
        Ok(move |x| on_prior!(self, p => p.ln_density(x, ln_normalizer)))
    }

    /// The natural logarithm of [`Integrate::integrate`] (the prior mass in
    /// `[lb, ub]`).
    pub fn ln_integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        self.validate()?;
        let (lb, ub) = requested_bounds(lb, ub, self.range_or_default())?;
        if let Prior::Point(point) = self {
            return Ok(if (lb..=ub).contains(&point.point) {
                0.0
            } else {
                f64::NEG_INFINITY
            });
        }
        ln_integrate_kernel(self.ln_kernel()?, lb, ub, self.peak_hint(), None)
    }
}

impl Integrate<IntegralError, PriorError> for Prior {
    fn integral(&self) -> Result<Auc, IntegralError> {
        Ok(Auc::from_ln(self.ln_integrate(None, None)?))
    }

    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        Ok(self.ln_integrate(lb, ub)?.exp())
    }
}

#[cfg(test)]
mod positive_scale_prior_tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn full_and_truncated_positive_priors_integrate_to_one() {
        let priors: [Prior; 2] = [
            BetaPrimePrior::new(0.5, (None, None)).into(),
            BetaPrimePrior::new(0.5, (Some(1.0), None)).into(),
        ];

        for prior in priors {
            assert_relative_eq!(prior.integral().unwrap().value, 1.0, epsilon = 1e-6);
        }
    }
}
