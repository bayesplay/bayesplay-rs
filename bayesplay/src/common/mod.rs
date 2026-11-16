use std::ops::{Div, Sub};

use approx::{AbsDiffEq, RelativeEq};

/// Trait for validating distribution parameters.
///
/// Types implementing this trait can verify that their parameters are valid
/// before performing calculations. This helps catch errors early and provides
/// meaningful error messages.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Valid parameters
/// let likelihood = NormalLikelihood::new(0.0, 1.0).unwrap();
/// assert!(likelihood.validate().is_ok());
///
/// // Invalid parameters (negative standard error)
/// let invalid = NormalLikelihood { mean: 0.0, se: -1.0 };
/// assert!(invalid.validate().is_err());
/// ```
pub trait Validate<E> {
    /// Validates the parameters of the implementing type.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if all parameters are valid
    /// - `Err(E)` with details about what validation failed
    fn validate(&self) -> Result<(), E>;
}

/// Trait for evaluating probability density or likelihood functions.
///
/// This is the core trait for computing the value of a distribution at a point.
/// It supports generic input and output types to handle both scalar and vector
/// evaluations.
///
/// # Type Parameters
///
/// - `T` - The input type (typically `f64` or `&[f64]`)
/// - `S` - The output type (typically `f64` or `Vec<Option<f64>>`)
/// - `E` - The error type
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
///
/// // Evaluate at a single point
/// let density = prior.function(0.0).unwrap();
///
/// // The density at the mean of a standard normal is 1/√(2π) ≈ 0.3989
/// assert!((density - 0.3989).abs() < 0.001);
/// ```
pub trait Function<T, S, E> {
    /// Evaluates the function at the given input.
    ///
    /// # Arguments
    ///
    /// - `x` - The point(s) at which to evaluate the function
    ///
    /// # Returns
    ///
    /// The function value(s) at the given point(s), or an error if evaluation fails
    fn function(&self, x: T) -> Result<S, E>;
}

/// Trait for evaluating the natural logarithm of a density or likelihood.
///
/// This is the primary way densities are implemented: working on the log scale
/// keeps values that would underflow an `f64` (e.g. a likelihood 40 standard
/// errors from its peak) usable. Every implementor gets [`Function`] for free,
/// as `exp(ln_function(x))`.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood = NormalLikelihood::new(40.0, 1.0).unwrap();
/// assert_eq!(likelihood.function(0.0).unwrap(), 0.0); // underflows
/// let log_density = likelihood.ln_function(0.0).unwrap();
/// assert!((log_density - (-800.0 - 0.5 * (2.0 * std::f64::consts::PI).ln())).abs() < 1e-9);
/// ```
pub trait LogFunction<E> {
    /// Evaluates the log density at `x`; `-inf` where the density is zero.
    fn ln_function(&self, x: f64) -> Result<f64, E>;
}

impl<T: LogFunction<E>, E> Function<f64, f64, E> for T {
    fn function(&self, x: f64) -> Result<f64, E> {
        Ok(self.ln_function(x)?.exp())
    }
}

/// Trait for managing the range (support) of a distribution.
///
/// Distributions can have bounded or unbounded support. This trait provides
/// methods to query and work with these bounds, which is essential for
/// numerical integration and truncated distributions.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// // Unbounded normal prior
/// let unbounded = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
/// assert!(unbounded.has_default_range());
///
/// // Truncated normal prior (positive values only)
/// let truncated = NormalPrior::new(0.0, 1.0, (Some(0.0), None)).unwrap();
/// assert!(!truncated.has_default_range());
/// assert_eq!(truncated.range(), (Some(0.0), None));
/// ```
pub trait Range {
    /// Returns the explicitly set range bounds.
    ///
    /// # Returns
    ///
    /// A tuple of `(lower_bound, upper_bound)` where `None` indicates no explicit bound.
    fn range(&self) -> (Option<f64>, Option<f64>);

    /// Returns the default (natural) range of the distribution.
    ///
    /// For unbounded distributions like Normal, this is `(-∞, +∞)`.
    /// For bounded distributions like Beta, this is `(0, 1)`.
    fn default_range(&self) -> (f64, f64);

    /// Returns the effective range, using defaults where bounds are not set.
    ///
    /// This is a convenience method that combines explicit bounds with defaults.
    fn range_or_default(&self) -> (f64, f64) {
        let (ll, ul) = self.range();
        let (default_ll, default_ul) = self.default_range();

        (ll.unwrap_or(default_ll), ul.unwrap_or(default_ul))
    }

    /// Checks if the distribution uses its default (untruncated) range.
    ///
    /// # Returns
    ///
    /// `true` if no explicit bounds have been set, `false` if the distribution is truncated.
    fn has_default_range(&self) -> bool {
        let (ll, ul) = self.range_or_default();
        let (default_ll, default_ul) = self.default_range();
        ll == default_ll && ul == default_ul
    }
}

/// The area under a curve: the result of integrating a model, prior or
/// posterior over its support.
///
/// For a [`Model`](crate::prelude::Model) this is the marginal likelihood, so
/// dividing two `Auc`s gives a Bayes factor:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).unwrap().into();
/// let h1: Auc = (likelihood * Prior::from(NormalPrior::new(0.0, 1.0, (None, None)).unwrap())).integral().unwrap();
/// let h0: Auc = (likelihood * Prior::from(PointPrior::new(0.0).unwrap())).integral().unwrap();
/// let bf10: f64 = h1 / h0;
/// assert!(bf10 > 1.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Auc {
    /// The value of the integral. This can underflow to 0 for extreme models;
    /// `log_value` is computed directly and remains accurate.
    pub value: f64,
    /// The natural logarithm of the integral.
    pub log_value: f64,
}

impl Auc {
    pub(crate) fn from_ln(log_value: f64) -> Self {
        Self {
            value: log_value.exp(),
            log_value,
        }
    }
}

impl PartialEq<f64> for Auc {
    fn eq(&self, other: &f64) -> bool {
        self.value == *other
    }
}

impl Div for Auc {
    type Output = f64;

    /// The ratio of two areas (e.g. a Bayes factor), computed from the log
    /// values so it stays accurate when either area underflows.
    fn div(self, rhs: Self) -> Self::Output {
        if self.log_value.is_finite() && rhs.log_value.is_finite() {
            (self.log_value - rhs.log_value).exp()
        } else {
            self.value / rhs.value
        }
    }
}

impl Div<Auc> for f64 {
    type Output = f64;

    fn div(self, rhs: Auc) -> Self::Output {
        self / rhs.value
    }
}

impl Sub<f64> for Auc {
    type Output = f64;

    fn sub(self, rhs: f64) -> Self::Output {
        self.value - rhs
    }
}

impl From<Auc> for f64 {
    fn from(auc: Auc) -> Self {
        auc.value
    }
}

impl AbsDiffEq for Auc {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.value.abs_diff_eq(&other.value, epsilon)
    }
}

impl RelativeEq for Auc {
    fn default_max_relative() -> Self::Epsilon {
        f64::EPSILON
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.value.relative_eq(&other.value, epsilon, max_relative)
    }
}

impl AbsDiffEq<f64> for Auc {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &f64, epsilon: Self::Epsilon) -> bool {
        self.value.abs_diff_eq(other, epsilon)
    }
}

impl RelativeEq<f64> for Auc {
    fn default_max_relative() -> Self::Epsilon {
        f64::EPSILON
    }

    fn relative_eq(
        &self,
        other: &f64,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.value.relative_eq(other, epsilon, max_relative)
    }
}

/// Trait for numerical integration of distributions.
///
/// This trait enables computing the area under a probability density function,
/// which is essential for normalization and computing marginal likelihoods.
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).unwrap().into();
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap().into();
/// let model: Model = likelihood * prior;
///
/// // Compute the marginal likelihood (area under the curve)
/// let marginal = model.integral().unwrap();
/// ```
pub trait Integrate<E, O>: Function<f64, f64, O> + Range {
    /// Computes the integral over the full range of the distribution.
    ///
    /// # Returns
    ///
    /// The area under the curve, or an error if integration fails.
    fn integral(&self) -> Result<Auc, E>;

    /// Computes the integral over a specified range.
    ///
    /// # Arguments
    ///
    /// - `lb` - Lower bound (uses default if `None`)
    /// - `ub` - Upper bound (uses default if `None`)
    ///
    /// # Returns
    ///
    /// The area under the curve within the specified bounds.
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, E>;
}

/// The log of the probability mass in `[lower, upper]`: the normalizing
/// constant of a truncated distribution. Computed from log CDF values so that
/// a truncation interval far in a tail (e.g. a normal truncated to
/// `[40, inf)`, mass ~1e-350) does not underflow.
///
/// `ln_cdf(x, lower_tail)` must return `ln P(X <= x)` (lower tail) or
/// `ln P(X > x)`. Returns `-inf` for an empty interval.
pub fn truncated_ln_normalization<E>(
    lower: f64,
    upper: f64,
    mut ln_cdf: impl FnMut(f64, bool) -> Result<f64, E>,
) -> Result<f64, E> {
    Ok(match (lower.is_infinite(), upper.is_infinite()) {
        (true, true) => 0.0,
        (false, true) => ln_cdf(lower, false)?,
        (true, false) => ln_cdf(upper, true)?,
        (false, false) => {
            // Subtract within whichever tail keeps both terms small, so the
            // difference does not cancel.
            let ln_lower = ln_cdf(lower, true)?;
            if ln_lower <= -std::f64::consts::LN_2 {
                let ln_upper = ln_cdf(upper, true)?;
                ln_upper + ln_one_minus_exp(ln_lower - ln_upper)
            } else {
                let (ln_sf_lower, ln_sf_upper) = (ln_cdf(lower, false)?, ln_cdf(upper, false)?);
                ln_sf_lower + ln_one_minus_exp(ln_sf_upper - ln_sf_lower)
            }
        }
    })
}

/// `ln(1 - exp(x))` for `x <= 0`, accurate for `x` near 0 and very negative.
fn ln_one_minus_exp(x: f64) -> f64 {
    if x > -std::f64::consts::LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

/// Implements `From<Inner>` for each `Variant(Inner)` of a wrapper enum.
macro_rules! impl_from_variants {
    ($enum:ident { $($variant:ident($inner:ty)),* $(,)? }) => {
        $(
            impl From<$inner> for $enum {
                fn from(inner: $inner) -> Self {
                    $enum::$variant(inner)
                }
            }
        )*
    };
}
pub(crate) use impl_from_variants;

/// Identifies which family a [`Prior`](crate::prelude::Prior) or
/// [`Likelihood`](crate::prelude::Likelihood) belongs to.
pub trait Family<T> {
    fn family(&self) -> T;
}
