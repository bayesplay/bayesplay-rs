use std::ops::{Div, Sub};

use approx::{AbsDiffEq, RelativeEq};

use crate::{prelude::Likelihood, prior::Prior};

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
/// let likelihood = NormalLikelihood::new(0.0, 1.0);
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
/// let prior = NormalPrior::new(0.0, 1.0, (None, None));
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
/// let unbounded = NormalPrior::new(0.0, 1.0, (None, None));
/// assert!(unbounded.has_default_range());
///
/// // Truncated normal prior (positive values only)
/// let truncated = NormalPrior::new(0.0, 1.0, (Some(0.0), None));
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Auc {
    pub value: f64,
    likelihood: Likelihood,
    prior: Prior,
}

impl Auc {
    pub(crate) fn new(value: f64, likelihood: Likelihood, prior: Prior) -> Self {
        Self {
            value,
            likelihood,
            prior,
        }
    }
}

impl PartialEq<f64> for Auc {
    fn eq(&self, other: &f64) -> bool {
        self.value == *other
    }
}

impl PartialOrd for Auc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
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

impl Div for Auc {
    type Output = f64;

    fn div(self, rhs: Self) -> Self::Output {
        self.value / rhs.value
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

impl From<Auc> for f64 {
    fn from(auc: Auc) -> Self {
        auc.value
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
/// let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
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

/// Compute the normalization constant for a truncated distribution.
///
/// This helper function calculates the probability mass within truncation bounds
/// for any distribution with a CDF. It's used by NormalPrior, CauchyPrior, StudentTPrior,
/// and any other truncated distributions.
///
/// # Arguments
/// * `lower` - Lower truncation bound
/// * `upper` - Upper truncation bound
/// * `cdf` - Cumulative distribution function that takes (x, lower_tail) -> Result<f64, E>
///
/// # Returns
/// The normalization constant (probability mass within bounds)
pub fn truncated_normalization<E>(
    lower: f64,
    upper: f64,
    mut cdf: impl FnMut(f64, bool) -> Result<f64, E>,
) -> Result<f64, E> {
    match (lower.is_infinite(), upper.is_infinite()) {
        (true, true) => Ok(1.0),
        (false, true) => cdf(lower, false), // P(X > lower) = 1 - F(lower)
        (true, false) => cdf(upper, true),  // P(X < upper) = F(upper)
        (false, false) => {
            let p_lower = cdf(lower, true)?;
            let p_upper = cdf(upper, true)?;
            Ok(p_upper - p_lower)
        }
    }
}



pub trait Family<T> {
    fn family(&self) -> T;
}
