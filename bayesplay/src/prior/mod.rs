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

use enum_dispatch::enum_dispatch;
use rmath::integrate;
use thiserror::Error;

pub mod beta;
pub mod cauchy;
pub mod normal;
pub mod point;
pub mod student_t;
pub mod uniform;

use crate::common::Auc;
use crate::common::Family;
use crate::common::Function;
use crate::common::Integrate;
use crate::common::Range;
use crate::common::Validate;
use crate::compute::model::IntegralError;
use crate::likelihood::NormalLikelihood;

pub use beta::BetaPrior;
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
    #[error("Invalid df ({0}). Must be greater than 1.")]
    InvalidDegreesOfFreedom(f64),
    #[error("Invalid range. Upper and lower bounds must be different.")]
    InvalidRange,
    #[error("Invalid range bounds. Must be between 0 and 1")]
    InvalidRangeBounds,
    #[error("Invalid shape parameter {0}. But be non-negative.")]
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
/// assert_eq!(prior.type_of(), PriorFamily::Normal);
///
/// let prior: Prior = PointPrior::new(0.0).into();
/// assert_eq!(prior.type_of(), PriorFamily::Point);
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
#[enum_dispatch]
pub trait Normalize {
    /// Computes the normalization constant for the distribution.
    ///
    /// Returns 1.0 for untruncated distributions. For truncated distributions,
    /// returns the probability mass within the truncation bounds.
    fn normalize(&self) -> Result<f64, PriorError>;
}

/// Concrete trait for prior function evaluation (used by enum_dispatch)
#[enum_dispatch]
pub trait PriorFn {
    fn prior_function(&self, x: f64) -> Result<f64, PriorError>;
}

/// Concrete trait for prior validation (used by enum_dispatch)
#[enum_dispatch]
pub trait PriorValidate {
    fn prior_validate(&self) -> Result<(), PriorError>;
}

/// Concrete trait for prior range (used by enum_dispatch)
#[enum_dispatch]
pub trait PriorRange {
    fn prior_range(&self) -> (Option<f64>, Option<f64>);
    fn prior_default_range(&self) -> (f64, f64);
}

// Implement the concrete traits for each prior type by delegating to the generic traits

impl PriorFn for NormalPrior {
    fn prior_function(&self, x: f64) -> Result<f64, PriorError> {
        Function::function(self, x)
    }
}

impl PriorFn for PointPrior {
    fn prior_function(&self, x: f64) -> Result<f64, PriorError> {
        Function::function(self, x)
    }
}

impl PriorFn for CauchyPrior {
    fn prior_function(&self, x: f64) -> Result<f64, PriorError> {
        Function::function(self, x)
    }
}

impl PriorFn for UniformPrior {
    fn prior_function(&self, x: f64) -> Result<f64, PriorError> {
        Function::function(self, x)
    }
}

impl PriorFn for StudentTPrior {
    fn prior_function(&self, x: f64) -> Result<f64, PriorError> {
        Function::function(self, x)
    }
}

impl PriorFn for BetaPrior {
    fn prior_function(&self, x: f64) -> Result<f64, PriorError> {
        Function::function(self, x)
    }
}

impl PriorValidate for NormalPrior {
    fn prior_validate(&self) -> Result<(), PriorError> {
        Validate::validate(self)
    }
}

impl PriorValidate for PointPrior {
    fn prior_validate(&self) -> Result<(), PriorError> {
        Validate::validate(self)
    }
}

impl PriorValidate for CauchyPrior {
    fn prior_validate(&self) -> Result<(), PriorError> {
        Validate::validate(self)
    }
}

impl PriorValidate for UniformPrior {
    fn prior_validate(&self) -> Result<(), PriorError> {
        Validate::validate(self)
    }
}

impl PriorValidate for StudentTPrior {
    fn prior_validate(&self) -> Result<(), PriorError> {
        Validate::validate(self)
    }
}

impl PriorValidate for BetaPrior {
    fn prior_validate(&self) -> Result<(), PriorError> {
        Validate::validate(self)
    }
}

impl PriorRange for NormalPrior {
    fn prior_range(&self) -> (Option<f64>, Option<f64>) {
        Range::range(self)
    }
    fn prior_default_range(&self) -> (f64, f64) {
        Range::default_range(self)
    }
}

impl PriorRange for PointPrior {
    fn prior_range(&self) -> (Option<f64>, Option<f64>) {
        Range::range(self)
    }
    fn prior_default_range(&self) -> (f64, f64) {
        Range::default_range(self)
    }
}

impl PriorRange for CauchyPrior {
    fn prior_range(&self) -> (Option<f64>, Option<f64>) {
        Range::range(self)
    }
    fn prior_default_range(&self) -> (f64, f64) {
        Range::default_range(self)
    }
}

impl PriorRange for UniformPrior {
    fn prior_range(&self) -> (Option<f64>, Option<f64>) {
        Range::range(self)
    }
    fn prior_default_range(&self) -> (f64, f64) {
        Range::default_range(self)
    }
}

impl PriorRange for StudentTPrior {
    fn prior_range(&self) -> (Option<f64>, Option<f64>) {
        Range::range(self)
    }
    fn prior_default_range(&self) -> (f64, f64) {
        Range::default_range(self)
    }
}

impl PriorRange for BetaPrior {
    fn prior_range(&self) -> (Option<f64>, Option<f64>) {
        Range::range(self)
    }
    fn prior_default_range(&self) -> (f64, f64) {
        Range::default_range(self)
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
/// Use the [`TypeOf`] trait to identify the type of a prior at runtime:
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let prior: Prior = PointPrior::new(0.0).into();
/// assert!(prior.is_point());
/// assert_eq!(prior.type_of(), PriorFamily::Point);
/// ```
#[enum_dispatch(Normalize, PriorFn, PriorValidate, PriorRange)]
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
}

// Note: From implementations are generated automatically by enum_dispatch

// Implement the generic Function trait for Prior using enum_dispatch
impl Function<f64, f64, PriorError> for Prior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        self.prior_function(x)
    }
}

impl Function<&[f64], Vec<Option<f64>>, PriorError> for Prior {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, PriorError> {
        Ok(x.iter()
            .map(|x| self.prior_function(*x).ok())
            .collect::<Vec<Option<_>>>())
    }
}

// Implement the generic Validate trait for Prior using enum_dispatch
impl Validate<PriorError> for Prior {
    fn validate(&self) -> Result<(), PriorError> {
        self.prior_validate()
    }
}

// Implement the Range trait for Prior using enum_dispatch
impl Range for Prior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        self.prior_range()
    }
    fn default_range(&self) -> (f64, f64) {
        self.prior_default_range()
    }
}

/// Trait for runtime type identification of prior distributions.
///
/// This trait allows querying the type of a prior at runtime, which is useful
/// for special handling of certain prior types (e.g., point priors).
///
/// # Examples
///
/// ```rust
/// use bayesplay::prelude::*;
///
/// let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
/// assert_eq!(prior.type_of(), PriorFamily::Normal);
/// assert!(!prior.is_point());
///
/// let point: Prior = PointPrior::new(0.0).into();
/// assert_eq!(point.type_of(), PriorFamily::Point);
/// assert!(point.is_point());
/// ```
pub trait TypeOf {
    /// Returns the family (type) of this prior distribution.
    fn type_of(&self) -> PriorFamily;

    /// Returns true if this is a point prior (Dirac delta).
    ///
    /// Point priors require special handling during integration since they
    /// represent a single value rather than a continuous distribution.
    fn is_point(&self) -> bool;
}

impl TypeOf for Prior {
    fn type_of(&self) -> PriorFamily {
        match self {
            Prior::Normal(_) => PriorFamily::Normal,
            Prior::Point(_) => PriorFamily::Point,
            Prior::Cauchy(_) => PriorFamily::Cauchy,
            Prior::Uniform(_) => PriorFamily::Uniform,
            Prior::StudentT(_) => PriorFamily::StudentT,
            Prior::Beta(_) => PriorFamily::Beta,
        }
    }

    fn is_point(&self) -> bool {
        matches!(self, Prior::Point(_))
    }
}

impl Integrate<IntegralError, PriorError> for Prior {
    fn integral(&self) -> Result<Auc, IntegralError> {
        let prior = *self;
        if prior.is_point() {
            return Ok(Auc::new(1.0, NormalLikelihood::new(0.0, 1.0).into(), prior));
        }
        let (lb, ub) = prior.range_or_default();
        let f = move |x| prior.function(x).unwrap();
        let h = integrate!(f = f, lower = lb, upper = ub);
        match h {
            Ok(v) => Ok(Auc::new(
                v.value,
                NormalLikelihood::new(0.0, 1.0).into(),
                prior,
            )),
            Err(e) => Err(IntegralError::Integration(e)),
        }
    }
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        let prior = *self;
        let lb = lb.unwrap_or(prior.range_or_default().0);
        let ub = ub.unwrap_or(prior.range_or_default().1);

        if prior.is_point() {
            let Prior::Point(point_prior) = prior else {
                unreachable!()
            };
            if (lb..=ub).contains(&point_prior.point) {
                return Ok(1.0);
            } else {
                return Ok(0.0);
            }
        }

        let f = move |x| prior.function(x).unwrap();
        let h = integrate!(f = f, lower = lb, upper = ub);
        match h {
            Ok(v) => Ok(v.value),
            Err(e) => Err(IntegralError::Integration(e)),
        }
    }
}

impl Family<PriorFamily> for Prior {
    fn family(&self) -> PriorFamily {
        self.type_of()
    }
}
