use enum_dispatch::enum_dispatch;
use rmath::integrate;
use thiserror::Error;

pub mod beta;
pub mod cauchy;
pub mod normal;
pub mod point;
pub mod student_t;
pub mod uniform;

use crate::common::Function;
use crate::common::Integrate;
use crate::common::Range;
use crate::common::Validate;
use crate::compute::model::IntegralError;

pub use beta::BetaPrior;
pub use cauchy::CauchyPrior;
pub use normal::NormalPrior;
pub use point::PointPrior;
pub use student_t::StudentTPrior;
pub use uniform::UniformPrior;

use serde::{Deserialize, Serialize};

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

#[derive(PartialEq, Debug, Serialize, Clone, Copy)]
pub enum PriorFamily {
    Cauchy,
    Normal,
    Point,
    Uniform,
    StudentT,
    Beta,
}

/// Trait for normalizing distributions (used by enum_dispatch)
#[enum_dispatch]
pub trait Normalize {
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

#[enum_dispatch(Normalize, PriorFn, PriorValidate, PriorRange)]
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum Prior {
    Normal(NormalPrior),
    Point(PointPrior),
    Cauchy(CauchyPrior),
    Uniform(UniformPrior),
    StudentT(StudentTPrior),
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

pub trait TypeOf {
    fn type_of(&self) -> PriorFamily;
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
    fn integral(&self) -> Result<f64, IntegralError> {
        let prior = *self;
        if prior.is_point() {
            return Ok(1.0);
        }
        let (lb, ub) = prior.range_or_default();
        let f = move |x| prior.function(x).unwrap();
        let h = integrate!(f = f, lower = lb, upper = ub);
        match h {
            Ok(v) => Ok(v.value),
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
