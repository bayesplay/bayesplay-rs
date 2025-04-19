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

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum Prior {
    Normal(NormalPrior),
    Point(PointPrior),
    Cauchy(CauchyPrior),
    Uniform(UniformPrior),
    StudentT(StudentTPrior),
    Beta(BetaPrior),
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
    #[error("Multiple errors: {0}, {1}]")]
    MultiError2(Box<PriorError>, Box<PriorError>),
    #[error("Multiple errors: {0}, {1}, {2}]")]
    MultiError3(Box<PriorError>, Box<PriorError>, Box<PriorError>),
    #[error("Multiple errors: {0}, {1}, {2}, {3}]")]
    MultiError4(
        Box<PriorError>,
        Box<PriorError>,
        Box<PriorError>,
        Box<PriorError>,
    ),
}

pub trait Normalize {
    fn normalize(&self) -> Result<f64, PriorError>;
}

impl Normalize for Prior {
    fn normalize(&self) -> Result<f64, PriorError> {
        match self {
            Prior::Normal(prior) => prior.normalize(),
            Prior::Point(prior) => prior.normalize(),
            Prior::Cauchy(prior) => prior.normalize(),
            Prior::Uniform(prior) => prior.normalize(),
            Prior::StudentT(prior) => prior.normalize(),
            Prior::Beta(prior) => prior.normalize(),
        }
    }
}

impl Validate<PriorError> for Prior {
    fn validate(&self) -> Result<(), PriorError> {
        match self {
            Prior::Normal(prior) => prior.validate(),
            Prior::Point(prior) => prior.validate(),
            Prior::Cauchy(prior) => prior.validate(),
            Prior::Uniform(prior) => prior.validate(),
            Prior::StudentT(prior) => prior.validate(),
            Prior::Beta(prior) => prior.validate(),
        }
    }
}

impl Function<f64, f64, PriorError> for Prior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        match self {
            Prior::Normal(prior) => prior.function(x),
            Prior::Point(prior) => prior.function(x),
            Prior::Cauchy(prior) => prior.function(x),
            Prior::Uniform(prior) => prior.function(x),
            Prior::StudentT(prior) => prior.function(x),
            Prior::Beta(prior) => prior.function(x),
        }
    }
}

impl Function<&[f64], Vec<Option<f64>>, PriorError> for Prior {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, PriorError> {
        Ok(x.iter()
            .map(|x| self.function(*x).ok())
            .collect::<Vec<Option<_>>>())
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

impl Range for Prior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        match self {
            Prior::Normal(prior) => prior.range(),
            Prior::Cauchy(prior) => prior.range(),
            Prior::Uniform(prior) => prior.range(),
            Prior::StudentT(prior) => prior.range(),
            Prior::Beta(prior) => prior.range(),
            Prior::Point(prior) => prior.range(),
        }
    }
    fn default_range(&self) -> (f64, f64) {
        match self {
            Prior::Normal(prior) => prior.default_range(),
            Prior::Point(prior) => prior.default_range(),
            Prior::Cauchy(prior) => prior.default_range(),
            Prior::Uniform(prior) => prior.default_range(),
            Prior::StudentT(prior) => prior.default_range(),
            Prior::Beta(prior) => prior.default_range(),
        }
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
            Err(e) => Err(IntegralError::Intergaration(e)),
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
            Err(e) => Err(IntegralError::Intergaration(e)),
        }
    }
}
