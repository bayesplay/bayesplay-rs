use gkquad::RuntimeError;
use gkquad::Tolerance;
use thiserror::Error;

use gkquad::single::Integrator;

#[derive(Error, Debug)]
pub enum IntegrationError {
    #[error("Roundoff Error")]
    RoundoffError,
    #[error("Insufficient Iteration")]
    InsufficientIteration,
    #[error("Subrange Too Small")]
    SubrangeTooSmall,
    #[error("Divergent")]
    Divergent,
    #[error("NanValueEncountered")]
    NanValueEncountered,
    #[error("OtherError")]
    OtherError,
}

pub fn integratewrapper<F>(args: Integrate<F>) -> Result<IntegratorResult, IntegrationError>
where
    F: Fn(f64) -> f64 + 'static,
{
    let a = args.lower; //.unwrap_or(NEG_INFINITY);
    let b = args.upper; //.unwrap_or(INFINITY);
    let f = args
        .f
        .expect("argument \"f\" is is missing, with no default");
    let limit = args.subdivisions.expect("error with \"subdivisions\"") as usize;
    let epsabs = args.abs_tol.expect("error with \"abs_tol\"");
    let epsrel = args.rel_tol.expect("error with \"rel_tol\"");

    let tol = Tolerance::AbsAndRel(epsabs, epsrel);

    let result = Integrator::new(f)
        .max_iters(limit)
        .tolerance(tol)
        .run(a.unwrap_or(f64::NEG_INFINITY)..b.unwrap_or(f64::INFINITY))
        .estimate_delta();

    match result {
        Ok((value, delta)) => Ok(IntegratorResult { value, delta }),
        Err(e) => match e {
            RuntimeError::InsufficientIteration => Err(IntegrationError::InsufficientIteration),
            RuntimeError::RoundoffError => Err(IntegrationError::RoundoffError),
            RuntimeError::SubrangeTooSmall => Err(IntegrationError::SubrangeTooSmall),
            RuntimeError::Divergent => Err(IntegrationError::Divergent),
            RuntimeError::NanValueEncountered => Err(IntegrationError::NanValueEncountered),
            _ => Err(IntegrationError::OtherError),
        },
    }
}

pub struct IntegratorResult {
    pub value: f64,
    pub delta: f64,
}

pub struct Integrate<F>
where
    F: Fn(f64) -> f64 + 'static,
{
    pub f: Option<F>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub subdivisions: Option<i64>,
    pub abs_tol: Option<f64>,
    pub rel_tol: Option<f64>,
}

impl<F> Default for Integrate<F>
where
    F: Fn(f64) -> f64 + 'static,
{
    fn default() -> Self {
        Integrate {
            f: None,
            lower: None,
            upper: None,
            subdivisions: Some(1000),
            abs_tol: Some(1.49e-8),
            rel_tol: Some(1.49e-8),
        }
    }
}
