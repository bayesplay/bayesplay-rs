use gkquad::RuntimeError;
use gkquad::Tolerance;
use thiserror::Error;

use gkquad::single::Integrator;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
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
    #[error("Integration bounds or split points must not be NaN")]
    NanBounds,
    #[error("OtherError")]
    OtherError,
}

/// Adaptive Gauss-Kronrod integration of `f` over `[lower, upper]` (either
/// bound may be infinite; missing bounds default to -inf/+inf).
pub fn integratewrapper<F>(args: Integrate<F>) -> Result<IntegratorResult, IntegrationError>
where
    F: FnMut(f64) -> f64,
{
    let Integrate {
        f,
        lower,
        upper,
        subdivisions,
        abs_tol,
        rel_tol,
        tolerance,
        points,
    } = args;
    let f = f.expect("argument \"f\" is missing, with no default");
    let lower = lower.unwrap_or(f64::NEG_INFINITY);
    let upper = upper.unwrap_or(f64::INFINITY);
    let points = points.unwrap_or_default();
    // gkquad panics on NaN bounds or points; report them as errors instead.
    if lower.is_nan() || upper.is_nan() || points.iter().any(|p| p.is_nan()) {
        return Err(IntegrationError::NanBounds);
    }

    let mut integrator = Integrator::new(f)
        .max_iters(subdivisions)
        .tolerance(match tolerance {
            ToleranceMode::Both => Tolerance::AbsAndRel(abs_tol, rel_tol),
            ToleranceMode::Either => Tolerance::AbsOrRel(abs_tol, rel_tol),
        });
    if !points.is_empty() {
        integrator = integrator.points(&points);
    }

    match integrator.run(lower..upper).estimate_delta() {
        Ok((value, delta)) => Ok(IntegratorResult { value, delta }),
        Err(RuntimeError::InsufficientIteration) => Err(IntegrationError::InsufficientIteration),
        Err(RuntimeError::RoundoffError) => Err(IntegrationError::RoundoffError),
        Err(RuntimeError::SubrangeTooSmall) => Err(IntegrationError::SubrangeTooSmall),
        Err(RuntimeError::Divergent) => Err(IntegrationError::Divergent),
        Err(RuntimeError::NanValueEncountered) => Err(IntegrationError::NanValueEncountered),
        Err(_) => Err(IntegrationError::OtherError),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegratorResult {
    pub value: f64,
    pub delta: f64,
}

/// How `abs_tol` and `rel_tol` combine into the convergence criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToleranceMode {
    /// The error estimate must satisfy both tolerances (the stricter one wins).
    #[default]
    Both,
    /// The error estimate must satisfy either tolerance. Useful when a piece of
    /// a larger integral only needs to be accurate relative to the whole.
    Either,
}

/// Arguments for [`integratewrapper`], usually built with the `integrate!` macro.
pub struct Integrate<F>
where
    F: FnMut(f64) -> f64,
{
    pub f: Option<F>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    /// Maximum number of subintervals.
    pub subdivisions: usize,
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub tolerance: ToleranceMode,
    /// Optional split points to force the integrator to subdivide at specific locations.
    /// Useful when the integrand has a known peak far from the integration origin.
    pub points: Option<Vec<f64>>,
}

impl<F> Default for Integrate<F>
where
    F: FnMut(f64) -> f64,
{
    fn default() -> Self {
        Integrate {
            f: None,
            lower: None,
            upper: None,
            subdivisions: 1000,
            abs_tol: 1.49e-8,
            rel_tol: 1.49e-8,
            tolerance: ToleranceMode::Both,
            points: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_bounds_are_errors_not_panics() {
        for (lower, upper, points) in [
            (Some(f64::NAN), Some(1.0), None),
            (Some(0.0), Some(f64::NAN), None),
            (Some(0.0), Some(1.0), Some(vec![f64::NAN])),
        ] {
            let result = integratewrapper(Integrate {
                f: Some(|x: f64| x),
                lower,
                upper,
                points,
                ..Default::default()
            });
            assert_eq!(result, Err(IntegrationError::NanBounds));
        }
    }

    #[test]
    fn closures_may_borrow_local_state() {
        let mut calls = 0;
        let result = integratewrapper(Integrate {
            f: Some(|x: f64| {
                calls += 1;
                x * x
            }),
            lower: Some(0.0),
            upper: Some(1.0),
            ..Default::default()
        })
        .unwrap();
        assert!((result.value - 1.0 / 3.0).abs() < 1e-12);
        assert!(calls > 0);
    }
}
