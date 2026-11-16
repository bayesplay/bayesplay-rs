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
/// bound may be infinite) with the default [`Quadrature`] settings.
///
/// ```
/// let result = rmath::integrate(|x: f64| (-x * x / 2.0).exp(), f64::NEG_INFINITY, f64::INFINITY).unwrap();
/// assert!((result.value - (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-8);
/// ```
pub fn integrate<F>(f: F, lower: f64, upper: f64) -> Result<IntegratorResult, IntegrationError>
where
    F: FnMut(f64) -> f64,
{
    Quadrature {
        lower,
        upper,
        ..Default::default()
    }
    .integrate(f)
}

/// The estimate of an integral and its estimated absolute error.
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

/// Settings for an adaptive Gauss-Kronrod integration. The defaults match R's
/// `integrate()`: the whole real line, 1000 subdivisions and tolerances of
/// `1.49e-8`.
///
/// ```
/// use rmath::Quadrature;
///
/// let result = Quadrature { lower: 0.0, upper: 1.0, ..Default::default() }
///     .integrate(|x: f64| x * x)
///     .unwrap();
/// assert!((result.value - 1.0 / 3.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Quadrature {
    pub lower: f64,
    pub upper: f64,
    /// Maximum number of subintervals.
    pub subdivisions: usize,
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub tolerance: ToleranceMode,
    /// Split points that force the integrator to subdivide at specific
    /// locations, e.g. a known peak far from the integration origin.
    pub points: Vec<f64>,
}

impl Default for Quadrature {
    fn default() -> Self {
        Quadrature {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            subdivisions: 1000,
            abs_tol: 1.49e-8,
            rel_tol: 1.49e-8,
            tolerance: ToleranceMode::Both,
            points: Vec::new(),
        }
    }
}

impl Quadrature {
    /// Integrates `f` with these settings.
    pub fn integrate<F>(&self, f: F) -> Result<IntegratorResult, IntegrationError>
    where
        F: FnMut(f64) -> f64,
    {
        // gkquad panics on NaN bounds or points; report them as errors instead.
        if self.lower.is_nan() || self.upper.is_nan() || self.points.iter().any(|p| p.is_nan()) {
            return Err(IntegrationError::NanBounds);
        }

        let mut integrator = Integrator::new(f)
            .max_iters(self.subdivisions)
            .tolerance(match self.tolerance {
                ToleranceMode::Both => Tolerance::AbsAndRel(self.abs_tol, self.rel_tol),
                ToleranceMode::Either => Tolerance::AbsOrRel(self.abs_tol, self.rel_tol),
            });
        if !self.points.is_empty() {
            integrator = integrator.points(&self.points);
        }

        match integrator.run(self.lower..self.upper).estimate_delta() {
            Ok((value, delta)) => Ok(IntegratorResult { value, delta }),
            Err(RuntimeError::InsufficientIteration) => {
                Err(IntegrationError::InsufficientIteration)
            }
            Err(RuntimeError::RoundoffError) => Err(IntegrationError::RoundoffError),
            Err(RuntimeError::SubrangeTooSmall) => Err(IntegrationError::SubrangeTooSmall),
            Err(RuntimeError::Divergent) => Err(IntegrationError::Divergent),
            Err(RuntimeError::NanValueEncountered) => Err(IntegrationError::NanValueEncountered),
            Err(_) => Err(IntegrationError::OtherError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_bounds_are_errors_not_panics() {
        for (lower, upper, points) in [
            (f64::NAN, 1.0, vec![]),
            (0.0, f64::NAN, vec![]),
            (0.0, 1.0, vec![f64::NAN]),
        ] {
            let result = Quadrature {
                lower,
                upper,
                points,
                ..Default::default()
            }
            .integrate(|x: f64| x);
            assert_eq!(result, Err(IntegrationError::NanBounds));
        }
    }

    #[test]
    fn closures_may_borrow_local_state() {
        let mut calls = 0;
        let result = integrate(
            |x: f64| {
                calls += 1;
                x * x
            },
            0.0,
            1.0,
        )
        .unwrap();
        assert!((result.value - 1.0 / 3.0).abs() < 1e-12);
        assert!(calls > 0);
    }
}
