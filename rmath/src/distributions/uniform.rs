use crate::error::{Distribution, Error, Result, require, require_finite};
use statrs::distribution::{Continuous, ContinuousCDF};

fn uniform(min: f64, max: f64) -> Result<statrs::distribution::Uniform> {
    require_finite(Distribution::Uniform, "min", min)?;
    require_finite(Distribution::Uniform, "max", max)?;
    require(
        max > min,
        Distribution::Uniform,
        "max",
        max,
        "must exceed min",
    )?;
    statrs::distribution::Uniform::new(min, max).map_err(|_| Error::InvalidParameter {
        distribution: Distribution::Uniform,
        parameter: "max".into(),
        value: max,
        requirement: "rejected by statrs".into(),
    })
}

pub fn punif(x: f64, min: f64, max: f64, lower_tail: bool, log_p: bool) -> Result<f64> {
    let dist = uniform(min, max)?;

    match (lower_tail, log_p) {
        (true, true) => Ok(dist.cdf(x).ln()),
        (true, false) => Ok(dist.cdf(x)),
        (false, true) => Ok((1.0 - dist.cdf(x)).ln()),
        (false, false) => Ok(1.0 - dist.cdf(x)),
    }
}

pub fn dunif(x: f64, min: f64, max: f64, log: bool) -> Result<f64> {
    let dist = uniform(min, max)?;

    match log {
        true => Ok(dist.ln_pdf(x)),
        false => Ok(dist.pdf(x)),
    }
}
