use crate::error::{Distribution, Error, Result, require_positive};
use statrs::distribution::{Continuous, ContinuousCDF};

/// `ln B(a, b)`, the log of the beta function.
pub fn lbeta(a: f64, b: f64) -> Result<f64> {
    require_positive(Distribution::Beta, "a", a)?;
    require_positive(Distribution::Beta, "b", b)?;
    Ok(statrs::function::beta::ln_beta(a, b))
}

fn beta(shape1: f64, shape2: f64) -> Result<statrs::distribution::Beta> {
    require_positive(Distribution::Beta, "shape1", shape1)?;
    require_positive(Distribution::Beta, "shape2", shape2)?;
    statrs::distribution::Beta::new(shape1, shape2).map_err(|_| Error::InvalidParameter {
        distribution: Distribution::Beta,
        parameter: "shape1, shape2".into(),
        value: shape1,
        requirement: "rejected by statrs".into(),
    })
}

pub fn pbeta(q: f64, shape1: f64, shape2: f64, lower_tail: bool, log_p: bool) -> Result<f64> {
    let dist = beta(shape1, shape2)?;

    let probability = if lower_tail { dist.cdf(q) } else { dist.sf(q) };
    Ok(if log_p { probability.ln() } else { probability })
}

pub fn dbeta(x: f64, shape1: f64, shape2: f64, log: bool) -> Result<f64> {
    let dist = beta(shape1, shape2)?;
    match log {
        true => Ok(dist.ln_pdf(x)),
        false => Ok(dist.pdf(x)),
    }
}
