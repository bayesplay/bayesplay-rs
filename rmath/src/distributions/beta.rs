use statrs::distribution::{Continuous, ContinuousCDF};

pub fn lbeta(a: f64, b: f64) -> Result<f64, &'static str> {
    statrs::function::beta::checked_ln_beta(a, b).map_err(|_| "Error evaluating log beta function")
}

pub fn pbeta(
    q: f64,
    shape1: f64,
    shape2: f64,
    lower_tail: bool,
    log_p: bool,
) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Beta::new(shape1, shape2)
        .map_err(|_| "Error creating Beta distribution")?;

    let probability = if lower_tail { dist.cdf(q) } else { dist.sf(q) };
    Ok(if log_p { probability.ln() } else { probability })
}

pub fn dbeta(x: f64, shape1: f64, shape2: f64, log: bool) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Beta::new(shape1, shape2)
        .map_err(|_| "Error creating Beta distribution")?;
    match log {
        true => Ok(dist.ln_pdf(x)),
        false => Ok(dist.pdf(x)),
    }
}
