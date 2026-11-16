use statrs::distribution::{Continuous, ContinuousCDF};

pub fn punif(
    x: f64,
    min: f64,
    max: f64,
    lower_tail: bool,
    log_p: bool,
) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Uniform::new(min, max)
        .map_err(|_| "Error creating Uniform distribution")?;

    match (lower_tail, log_p) {
        (true, true) => Ok(dist.cdf(x).ln()),
        (true, false) => Ok(dist.cdf(x)),
        (false, true) => Ok((1.0 - dist.cdf(x)).ln()),
        (false, false) => Ok(1.0 - dist.cdf(x)),
    }
}

pub fn dunif(x: f64, min: f64, max: f64, log: bool) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Uniform::new(min, max)
        .map_err(|_| "Error creating Uniform distribution")?;

    match log {
        true => Ok(dist.ln_pdf(x)),
        false => Ok(dist.pdf(x)),
    }
}
